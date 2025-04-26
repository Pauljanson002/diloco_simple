import os
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
from flax.training import train_state
from flax import core, struct
from flax.training.common_utils import shard_prng_key, shard
from functools import partial
from cyclopts import App
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    FlaxLlamaForCausalLM,
    LlamaConfig,
    DataCollatorForLanguageModeling,
)

app = App()


def create_learning_rate_scheduler(lr, warmup_steps, total_steps, cosine_decay=True):
    """Creates learning rate schedule."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=lr, transition_steps=warmup_steps
    )

    if cosine_decay:
        decay_fn = optax.cosine_decay_schedule(
            init_value=lr, decay_steps=total_steps - warmup_steps
        )
    else:
        decay_fn = optax.linear_schedule(
            init_value=lr, end_value=0, transition_steps=total_steps - warmup_steps
        )

    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
    )

    return schedule_fn


def create_adamw_optimizer(learning_rate_schedule, weight_decay, beta1=0.9, beta2=0.95):
    """Create AdamW optimizer with specified parameters."""
    optimizer = optax.adamw(
        learning_rate=learning_rate_schedule,
        b1=beta1,
        b2=beta2,
        weight_decay=weight_decay,
        mask=lambda p: jax.tree_map(lambda x: x.ndim > 1, p),
    )
    return optimizer


def create_sgd_optimizer(learning_rate_schedule, momentum=0.9, nesterov=True):
    """Create SGD optimizer with Nesterov momentum for outer optimization."""
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule,
        momentum=momentum,
        nesterov=nesterov,
    )
    return optimizer


class LowCommTrainState(train_state.TrainState):
    """Custom train state that includes both inner and outer optimizers."""

    outer_params: flax.core.FrozenDict
    outer_opt_state: optax.OptState  # Store outer optimizer state
    outer_tx: optax.GradientTransformation = struct.field(
        pytree_node=False
    )  # Mark as static
    inner_steps_counter: int
    inner_steps_max: int


def create_low_comm_train_state(
    model, inner_optimizer, outer_optimizer, inner_steps_max, params=None
):
    """Creates initial `LowCommTrainState` for model with inner and outer optimizers."""
    if params is None:
        params = model.params

    # Initialize outer optimizer state
    outer_opt_state = outer_optimizer.init(params)

    print(type(outer_opt_state))
    print(type(outer_optimizer))
    # Initialize with both params the same
    return LowCommTrainState.create(
        apply_fn=model.__call__,
        params=params,  # Current parameters used by inner optimizer
        tx=inner_optimizer,  # Inner optimizer
        outer_params=params,  # Last globally synced parameters
        outer_opt_state=outer_opt_state,  # Outer optimizer state
        outer_tx=outer_optimizer,  # Store outer optimizer
        inner_steps_counter=0,  # Counter for inner optimization steps
        inner_steps_max=inner_steps_max,  # Max steps before sync
    )


def compute_metrics(logits, labels, padding_mask):
    """Compute metrics for training."""
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels,
    )

    # Apply padding mask
    loss = loss * padding_mask

    # Calculate mean loss over non-padding tokens
    loss = jnp.sum(loss) / jnp.sum(padding_mask)

    # Calculate perplexity
    perplexity = jnp.exp(loss)

    return {"loss": loss, "perplexity": perplexity}


def create_padding_mask(targets):
    """Create a padding mask for the targets."""
    return jnp.where(targets > 0, 1.0, 0.0)


def train_step_single_device(state, batch, dropout_rng):
    """Train for a single step on a single device with low communication."""
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        batch_copy = dict(
            batch
        )  # Make a copy of the batch to avoid modifying the original
        labels = batch_copy.pop("labels")
        padding_mask = create_padding_mask(labels)

        logits = state.apply_fn(
            **batch_copy,
            params=params,
            dropout_rng=dropout_rng,
            train=True,
        )[0]

        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]
        shift_padding_mask = padding_mask[:, 1:]

        metrics = compute_metrics(shift_logits, shift_labels, shift_padding_mask)
        return metrics["loss"], metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)

    # Clip gradients by global norm
    grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)

    # Update state with gradients (without pmean synchronization)
    new_state = state.apply_gradients(grads=grads)

    # Increment the inner steps counter
    new_inner_steps_counter = state.inner_steps_counter + 1

    # Check if we need to synchronize with global parameters
    needs_sync = new_inner_steps_counter >= state.inner_steps_max

    # Define the two branches for JAX's conditional
    def sync_branch(args):
        new_state, state = args
        # Calculate parameter differences between current and last global sync
        param_diff = jax.tree_map(
            lambda inner, outer: inner - outer, new_state.params, state.outer_params
        )

        # Use pmean to average the parameter differences across devices
        param_diff_mean = jax.lax.pmean(param_diff, axis_name="batch")

        # Create updates for outer optimizer using the maintained outer optimizer state
        outer_updates, new_outer_opt_state = state.outer_tx.update(
            param_diff_mean, state.outer_opt_state
        )

        # Apply the updates to the outer parameters
        new_outer_params = optax.apply_updates(state.outer_params, outer_updates)

        # Reset inner state to use the new global parameters
        return new_state.replace(
            params=new_outer_params,
            outer_params=new_outer_params,
            outer_opt_state=new_outer_opt_state,  # Update the outer optimizer state
            inner_steps_counter=0,  # Reset counter after sync
        )

    def no_sync_branch(args):
        new_state, _ = args
        # Just update the counter if no sync is needed
        return new_state.replace(inner_steps_counter=new_inner_steps_counter)

    # Use JAX's conditional instead of Python if
    new_state = jax.lax.cond(
        needs_sync, sync_branch, no_sync_branch, (new_state, state)
    )

    return new_state, metrics, dropout_rng, needs_sync


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def train_step(state, batch, dropout_rng):
    """Train for a single step across multiple devices with low communication."""
    new_state, metrics, dropout_rng, needs_sync = train_step_single_device(
        state, batch, dropout_rng
    )

    # All-reduce metrics across devices
    metrics = jax.lax.pmean(metrics, axis_name="batch")

    # Synchronize the needs_sync flag across devices with psum - all devices sync if any need to sync
    # Using psum to count how many devices need sync, then checking if any do
    needs_sync = jax.lax.pmin(needs_sync, axis_name="batch")

    return new_state, metrics, dropout_rng, needs_sync


def prepare_batch_for_jax(batch, devices=None):
    """Convert a batch of PyTorch tensors to JAX arrays and shard across devices."""
    jax_batch = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            jax_batch[k] = jnp.array(v)
        else:
            jax_batch[k] = jnp.array(v.numpy())

    # If devices are provided, shard the batch across them
    if devices is not None:
        jax_batch = shard(jax_batch)

    return jax_batch


@app.default
def main(
    batch_size: int = 128,
    per_device_batch_size: int = 64,
    seq_length: int = 1024,
    warmup_steps: int = 1000,
    total_steps: int = 88_000,
    config_path: str = "config_14m.json",
    lr: float = 4e-4,
    weight_decay: float = 0.1,
    inner_steps: int = 10,  # Number of inner steps before synchronization
    outer_lr: float = 1e-3,  # Learning rate for outer optimizer
    outer_momentum: float = 0.9,  # Momentum for outer optimizer
):
    # Calculate gradient accumulation steps and device batch size
    num_devices = jax.device_count()
    device_batch_size = per_device_batch_size
    total_batch_size = device_batch_size * num_devices
    gradient_accumulation_steps = batch_size // total_batch_size

    if batch_size % total_batch_size != 0:
        print(
            f"Warning: batch_size {batch_size} is not divisible by {total_batch_size} (device_batch_size * num_devices)"
        )
        batch_size = gradient_accumulation_steps * total_batch_size
        print(f"Adjusting batch_size to {batch_size}")

    # Print device info
    print(f"Number of devices: {num_devices}")
    print(f"Device type: {jax.devices()[0].platform}")
    print(f"Per-device batch size: {device_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Total batch size: {batch_size}")
    print(f"Inner steps before synchronization: {inner_steps}")

    # Initialize random keys
    rng = jax.random.PRNGKey(0)
    rng, dropout_rng, init_rng = jax.random.split(rng, 3)

    # Create a separate dropout key for each device
    dropout_rng = jax.random.split(dropout_rng, num_devices)

    # Load model configuration
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=config_path)

    # Initialize model
    model = FlaxLlamaForCausalLM(config)
    print(
        f"Model loaded with {sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(model.params)):,} parameters"
    )

    # Setup learning rate schedulers
    inner_lr_schedule = create_learning_rate_scheduler(
        lr=lr, warmup_steps=warmup_steps, total_steps=total_steps, cosine_decay=True
    )

    outer_lr_schedule = create_learning_rate_scheduler(
        lr=outer_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps // inner_steps,
        cosine_decay=True,
    )

    # Setup optimizers
    inner_optimizer = create_adamw_optimizer(
        learning_rate_schedule=inner_lr_schedule,
        weight_decay=weight_decay,
        beta1=0.9,
        beta2=0.95,
    )

    outer_optimizer = create_sgd_optimizer(
        learning_rate_schedule=outer_lr_schedule,
        momentum=outer_momentum,
        nesterov=True,
    )

    # Create training state and replicate across devices
    state = create_low_comm_train_state(
        model, inner_optimizer, outer_optimizer, inner_steps
    )
    state = flax.jax_utils.replicate(state)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1", use_fast=True
    )
    tokenizer.pad_token = "</s>"  # Ensure pad token is set

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("PrimeIntellect/c4-tiny", "en", verification_mode="no_checks")

    # Tokenize dataset
    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_length)
        return outputs

    print("Tokenizing dataset...")
    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]
    )

    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create data iterator that produces batches for all devices
    def data_generator():
        train_dataset = tokenized_datasets["train"]
        indices = np.random.permutation(len(train_dataset))

        # Create batches for all devices
        # Each global batch needs to have (device_batch_size * num_devices) examples
        global_batch_size = device_batch_size * num_devices

        for i in range(0, len(indices), global_batch_size):
            if i + global_batch_size > len(indices):
                # Skip incomplete batches at the end
                continue

            batch_indices = indices[i : i + global_batch_size]
            examples = [train_dataset[int(idx)] for idx in batch_indices]
            batch = data_collator(examples)

            # Reshape batch to match device count
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    # Reshape (global_batch_size, ...) -> (num_devices, device_batch_size, ...)
                    batch[k] = v.reshape(num_devices, device_batch_size, *v.shape[1:])
                else:
                    # Convert PyTorch tensor to numpy and reshape
                    array = v.numpy()
                    batch[k] = array.reshape(
                        num_devices, device_batch_size, *array.shape[1:]
                    )

            yield batch

    train_iter = data_generator()

    print(f"Starting training with effective batch size of {batch_size}")

    # Training loop
    global_step = 0
    accumulated_metrics = {"loss": 0.0, "perplexity": 0.0}
    sync_count = 0

    while global_step < total_steps:
        # Accumulate gradients over multiple steps
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = data_generator()
                batch = next(train_iter)

            # No need to prepare_batch_for_jax as our data generator already provides the right shapes
            # We just need to create JAX arrays
            jax_batch = {k: jnp.array(v) for k, v in batch.items()}

            # Train step across all devices
            state, metrics, dropout_rng, did_sync = train_step(
                state, jax_batch, dropout_rng
            )

            # Extract metrics from first device (they're the same across all devices due to pmean)
            metrics = {k: float(v[0]) for k, v in metrics.items()}

            # Accumulate metrics
            for key in accumulated_metrics:
                accumulated_metrics[key] += metrics[key] / gradient_accumulation_steps

            # Log if sync happened
            # Only print sync message from the first device (host_id 0)
            if did_sync.all() and jax.process_index() == 0:
                sync_count += 1
                print(
                    f"Global synchronization completed (sync #{sync_count}) at step {global_step}"
                )

        global_step += 1

        # Log metrics
        # Get current learning rate from first device
        current_lr = inner_lr_schedule(int(jax.device_get(state.step[0])))
        print(
            f"Step: {global_step}, "
            f"Loss: {accumulated_metrics['loss']:.4f}, "
            f"Perplexity: {accumulated_metrics['perplexity']:.2f}, "
            f"LR: {current_lr:.6f}"
        )

        # Reset accumulated metrics
        accumulated_metrics = {"loss": 0.0, "perplexity": 0.0}

        # Save checkpoint periodically
        if global_step % 1000 == 0:
            # Save model checkpoint - unreplicate from first device
            params = jax.device_get(flax.jax_utils.unreplicate(state).params)
            model_output_dir = f"./checkpoints/step_{global_step}"
            os.makedirs(model_output_dir, exist_ok=True)
            model.save_pretrained(model_output_dir, params=params)
            tokenizer.save_pretrained(model_output_dir)
            print(f"Saved checkpoint at step {global_step}")

    print("Training completed.")

    # Save final model - unreplicate from first device
    params = jax.device_get(flax.jax_utils.unreplicate(state).params)
    model_output_dir = "./checkpoints/final"
    os.makedirs(model_output_dir, exist_ok=True)
    model.save_pretrained(model_output_dir, params=params)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Saved final model to {model_output_dir}")


if __name__ == "__main__":
    app()
