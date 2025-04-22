import os
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
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


def create_train_state(model, optimizer, params=None):
    """Creates initial `TrainState` for model."""
    if params is None:
        params = model.params

    return train_state.TrainState.create(
        apply_fn=model.__call__,
        params=params,
        tx=optimizer,
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


@jax.jit
def train_step(state, batch, dropout_rng):
    """Train for a single step."""
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        labels = batch.pop("labels")
        padding_mask = create_padding_mask(labels)

        logits = state.apply_fn(
            **batch,
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

    # Update state with gradients
    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics, dropout_rng


def prepare_batch_for_jax(batch, device):
    """Convert a batch of PyTorch tensors to JAX arrays."""
    jax_batch = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            jax_batch[k] = jnp.array(v)
        else:
            jax_batch[k] = jnp.array(v.numpy())

    return jax_batch


@app.default
def main(
    batch_size: int = 512,
    per_device_batch_size: int = 32,
    seq_length: int = 1024,
    warmup_steps: int = 1000,
    total_steps: int = 88_000,
    config_path: str = "config_14m.json",
    lr: float = 4e-4,
    weight_decay: float = 0.1,
):
    # Calculate gradient accumulation steps
    gradient_accumulation_steps = batch_size // per_device_batch_size

    # Print device info
    print(f"Number of devices: {jax.device_count()}")
    print(f"Device type: {jax.devices()[0].platform}")

    # Initialize random keys
    rng = jax.random.PRNGKey(0)
    rng, dropout_rng, init_rng = jax.random.split(rng, 3)

    # Load model configuration
    config = LlamaConfig.from_pretrained(pretrained_model_name_or_path=config_path)

    # Initialize model
    model = FlaxLlamaForCausalLM(config)
    print(
        f"Model loaded with {sum(np.prod(p.shape) for p in jax.tree_util.tree_map(model.params)):,} parameters"
    )

    # Setup learning rate scheduler
    lr_schedule = create_learning_rate_scheduler(
        lr=lr, warmup_steps=warmup_steps, total_steps=total_steps, cosine_decay=True
    )

    # Setup optimizer
    optimizer = create_adamw_optimizer(
        learning_rate_schedule=lr_schedule,
        weight_decay=weight_decay,
        beta1=0.9,
        beta2=0.95,
    )

    # Create training state
    state = create_train_state(model, optimizer)

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

    # Create data iterator
    def data_generator():
        train_dataset = tokenized_datasets["train"]
        indices = np.random.permutation(len(train_dataset))
        
        batch_indices = [
            indices[i : i + per_device_batch_size]
            for i in range(0, len(indices), per_device_batch_size)
        ]

        for batch_idx in batch_indices:
            batch = [train_dataset[int(idx)] for idx in batch_idx]
            batch = data_collator(batch)
            jax_batch = prepare_batch_for_jax(batch, device=None)
            yield jax_batch

    train_iter = data_generator()

    print(f"Starting training with effective batch size of {batch_size}")

    # Training loop
    global_step = 0
    accumulated_metrics = {"loss": 0.0, "perplexity": 0.0}

    while global_step < total_steps:
        # Accumulate gradients over multiple steps
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = data_generator()
                batch = next(train_iter)

            state, metrics, dropout_rng = train_step(state, batch, dropout_rng)

            # Accumulate metrics
            for key in accumulated_metrics:
                accumulated_metrics[key] += metrics[key] / gradient_accumulation_steps

        global_step += 1

        # Log metrics
        current_lr = lr_schedule(state.step)
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
            # Save model checkpoint
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model_output_dir = f"./checkpoints/step_{global_step}"
            os.makedirs(model_output_dir, exist_ok=True)
            model.save_pretrained(model_output_dir, params=params)
            tokenizer.save_pretrained(model_output_dir)
            print(f"Saved checkpoint at step {global_step}")

    print("Training completed.")

    # Save final model
    params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
    model_output_dir = "./checkpoints/final"
    os.makedirs(model_output_dir, exist_ok=True)
    model.save_pretrained(model_output_dir, params=params)
    tokenizer.save_pretrained(model_output_dir)
    print(f"Saved final model to {model_output_dir}")


if __name__ == "__main__":
    app()
