from learned_optimization.research.general_lopt import prefab
import optax
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
# defining an optimizer that targets 1000 training steps
NUM_STEPS = 1000
# opt = prefab.optax_lopt(NUM_STEPS)

# print("Optimizer:", opt)
# print("type:", type(opt))

# exit(0)

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tqdm
from flax.training import train_state
import jax.random as random


class LinenModel(nn.Module):
    dmid: int
    dout: int

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Dense(features=self.dmid)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.Dense(features=self.dout)(x)
        return x


class TrainState(train_state.TrainState):
    """Custom train state that includes batch statistics for BatchNorm."""

    batch_stats: any
    
    def apply_gradients(self, *, grads, **kwargs):
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, extra_args={'loss': kwargs.pop('loss')}
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
            'params': new_params_with_opt,
            OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def create_train_state(rng, input_shape, model, tx):
    """Creates initial `TrainState` with model parameters and optimizer."""
    params_rng, dropout_rng = jax.random.split(rng)

    # Initialize parameters
    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng}, jnp.ones(input_shape)
    )

    # Separate parameters and batch_stats
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # Create TrainState
    return TrainState.create(
        apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx
    )


@jax.jit
def train_step(state, x, y, dropout_rng):
    """Train for a single step."""
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        variables = {"params": params, "batch_stats": state.batch_stats}
        y_pred, new_model_state = state.apply_fn(
            variables,
            x,
            training=True,
            rngs={"dropout": dropout_rng},
            mutable=["batch_stats"],
        )
        loss = jnp.mean((y_pred - y) ** 2)
        return loss, new_model_state

    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )

    # Update state with new parameters and batch statistics
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"],loss=loss
    )

    return new_state, loss


@jax.jit
def eval_step(state, x, y):
    """Evaluate for a single step."""
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    y_pred = state.apply_fn(variables, x, training=False)
    loss = jnp.mean((y_pred - y) ** 2)
    return loss


def main():
    # Model parameters
    din, dmid, dout = 2, 64, 3

    # Initialize model
    model = LinenModel(dmid=dmid, dout=dout)

    # Initialize optimizer
    tx = optax.adam(learning_rate=1e-3)
    opt = prefab.optax_lopt(NUM_STEPS)
    tx = opt
    print(f"Optimizer: {tx}")
    print(f"Optimizer type: {type(tx)}")
    # Initialize random key
    rng = jax.random.PRNGKey(0)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    # Create train state
    state = create_train_state(init_rng, (1, din), model, tx)

    # Generate synthetic data
    num_samples = 1000

    # Input features: random points in 2D space
    data_rng, subkey = jax.random.split(data_rng)
    x = jax.random.uniform(subkey, shape=(num_samples, din), minval=-5, maxval=5)

    # Target values: a quadratic function of the inputs
    # y = a*x1^2 + b*x2^2 + c*x1*x2 + d*x1 + e*x2 + f + noise
    a, b, c, d, e, f = 1.0, 0.5, 0.3, -2.0, 1.5, 0.5  # Coefficients

    # Compute y values using the quadratic function
    y = (
        a * x[:, 0] ** 2
        + b * x[:, 1] ** 2
        + c * x[:, 0] * x[:, 1]
        + d * x[:, 0]
        + e * x[:, 1]
        + f
    )

    # Reshape to match expected output dimension (num_samples, 3)
    y = jnp.tile(y[:, jnp.newaxis], (1, dout))

    # Add some noise to make it more realistic
    data_rng, subkey = jax.random.split(data_rng)
    noise = 0.1 * jax.random.normal(subkey, shape=y.shape)
    y = y + noise

    print(f"Generated {num_samples} samples from a quadratic function")
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # Training parameters
    batch_size = 32
    num_epochs = 500
    dropout_rng = jax.random.PRNGKey(1)

    # Training loop
    for epoch in range(num_epochs):
        # Shuffle data
        data_rng, subkey = jax.random.split(data_rng)
        indices = jax.random.permutation(subkey, num_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Batch processing
        total_loss = 0.0
        num_batches = num_samples // batch_size

        for i in tqdm.tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = x_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            state, loss = train_step(state, batch_x, batch_y, dropout_rng)
            total_loss += loss

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
