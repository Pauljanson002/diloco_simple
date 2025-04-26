import jax
import jax.numpy as jnp
from jax import random, pmap, local_device_count
import optax
from functools import partial
import numpy as np

# Enable more verbose JAX logging
jax.config.update("jax_log_compiles", True)


def print_device_values(prefix, values):
    """Print values across devices with labels"""
    if isinstance(values, jax.Array) and values.sharding.device_set:
        for i, device in enumerate(values.sharding.device_set):
            device_str = str(device).split("/")[-1]  # Get just the device ID part
            print(f"{prefix} on {device_str}: {jax.device_get(values[i])}")
    else:
        print(f"{prefix} (not sharded): {values}")


# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")
num_devices = len(devices)
print(f"Number of devices: {num_devices}")


# Create a simple MLP model
def init_model_params(key, input_dim=8, hidden_dim=32, output_dim=1):
    """Initialize model parameters"""
    k1, k2 = random.split(key)
    w1 = random.normal(k1, (input_dim, hidden_dim))
    w2 = random.normal(k2, (hidden_dim, output_dim))
    return {"w1": w1, "w2": w2}


# Forward pass for our simple MLP
def forward(params, x):
    """Simple 2-layer MLP forward pass"""
    hidden = jnp.dot(x, params["w1"])
    hidden = jnp.tanh(hidden)
    output = jnp.dot(hidden, params["w2"])
    return output


# Loss function
def loss_fn(params, x, y):
    """Mean squared error loss"""
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)


# Create a training step function that will be pmapped
@partial(pmap, axis_name="batch")
def train_step(params, opt_state, x, y, step_size):
    """Single training step with extensive printing"""
    # Print inputs on each device
    batch_idx = jax.lax.axis_index("batch")
    # Use host_id() to identify the process
    host_id = jax.process_index()

    # Print input data for each device
    x_mean = jnp.mean(x)
    jax.debug.print("Device {d}: Step data mean: {m}", d=batch_idx, m=x_mean)

    # Compute loss and gradients
    loss_val, grads = jax.value_and_grad(loss_fn)(params, x, y)

    # Print loss for each device
    jax.debug.print("Device {d}: Loss: {l}", d=batch_idx, l=loss_val)

    # Calculate mean gradients across devices using psum
    grads = jax.lax.pmean(grads, axis_name="batch")

    # Print averaged gradients
    jax.debug.print(
        "Device {d}: Mean grad w1 norm: {g}",
        d=batch_idx,
        g=jnp.linalg.norm(grads["w1"]),
    )

    # Update parameters using optax
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Print parameter update magnitude
    param_change = jnp.linalg.norm(jax.tree_util.tree_leaves(updates)[0])
    jax.debug.print(
        "Device {d}: Parameter update magnitude: {p}", d=batch_idx, p=param_change
    )

    return new_params, new_opt_state, loss_val


# Generate some random data
key = random.key(42)
key, subkey = random.split(key)

# Create data and split across devices
batch_size_per_device = 16
total_batch_size = num_devices * batch_size_per_device
input_dim = 8

x = random.normal(key, (total_batch_size, input_dim))
key, subkey = random.split(key)
true_w = random.normal(key, (input_dim, 1))
y = jnp.dot(x, true_w) + 0.1 * random.normal(subkey, (total_batch_size, 1))

# Reshape data for pmap (devices, batch_per_device, ...)
x_split = x.reshape(num_devices, batch_size_per_device, input_dim)
y_split = y.reshape(num_devices, batch_size_per_device, 1)

print(f"Data shapes after splitting: x={x_split.shape}, y={y_split.shape}")

# Initialize model on each device
key, subkey = random.split(key)
init_keys = random.split(subkey, num_devices)
init_params_pmap = pmap(init_model_params)(init_keys)

# Print initial parameters on each device
print("\n=== Initial Parameters ===")
for i, device in enumerate(devices):
    print(f"Device {i} params:")
    print(f"  w1 shape: {init_params_pmap['w1'][i].shape}")
    print(f"  w1 mean: {jnp.mean(init_params_pmap['w1'][i])}")
    print(f"  w2 shape: {init_params_pmap['w2'][i].shape}")
    print(f"  w2 mean: {jnp.mean(init_params_pmap['w2'][i])}")

# Create optimizer
learning_rate = 0.01
# Create a replicated learning rate for each device
learning_rate_replicated = jnp.ones((num_devices,)) * learning_rate
optimizer = optax.adam(learning_rate)
opt_state = pmap(optimizer.init)(init_params_pmap)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # For demonstration, recreate random data each epoch
    key, subkey = random.split(key)
    x = random.normal(key, (total_batch_size, input_dim))
    key, subkey = random.split(key)
    y = jnp.dot(x, true_w) + 0.1 * random.normal(subkey, (total_batch_size, 1))

    # Reshape for pmap
    x_split = x.reshape(num_devices, batch_size_per_device, input_dim)
    y_split = y.reshape(num_devices, batch_size_per_device, 1)

    # Perform training step
    init_params_pmap, opt_state, losses = train_step(
        init_params_pmap, opt_state, x_split, y_split, learning_rate_replicated
    )

    # Print epoch summary (gather losses from all devices)
    mean_loss = jnp.mean(losses)
    print(f"\nEpoch {epoch+1}/{num_epochs}, Mean Loss: {mean_loss}")

    # Print parameter stats across devices for every few epochs
    if epoch % 3 == 0 or epoch == num_epochs - 1:
        print(f"=== Parameters after epoch {epoch+1} ===")
        for i, device in enumerate(devices):
            print(f"Device {i} params:")
            print(f"  w1 mean: {jnp.mean(init_params_pmap['w1'][i])}")
            print(f"  w1 norm: {jnp.linalg.norm(init_params_pmap['w1'][i])}")
            print(f"  w2 mean: {jnp.mean(init_params_pmap['w2'][i])}")
            print(f"  w2 norm: {jnp.linalg.norm(init_params_pmap['w2'][i])}")


# Final evaluation
@pmap
def evaluate(params, x, y):
    pred = forward(params, x)
    mse = jnp.mean((pred - y) ** 2)
    return mse


final_loss = evaluate(init_params_pmap, x_split, y_split)
print("\n=== Final Evaluation ===")
for i, device in enumerate(devices):
    print(f"Device {i} final loss: {final_loss[i]}")

print("\n=== Final Parameters ===")
# Compare parameters across devices to verify they're identical after pmap training
w1_device0 = init_params_pmap["w1"][0]
w2_device0 = init_params_pmap["w2"][0]

for i in range(1, num_devices):
    w1_match = jnp.allclose(w1_device0, init_params_pmap["w1"][i])
    w2_match = jnp.allclose(w2_device0, init_params_pmap["w2"][i])
    print(f"Device {i} parameters match device 0: w1={w1_match}, w2={w2_match}")

print("\nTraining complete!")
