"""
jax
import jax.numpy as jnp
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

# Dimensions
DOFS = 2 # x, y
STATE_DIM = 4 # x, y, theta, v
ACTION_DIM = 2 # a, steering

# Constants (from Config)
UNI_LENGTH = 1.0
W_SIZE = 100.0

# Limits
# State: x, y, theta, v
# Limits for NON-POS: theta, v
NOPOS_STATE_MIN = jnp.array([-jnp.pi, -0.3])
NOPOS_STATE_MAX = jnp.array([jnp.pi, 0.3])

# Action: a, steering
ACTION_MIN = jnp.array([-0.2, -jnp.pi/2])
ACTION_MAX = jnp.array([0.2, jnp.pi/2])

def forward(params, state, action, dt):
    x, y, theta, v = state
    a, steering = action
    
    # Dynamics
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    tan_steering = jnp.tan(steering)
    
    next_x = x + v * cos_theta * dt
    next_y = y + v * sin_theta * dt
    next_theta = theta + (v / UNI_LENGTH) * tan_steering * dt
    next_v = v + a * dt
    
    # Wrap theta?
    next_theta = (next_theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    return jnp.array([next_x, next_y, next_theta, next_v])

def collide(params, old_state, new_state):
    # state: (4,)
    old_pos = old_state[:DOFS] # x, y
    new_pos = new_state[:DOFS] # x, y
    
    # Bounds check (using STATE_MIN/MAX)
    # Check x, y, v against limits?
    # Usually collision only checks position bounds.
    # But let's check basic bounds.
    bounds_coll = jnp.any((new_state < params.state_min) | (new_state > params.state_max))

    # Obstacle check
    is_free = params.collide_fn(old_pos, new_pos, params.obs_data)
    
    return bounds_coll | (~is_free)

def metric(states, goal_states):
    # Distance in x,y + some weight on theta, v?
    # Point mass metric: pos_diff + 0.1 * vel_diff
    # Here: pos (2), theta(1), v(1).
    # Simple Euclidean on pos?
    
    pos_diff = states[..., :2] - goal_states[..., :2]
    # ignoring theta/v for metric usually works for RRT in position space, 
    # but for Kino-RRT we need full state distance often.
    # Let's use weighted.
    
    rest_diff = states[..., 2:] - goal_states[..., 2:]
    
    return jnp.linalg.norm(pos_diff, axis=-1) + 0.1 * jnp.linalg.norm(rest_diff, axis=-1)

def draw_robot(vis, state):
    x, y, theta = state[0], state[1], state[2]
    
    name = "robots/unicycle"
    
    # Clean up old? meshcat handles replacements.
    
    # Box for body
    vis[name]["body"].set_object(g.Box([0.5, 0.2, 0.1]), g.MeshLambertMaterial(color=0x0000ff))
    
    # Transform
    # Rotate by theta around Z
    mat = tf.translation_matrix([x, y, 0.1]).dot(tf.rotation_matrix(theta, [0, 0, 1]))
    vis[name].set_transform(mat)
"""