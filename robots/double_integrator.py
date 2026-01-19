import jax
import jax.numpy as jnp
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

# Dimensions
DOFS = 3 # x, y, z
STATE_DIM = 6 # x,y,z, vx,vy,vz
ACTION_DIM = 3 # ax, ay, az

# Limits
# State: x,y,z, vx,vy,vz
# Limits for NON-POS: vx,vy,vz
NOPOS_STATE_MIN = jnp.array([-30.0, -30.0, -30.0])
NOPOS_STATE_MAX  = jnp.array([30.0, 30.0, 30.0])

# Action
# Ref A: -0.2 to 0.2 (for W=1).
ACTION_MIN = jnp.array([-0.1, -0.1, -0.1])
ACTION_MAX = jnp.array([0.1, 0.1, 0.1])

def forward(params, state, action, dt):
    pos = state[:3]
    vel = state[3:]
    acc = action
    
    # Exact integration for constant acceleration
    new_pos = pos + vel * dt + 0.5 * acc * dt * dt
    new_vel = vel + acc * dt
    
    # Clamp velocity
    min_v = params.state_min[3:]
    max_v = params.state_max[3:]
    new_vel = jnp.clip(new_vel, min_v, max_v)
    
    return jnp.concatenate([new_pos, new_vel])

def collide(params, old_state, new_state):
    old_pos = old_state[:DOFS]
    new_pos = new_state[:DOFS]
    
    # Only check new_state, as old_state is assumed valid from the last collide() check
    # and the base case (start and goal state) are assumed valid
    bounds_coll = jnp.any((new_state < params.state_min) | (new_state > params.state_max))

    is_free = params.collide_fn(old_pos, new_pos, params.obs_data)
    
    return bounds_coll | (~is_free)

def metric(states, goal_states):
    pos_diff = states[..., :DOFS] - goal_states[..., :DOFS]
    vel_diff = states[..., DOFS:] - goal_states[..., DOFS:]
    return jnp.linalg.norm(pos_diff, axis=-1) + 0.1 * jnp.linalg.norm(vel_diff, axis=-1)

def draw_robot(vis, state, color=0x0000ff, name="double_integrator"):
    pos = state[:3]
    
    name = f"robots/{name}"
    vis[name].set_object(g.Sphere(0.1), g.MeshLambertMaterial(color=color))
    vis[name].set_transform(tf.translation_matrix(pos))
