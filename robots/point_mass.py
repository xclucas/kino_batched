import jax
import jax.numpy as jnp
import meshcat.geometry as g
import meshcat.transformations as tf

# Dimensions
DOFS = 3
STATE_DIM = 6 # x, y, z, vx, vy, vz
ACTION_DIM = 3 # ax, ay, az

# Limits for position x, y, z
NOPOS_STATE_MIN = jnp.array([-2.0, -2.0, -2.0])
NOPOS_STATE_MAX = jnp.array([2.0, 2.0, 2.0])
ACTION_MIN = jnp.array([-10.0, -10.0, -10.0])
ACTION_MAX = jnp.array([10.0, 10.0, 10.0])

def forward(params, state, action, dt):
    pos = state[:DOFS]
    vel = state[DOFS:]
    
    # Simple Euler integration
    new_vel = vel + action * dt
    new_pos = pos + new_vel * dt
    
    # Clamp velocity
    min_v = params.state_min[DOFS:]
    max_v = params.state_max[DOFS:]
    
    new_vel = jnp.clip(new_vel, min_v, max_v)

    return jnp.concatenate([new_pos, new_vel])

def collide(params, state):
    # state: (4,) [x, y, vx, vy]
    pos = state[:DOFS]
    
    # Bounds check
    bounds_coll = jnp.any((state < params.state_min) | (state > params.state_max))
    
    # Obstacle check
    # params.obs_data is expected to be array of obstacles.
    # We assume obstacles are [min_x, min_y, min_z, max_x, max_y, max_z]
    
    # check if point is inside any obstacle (broad phase valid means separated)
    # so if valid -> no collision
    is_free = params.collide_fn(pos, params.obs_data)
    
    return bounds_coll | (~is_free)

def metric(states, goal_states):
    '''Compute the distance metric between states and goal_states'''
    pos_diff = states[..., :DOFS] - goal_states[..., :DOFS]
    vel_diff = states[..., DOFS:] - goal_states[..., DOFS:]
    return jnp.linalg.norm(pos_diff, axis=-1) + 0.1 * jnp.linalg.norm(vel_diff, axis=-1)

def draw_robot(vis, state, color=0x0000ff, name="point_mass"):
    # state: (4,)
    pos = state[:DOFS]
    
    name = f"robots/{name}"
    # Basic sphere
    vis[name].set_object(g.Sphere(0.1), g.MeshLambertMaterial(color=color))
    vis[name].set_transform(tf.translation_matrix([pos[0], pos[1], pos[2]]))
