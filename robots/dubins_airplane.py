import jax
import jax.numpy as jnp
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

# Dimensions
DOFS = 3 # x, y, z
STATE_DIM = 6 # x, y, z, yaw, pitch, v
ACTION_DIM = 3 # yaw_rate, pitch_rate, a

# Constants
W_SIZE = 100.0

# Limits
# State: x, y, z, yaw, pitch, v
# Limits for NON-POS: yaw, pitch, v
NOPOS_STATE_MIN = jnp.array([-jnp.pi, -jnp.pi/3, 0.1])
NOPOS_STATE_MAX = jnp.array([jnp.pi, jnp.pi/3, 30.0])

# Action: yaw_rate, pitch_rate, a
ACTION_MIN = jnp.array([-jnp.pi/4, -jnp.pi/4, -0.5/3])
ACTION_MAX = jnp.array([jnp.pi/4, jnp.pi/4, 0.5/3])

def forward(params, state, action, dt):
    # state: x, y, z, yaw, pitch, v
    # action: yaw_rate, pitch_rate, a
    x, y, z, yaw, pitch, v = state
    yaw_rate, pitch_rate, a = action
    
    # Dynamics (Euler is usually fine for small steps, or RK4)
    # Using RK4 like reference for stability
    def dynamics(s, act):
        _yaw = s[3]
        _pitch = s[4]
        _v = s[5]
        
        dx = _v * jnp.cos(_pitch) * jnp.cos(_yaw)
        dy = _v * jnp.cos(_pitch) * jnp.sin(_yaw)
        dz = _v * jnp.sin(_pitch)
        dyaw = act[0]
        dpitch = act[1]
        dv = act[2]
        return jnp.array([dx, dy, dz, dyaw, dpitch, dv])
    
    k1 = dynamics(state, action)
    k2 = dynamics(state + 0.5 * dt * k1, action)
    k3 = dynamics(state + 0.5 * dt * k2, action)
    k4 = dynamics(state + dt * k3, action)
    
    s_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Normalize angles?
    nx, ny, nz, nyaw, npitch, nv = s_next
    
    # Wrap yaw
    nyaw = (nyaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # Clamp pitch and v using PARAMS
    # [min_x, min_y, min_z, min_yaw, min_pitch, min_v]
    min_b = params.state_min
    max_b = params.state_max
    
    npitch = jnp.clip(npitch, min_b[4], max_b[4])
    nv = jnp.clip(nv, min_b[5], max_b[5])
    
    return jnp.array([nx, ny, nz, nyaw, npitch, nv])

def collide(params, state):
    pos = state[:3] # x,y,z
    
    bounds_coll = jnp.any((state < params.state_min) | (state > params.state_max))

    is_free = params.collide_fn(pos, params.obs_data)
    
    return bounds_coll | (~is_free)

def metric(states, goal_states):
    pos_diff = states[..., :3] - goal_states[..., :3]
    rest_diff = states[..., 3:] - goal_states[..., 3:]
    return jnp.linalg.norm(pos_diff, axis=-1) + 0.1 * jnp.linalg.norm(rest_diff, axis=-1)

def draw_robot(vis, state, color=0x00ffff, name="dubins_airplane"):
    x, y, z, yaw, pitch = state[:5]
    
    name = f"robots/{name}"
    vis[name]["body"].set_object(g.Box([1.0, 0.2, 0.1]), g.MeshLambertMaterial(color=color))
    # Transform: Translate then Rotate (Yaw then Pitch)
    # R = Rz(yaw) * Ry(-pitch) ?
    # Standard: yaw around Z, pitch around Y (body).
    
    mat = tf.translation_matrix([x, y, z])
    rot = tf.euler_matrix(yaw, pitch, 0.0, 'rzyx') # Z Y X order?
    # Dubins: yaw (heading), pitch (flight path angle).
    
    vis[name].set_transform(mat.dot(rot))
