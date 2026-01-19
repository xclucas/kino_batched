import jax
import jax.numpy as jnp
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

# Dimensions
DOFS = 3
STATE_DIM = 12 # x, y, z, phi, theta, psi, u, v, w, p, q, r
ACTION_DIM = 4 # Zc, Lc, Mc, Nc

# Constants
GRAVITY = -9.81
MASS = 1.0
MASS_INV = 1.0 / MASS
IX = 1.0
IY = 1.0
IZ = 2.0
NU = 10e-3
MU = 2e-6

# Limits
# State: pos(3), ang(3), lin_vel(3), ang_vel(3)
# Limits for NON-POS: ang(3), lin_vel(3), ang_vel(3)
NOPOS_STATE_MIN = jnp.array(
                      [-jnp.pi, -jnp.pi, -jnp.pi, 
                       -0.5/3, -0.5/3, -0.5/3, 
                       -0.5/3, -0.5/3, -0.5/3])
NOPOS_STATE_MAX = jnp.array(
                      [jnp.pi, jnp.pi, jnp.pi, 
                       0.5/3, 0.5/3, 0.5/3, 
                       0.5/3, 0.5/3, 0.5/3])

# Action: Zc, Lc, Mc, Nc
ACTION_MIN = jnp.array([0.0, -jnp.pi, -jnp.pi, -jnp.pi])
ACTION_MAX = jnp.array([0.5/3, jnp.pi, jnp.pi, jnp.pi])

def quad_ode(state, action):
    # State: x, y, z, phi, theta, psi, u, v, w, p, q, r
    phi, theta, psi = state[3], state[4], state[5]
    u, v, w = state[6], state[7], state[8]
    p, q, r = state[9], state[10], state[11]
    
    Zc, Lc, Mc, Nc = action
    
    # Rotation matrix components
    c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
    c_th, s_th = jnp.cos(theta), jnp.sin(theta)
    c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
    t_th = jnp.tan(theta)
    
    # 1. Position Derivatives (World frame velocity)
    # R * [u, v, w]^T
    x_dot = (c_th * c_psi) * u + \
            (s_phi * s_th * c_psi - c_phi * s_psi) * v + \
            (c_phi * s_th * c_psi + s_phi * s_psi) * w
            
    y_dot = (c_th * s_psi) * u + \
            (s_phi * s_th * s_psi + c_phi * c_psi) * v + \
            (c_phi * s_th * s_psi - s_phi * c_psi) * w
            
    z_dot = -s_th * u + s_phi * c_th * v + c_phi * c_th * w
    
    # 2. Angle Derivatives (Euler rates)
    phi_dot = p + (q * s_phi + r * c_phi) * t_th
    theta_dot = q * c_phi - r * s_phi
    psi_dot = (q * s_phi + r * c_phi) / jnp.cos(theta) # Fix potential div by zero if necessary
    
    # 3. Linear Velocity Derivatives (Body frame acceleration)
    v_norm = jnp.sqrt(u**2 + v**2 + w**2)
    XYZ = -NU * v_norm
    
    u_dot = (r * v - q * w) - GRAVITY * s_th + MASS_INV * XYZ * u
    v_dot = (p * w - r * u) + GRAVITY * c_th * s_phi + MASS_INV * XYZ * v
    w_dot = (q * u - p * v) + GRAVITY * c_th * c_phi + \
            MASS_INV * XYZ * w + MASS_INV * Zc
            
    # 4. Angular Velocity Derivatives (Body rates)
    omega_norm = jnp.sqrt(p**2 + q**2 + r**2)
    LMN = -MU * omega_norm
    
    L = LMN * p
    M = LMN * q
    N = LMN * r
    
    p_dot = (IY - IZ) / IX * q * r + (1.0/IX) * L + (1.0/IX) * Lc
    q_dot = (IZ - IX) / IY * p * r + (1.0/IY) * M + (1.0/IY) * Mc
    r_dot = (IX - IY) / IZ * p * q + (1.0/IZ) * N + (1.0/IZ) * Nc
    
    return jnp.array([
        x_dot, y_dot, z_dot,
        phi_dot, theta_dot, psi_dot,
        u_dot, v_dot, w_dot,
        p_dot, q_dot, r_dot
    ])

def forward(params, state, action, dt):
    k1 = quad_ode(state, action)
    k2 = quad_ode(state + 0.5 * dt * k1, action)
    k3 = quad_ode(state + 0.5 * dt * k2, action)
    k4 = quad_ode(state + dt * k3, action)
    
    s_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return s_next

def collide(params, state):
    pos = state[:3] # x,y,z
    bounds_coll = jnp.any((state < params.state_min) | (state > params.state_max))
    is_free = params.collide_fn(pos, params.obs_data)
    return bounds_coll | (~is_free)

def metric(states, goal_states):
    pos_diff = states[..., :3] - goal_states[..., :3]
    rest_diff = states[..., 3:] - goal_states[..., 3:]
    return jnp.linalg.norm(pos_diff, axis=-1) + 0.1 * jnp.linalg.norm(rest_diff, axis=-1)

def draw_robot(vis, state, color=0xff0000, name="quadrotor"):
    x, y, z, phi, theta, psi = state[:6]
    
    name = f"robots/{name}"
    
    # Arms
    vis[name]["arm1"].set_object(g.Box([0.5, 0.05, 0.02]), g.MeshLambertMaterial(color=color))
    vis[name]["arm2"].set_object(g.Box([0.05, 0.5, 0.02]), g.MeshLambertMaterial(color=color))
    
    mat = tf.translation_matrix([x, y, z])
    rot = tf.euler_matrix(psi, theta, phi, 'rzyx') # Verify order
    
    vis[name].set_transform(mat.dot(rot))
