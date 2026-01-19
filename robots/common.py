import jax
import jax.numpy as jnp


def is_broad_phase_valid(bb_min, bb_max, obs):
    """
    Checks if AABB (bb_min, bb_max) overlaps with obstacle obs.
    obs format: [min_x, min_y, min_z, max_x, max_y, max_z]
    Returns True if VALID (NO collision), False if collision.
    """
    # obs[:3] is obs_min, obs[3:] is obs_max
    obs_min = obs[:3]
    obs_max = obs[3:]
    
    # Ensure inputs are 3D
    # Check for separation in any dimension
    # If bb_max <= obs_min OR obs_max <= bb_min in any dim, then separated.
    separated = jnp.any((bb_max <= obs_min) | (obs_max <= bb_min))
    return separated


def is_motion_valid(bb_min, bb_max, obstacles):
    """
    Checks if the bounding box (bb_min, bb_max) is valid against all obstacles.
    Returns True if valid (no collision).
    """
    # Vectorized check against all obstacles
    safeties = jax.vmap(is_broad_phase_valid, in_axes=(None, None, 0))(bb_min, bb_max, obstacles)
    return jnp.all(safeties)