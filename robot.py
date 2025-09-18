
from dataclasses import dataclass
import jax
import jax.numpy as jnp

def forward(params, pos, vel, action):
    assert pos.shape == (params.dofs,)
    assert vel.shape == (params.dofs,)
    assert action.shape == (params.dofs,)
    
    new_vels = vel + action * params.dt
    new_poses = pos + new_vels * params.dt
    
    vel *= 0.8
    
    # new_vels = action
    # new_poses = pos + new_vels * params.dt
    
    return new_poses, new_vels


def sample(params, key, num):
    # Sample in params.bound_min and params.bound_max
    poses = jax.random.uniform(key, (num, params.dofs), minval=params.pos_min, maxval=params.pos_max)
    return poses
    
def metric(pos, goal_pos):
    return jnp.linalg.norm(pos - goal_pos)

def metric_batch(poses, goal_poses):
    return jnp.linalg.norm(poses - goal_poses, axis=-1)