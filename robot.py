
from dataclasses import dataclass
import jax
import jax.numpy as jnp

def forward(params, pos, vel, action, key):
    assert pos.shape == (params.dofs,)
    assert vel.shape == (params.dofs,)
    assert action.shape == (params.dofs,)
    
    # dt = jax.random.uniform(key, minval=0.0, maxval=params.dt)
    dt = params.dt
    
    new_vels = vel + action * dt
    new_poses = pos + new_vels * dt
    
    # vel *= 0.8
    
    # new_vels = action
    # new_poses = pos + new_vels * params.dt
    
    return new_poses, new_vels


def sample(params, key, num):
    '''Sample a state in params.bound_min and params.bound_max'''
    key1, key2 = jax.random.split(key)
    poses = jax.random.uniform(key1, (num, params.dofs), minval=params.pos_min, maxval=params.pos_max)
    vels = jax.random.uniform(key2, (num, params.dofs), minval=-1.0, maxval=1.0)
    return jnp.concatenate([poses, vels], axis=-1)
    
def metric(state, goal_state):
    '''Compute the distance metric between state and goal_state
        start: [pos, vel]
        goal: [pos, vel]
    '''
    dofs = state.shape[0] // 2
    pos_diff = state[:dofs] - goal_state[:dofs]
    vel_diff = state[dofs:] - goal_state[dofs:]
    return jnp.linalg.norm(pos_diff) + 1.0 * jnp.linalg.norm(vel_diff)

def metric_batch(states, goal_states):
    '''Compute the distance metric between states and goal_states
        states: [..., pos, vel]
        goal_states: [..., pos, vel]
        TODO probably same as metric(), too lazy to check
    '''
    dofs = states.shape[-1] // 2
    pos_diff = states[..., :dofs] - goal_states[..., :dofs]
    vel_diff = states[..., dofs:] - goal_states[..., dofs:]
    return jnp.linalg.norm(pos_diff, axis=-1) + 1.0 * jnp.linalg.norm(vel_diff, axis=-1)