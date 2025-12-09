
from functools import partial
import signal
import time
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

import env
from robot import forward, metric_batch, sample, metric
import argparse

class Params:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.dofs = 2
        
        self.pos_min = jnp.array([0, 0])
        self.pos_max = jnp.array([50, 50])

        self.action_min = jnp.array([-1, -1])
        self.action_max = jnp.array([1, 1])
        
        self.obs_data = None

        # Concurrent extensions 
        self.batch_size = 100
        
        self.max_tree_size = 1000
        
        # How many actions to sample when extending ONE node
        self.num_sample_actions = 1000 # Can be a large as fits in GPU lane
        self.sim_steps = 40
        
        # Debug visualization
        self.viewopt = False

def get_closest_k_edges(poses, poses_size, targets, k):
    '''
    Given 2 sets of positions: poses and targets
    Get the k closest EDGES in the full bipartite graph
    Returns k closest poses and their k corresponding targets
    '''
    assert poses.shape[-1] == targets.shape[-1], f"Dimension mismatch {poses.shape} {targets.shape}"
    
    valid_poses = jnp.arange(poses.shape[0]) < poses_size
    
    # (targets size, poses size)
    edge_lens = metric_batch(targets[:, None, :], poses[None, :, :])
    
    # Mask out invalid targets and poses
    edge_lens = jnp.where(valid_poses[None, :], edge_lens, jnp.inf)
    # edge_lens = jnp.where(valid_targets[:, None], edge_lens, jnp.inf)

    # Use top_k to get the smallest differences. top_k finds largest, so we negate.
    flat_edge_lens = edge_lens.flatten()
    _, flat_indices = jax.lax.top_k(-flat_edge_lens, k=k)
    
    target_indices, pose_indices = jnp.unravel_index(flat_indices, edge_lens.shape)
    
    return pose_indices, target_indices

def get_k_noncolliding_targets(params, targets, k):
    '''
    From a set of targets, get at most k noncolliding,
    if there are not enough, return as many as possible and then
    random colliding ones
    '''
    valid_mask = jax.vmap(lambda pos: ~collide(params, pos))(targets)
    
    # Get min k valid targets
    _, idx = jax.lax.top_k(valid_mask, k)
    
    return idx

def collide(params, pos):
    # Collision check with [x, y, x2, y2]
    x, y, x2, y2 = params.obs_data.T
    box_coll = jnp.any((pos[0] >= x) & (pos[0] <= x2) & (pos[1] >= y) & (pos[1] <= y2))
    bounds_coll = jnp.any((pos < params.pos_min) | (pos > params.pos_max))
    return box_coll | bounds_coll


def extend_one(params, start_state, action, goal_state):
    '''
    Extend one state by one random action, for N steps
    Returns the state closest to the goal
    '''
    def extend_one_step(i, val):
        '''Extend one state for one step, tracking closest so far'''
        
        state, nearest_distance, nearest_state = val
        pos = state[:params.dofs]
        vel = state[params.dofs:]
        
        new_pose, new_vel = forward(params, pos, vel, action)
        new_state = jnp.concatenate([new_pose, new_vel])
        
        # Update pos
        did_collide = collide(params, new_pose)
        state = jnp.where(did_collide, state, new_state)

        # Update nearest
        new_distance = metric(state, goal_state)
        new_nearest = new_distance < nearest_distance

        nearest_distance = jnp.where(new_nearest, new_distance, nearest_distance)
        nearest_state = jnp.where(new_nearest, state, nearest_state)

        return state, nearest_distance, nearest_state

    initial_distance = metric(start_state, goal_state)

    last_state, nearest_distance, nearest_state = \
        jax.lax.fori_loop(0, params.sim_steps, extend_one_step, (start_state, initial_distance, start_state))
        
    return nearest_distance, nearest_state

def extend_one_multiple_actions(params, start_state, actions, goal_state):
    '''
    Extend one state by many random actions, for N steps per action
    returns the single state closest to the goal
    '''
    extend_one_vmap = jax.vmap(extend_one, in_axes=(None, None, 0, None))
    closest_distances, closest_states = extend_one_vmap(params, start_state, actions, goal_state)
        
    best_index = jnp.argmin(closest_distances)
    return closest_states[best_index] 
        
        
@partial(jax.jit, static_argnames=['params'])
def voronoi_kino(params, key, tree, tree_len):
    
    # Tree helpers
    dofs2 = params.dofs * 2

    def step(items):
        '''
        One step of the algorithm
        '''
        # nonlocal targets
        key, tree, tree_len = items
        
        targets = sample(params, key, params.batch_size)
        target_poses = targets
        
        # # Get non colliding targets
        # free_idx = get_k_noncolliding_targets(params, targets, params.batch_size)
        # targets = targets[free_idx]
    
        # # Which tree nodes to try to extend? And what targets to extend to?
        
        # Get closest tree node for each target
        distances = metric_batch(targets[:, None, :], tree[None, :, :dofs2])
        tree_valid = jnp.arange(tree.shape[0]) < tree_len
        distances = jnp.where(tree_valid, distances, jnp.inf)
        
        pose_indices = jnp.argmin(distances, axis=1)
        extend_states = tree[pose_indices, :dofs2]
        target_poses = targets
        
        # pose_indices, target_indices = get_closest_k_edges(tree[:, :dofs], tree_len, targets, params.batch_size)
        # extend_poses = tree[pose_indices, :dofs]
        # extend_vels = tree[pose_indices, dofs:dofs2]
        # target_poses = targets[target_indices]
       
        # Actions for EACH node
        actions = jax.random.uniform(key, (params.num_sample_actions, params.dofs), minval=params.action_min, maxval=params.action_max)
        extend_many = jax.vmap(extend_one_multiple_actions, in_axes=(None, 0, None, 0))
        closest_states = extend_many(params, extend_states, actions, target_poses)
                
        # Add to tree (closest_poses and closest_vels)
        add_idx = tree_len + jnp.arange(closest_states.shape[0])
        tree = tree.at[add_idx, :dofs2].set(closest_states)
        tree = tree.at[add_idx, -1].set(pose_indices)
        tree_len += closest_states.shape[0]
        
        # New key
        new_key, _ = jax.random.split(key)
        
        if params.viewopt:
            def callback(tree, tree_idx, target_poses):
                tree = np.asarray(tree)
                target_poses = np.asarray(target_poses)
                # Render new tree nodes
                vis.render_tree(tree, tree_idx)
                # Redden invalidated targets
                vis.render_targets(target_poses)
                plt.pause(0.001)
            
            jax.debug.callback(callback, tree, add_idx, target_poses)

        return new_key, tree, tree_len
    
    def cond(items):
        key, tree, tree_len = items
        return tree_len + params.batch_size < params.max_tree_size
    
    # Run 20 steps
    key, tree, tree_len = \
        jax.lax.while_loop(cond, step, (key, tree, tree_len))
        
    return key, tree, tree_len

def init_tree(params, start_state):
     
    # Tree helpers
    dofs2 = params.dofs * 2
    
    # Tree is a [max_tree_size, params.dof * 2 + 1] array
    # Where each row is [pos, vel, parent_index]
    tree = jnp.zeros((params.max_tree_size, params.dofs * 2 + 1))
    tree_len = 0
    
    # Add start
    tree = tree.at[0, :dofs2].set(start_state)
    tree = tree.at[0, -1].set(-1)
    tree_len += 1
    
    return tree, tree_len

tree_history = []
new_sampled_idx = []

@partial(jax.jit, static_argnames=['params'])
def voronoi_kino_star(params, start_state, key):
   
    tree, tree_len = init_tree(params, start_state)
    
    def opt_step(i, items):
        key, tree, tree_len = items
        key, tree, tree_len = voronoi_kino(params, key, tree, tree_len)
        
        # Sample 50% of the tree
        sample_size = int(params.max_tree_size * 0.3)
        sample_indices = jax.random.choice(key, params.max_tree_size, (sample_size,), replace=False)
        
        new_tree = tree.at[:sample_size].set(tree[sample_indices])
        new_tree_len = sample_size
        
        # FIXME set parent indices to -1
        new_tree = new_tree.at[:sample_size, -1].set(-1)
        
        # Add tree
        def save_tree(tree):
            tree_history.append(np.asarray(tree))
            new_sampled_idx.append(sample_indices)
        jax.debug.callback(save_tree, new_tree)
        
        if params.viewopt:
            def callback(tree, tree_len):
                print('Resampling tree')
                vis.reset()
                # Draw new tree nodes
                vis.render_tree(tree, jnp.arange(tree_len))
                plt.pause(0.001)
            jax.debug.callback(callback, new_tree, new_tree_len)
        
        return key, new_tree, new_tree_len
        
    # Run 10 iterations of voronoi_kino
    key, tree, tree_len = jax.lax.fori_loop(0, 1, opt_step, (key, tree, tree_len))
    
    # FIXME
    # def save_final_tree(tree):
    #     tree_history.append(np.asarray(tree))
    # jax.debug.callback(save_final_tree, tree)

vis = None

if __name__ == "__main__":
    # Override matplotlib siginal handler to ctrl + c
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewopt', action='store_true', help='Enable debug visualization')
    args = parser.parse_args()
    seed = time.time_ns()
     
    params = Params()
    params.viewopt = args.viewopt
    
    vis = env.RectangleEnv(seed=seed, width=params.pos_max[0], height=params.pos_max[1])
    params.obs_data = np.asarray(vis.obs_data)
    vis.render_obs()
    plt.pause(0.001)
    
    start_pos = jnp.array([1, 1])
    start_vel = jnp.zeros(2)
    start_state = jnp.concatenate([start_pos, start_vel])
    key = jax.random.key(seed)
    
    # Warmup
    voronoi_kino_star(params, start_state, key)
    
    start_time = time.perf_counter()
    voronoi_kino_star(params, start_state, key)
    time_ms = (time.perf_counter() - start_time) * 1000
    print(f"Planning took {time_ms :.2f} ms")
    
    # Show the final tree
    time.sleep(1000)