
from functools import partial
import signal
import time
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np

import env
from robots.point_mass import forward, sample, metric, collide
import argparse

class Params:
    def __init__(self):
        self.dt = 0.7
        self.dofs = 2
        
        self.obs_data = None

        # Concurrent extensions 
        self.batch_size = 50
        
        # Runs until tree is this big
        self.max_tree_size = 500
        
        # How many actions to sample when extending ONE node
        self.num_sample_actions = 100 # Can be a large as fits in GPU lane
        self.sim_steps = 10
        
        # Debug visualization
        self.viewopt = False

def extend_one(params, start_state, action, goal_state, key):
    '''
    Extend one state by one random action, for N steps
    Returns the state closest to the goal
    '''
    def extend_one_step(i, val):
        '''Extend one state for one step, tracking closest so far'''
        
        state, nearest_distance, nearest_state, key = val

        step_key, key = jax.random.split(key)
        new_state = forward(params, state, action, params.dt)
        new_pose = new_state[:params.dofs]
        
        # Update pos
        did_collide = collide(params, new_pose)
        state = jnp.where(did_collide, state, new_state)

        # Update nearest
        new_distance = metric(state, goal_state)
        new_nearest = new_distance < nearest_distance

        nearest_distance = jnp.where(new_nearest, new_distance, nearest_distance)
        nearest_state = jnp.where(new_nearest, state, nearest_state)

        return state, nearest_distance, nearest_state, key

    initial_distance = metric(start_state, goal_state)

    last_state, nearest_distance, nearest_state, _ = \
        jax.lax.fori_loop(0, params.sim_steps, extend_one_step, (start_state, initial_distance, start_state, key))
        
    return nearest_distance, nearest_state

def extend_one_multiple_actions(params, start_state, goal_state, key):
    '''
    Extend one state by many random actions, for N steps per action
    returns the single state closest to the goal
    '''
    actions = jax.random.uniform(key, (params.num_sample_actions, params.dofs), minval=params.action_min, maxval=params.action_max)
    keys = jax.random.split(key, params.num_sample_actions)
    extend_one_vmap = jax.vmap(extend_one, in_axes=(None, None, 0, None, 0))
    closest_distances, closest_states = extend_one_vmap(params, start_state, actions, goal_state, keys)
        
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
        
        key, sample_key, action_key, extend_key = jax.random.split(key, 4)
        
        targets = sample(params, sample_key, params.batch_size)
        target_poses = targets
        
        # # Get non colliding targets
        # free_idx = get_k_noncolliding_targets(params, targets, params.batch_size)
        # targets = targets[free_idx]
    
        # # Which tree nodes to try to extend? And what targets to extend to?
        
        # Get closest tree node for each target
        distances = metric(targets[:, None, :], tree[None, :, :dofs2])
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
        
        batch_keys = jax.random.split(extend_key, extend_states.shape[0])
        extend_many = jax.vmap(extend_one_multiple_actions, in_axes=(None, 0, 0, 0))
        closest_states = extend_many(params, extend_states, target_poses, batch_keys)
                
        # Add to tree (closest_poses and closest_vels)
        add_idx = tree_len + jnp.arange(closest_states.shape[0])
        tree = tree.at[add_idx, :dofs2].set(closest_states)
        tree = tree.at[add_idx, -1].set(pose_indices)
        tree_len += closest_states.shape[0]
        
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

        return key, tree, tree_len
    
    def cond(items):
        key, tree, tree_len = items
        return tree_len + params.batch_size <= params.max_tree_size
    
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
    if not params.viewopt:
        voronoi_kino_star(params, start_state, key)
    
    start_time = time.perf_counter()
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    voronoi_kino_star(params, start_state, key)
    time_ms = (time.perf_counter() - start_time) * 1000
    print(f"Planning took {time_ms :.2f} ms")
    
    if params.viewopt:
        # Show the final tree
        plt.savefig('planning_result.png')
        time.sleep(1000)
    else:
        # FIXME doesnt work
        params.viewopt = True
        voronoi_kino_star(params, start_state, key)
        plt.savefig('planning_result.png')