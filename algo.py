from functools import partial
import signal
import time
import jax
import jax.numpy as jnp
import argparse

from obstacles import common
from robots import point_mass, double_integrator, dubins_airplane, quadrotor
import draw

ROBOTS = {
    'point_mass': point_mass,
    'double_integrator': double_integrator,
    'dubins_airplane': dubins_airplane,
    'quadrotor': quadrotor
}

# Global vis instance
vis = None

def collide_fn(p1, p2, obstacles):
    # p1, p2: (3,)
    # obstacles: (N, 6)
    assert p1.shape == (3,)
    assert p2.shape == (3,)
    
    # Bounding box of the movement
    p_min = jnp.minimum(p1, p2)
    p_max = jnp.maximum(p1, p2)
    
    obs_min = obstacles[:, :3]
    obs_max = obstacles[:, 3:6]
    
    # Check intersection of AABBs
    # p_min <= obs_max AND obs_min <= p_max
    overlap = jnp.all((p_max >= obs_min) & (p_min <= obs_max), axis=1)
    
    # TODO in the future we may account for robot radius, but kinopax doesn't 
    # do that so we won't either
        
    return ~jnp.any(overlap)

class Params:
    def __init__(self, robot_module, obs_file):
        self.dt = 0.1 # Default, usually robot specific or 0.1
        
        self.obs_data = None # To be loaded
        self.collide_fn = collide_fn

        # Concurrent extensions 
        self.batch_size = 50
        
        # Runs until tree is this big
        self.max_tree_size = 500
        
        # How many actions to sample when extending ONE node
        self.num_sample_actions = 50 # Can be a large as fits in GPU lane
        self.sim_steps = 15 # Steps per extension
        
        # The sample is clamped to be at most this far from the tree
        self.max_sample_from_distance = jnp.inf
        
        # Visualize as we extend
        self.viewopt = False
        
        self.dofs = robot_module.DOFS
        self.state_dim = robot_module.STATE_DIM
        self.action_dim = robot_module.ACTION_DIM
        
        self.action_min = robot_module.ACTION_MIN
        self.action_max = robot_module.ACTION_MAX
        
        assert robot_module.DOFS == 3, "Only 3D position robots supported"
        
        # Load obstacles
        obs_file = f"obstacles/{obs_file}.csv"
        start, goal, bounds, obstacles = common.load_obstacles(obs_file)
        self.obs_data = obstacles
        
        # Concatenate bounds: pos (from file) + vel (from robot)
        self.state_min = jnp.concatenate([jnp.array(bounds[0][:robot_module.DOFS]), robot_module.NOPOS_STATE_MIN])
        self.state_max = jnp.concatenate([jnp.array(bounds[1][:robot_module.DOFS]), robot_module.NOPOS_STATE_MAX])
        
        assert self.state_min.shape[0] == robot_module.STATE_DIM, f"State min dim {self.state_min.shape[0]} != robot dim {robot_module.STATE_DIM}"
        assert self.state_max.shape[0] == robot_module.STATE_DIM, f"State max dim {self.state_max.shape[0]} != robot dim {robot_module.STATE_DIM}"

        # Start/Goal state: pos (from file) + zeros (vel)
        self.start = jnp.concatenate([jnp.array(start[:robot_module.DOFS]), jnp.zeros(robot_module.STATE_DIM - robot_module.DOFS)])
        self.goal = jnp.concatenate([jnp.array(goal[:robot_module.DOFS]), jnp.zeros(robot_module.STATE_DIM - robot_module.DOFS)])
        
        assert jnp.all((self.start >= self.state_min) & (self.start <= self.state_max)), "Start state out of bounds"
        assert jnp.all((self.goal >= self.state_min) & (self.goal <= self.state_max)), "Goal state out of bounds"
        
        assert robot_module.collide(self, self.start, self.start) == False, "Start state in collision"
        assert robot_module.collide(self, self.goal, self.goal) == False, "Goal state in collision"

def sample(params, key, batch_size):
    '''
    Sample random states uniformly within bounds
    '''
    return jax.random.uniform(key, (batch_size, params.state_dim), 
                              minval=params.state_min, maxval=params.state_max)

def extend_one(robot, params, start_state, action, goal_state, key):
    '''
    Extend one state by one random action, for N steps
    Returns the state closest to the goal, and the number of steps forward that were valid
    '''
    def extend_one_step(i, val):
        state, nearest_distance, nearest_state, nearest_step, key = val

        # FIXME in the past we had random dt
        new_state = robot.forward(params, state, action, params.dt)
        did_collide = robot.collide(params, state, new_state)
        state = jnp.where(did_collide, state, new_state)

        # Update nearest
        new_distance = robot.metric(state, goal_state)
        new_nearest = new_distance < nearest_distance

        nearest_distance = jnp.where(new_nearest, new_distance, nearest_distance)
        nearest_state = jnp.where(new_nearest, state, nearest_state)
        # i goes 0..steps-1. This is the (i+1)-th state in the sequence
        nearest_step = jnp.where(new_nearest, i + 1, nearest_step)

        return state, nearest_distance, nearest_state, nearest_step, key

    initial_distance = robot.metric(start_state, goal_state)

    # Initial step is 0
    last_state, nearest_distance, nearest_state, nearest_step, _ = \
        jax.lax.fori_loop(0, params.sim_steps, extend_one_step, (start_state, initial_distance, start_state, 0, key))
        
    return nearest_distance, nearest_state, nearest_step

def extend_one_multiple_actions(robot, params, start_state, goal_state, key):
    '''
    Extend one state by many random actions, for N steps per action
    returns the single state closest to the goal, the action used, and the elapsed time
    '''
    # Sample actions
    # Split key explicitly for sampling vs vmap
    sample_key, key = jax.random.split(key)
    
    actions = jax.random.uniform(sample_key, (params.num_sample_actions, params.action_dim), minval=params.action_min, maxval=params.action_max)
    
    keys = jax.random.split(key, params.num_sample_actions)
    
    # Exclude robot from arglist
    extend_one_fn = partial(extend_one, robot)
    extend_one_vmap = jax.vmap(extend_one_fn, in_axes=(None, None, 0, None, 0))
    
    closest_distances, closest_states, closest_steps = extend_one_vmap(params, start_state, actions, goal_state, keys)
    
    # Pick best action
    best_index = jnp.argmin(closest_distances)
    return closest_states[best_index], actions[best_index], closest_steps[best_index]

def reverse_extend_one(robot, params, start_state, action, goal_state, key):
    '''
    Extend one state backwards by one random action, for N steps
    Note! This extends from 'start_state' backwards towards 'goal_state', so note
    the argument order.
    returns the state closest to goal_state.
    '''
    def extend_one_step(i, val):
        state, nearest_distance, nearest_state, nearest_step, key = val

        # Backward Euler integration approximation
        # Estimate gradients wrt time to get velocity vector
        # x_prev = x - x_dot * dt
        def forward_t(t):
            return robot.forward(params, state, action, t)
            
        # Use JVP to get derivative d(forward)/dt
        _, x_dot = jax.jvp(forward_t, (0.0,), (1.0,))
        
        delta = x_dot * params.dt
        new_state = state - delta
        
        # Check collision for the segment 
        did_collide = robot.collide(params, state, new_state)
        state = jnp.where(did_collide, state, new_state)

        # Update nearest
        new_distance = robot.metric(state, goal_state)
        new_nearest = new_distance < nearest_distance

        nearest_distance = jnp.where(new_nearest, new_distance, nearest_distance)
        nearest_state = jnp.where(new_nearest, state, nearest_state)
        nearest_step = jnp.where(new_nearest, i + 1, nearest_step)

        return state, nearest_distance, nearest_state, nearest_step, key

    initial_distance = robot.metric(start_state, goal_state)

    last_state, nearest_distance, nearest_state, nearest_step, _ = \
        jax.lax.fori_loop(0, params.sim_steps, extend_one_step, (start_state, initial_distance, start_state, 0, key))
        
    return nearest_distance, nearest_state, nearest_step

def reverse_extend_one_multiple_actions(robot, params, start_state, goal_state, key):
    '''
    Extend one state by many random actions, for N steps per action
    returns the single state closest to the goal, the action used, and elapsed time
    '''
    # Sample actions
    # Split key explicitly for sampling vs vmap
    sample_key, key = jax.random.split(key)
    
    actions = jax.random.uniform(sample_key, (params.num_sample_actions, params.action_dim), minval=params.action_min, maxval=params.action_max)
    
    keys = jax.random.split(key, params.num_sample_actions)
    
    # Exclude robot from arglist
    extend_one_fn = partial(reverse_extend_one, robot)
    extend_one_vmap = jax.vmap(extend_one_fn, in_axes=(None, None, 0, None, 0))
    
    closest_distances, closest_states, closest_steps = extend_one_vmap(params, start_state, actions, goal_state, keys)
    
    # Pick best action
    best_index = jnp.argmin(closest_distances)
    return closest_states[best_index], actions[best_index], closest_steps[best_index]

@partial(jax.jit, static_argnames=['robot', 'vis_callback', 'params'])
def forward_tree(robot, vis_callback, params, key, tree, tree_len):
    '''
    Expands a tree outwards until it reaches max_tree_size
    Returns (key, tree, tree_len)
    '''
    def step(items):
        key, tree, tree_len = items
        state_dim = params.state_dim

        key, sample_key, extend_key = jax.random.split(key, 3)
        
        # Sample targets
        samples = sample(params, sample_key, params.batch_size)
        
        # Get closest tree node for each sample
        # tree: [MAX, state_dim + 1]
        tree_states = tree[:, :state_dim]
        
        # Vectorized metric: samples(B, S), tree(T, S)
        # Broadcast: (B, 1, S), (1, T, S)
        distances = robot.metric(samples[:, None, :], tree_states[None, :, :])
        
        # Mask invalid tree nodes
        tree_valid = jnp.arange(tree.shape[0]) < tree_len
        distances = jnp.where(tree_valid, distances, jnp.inf)
        
        pose_indices = jnp.argmin(distances, axis=1) # (B,) indices of closest tree nodes
        origin_states = tree[pose_indices, :state_dim]

        # Clamp samples
        # TODO keep or no
        dists_to_origin = jnp.min(distances, axis=1)
        how_much_too_far = jnp.maximum(1.0, dists_to_origin / (params.max_sample_from_distance + 1e-6))
        samples = origin_states + (samples - origin_states) / how_much_too_far[:, None]
        
        # Actions for EACH node
        batch_keys = jax.random.split(extend_key, params.batch_size)
        
        extend_many = jax.vmap(extend_one_multiple_actions, in_axes=(None, None, 0, 0, 0))
        new_states, new_actions, new_elapsed_steps = extend_many(robot, params, origin_states, samples, batch_keys)
        
        # Add to tree
        add_idx = tree_len + jnp.arange(params.batch_size)
                
        tree = tree.at[add_idx, :state_dim].set(new_states)
        tree = tree.at[add_idx, state_dim:state_dim+params.action_dim].set(new_actions)
        tree = tree.at[add_idx, state_dim+params.action_dim].set(new_elapsed_steps)
        tree = tree.at[add_idx, -1].set(pose_indices)
        
        if params.viewopt:
            # Extract parent states for visualization
            parent_states = tree[pose_indices, :state_dim]
            jax.debug.callback(vis_callback, new_states, parent_states, new_actions, new_elapsed_steps, samples, tree_len, "forward")
        
        tree_len += params.batch_size

        return key, tree, tree_len
    
    def cond(items):
        key, tree, tree_len = items
        return tree_len + params.batch_size <= params.max_tree_size
    
    # Run 20 steps
    key, tree, tree_len = \
        jax.lax.while_loop(cond, step, (key, tree, tree_len))
        
    return key, tree, tree_len

@partial(jax.jit, static_argnames=['robot', 'vis_callback', 'params'])
def reverse_tree(robot, vis_callback, params, key, tree, tree_len):
    '''
    Expands a tree outwards and backwards until it reaches max_tree_size
    Returns (key, tree, tree_len)
    '''
    def step(items):
        key, tree, tree_len = items
        state_dim = params.state_dim

        key, sample_key, extend_key = jax.random.split(key, 3)
        
        # Sample targets
        samples = sample(params, sample_key, params.batch_size)
        
        # Get closest tree node for each sample
        # tree: [MAX, state_dim + 1]
        tree_states = tree[:, :state_dim]
        
        # Vectorized metric: samples(B, S), tree(T, S)
        # Broadcast: (B, 1, S), (1, T, S)
        distances = robot.metric(samples[:, None, :], tree_states[None, :, :])
        
        # Mask invalid tree nodes
        tree_valid = jnp.arange(tree.shape[0]) < tree_len
        distances = jnp.where(tree_valid, distances, jnp.inf)
        
        pose_indices = jnp.argmin(distances, axis=1) # (B,) indices of closest tree nodes
        origin_states = tree[pose_indices, :state_dim]
        
        # Clamp samples
        # TODO keep or no
        dists_to_origin = jnp.min(distances, axis=1)
        how_much_too_far = jnp.maximum(1.0, dists_to_origin / (params.max_sample_from_distance + 1e-6))
        samples = origin_states + (samples - origin_states) / how_much_too_far[:, None]
        
        # Actions for EACH node
        batch_keys = jax.random.split(extend_key, params.batch_size)
        
        # Use REVERSE extension
        extend_many = jax.vmap(reverse_extend_one_multiple_actions, in_axes=(None, None, 0, 0, 0))
        new_states, new_actions, new_elapsed_steps = extend_many(robot, params, origin_states, samples, batch_keys)
        
        # Add to tree
        add_idx = tree_len + jnp.arange(params.batch_size)
                
        tree = tree.at[add_idx, :state_dim].set(new_states)
        tree = tree.at[add_idx, state_dim:state_dim+params.action_dim].set(new_actions)
        tree = tree.at[add_idx, state_dim+params.action_dim].set(new_elapsed_steps)
        tree = tree.at[add_idx, -1].set(pose_indices)
        
        if params.viewopt:
            # Extract parent states for visualization
            parent_states = tree[pose_indices, :state_dim]
            jax.debug.callback(vis_callback, new_states, parent_states, new_actions, new_elapsed_steps, samples, tree_len, "reverse")

        tree_len += params.batch_size

        return key, tree, tree_len
    
    def cond(items):
        key, tree, tree_len = items
        return tree_len + params.batch_size <= params.max_tree_size
    
    # Run loop
    key, tree, tree_len = \
        jax.lax.while_loop(cond, step, (key, tree, tree_len))
        
    return key, tree, tree_len

def init_tree(params, start_state):
    # Tree is [max_tree_size, state_dim + action_dim + 1 + 1]
    # Which holds [:, state + action + elapsed_steps + parent_index]
    tree = jnp.zeros((params.max_tree_size, params.state_dim + params.action_dim + 1 + 1))
    tree_len = 0
    
    tree = tree.at[0, :params.state_dim].set(start_state)
    tree = tree.at[0, -1].set(-1)
    tree_len += 1
    
    return tree, tree_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='point_mass', choices=ROBOTS.keys())
    parser.add_argument('--scene', type=str, default='house', help="house, trees, narrowPassage")
    parser.add_argument('--viewopt', action='store_true', help="View the tree as it is built")
    args = parser.parse_args()
    
    robot_module = ROBOTS[args.robot]
    params = Params(robot_module, args.scene)
    params.viewopt = args.viewopt
    
    key = jax.random.key(time.time_ns())
    
    # print(f"Running {args.robot} in scene {args.scene}")
    # print(f"Start State: {params.start}")
    # print(f"Goal State: {params.goal}")
    # print(f"State Min {params.state_min}\n      Max: {params.state_max}")
    
    # Draw env
    vis = draw.init_vis()
    draw.draw_obstacles(params.obs_data)
    
    robot_module.draw_robot(vis, params.start, color=0x0000ff, name="start")
    robot_module.draw_robot(vis, params.goal, color=0xff0000, name="goal")

    def vis_callback(new_states, parent_states, actions, elapsed_steps, targets, tree_len, mode):
        draw.draw_edges(parent_states, new_states, actions, elapsed_steps, tree_len, mode, params, robot_module)
        draw.draw_targets(targets, robot_module)
        time.sleep(0.3)

    # FIXME we incur a penalty to copy this to gpu, in the
    # future star version we should call init_tree on gpu directly
    tree, tree_len = init_tree(params, params.start)

    # If not visualizing, then warmup for benchmark
    if not params.viewopt:
        forward_tree(robot_module, vis_callback, params, key, tree, tree_len)
        
    # Real shit
    start_time = time.perf_counter()
    key, tree, tree_len = forward_tree(robot_module, vis_callback, params, key, tree, tree_len)
    tree_len.block_until_ready()
    time_ms = (time.perf_counter() - start_time) * 1000
    
    if not params.viewopt:
        print(f"Planning took {time_ms :.2f} ms")
        
    print(f"Final tree size: {tree_len} / {params.max_tree_size}")

    # Extract information from tree
    valid_tree = tree[:tree_len]
    parent_indices = valid_tree[:, -1].astype(jnp.int32)
    parent_states = valid_tree[parent_indices, :params.state_dim]
    new_states = valid_tree[:, :params.state_dim]
    actions = valid_tree[:, params.state_dim:params.state_dim+params.action_dim]
    elapsed_steps = valid_tree[:, params.state_dim+params.action_dim]    
    targets = jnp.zeros_like(new_states) # Not available here
    
    draw.draw_edges(parent_states, new_states, actions, elapsed_steps, tree_len, "forward", params, robot_module)

