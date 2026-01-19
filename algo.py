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

def collide_fn(p1, obstacles):
    # p1: (3,)
    # obstacles: (N, 6)
    assert p1.shape[0] == 3
    
    # obstacles (N, 6)
    obs_min = obstacles[:, :3]
    obs_max = obstacles[:, 3:6]
    
    # Check if point is INSIDE any obstacle
    # (N,) boolean
    inside = jnp.all((p1 >= obs_min) & (p1 <= obs_max), axis=1)
        
    return ~jnp.any(inside)

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
        self.num_sample_actions = 100 # Can be a large as fits in GPU lane
        self.sim_steps = 10 # Steps per extension
        
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
        
        assert robot_module.collide(self, self.start) == False, "Start state in collision"
        assert robot_module.collide(self, self.goal) == False, "Goal state in collision"

def sample(params, key, batch_size):
    '''
    Sample random states uniformly within bounds
    '''
    return jax.random.uniform(key, (batch_size, params.state_dim), 
                              minval=params.state_min, maxval=params.state_max)

def extend_one(robot, params, start_state, action, goal_state, key):
    '''
    Extend one state by one random action, for N steps
    Returns the state closest to the goal
    '''
    def extend_one_step(i, val):
        state, nearest_distance, nearest_state, key = val

        # FIXME in the past we had random dt
        new_state = robot.forward(params, state, action, params.dt)
        did_collide = robot.collide(params, new_state)
        state = jnp.where(did_collide, state, new_state)

        # Update nearest
        new_distance = robot.metric(state, goal_state)
        new_nearest = new_distance < nearest_distance

        nearest_distance = jnp.where(new_nearest, new_distance, nearest_distance)
        nearest_state = jnp.where(new_nearest, state, nearest_state)

        return state, nearest_distance, nearest_state, key

    initial_distance = robot.metric(start_state, goal_state)

    last_state, nearest_distance, nearest_state, _ = \
        jax.lax.fori_loop(0, params.sim_steps, extend_one_step, (start_state, initial_distance, start_state, key))
        
    return nearest_distance, nearest_state

def extend_one_multiple_actions(robot, params, start_state, goal_state, key):
    '''
    Extend one state by many random actions, for N steps per action
    returns the single state closest to the goal
    '''
    # Sample actions
    # Split key explicitly for sampling vs vmap
    sample_key, key = jax.random.split(key)
    
    actions = jax.random.uniform(sample_key, (params.num_sample_actions, params.action_dim), minval=params.action_min, maxval=params.action_max)
    
    keys = jax.random.split(key, params.num_sample_actions)
    
    # Exclude robot from arglist
    extend_one_fn = partial(extend_one, robot)
    extend_one_vmap = jax.vmap(extend_one_fn, in_axes=(None, None, 0, None, 0))
    
    closest_distances, closest_states = extend_one_vmap(params, start_state, actions, goal_state, keys)
    
    # Pick best action
    best_index = jnp.argmin(closest_distances)
    return closest_states[best_index]

@partial(jax.jit, static_argnames=['robot', 'vis_callback', 'params'])
def voronoi_kino(robot, vis_callback, params, key, tree, tree_len):
    '''
    Creates an expands a tree until it reaches max_tree_size
    Returns (key, tree, tree_len)
    '''
    def step(items):
        key, tree, tree_len = items
        state_dim = params.state_dim

        key, sample_key, extend_key = jax.random.split(key, 3)
        
        # Sample targets
        targets = sample(params, sample_key, params.batch_size)
        
        # Get closest tree node for each target
        # tree: [MAX, state_dim + 1]
        tree_states = tree[:, :state_dim]
        
        # Vectorized metric: targets(B, S), tree(T, S)
        # Broadcast: (B, 1, S), (1, T, S)
        distances = robot.metric(targets[:, None, :], tree_states[None, :, :])
        
        # Mask invalid tree nodes
        tree_valid = jnp.arange(tree.shape[0]) < tree_len
        distances = jnp.where(tree_valid, distances, jnp.inf)
        
        pose_indices = jnp.argmin(distances, axis=1) # (B,) indices of closest tree nodes
        origin_states = tree[pose_indices, :state_dim]
        
        # Actions for EACH node
        batch_keys = jax.random.split(extend_key, params.batch_size)
        
        extend_many = jax.vmap(extend_one_multiple_actions, in_axes=(None, None, 0, 0, 0))
        new_states = extend_many(robot, params, origin_states, targets, batch_keys)
                
        # Check collision again? extend_one already returns valid states (or start state).
        # We assume extend_one logic ensures validity.
        
        # Add to tree
        add_idx = tree_len + jnp.arange(params.batch_size)
                
        tree = tree.at[add_idx, :state_dim].set(new_states)
        tree = tree.at[add_idx, -1].set(pose_indices)
        
        if params.viewopt:
             # Extract parent states for visualization
             parent_states = tree[pose_indices, :state_dim]
             jax.debug.callback(vis_callback, new_states, parent_states, targets, tree_len)

        tree_len += params.batch_size

        return key, tree, tree_len
    
    def cond(items):
        key, tree, tree_len = items
        return tree_len + params.batch_size <= params.max_tree_size
    
    # Run 20 steps
    key, tree, tree_len = \
        jax.lax.while_loop(cond, step, (key, tree, tree_len))
        
    return key, tree, tree_len
        
def init_tree(params, start_state):
    # Tree is [max_tree_size, state_dim + 1]
    # Which holds [:, state + parent_index]
    tree = jnp.zeros((params.max_tree_size, params.state_dim + 1))
    tree_len = 0
    
    tree = tree.at[0, :params.state_dim].set(start_state)
    tree = tree.at[0, -1].set(-1)
    tree_len += 1
    
    return tree, tree_len

if __name__ == "__main__":
    # Override matplotlib siginal handler to ctrl + c
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='point_mass', choices=ROBOTS.keys())
    parser.add_argument('--scene', type=str, default='house', help="house, trees, narrowPassage")
    parser.add_argument('--viewopt', action='store_true')
    args = parser.parse_args()
    
    robot_module = ROBOTS[args.robot]
    params = Params(robot_module, args.scene)
    params.viewopt = args.viewopt
    
    key = jax.random.key(time.time_ns())
    
    print(f"Running {args.robot} in scene {args.scene}")
    print(f"Start State: {params.start}")
    print(f"Goal State: {params.goal}")
    print(f"State Min {params.state_min}\n      Max: {params.state_max}")
    
    # Draw env
    if args.viewopt:
        vis = draw.init_vis()
        draw.draw_obstacles(params.obs_data)
        
        robot_module.draw_robot(vis, params.start, color=0x0000ff, name="start")
        robot_module.draw_robot(vis, params.goal, color=0xff0000, name="goal")


    def vis_callback(new_states, parent_states, targets, tree_len):
        draw.draw_edges(parent_states, new_states, tree_len, robot_module)
        draw.draw_targets(targets, robot_module)

    # FIXME we incur a penalty to copy this to gpu, in the
    # future star version we should call init_tree on gpu directly
    tree, tree_len = init_tree(params, params.start)

    # If not visualizing, then warmup for benchmark
    if not params.viewopt:
        voronoi_kino(robot_module, vis_callback, params, key, tree, tree_len)
        
    # Real shit
    start_time = time.perf_counter()
    key, tree, tree_len = voronoi_kino(robot_module, vis_callback, params, key, tree, tree_len)
    tree_len.block_until_ready()
    time_ms = (time.perf_counter() - start_time) * 1000
    print(f"Planning took {time_ms :.2f} ms")
    print(f"Final tree size: {tree_len} / {params.max_tree_size}")