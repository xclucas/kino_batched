
import jax
import meshcat
import numpy as np
import jax.numpy as jnp
import meshcat.geometry as g
import meshcat.transformations as tf
import jax
from functools import partial

vis = None

def init_vis():
    global vis
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    vis.delete()
    return vis


def draw_obstacles(obstacles, name="obstacles", color=0x00ff00, opacity=0.5):
    """
    Draws obstacles in MeshCat.
    obstacles: (N, 6) numpy array or list
    """
    vis[name].delete()
    for i, obs in enumerate(obstacles):
        # obs: min_x, min_y, min_z, max_x, max_y, max_z
        min_pt = np.array(obs[:3])
        max_pt = np.array(obs[3:6])
        dims = max_pt - min_pt
        center = min_pt + dims / 2.0
        
        # Avoid 0 dimensions for visibility
        dims = np.maximum(dims, 0.01).astype(np.float64)
        center = center.astype(np.float64)
        
        box = g.Box(dims)
        vis[f"{name}/obs_{i}"].set_object(box, g.MeshLambertMaterial(color=color, opacity=opacity))
        vis[f"{name}/obs_{i}"].set_transform(tf.translation_matrix(center))


def draw_edges(parent_states, new_states, actions, elapsed_steps, tree_len_start, mode, params, robot_module=None):
    """
    Draws new edges and nodes.
    from: (B, Dim)
    to: (B, Dim)
    tree_len_start: int, starting index for naming
    """
    dofs = robot_module.DOFS
    B = new_states.shape[0]

    # Integrate fine edges on CPU/host using JAX calls
    traj_points = []
    current_state = parent_states

    # Append initial point
    traj_points.append(current_state[:, :3])

    if mode == "forward":
        forward_batch = jax.vmap(robot_module.forward, in_axes=(None, 0, 0, None))
        for _ in range(params.sim_steps):
            current_state = forward_batch(params, current_state, actions, params.dt)
            traj_points.append(current_state[:, :3])
            
    elif mode == "reverse":
        # Define batched JVP for velocity
        def get_velocity(s, a):
            forward_dt = partial(robot_module.forward, params, s, a)
            _, x_dot = jax.jvp(forward_dt, (0.0,), (1.0,))
            return x_dot

        get_velocity_batch = jax.vmap(get_velocity)
        
        for _ in range(params.sim_steps):
            vel = get_velocity_batch(current_state, actions)
            current_state = current_state - vel * params.dt
            traj_points.append(current_state[:, :3])
    else:
        raise ValueError(f"Invalid mode {mode} for drawing edges.")
    
    # Convert to segments for drawing
    traj_points = np.array(traj_points) # (Steps+1, B, 3)
    
    # Flatten into segments: (Steps * B, 2, 3) effective
    # p0: [t0...t_end-1], p1: [t1...t_end]
    p0s = traj_points[:-1].reshape(-1, 3).astype(np.float64)
    p1s = traj_points[1:].reshape(-1, 3).astype(np.float64)
    
    # Mask segments beyond elapsed_steps
    step_indices = np.arange(params.sim_steps)[:, None] # (Steps, 1)
    # Broadcast to (Steps, B)
    step_indices = np.tile(step_indices, (1, B))
    
    # Threshold steps is already in integer steps
    elapsed_steps = elapsed_steps.astype(int)[None, :] # (1, B)
    
    keep_mask = step_indices < elapsed_steps
    keep_mask = keep_mask.reshape(-1) # Flatten to match p0s
    
    # Validate NaN too
    # valid_mask = ~np.isnan(p0s).any(axis=1) & ~np.isnan(p1s).any(axis=1)
    
    # # Combine masks
    # final_mask = valid_mask & keep_mask
    
    p0s = p0s[keep_mask]
    p1s = p1s[keep_mask]
    
    N_segs = p0s.shape[0]
    
    # Accumulate all lines into one object
    vertices = np.zeros((3, 2 * N_segs), dtype=np.float64)
    vertices[:, 0::2] = p0s.T
    vertices[:, 1::2] = p1s.T
    
    vis[f"tree/batch_{tree_len_start}"].set_object(
        g.LineSegments(
            g.PointsGeometry(vertices),
            g.LineBasicMaterial(color=0x000000)
        )
    )

def draw_targets(targets, robot_module=None):
    """
    Draws targets as points or spheres.
    targets: (B, Dim)
    """
    dofs = robot_module.DOFS
        
    pts = np.zeros((targets.shape[0], 3), dtype=np.float64)
    pts[:, :dofs] = targets[:, :dofs]
    pts = np.asarray(pts)
        
    vis["targets"].set_object(
        g.Points(
            g.PointsGeometry(pts.T.astype(np.float64)),
            g.PointsMaterial(size=0.05, color=0xff0000)
        )
    )
