
import meshcat
import numpy as np
import jax.numpy as jnp
import meshcat.geometry as g
import meshcat.transformations as tf

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


def draw_edges(parent_states, new_states, tree_len_start, robot_module=None):
    """
    Draws new edges and nodes.
    from: (B, Dim)
    to: (B, Dim)
    tree_len_start: int, starting index for naming
    """
    dofs = robot_module.DOFS
    B = new_states.shape[0]

    # X, Y -> X, Y, 0
    p1s = np.zeros((B, 3), dtype=np.float64)
    p0s = np.zeros((B, 3), dtype=np.float64)
    p1s[:, :dofs] = new_states[:, :dofs]
    p0s[:, :dofs] = parent_states[:, :dofs]
    

    # Accumulate all lines into one object "tree_edges"
    vertices = np.zeros((3, 2 * B), dtype=np.float64)
    vertices[:, 0::2] = p0s.T
    vertices[:, 1::2] = p1s.T
    
    # print('drawing edges:', vertices)
    
    # We might need to convert JAX arrays to numpy if they aren't already
    vertices = np.asarray(vertices)
    
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
