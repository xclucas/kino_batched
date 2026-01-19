import sys
import os
import argparse

# Adjust path to find robots module (parent of obstacles dir)
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from obstacles import common
import draw

def main():
    parser = argparse.ArgumentParser(description="Visualize obstacles CSV in MeshCat")
    parser.add_argument("file", type=str, help="Obstacle file path (e.g. 'house")
    args = parser.parse_args()

    filename = args.file
    if not filename.endswith('.csv'):
        filename += '.csv'

    # Candidates
    # 1. obstacles/FILE (Relative to workspace root)
    path1 = os.path.join(workspace_root, 'obstacles', filename)
    # 2. FILE (Relative to CWD)
    path2 = os.path.abspath(filename)

    final_path = None
    if os.path.exists(path1):
        final_path = path1
    elif os.path.exists(path2):
        final_path = path2
    else:
        print(f"Error: Could not find obstacle file.")
        print(f"Checked:\n  {path1}\n  {path2}")
        sys.exit(1)

    print(f"Loading obstacles from: {final_path}")
    start, goal, bounds, obstacles = common.load_obstacles(final_path)
    print(f"Loaded {len(obstacles)} obstacles.")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Bounds: {bounds}")

    vis = draw.init_vis()
    draw.draw_obstacles(obstacles)
    
    # Draw Start and Goal
    import meshcat.geometry as g
    import meshcat.transformations as tf
    import numpy as np

    # Start (Blue)
    vis["start"].set_object(g.Sphere(0.1), g.MeshLambertMaterial(color=0x0000ff))
    vis["start"].set_transform(tf.translation_matrix(start[:3]))
    
    # Goal (Red)
    vis["goal"].set_object(g.Sphere(0.1), g.MeshLambertMaterial(color=0xff0000))
    vis["goal"].set_transform(tf.translation_matrix(goal[:3]))


if __name__ == "__main__":
    main()
