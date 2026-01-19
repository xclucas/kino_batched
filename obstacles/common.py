
import csv
import sys
import numpy as np

def load_obstacles(filename):
    """
    Reads obstacles from a CSV file.
    Expected Format:
    Row 0: Start State
    Row 1: Goal State
    Row 2: Min Bounds (pos only used)
    Row 3: Max Bounds (pos only used)
    Row 4+: Obstacles (min_x, min_y, min_z, max_x, max_y, max_z)
    
    Returns:
        start (np.array)
        goal (np.array)
        bounds (tuple of np.array): (min_pos, max_pos)
        obstacles (np.array of shape (N, 6))
    """
    rows = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # Filter empty strings
                cl = [x.strip() for x in row if x.strip() != '']
                rows.append(cl)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        raise
        
    if not rows:
        raise ValueError("Empty file")
    
    # Simple assertion: rows > 5
    assert len(rows) > 5, f"File must have > 5 rows (found {len(rows)})"
    assert len(rows[0]) == 3, f"Start row must have exactly 3 items, {rows[0]}"
    assert len(rows[1]) == 3, f"Goal row must have exactly 3 items, {rows[1]}"
    assert len(rows[2]) >= 3, f"Min bounds row must have at least 3 items, {rows[2]}"
    assert len(rows[3]) >= 3, f"Max bounds row must have at least 3 items, {rows[3]}"

    start = np.array([float(x) for x in rows[0]], dtype=np.float32)
    goal = np.array([float(x) for x in rows[1]], dtype=np.float32)
    
    # Consistent dims for start/goal
    if len(start) != len(goal):
            raise ValueError(f"Start dim ({len(start)}) != Goal dim ({len(goal)})")

    # Bounds: pos only (first 3)
    min_b_full = [float(x) for x in rows[2]]
    max_b_full = [float(x) for x in rows[3]]
    
    min_b = np.array(min_b_full[:3], dtype=np.float32)
    max_b = np.array(max_b_full[:3], dtype=np.float32)
    
    if len(min_b) < 3 or len(max_b) < 3:
        raise ValueError("Bounds need at least 3 dimensions")
        
    bounds = (min_b, max_b)
    
    # Obstacles
    obs_list = []
    for r in rows[4:]:
        obs_list.append([float(x) for x in r])
        
    obstacles = np.array(obs_list, dtype=np.float32)
        

    return start, goal, bounds, obstacles