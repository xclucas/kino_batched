import argparse
import csv
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Scale obstacles proportionally so X fits 0-target.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", nargs='?', help="Path to output CSV file (prints to stdout if omitted)")
    parser.add_argument("--target_x", type=float, default=5.0, help="Target size for X dimension (default: 5.0)")
    
    args = parser.parse_args()

    # Load data
    obstacles = []
    try:
        with open(args.input_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # Parse floats
                try:
                    vals = [float(x.strip()) for x in row if x.strip() != '']
                    if len(vals) == 6:
                        obstacles.append(vals)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found.", file=sys.stderr)
        sys.exit(1)

    if not obstacles:
        print("No valid obstacles found.", file=sys.stderr)
        sys.exit(1)

    arr = np.array(obstacles)
    
    # Columns: min_x, min_y, min_z, max_x, max_y, max_z
    x_mins = arr[:, 0]
    x_maxs = arr[:, 3]
    
    global_min_x = np.min(x_mins)
    global_max_x = np.max(x_maxs)
    
    current_width = global_max_x - global_min_x
    if current_width == 0:
        print("Error: X width is 0, cannot scale.", file=sys.stderr)
        sys.exit(1)
        
    scale_factor = args.target_x / current_width
    
    # Shift X to 0 first (translation)
    # We subtract global_min_x from x coordinates
    arr[:, 0] -= global_min_x
    arr[:, 3] -= global_min_x
    
    # We also need to decide what to do with Y and Z translation.
    # To maintain relative positions perfectly, we usually just scale. Use input origin?
    # But "scale so x is 0-5" strongly implies standardizing X position.
    # To keep "proportions", Y and Z must be scaled by the same factor.
    # Should Y be shifted? If we don't, and Y starts at 100, it will end at 500 (if scale 5).
    # If the original data was in a unit box (0-1) in both X and Y, 
    # and we verify that:
    
    # X is now [0, width]. 
    # Apply scale to everything.
    # new_coord = old_coord * scale
    
    arr *= scale_factor
    
    # Output
    if args.output_file:
        with open(args.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(arr)
        print(f"Scaled obstacles to {args.output_file}. X range: [0.0, {args.target_x}]", file=sys.stderr)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerows(arr)

if __name__ == "__main__":
    main()
