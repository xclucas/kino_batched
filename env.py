import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class RectangleEnv:
    """
    A 2D motion planning environment with rectangular obstacles.
    """

    def __init__(self, width=50, height=50, num_obstacles=10, seed=None):
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.obstacles = []

        if seed is not None:
            np.random.seed(seed % (2**32 - 1))
            
        self.obs_data = []

        for _ in range(self.num_obstacles):
            while True:
                width = np.random.uniform(6, 13)
                height = np.random.uniform(6, 13)
                x = np.random.uniform(-width, self.width)
                y = np.random.uniform(-height, self.height)
                
                # Check if collide 1, 1
                if (x < 1 < x + width) and (y < 1 < y + height):
                    continue
                
                self.obs_data.append((x, y, x + width, y + height))
                
                new_obstacle = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='r')
                
                if not any(self._check_collision(new_obstacle, obs) for obs in self.obstacles):
                    self.obstacles.append(new_obstacle)
                    break
        
        plt.figure(figsize=(8, 8))
        self.render_obs()

    def _check_collision(self, rect1, rect2):
        return (rect1.get_x() < rect2.get_x() + rect2.get_width() and
                rect1.get_x() + rect1.get_width() > rect2.get_x() and
                rect1.get_y() < rect2.get_y() + rect2.get_height() and
                rect1.get_y() + rect1.get_height() > rect2.get_y())

    def render_obs(self):
        
        ax = plt.gca()
        
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')
        plt.xticks(np.arange(0, self.width + 1, 10))
        plt.yticks(np.arange(0, self.height + 1, 10))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        for obs in self.obstacles:
            ax.add_patch(patches.Rectangle((obs.get_x(), obs.get_y()), obs.get_width(), obs.get_height(), 
                                           linewidth=1, edgecolor='r', facecolor='r'))

    def render_path(self, path):
        if not path:
            return
        
        path_points = np.array(path)
        
        # Plot lines
        plt.plot(path_points[:, 0], path_points[:, 1], 'b-', linewidth=2)
        
        # Plot points
        plt.plot(path_points[:, 0], path_points[:, 1], 'go', markersize=5)

    def render_tree(self, tree, idx):
        """
        Draw the nodes at idx of the tree and their connections.
        """

        # Plot nodes
        plt.plot(tree[idx, 0], tree[idx, 1], 'ko', markersize=3, label='Tree Nodes')

        # Plot edges
        segments = []
        for i in idx:
            parent_idx = int(tree[i, 4])
            if parent_idx >= 0:
                segments.append([tree[i, 0:2], tree[parent_idx, 0:2]])
        
        if segments:
            lines = np.array(segments)
            plt.plot(lines[:, :, 0].T, lines[:, :, 1].T, 'k-', linewidth=0.5)

    def render_targets(self, targets):
        """
        Renders the targets on the given axes.
        """
        # Clear previous targets
        if hasattr(self, 'lines'):
            for line in self.lines:
                line.remove()

        self.lines = plt.plot(targets[:, 0], targets[:, 1], 'gx', markersize=5)
    
    def reset(self):
        self.lines = []
        plt.clf()
        self.render_obs()
        
    def get_bound_min(self):
        return np.array([0, 0])

    def get_bound_max(self):
        return np.array([self.width, self.height])



if __name__ == '__main__':
    # Create the environment
    env = RectangleEnv(width=50, height=50, num_obstacles=15, seed=42)
    
    # Render the environment
    env.render()
    
    # Define a sample path
    sample_path = [
        (5, 5),
        (15, 20),
        (25, 35),
        (40, 25),
        (45, 45)
    ]
    
    # Render the path
    env.render_path(sample_path)
    
    # Add legend and title
    plt.title("Voronoi Kino Vis")
    
    # Show the plot
    plt.show()
