"""
Some stuff that isn't used any more but might be inspirational later
"""

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
    
    
"""
A bad attempt at an optimal planner. Is this actually optimal?
"""

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
