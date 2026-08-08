import numpy as np
from numba import njit
from typing import Tuple

@njit(cache=True)
def compute_element_colors(conn: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy element coloring algorithm for thread-safe scatter operations.
    Groups elements such that no two elements in the same color share a node.

    Args:
        conn: (n_elems, nodes_per_elem) integer array of node connectivity.
        num_nodes: Total number of nodes in the model (to size the lock array).

    Returns:
        color_indices: (n_elems,) array of element indices sorted by color.
        color_offsets: (num_colors + 1,) array denoting the start and end
                       indices of each color in `color_indices`.
    """
    num_elems = conn.shape[0]
    elem_color = np.zeros(num_elems, dtype=np.int32)
    node_last_color = np.full(num_nodes, -1, dtype=np.int32)
    
    max_color = 0
    for e in range(num_elems):
        color = 0
        while True:
            conflict = False
            for idx in range(conn.shape[1]):
                node_id = conn[e, idx]
                if node_last_color[node_id] == color:
                    conflict = True
                    break
            if not conflict:
                break
            color += 1
        
        elem_color[e] = color
        if color > max_color:
            max_color = color
            
        for idx in range(conn.shape[1]):
            node_id = conn[e, idx]
            node_last_color[node_id] = color
            
    num_colors = max_color + 1
    
    # Sort elements by color
    color_counts = np.zeros(num_colors, dtype=np.int32)
    for e in range(num_elems):
        color_counts[elem_color[e]] += 1
        
    color_offsets = np.zeros(num_colors + 1, dtype=np.int32)
    for c in range(num_colors):
        color_offsets[c + 1] = color_offsets[c] + color_counts[c]
        
    color_indices = np.zeros(num_elems, dtype=np.int32)
    current_offsets = color_offsets[:-1].copy()
    
    for e in range(num_elems):
        c = elem_color[e]
        color_indices[current_offsets[c]] = e
        current_offsets[c] += 1
        
    return color_indices, color_offsets
