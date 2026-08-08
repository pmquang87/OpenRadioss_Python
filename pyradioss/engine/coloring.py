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
    
    # 1. Build node to element mapping
    node_degree = np.zeros(num_nodes, dtype=np.int32)
    for e in range(num_elems):
        for i in range(conn.shape[1]):
            node_degree[conn[e, i]] += 1
            
    node_offset = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(num_nodes):
        node_offset[i + 1] = node_offset[i] + node_degree[i]
        
    node_elems = np.zeros(node_offset[-1], dtype=np.int32)
    current_offset = node_offset[:-1].copy()
    
    for e in range(num_elems):
        for i in range(conn.shape[1]):
            n_id = conn[e, i]
            node_elems[current_offset[n_id]] = e
            current_offset[n_id] += 1
            
    # 2. Greedy coloring
    elem_color = np.full(num_elems, -1, dtype=np.int32)
    max_color = 0
    # Marker array to track which colors are used by neighbors of the current element.
    # color_used[c] == e means color c is used by a neighbor of element e.
    color_used = np.full(1000, -1, dtype=np.int32) 
    
    for e in range(num_elems):
        for i in range(conn.shape[1]):
            n_id = conn[e, i]
            for j in range(node_offset[n_id], node_offset[n_id + 1]):
                nbr = node_elems[j]
                c = elem_color[nbr]
                if c != -1:
                    if c >= len(color_used):
                        new_color_used = np.full(len(color_used) * 2, -1, dtype=np.int32)
                        new_color_used[:len(color_used)] = color_used
                        color_used = new_color_used
                    color_used[c] = e
                    
        color = 0
        while color < len(color_used) and color_used[color] == e:
            color += 1
            
        elem_color[e] = color
        if color > max_color:
            max_color = color
            
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
