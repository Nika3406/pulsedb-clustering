"""
closest_pair.py - Find the closest pair in a cluster
"""

import numpy as np
from dtw import dtw_distance  # Changed from relative import


def find_closest_pair(cluster):
    """
    Find the two most similar time series in a cluster.
    
    Args:
        cluster: list of time-series arrays
        
    Returns:
        (idx1, idx2, distance): indices of closest pair and their DTW distance
    """
    if len(cluster) < 2:
        return None, None, float('inf')
    
    min_dist = float('inf')
    best_i, best_j = 0, 1
    
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            dist = dtw_distance(cluster[i], cluster[j])
            if dist < min_dist:
                min_dist = dist
                best_i, best_j = i, j
    
    return best_i, best_j, min_dist