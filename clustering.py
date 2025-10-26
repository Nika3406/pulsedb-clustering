"""
clustering.py – Divide-and-Conquer Time-Series Clustering

Implements a simple recursive clustering method based on average pairwise distance.
Each cluster is recursively split until its internal similarity exceeds a threshold.
"""

import numpy as np
from dtw import dtw_distance


def cluster_similarity(cluster):
    """Compute the average pairwise similarity within a cluster."""
    n = len(cluster)
    if n < 2:
        return 1.0
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += dtw_distance(cluster[i], cluster[j])
            count += 1
    return total / count


def divide_and_conquer_cluster(data, threshold=0.5):
    """
    Recursively cluster time-series segments using divide-and-conquer.
    Arguments:
        data: ndarray of shape (n_segments, length)
        threshold: maximum allowed average distance in a cluster
    Returns:
        List of clusters (each cluster is a list of time-series arrays)
    """
    # Base case
    if len(data) <= 2:
        return [data]

    # Compute similarity of the whole cluster
    avg_dist = cluster_similarity(data)
    if avg_dist < threshold:
        return [data]  # cluster is cohesive enough

    # Divide step – split into two halves
    mid = len(data) // 2
    left = data[:mid]
    right = data[mid:]

    # Conquer step – recursively cluster each half
    left_clusters = divide_and_conquer_cluster(left, threshold)
    right_clusters = divide_and_conquer_cluster(right, threshold)

    # Combine results
    return left_clusters + right_clusters