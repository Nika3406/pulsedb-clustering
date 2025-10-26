"""
viz.py

Plotting helpers for clusters, representative pair, and Kadane intervals.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_cluster_series(t: np.ndarray, data: np.ndarray, cluster_inds: List[int], max_plot: int = 60, title: str = None, save_path: str = None):
    plt.figure(figsize=(10, 4))
    for idx in cluster_inds[:max_plot]:
        plt.plot(t, data[idx], alpha=0.25)
    plt.title(title or f"Cluster (size={len(cluster_inds)})")
    plt.xlabel("time (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

def plot_pair(t: np.ndarray, s1: np.ndarray, s2: np.ndarray, labels=('a', 'b'), title: str = None, save_path: str = None):
    plt.figure(figsize=(10, 4))
    plt.plot(t, s1, label=f"member {labels[0]}")
    plt.plot(t, s2, label=f"member {labels[1]}")
    plt.legend()
    plt.title(title or "Closest pair (z-normalized)")
    plt.xlabel("time (s)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()

def highlight_kadane(t: np.ndarray, s: np.ndarray, l: int, r: int, title: str = None, save_path: str = None):
    plt.figure(figsize=(10, 3.5))
    plt.plot(t, s, label='series')
    plt.axvspan(t[l], t[r], alpha=0.2, color='orange', label='Kadane interval')
    plt.title(title or "Kadane maximum subarray interval")
    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()
