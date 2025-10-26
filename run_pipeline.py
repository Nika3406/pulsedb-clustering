"""
run_pipeline.py

Main entry point to run pipeline on PulseDB segments.

Example usage:
    python run_pipeline.py --data_dir /path/to/Segment_Files --field ABP_F --n_segments 1000
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# If you plan to run as a package, keep relative imports; otherwise change to direct imports.
try:
    from data_loader import load_pulsedb_segments
    from clustering import top_down_clustering, centroid
    from closest_pair import closest_pair_in_cluster
    from kadane import kadane
    from viz import plot_cluster_series, plot_pair, highlight_kadane
except Exception:
    # fallback for relative import issues (e.g., running from different cwd)
    from .data_loader import load_pulsedb_segments
    from .clustering import top_down_clustering, centroid
    from .closest_pair import closest_pair_in_cluster
    from .kadane import kadane
    from .viz import plot_cluster_series, plot_pair, highlight_kadane

def z_normalize(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(axis=1, keepdims=True)
    sigma = arr.std(axis=1, keepdims=True) + 1e-9
    return (arr - mu) / sigma

def main(args):
    print("Loading segments...")
    data, meta = load_pulsedb_segments(args.data_dir, n_segments=args.n_segments, field=args.field, random_sample=args.random, skip_bad=True)
    print(f"Loaded {data.shape[0]} segments of length {data.shape[1]}")
    data_z = z_normalize(data)

    root_inds = list(range(data_z.shape[0]))
    print("Clustering (top-down divide-and-conquer)...")
    clusters = top_down_clustering(root_inds, data_z, max_leaf_size=args.leaf_size, max_depth=args.max_depth)
    clusters_sorted = sorted(clusters, key=lambda c: -len(c))

    print(f"Total clusters: {len(clusters_sorted)}")
    sizes = [len(c) for c in clusters_sorted]
    print("Largest clusters (top 10):", sizes[:10])

    results = []
    t = np.linspace(0, args.duration, data.shape[1], endpoint=False)

    # analyze top K clusters
    for idx, cl in enumerate(tqdm(clusters_sorted[:args.top_k], desc="Analyzing clusters")):
        i, j, d = closest_pair_in_cluster(cl, data_z, small_threshold=args.small_threshold, shortlist_k=args.shortlist_k)
        cent = centroid(data_z, cl)
        kad_cent = kadane(cent)
        kad_i = kadane(data_z[i])
        kad_j = kadane(data_z[j])

        # save plots per cluster to output_dir
        os.makedirs(args.output_dir, exist_ok=True)
        plot_cluster_series(t, data, cl, max_plot=60, title=f"Cluster {idx} (size={len(cl)})", save_path=os.path.join(args.output_dir, f"cluster_{idx}_members.png"))
        plot_pair(t, data_z[i], data_z[j], labels=(str(i), str(j)), title=f"Cluster {idx} closest pair DTW={d:.3f}", save_path=os.path.join(args.output_dir, f"cluster_{idx}_pair.png"))
        # kadane highlights for pair members
        _, li, ri = kad_i
        highlight_kadane(t, data_z[i], li, ri, title=f"Cluster {idx} member {i} Kadane [{li},{ri}]", save_path=os.path.join(args.output_dir, f"cluster_{idx}_member_{i}_kadane.png"))
        _, lj, rj = kad_j
        highlight_kadane(t, data_z[j], lj, rj, title=f"Cluster {idx} member {j} Kadane [{lj},{rj}]", save_path=os.path.join(args.output_dir, f"cluster_{idx}_member_{j}_kadane.png"))

        results.append({
            'cluster_id': idx,
            'size': len(cl),
            'closest_i': int(i),
            'closest_j': int(j),
            'closest_dtw': float(d),
            'kad_centroid': kad_cent,
            'kad_i': kad_i,
            'kad_j': kad_j
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "cluster_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to {csv_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PulseDB divide-and-conquer clustering pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help='PulseDB Segment_Files folder')
    parser.add_argument('--field', type=str, default='ABP_F', help='field in .mat to use (e.g., ABP_F)')
    parser.add_argument('--n_segments', type=int, default=1000, help='number of segments to load')
    parser.add_argument('--random', action='store_true', help='random sample segments from available files')
    parser.add_argument('--leaf_size', type=int, default=20, help='max leaf cluster size')
    parser.add_argument('--max_depth', type=int, default=12, help='max recursion depth for clustering')
    parser.add_argument('--top_k', type=int, default=10, help='analyze top K largest clusters')
    parser.add_argument('--small_threshold', type=int, default=120, help='small cluster threshold for brute DTW')
    parser.add_argument('--shortlist_k', type=int, default=300, help='number of candidate pairs to DTW for large clusters')
    parser.add_argument('--duration', type=float, default=10.0, help='duration in seconds (for plotting x-axis)')
    parser.add_argument('--output_dir', type=str, default='results', help='directory to save plots and CSV')
    args = parser.parse_args()
    main(args)
