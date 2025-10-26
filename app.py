import numpy as np
import pandas as pd
from data_loader import load_pulsedb_segments
from clustering import divide_and_conquer_cluster
from closest_pair import find_closest_pair
from kadane import kadane_max_subarray
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ==========================================================
# CONFIGURATION
# ==========================================================
DATA_DIR = "PulseDB_MIMIC"
SIGNAL_TYPE = "ABP_F"      # 'PPG_F' or 'ECG_F' also available
N_SEGMENTS = 5             # Set to 1000 for final run
CLUSTER_THRESHOLD = 2.0    # distance threshold for recursive clustering
OUTPUT_DIR = "results"     # Directory to save results

# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=== Time-Series Clustering & Segment Analysis ===")
    print(f"Output directory: {OUTPUT_DIR}")

    # ------------------------------------------------------
    # STEP 1 – LOAD DATA
    # ------------------------------------------------------
    print("\n[1] Loading PulseDB segments ...")
    data, metadata = load_pulsedb_segments(DATA_DIR, field=SIGNAL_TYPE, n_segments=N_SEGMENTS)
    print(f"Loaded dataset shape: {data.shape}")
    
    # Save data loading info to CSV
    data_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_dir': DATA_DIR,
        'signal_type': SIGNAL_TYPE,
        'n_segments_requested': N_SEGMENTS,
        'n_segments_loaded': data.shape[0],
        'segment_length': data.shape[1],
        'cluster_threshold': CLUSTER_THRESHOLD
    }
    pd.DataFrame([data_info]).to_csv(os.path.join(OUTPUT_DIR, 'data_info.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/data_info.csv")

    # ------------------------------------------------------
    # STEP 2 – DIVIDE-AND-CONQUER CLUSTERING
    # ------------------------------------------------------
    print("\n[2] Performing divide-and-conquer clustering ...")
    clusters = divide_and_conquer_cluster(data, threshold=CLUSTER_THRESHOLD)
    print(f"Generated {len(clusters)} clusters.")
    
    # Prepare cluster summary data
    cluster_summary = []
    for i, cluster in enumerate(clusters):
        cluster_size = len(cluster)
        print(f"  Cluster {i+1}: {cluster_size} members")
        cluster_summary.append({
            'cluster_id': i + 1,
            'cluster_size': cluster_size,
            'percentage': (cluster_size / data.shape[0]) * 100
        })
    
    # Save cluster summary to CSV
    cluster_df = pd.DataFrame(cluster_summary)
    cluster_df.to_csv(os.path.join(OUTPUT_DIR, 'cluster_summary.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/cluster_summary.csv")

    # ------------------------------------------------------
    # STEP 3 – CLOSEST PAIR ANALYSIS WITHIN EACH CLUSTER
    # ------------------------------------------------------
    print("\n[3] Identifying closest pairs within clusters ...")
    closest_pairs_data = []
    
    for i, cluster in enumerate(clusters):
        if len(cluster) < 2:
            print(f"Cluster {i+1}: only one segment, skipping closest-pair check.")
            closest_pairs_data.append({
                'cluster_id': i + 1,
                'cluster_size': len(cluster),
                'closest_pair_idx1': None,
                'closest_pair_idx2': None,
                'dtw_distance': None,
                'status': 'single_member'
            })
            continue

        idx1, idx2, dist = find_closest_pair(cluster)
        print(f"Cluster {i+1}: Closest pair indices ({idx1}, {idx2}) with distance {dist:.4f}")
        
        closest_pairs_data.append({
            'cluster_id': i + 1,
            'cluster_size': len(cluster),
            'closest_pair_idx1': idx1,
            'closest_pair_idx2': idx2,
            'dtw_distance': dist,
            'status': 'computed'
        })
    
    # Save closest pairs to CSV
    closest_pairs_df = pd.DataFrame(closest_pairs_data)
    closest_pairs_df.to_csv(os.path.join(OUTPUT_DIR, 'closest_pairs.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/closest_pairs.csv")

    # ------------------------------------------------------
    # STEP 4 – KADANE'S ALGORITHM (MAX SUBARRAY)
    # ------------------------------------------------------
    print("\n[4] Applying Kadane's algorithm to detect active intervals ...")
    kadane_results = []
    
    # Apply Kadane to all segments (or first 100 for speed if needed)
    max_segments_to_analyze = min(100, data.shape[0])
    print(f"  Analyzing first {max_segments_to_analyze} segments...")
    
    for idx in range(max_segments_to_analyze):
        segment = data[idx]
        max_sum, start, end = kadane_max_subarray(segment)
        
        kadane_results.append({
            'segment_id': idx,
            'max_sum': max_sum,
            'start_index': start,
            'end_index': end,
            'interval_length': end - start + 1,
            'interval_percentage': ((end - start + 1) / len(segment)) * 100
        })
        
        if idx == 0:
            print(f"Example segment {idx}: max activity sum = {max_sum:.3f}, range = [{start}:{end}]")
    
    # Save Kadane results to CSV
    kadane_df = pd.DataFrame(kadane_results)
    kadane_df.to_csv(os.path.join(OUTPUT_DIR, 'kadane_results.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/kadane_results.csv")
    
    # Calculate and save Kadane statistics
    kadane_stats = {
        'mean_max_sum': kadane_df['max_sum'].mean(),
        'std_max_sum': kadane_df['max_sum'].std(),
        'mean_interval_length': kadane_df['interval_length'].mean(),
        'mean_interval_percentage': kadane_df['interval_percentage'].mean(),
        'min_max_sum': kadane_df['max_sum'].min(),
        'max_max_sum': kadane_df['max_sum'].max()
    }
    pd.DataFrame([kadane_stats]).to_csv(os.path.join(OUTPUT_DIR, 'kadane_statistics.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/kadane_statistics.csv")

    # ------------------------------------------------------
    # STEP 5 – VISUALIZATION
    # ------------------------------------------------------
    print("\n[5] Generating visualizations ...")
    
    # Plot clusters
    fig, axes = plt.subplots(len(clusters), 1, figsize=(10, 3 * len(clusters)))
    if len(clusters) == 1:
        axes = [axes]
    
    for i, cluster in enumerate(clusters):
        for ts in cluster:
            axes[i].plot(ts, alpha=0.5)
        axes[i].set_title(f'Cluster {i+1} ({len(cluster)} members, {cluster_summary[i]["percentage"]:.1f}%)')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    cluster_plot_path = os.path.join(OUTPUT_DIR, 'clusters.png')
    plt.savefig(cluster_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {cluster_plot_path}")
    
    # Plot Kadane example
    example_idx = 0
    segment = data[example_idx]
    max_sum, start, end = kadane_max_subarray(segment)
    
    plt.figure(figsize=(10, 4))
    plt.plot(segment, label='Signal', linewidth=1.5)
    plt.axvspan(start, end, alpha=0.3, color='orange', label=f'Kadane interval [{start}:{end}]')
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title(f'Segment {example_idx} with Maximum Subarray Interval (Sum={max_sum:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    kadane_plot_path = os.path.join(OUTPUT_DIR, 'kadane_example.png')
    plt.savefig(kadane_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {kadane_plot_path}")
    
    # Plot cluster size distribution
    plt.figure(figsize=(10, 6))
    cluster_sizes = [len(c) for c in clusters]
    plt.bar(range(1, len(clusters) + 1), cluster_sizes, color='steelblue', alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Members')
    plt.title(f'Cluster Size Distribution ({len(clusters)} clusters)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    dist_plot_path = os.path.join(OUTPUT_DIR, 'cluster_distribution.png')
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {dist_plot_path}")
    
    # Plot Kadane statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(kadane_df['max_sum'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Maximum Sum')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Maximum Subarray Sums')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(kadane_df['interval_length'], bins=30, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Interval Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Kadane Interval Lengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].scatter(kadane_df['interval_length'], kadane_df['max_sum'], alpha=0.5, color='green')
    axes[1, 0].set_xlabel('Interval Length')
    axes[1, 0].set_ylabel('Maximum Sum')
    axes[1, 0].set_title('Interval Length vs Maximum Sum')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(kadane_df['interval_percentage'], bins=30, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Interval Percentage (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Interval Coverage')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    kadane_stats_path = os.path.join(OUTPUT_DIR, 'kadane_statistics.png')
    plt.savefig(kadane_stats_path, dpi=150)
    plt.close()
    print(f"  Saved: {kadane_stats_path}")

    # ------------------------------------------------------
    # STEP 6 – GENERATE SUMMARY REPORT
    # ------------------------------------------------------
    print("\n[6] Generating summary report ...")
    
    summary_report = {
        'Pipeline Execution Summary': '',
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '': '',
        'Data Information': '',
        'Signal Type': SIGNAL_TYPE,
        'Segments Loaded': data.shape[0],
        'Segment Length': data.shape[1],
        ' ': '',
        'Clustering Results': '',
        'Number of Clusters': len(clusters),
        'Cluster Threshold': CLUSTER_THRESHOLD,
        'Largest Cluster Size': max(cluster_sizes),
        'Smallest Cluster Size': min(cluster_sizes),
        'Average Cluster Size': np.mean(cluster_sizes),
        '  ': '',
        'Kadane Analysis': '',
        'Segments Analyzed': max_segments_to_analyze,
        'Mean Max Sum': kadane_stats['mean_max_sum'],
        'Std Max Sum': kadane_stats['std_max_sum'],
        'Mean Interval Length': kadane_stats['mean_interval_length'],
        'Mean Interval Coverage (%)': kadane_stats['mean_interval_percentage'],
        '   ': '',
        'Output Files': '',
        'CSV Files': 'data_info.csv, cluster_summary.csv, closest_pairs.csv, kadane_results.csv, kadane_statistics.csv',
        'PNG Files': 'clusters.png, kadane_example.png, cluster_distribution.png, kadane_statistics.png'
    }
    
    summary_df = pd.DataFrame(list(summary_report.items()), columns=['Metric', 'Value'])
    summary_path = os.path.join(OUTPUT_DIR, 'pipeline_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print("\n=== Pipeline complete. All results saved to", OUTPUT_DIR, "===")
    print("\nGenerated files:")
    print("  CSV files:")
    print("    - data_info.csv: Data loading configuration")
    print("    - cluster_summary.csv: Cluster sizes and percentages")
    print("    - closest_pairs.csv: Closest pair analysis per cluster")
    print("    - kadane_results.csv: Kadane algorithm results for each segment")
    print("    - kadane_statistics.csv: Statistical summary of Kadane results")
    print("    - pipeline_summary.csv: Overall pipeline execution summary")
    print("  PNG files:")
    print("    - clusters.png: Visualization of all clusters")
    print("    - kadane_example.png: Example of Kadane interval detection")
    print("    - cluster_distribution.png: Bar chart of cluster sizes")
    print("    - kadane_statistics.png: Statistical plots of Kadane analysis")


if __name__ == "__main__":
    main()