# Time-Series Clustering on PulseDB

Divide-and-conquer clustering and analysis of physiological time-series data from PulseDB using algorithmic approaches (DTW, Kadane's algorithm).

## Features

- Recursive divide-and-conquer clustering
- DTW-based similarity measurement
- Closest pair validation
- Kadane's algorithm for peak detection
- CSV exports and visualizations

## Installation

```bash
# Clone repository
git clone https://github.com/Nika3406/pulsedb-clustering.git
cd pulsedb-clustering

# Install dependencies
pip install -r requirements.txt

# Download PulseDB dataset (see instructions below)
```

### Download PulseDB Dataset

Download from [PulseDB GitHub](https://github.com/pulselabteam/PulseDB) or [Box](https://rutgers.app.box.com/s/sw3c51fr5oybz6mhqsphh5zg8ibxw800)

Extract to `PulseDB_MIMIC/` folder in project root.

## Quick Start

```bash
# Run with default settings (5 segments)
python app.py

# Run with 1000 segments
# Edit app.py: N_SEGMENTS = 1000
python app.py
```

## Usage

**Basic pipeline:**
```bash
python app.py
```

**Advanced pipeline:**
```bash
python run_pipeline.py --data_dir PulseDB_MIMIC --field ABP_F --n_segments 1000
```

## Project Structure

```
├── app.py                 # Main pipeline
├── data_loader.py         # Load PulseDB data
├── clustering.py          # Divide-and-conquer clustering
├── closest_pair.py        # Closest pair algorithm
├── kadane.py             # Maximum subarray
├── dtw.py                # Dynamic Time Warping
├── viz.py                # Visualizations
├── requirements.txt      # Dependencies
└── results/              # Output directory
    ├── *.csv            # Statistical results
    └── *.png            # Visualizations
```

## Algorithms

### 1. Divide-and-Conquer Clustering
Recursively partitions time-series based on DTW similarity until clusters are cohesive (below threshold).

### 2. Dynamic Time Warping (DTW)
Measures similarity between time-series allowing temporal warping. O(nm) complexity.

### 3. Closest Pair
Finds most similar pair within each cluster to validate cohesion. O(n²) per cluster.

### 4. Kadane's Algorithm
Detects maximum sum subarray in O(n) time to identify peak activity intervals.

## Output Files

**CSV Files:**
- `data_info.csv` - Loading configuration
- `cluster_summary.csv` - Cluster sizes
- `closest_pairs.csv` - Pair analysis per cluster
- `kadane_results.csv` - Peak detection results
- `kadane_statistics.csv` - Statistical summary
- `pipeline_summary.csv` - Overall execution report

**Visualizations:**
- `clusters.png` - All clusters
- `kadane_example.png` - Peak detection example
- `cluster_distribution.png` - Cluster sizes
- `kadane_statistics.png` - Statistical plots

## Documentation

Full report available in `docs/report.pdf`

## Requirements

- Python 3.7+
- numpy, scipy, matplotlib, pandas, tqdm, h5py

## License

MIT License - Dataset follows PulseDB licenses (ODbL for MIMIC, CC BY-NC-SA 4.0 for VitalDB)

## References

PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB. *Frontiers in Digital Health*, 2023.