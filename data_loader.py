"""
data_loader.py

Functions to load PulseDB .mat segment files and construct a (N, L) numpy array
of univariate time-series (ABP by default).

PulseDB segment .mat files typically contain fields like 'ABP_F', 'PPG_F', etc.
This loader finds .mat files in a directory, reads a given field, and returns
an array of segments and optional metadata list.
"""

import os
import h5py
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm


def _list_mat_files(data_dir: str) -> List[str]:
    """
    Recursively find all .mat files in a directory.
    
    Args:
        data_dir: root directory to search
        
    Returns:
        Sorted list of full paths to .mat files
    """
    files = []
    for root, _, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.lower().endswith('.mat'):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def load_pulsedb_segments(data_dir: str, 
                         n_segments: int = 1000, 
                         field: str = 'ABP_F', 
                         random_sample: bool = False, 
                         skip_bad: bool = True) -> Tuple[np.ndarray, List[dict]]:
    """
    Load up to n_segments segments from PulseDB .mat files.
    
    This function handles both:
    1. Direct signal storage (signal_type at top level)
    2. Reference-based storage (Subj_Wins structure)
    
    Args:
        data_dir: directory containing PulseDB .mat Segment_Files
        n_segments: number of segments to load (will stop early if not enough files)
        field: which variable to read from .mat (e.g., 'ABP_F', 'PPG_F', 'ECG_F')
        random_sample: sample random files instead of first n
        skip_bad: skip segments that do not contain the requested field or are ill-shaped
        
    Returns:
        segments: numpy array of shape (M, L) where M <= n_segments
        metadata: list of dict with keys {path, original_length, loaded_field}
    """
    mat_files = _list_mat_files(data_dir)
    
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    
    print(f"Found {len(mat_files)} .mat files in {data_dir}")
    
    if random_sample:
        import random
        mat_files = mat_files.copy()
        random.shuffle(mat_files)
    
    segments = []
    metadata = []
    
    # Try to load from files
    for path in tqdm(mat_files, desc=f"Loading {field}"):
        if len(segments) >= n_segments:
            break
        
        try:
            with h5py.File(path, 'r') as f:
                seg = None
                
                # --- Case 1: signal directly stored at top level ---
                if field in f:
                    seg = np.array(f[field]).squeeze()
                
                # --- Case 2: reference structure (Subj_Wins) ---
                elif "Subj_Wins" in f:
                    subj = f["Subj_Wins"]
                    first_key = list(subj.keys())[0]
                    ref = subj[first_key][0][0]
                    deref = f[ref]
                    
                    # Some files point directly to a dataset
                    if isinstance(deref, h5py.Dataset):
                        seg = np.array(deref).squeeze()
                    # Others point to a group with named signals
                    elif isinstance(deref, h5py.Group) and field in deref:
                        seg = np.array(deref[field]).squeeze()
                
                # --- Validate and store ---
                if seg is not None and seg.ndim == 1 and np.isfinite(seg).all():
                    # Z-normalize
                    seg_normalized = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)
                    
                    segments.append(seg_normalized)
                    metadata.append({
                        'path': path,
                        'original_length': len(seg),
                        'loaded_field': field
                    })
                elif not skip_bad:
                    raise ValueError(f"Invalid segment in {path}")
                    
        except Exception as e:
            if not skip_bad:
                raise
            print(f"[WARN] Skipping {os.path.basename(path)}: {e}")
            continue
    
    if len(segments) == 0:
        raise RuntimeError(
            f"No valid segments loaded from {data_dir}. "
            f"Check file structure or signal type '{field}'."
        )
    
    # Convert to numpy array
    # Find minimum length to handle variable-length segments
    min_len = min(len(s) for s in segments)
    segments_array = np.array([s[:min_len] for s in segments])
    
    print(f"✓ Loaded {len(segments)} segments of length {min_len}")
    
    return segments_array, metadata


def get_available_fields(data_dir: str, sample_size: int = 10) -> List[str]:
    """
    Inspect a sample of .mat files to find available fields.
    
    Args:
        data_dir: directory containing .mat files
        sample_size: number of files to sample
        
    Returns:
        List of field names found
    """
    mat_files = _list_mat_files(data_dir)[:sample_size]
    fields = set()
    
    for path in mat_files:
        try:
            with h5py.File(path, 'r') as f:
                # Check top-level keys
                for key in f.keys():
                    if key not in ['Subj_Wins']:
                        fields.add(key)
                
                # Check reference structure
                if 'Subj_Wins' in f:
                    subj = f['Subj_Wins']
                    first_key = list(subj.keys())[0]
                    ref = subj[first_key][0][0]
                    deref = f[ref]
                    if isinstance(deref, h5py.Group):
                        for key in deref.keys():
                            fields.add(key)
        except:
            continue
    
    return sorted(list(fields))


if __name__ == "__main__":
    # Test the loader
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "PulseDB_MIMIC"
    
    print(f"Testing data loader with: {data_dir}\n")
    
    # Check available fields
    print("Checking available fields...")
    try:
        fields = get_available_fields(data_dir)
        print(f"Available fields: {fields}\n")
    except Exception as e:
        print(f"Could not determine fields: {e}\n")
    
    # Try loading ABP_F
    try:
        print("Attempting to load ABP_F segments...")
        segments, meta = load_pulsedb_segments(data_dir, n_segments=10, field='ABP_F')
        
        print(f"\n✓ Successfully loaded data!")
        print(f"  Shape: {segments.shape}")
        print(f"  Mean: {np.mean(segments):.4f}")
        print(f"  Std: {np.std(segments):.4f}")
        print(f"  Min: {np.min(segments):.4f}")
        print(f"  Max: {np.max(segments):.4f}")
        
        print(f"\nFirst metadata entry:")
        print(f"  {meta[0]}")
        
    except Exception as e:
        print(f"\n✗ Failed to load data: {e}")