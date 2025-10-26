"""
kadane.py

Kadane's maximum subarray algorithm for 1D arrays.

Exports:
    kadane(arr) -> (max_sum, start_idx, end_idx)
"""

import numpy as np
from typing import Tuple

def kadane(arr: np.ndarray) -> Tuple[float, int, int]:
    if arr.size == 0:
        return 0.0, 0, -1
    max_ending = arr[0]
    max_sofar = arr[0]
    start = 0
    best_l = 0
    best_r = 0
    for i in range(1, len(arr)):
        if max_ending + arr[i] < arr[i]:
            max_ending = arr[i]
            start = i
        else:
            max_ending += arr[i]
        if max_ending > max_sofar:
            max_sofar = max_ending
            best_l = start
            best_r = i
    return float(max_sofar), int(best_l), int(best_r)

kadane_max_subarray = kadane
