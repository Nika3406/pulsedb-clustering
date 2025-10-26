"""
dtw.py — Basic Dynamic Time Warping distance computation
"""
import numpy as np

def dtw_distance(ts1, ts2):
    """Compute a basic DTW distance between two time-series arrays."""
    n, m = len(ts1), len(ts2)
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ts1[i - 1] - ts2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return dp[n, m] / (n + m)
