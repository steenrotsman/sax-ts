"""Reference implementation of SAX using numpy."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm, zscore

# windows = (n - window) / stride + 1
# paa in O(windows)


def discretise(ts, window, stride, w, alpha):
    # Create sliding windows and z-normalise each sliding window
    windows = sliding_window_view(ts, window)[::stride]
    windows = zscore(windows, axis=1)

    # PAA
    indices = np.arange(window * w) // w
    indices = indices.reshape(w, -1)
    paa = windows[:, indices].mean(axis=2)

    # SAX
    breakpoints = norm.ppf(np.arange(1, alpha) / alpha, loc=0)
    sax = np.digitize(paa, breakpoints)

    return sax.tolist()
