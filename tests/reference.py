"""Reference implementation of SAX using numpy."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPSILON = 1e-6


def discretise(ts, window, stride, w, alpha):
    # Create sliding windows and z-normalise each sliding window
    windows = sliding_window_view(ts, window)[::stride]
    windows = zscore(windows, axis=1)

    # PAA
    indices = np.arange(window * w) // w
    indices = indices.reshape(w, -1)
    paa = windows[:, indices].mean(axis=2)

    # SAX
    paa[abs(paa) < EPSILON] = 0.0
    sax = np.digitize(paa, np.array(breakpoints[alpha], dtype=np.float32))

    return sax.tolist()


def paa_window(ts, window, stride, w):
    # Create sliding windows and z-normalise each sliding window
    windows = sliding_window_view(ts, window)[::stride]
    windows = zscore(windows, axis=1)

    # PAA
    indices = np.arange(window * w) // w
    indices = indices.reshape(w, -1)
    paa = windows[:, indices].mean(axis=2)

    return paa.tolist()


def zscore(ts, axis):
    mean = np.mean(ts, axis=axis, keepdims=True)
    std = np.std(ts, axis=axis, keepdims=True)
    return (ts - mean) / std


breakpoints = [
    [],
    [],  # alpha = 0 and alpha = 1 are invalid
    [0.0],
    [-0.43072729929545756, 0.43072729929545744],
    [-0.6744897501960817, 0.0, 0.6744897501960817],
    [-0.8416212335729142, -0.2533471031357997, 0.2533471031357997, 0.8416212335729143],
    [
        -0.967421566101701,
        -0.43072729929545756,
        0.0,
        0.43072729929545744,
        0.967421566101701,
    ],
    [
        -1.0675705238781414,
        -0.5659488219328631,
        -0.1800123697927051,
        0.18001236979270496,
        0.5659488219328631,
        1.0675705238781412,
    ],
    [
        -1.1503493803760079,
        -0.6744897501960817,
        -0.31863936396437514,
        0.0,
        0.31863936396437514,
        0.6744897501960817,
        1.1503493803760079,
    ],
    [
        -1.22064034884735,
        -0.7647096737863871,
        -0.43072729929545756,
        -0.13971029888186212,
        0.13971029888186212,
        0.43072729929545744,
        0.7647096737863871,
        1.2206403488473496,
    ],
    [
        -1.2815515655446004,
        -0.8416212335729142,
        -0.5244005127080409,
        -0.2533471031357997,
        0.0,
        0.2533471031357997,
        0.5244005127080407,
        0.8416212335729143,
        1.2815515655446004,
    ],
    [
        -1.335177736118937,
        -0.9084578685373851,
        -0.6045853465832371,
        -0.3487556955170447,
        -0.11418529432142838,
        0.11418529432142824,
        0.3487556955170447,
        0.6045853465832371,
        0.9084578685373853,
        1.3351777361189363,
    ],
    [
        -1.382994127100638,
        -0.967421566101701,
        -0.6744897501960817,
        -0.43072729929545756,
        -0.2104283942479247,
        0.0,
        0.21042839424792484,
        0.43072729929545744,
        0.6744897501960817,
        0.967421566101701,
        1.382994127100638,
    ],
]
