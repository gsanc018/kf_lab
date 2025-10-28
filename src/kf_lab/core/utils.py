import numpy as np


def block_diag(*arrs: np.ndarray) -> np.ndarray:
    """Small helper for block diagonal concatenation."""
    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=float)
    r, c = 0, 0
    for a in arrs:
        rr, cc = a.shape
        out[r : r + rr, c : c + cc] = a
        r += rr
        c += cc
    return out
