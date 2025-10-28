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


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def finite_diff_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian for quick checks: df/dx at x."""
    x = x.astype(float)
    y0 = f(x)
    J = np.zeros((y0.size, x.size), dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        J[:, i] = (f(xp) - f(xm)) / (2.0 * eps)
    return J
