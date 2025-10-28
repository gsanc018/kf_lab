import numpy as np


def nees(x_err: np.ndarray, P_seq: list[np.ndarray]) -> np.ndarray:
    """
    x_err: (N, n) state error vectors
    P_seq: list of (n,n) covs at each step
    Returns per-step NEES values.
    """
    N, n = x_err.shape
    vals = np.empty(N)
    for k in range(N):
        e = x_err[k]
        P = P_seq[k]
        vals[k] = float(e.T @ np.linalg.inv(P) @ e)
    return vals


def nis(innov: np.ndarray, S_seq: list[np.ndarray]) -> np.ndarray:
    """
    innov: (N, m) innovations
    S_seq: list of (m,m) innovation covariances
    """
    N, m = innov.shape
    vals = np.empty(N)
    for k in range(N):
        y = innov[k]
        S = S_seq[k]
        vals[k] = float(y.T @ np.linalg.inv(S) @ y)
    return vals
