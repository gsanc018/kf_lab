import numpy as np


def nearest_on_times(t_truth: np.ndarray, truth: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_truth, t_query, side="left")
    idx = np.clip(idx, 0, len(t_truth) - 1)
    return truth[idx]
