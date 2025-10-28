import numpy as np


# def rmse_pos_vel(truth: np.ndarray, est: np.ndarray) -> dict:
#     """
#     truth, est shape (N, 4) with [px, py, vx, vy].
#     Returns dict of RMSE for pos and vel.
#     """
#     assert truth.shape == est.shape
#     err = est - truth
#     px, py, vx, vy = err.T
#     rmse = lambda a: float(np.sqrt(np.mean(a**2)))
#     return {
#         "rmse_px": rmse(px),
#         "rmse_py": rmse(py),
#         "rmse_pos": rmse(np.hypot(px, py)),
#         "rmse_vx": rmse(vx),
#         "rmse_vy": rmse(vy),
#         "rmse_vel": rmse(np.hypot(vx, vy)),
#     }


import numpy as np
from numpy.typing import NDArray
from typing import Dict


def rmse_pos_vel(truth: NDArray[np.floating], est: NDArray[np.floating]) -> Dict[str, float]:
    assert truth.shape == est.shape, f"shape mismatch: {truth.shape} vs {est.shape}"
    err = est - truth
    px, py, vx, vy = err.T
    rmse = lambda a: float(np.sqrt(np.mean(a**2)))
    return {
        "rmse_px": rmse(px),
        "rmse_py": rmse(py),
        "rmse_pos": rmse(np.hypot(px, py)),
        "rmse_vx": rmse(vx),
        "rmse_vy": rmse(vy),
        "rmse_vel": rmse(np.hypot(vx, vy)),
    }
