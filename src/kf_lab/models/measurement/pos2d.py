import numpy as np


class Pos2D:
    """Measures [px, py] with Gaussian noise."""

    def __init__(self, sigma_px: float = 5.0, sigma_py: float = 5.0):
        self.sigma_px = sigma_px
        self.sigma_py = sigma_py

    def h(self, x: np.ndarray) -> np.ndarray:
        return x[:2]

    def H(self, x: np.ndarray) -> np.ndarray:
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def R(self, **params) -> np.ndarray:
        spx = params.get("sigma_px", self.sigma_px)
        spy = params.get("sigma_py", self.sigma_py)
        return np.diag([spx**2, spy**2])
