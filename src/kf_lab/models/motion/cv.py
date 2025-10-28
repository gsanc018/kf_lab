import numpy as np


class CV2D:
    """Constant Velocity in 2D with state [px, py, vx, vy]."""

    def __init__(self, q: float = 0.1):
        self.q = q

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        F = self.F(x, dt)
        return F @ x

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    def Q(self, dt: float, **params) -> np.ndarray:
        q = params.get("q", self.q)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q11 = dt4 / 4
        q13 = dt3 / 2
        q33 = dt2
        Q1 = np.array(
            [[q11, 0, q13, 0], [0, q11, 0, q13], [q13, 0, q33, 0], [0, q13, 0, q33]], dtype=float
        )
        return q * Q1
