import numpy as np


class CA2D:
    """
    Constant Acceleration in 2D.
    State x = [px, py, vx, vy, ax, ay].
    Units: m, s, rad.
    Process noise modeled as white jerk -> standard discretization with scalar q.
    """

    def __init__(self, q: float = 0.01):
        self.q = q

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        dt2 = dt * dt
        return np.array(
            [
                [1, 0, dt, 0, 0.5 * dt2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        return self.F(x, dt) @ x

    def Q(self, dt: float, **params) -> np.ndarray:
        # White-jerk spectral density q (scalar)
        q = params.get("q", self.q)
        dt2, dt3, dt4, dt5 = dt * dt, dt**3, dt**4, dt**5
        q11 = dt5 / 20
        q13 = dt4 / 8
        q15 = dt3 / 6
        q33 = dt3 / 3
        q35 = dt2 / 2
        q55 = dt
        Q1 = np.array(
            [
                [q11, 0, q13, 0, q15, 0],
                [0, q11, 0, q13, 0, q15],
                [q13, 0, q33, 0, q35, 0],
                [0, q13, 0, q33, 0, q35],
                [q15, 0, q35, 0, q55, 0],
                [0, q15, 0, q35, 0, q55],
            ],
            dtype=float,
        )
        return q * Q1
