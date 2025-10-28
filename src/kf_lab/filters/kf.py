import numpy as np


class KalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, motion_model, meas_model, q: float = 0.1):
        self.x = x0.copy()
        self.P = P0.copy()
        self.f = motion_model
        self.h = meas_model
        self.q = q

    def predict(self, dt: float) -> None:
        F = self.f.F(self.x, dt)
        Q = self.f.Q(dt, q=self.q)
        self.x = self.f.propagate(self.x, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray) -> None:
        H = self.h.H(self.x)
        R = self.h.R()
        y = z - self.h.h(self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
