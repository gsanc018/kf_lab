import numpy as np
from kf_lab.core.utils import wrap_angle


class EKF:
    """
    Generic EKF using provided motion and measurement models.
    Models must implement F(x,dt), Q(dt), h(x), H(x), R().
    """

    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        motion_model,
        meas_model,
        q_params: dict | None = None,
        r_params: dict | None = None,
    ):
        self.x = x0.copy()
        self.P = P0.copy()
        self.f = motion_model
        self.h = meas_model
        self.q_params = q_params or {}
        self.r_params = r_params or {}
        self.innovations: list[np.ndarray] = []
        self.S_seq: list[np.ndarray] = []
        self.P_seq: list[np.ndarray] = []

    def predict(self, dt: float) -> None:
        F = self.f.F(self.x, dt)
        Q = self.f.Q(dt, **self.q_params)
        # Nonlinear propagate:
        self.x = self.f.propagate(self.x, dt)
        self.P = F @ self.P @ F.T + Q
        self.P_seq.append(self.P.copy())

    def update(self, z: np.ndarray) -> None:
        H = self.h.H(self.x)
        R = self.h.R(self.x, **self.r_params)
        z_pred = self.h.h(self.x)
        y = z - z_pred
        # angle component wrap if measurement includes bearing (2nd element)
        if y.size >= 2:
            y[1] = wrap_angle(float(y[1]))
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
        self.innovations.append(y.copy())
        self.S_seq.append(S.copy())
