import numpy as np
from kf_lab.core.utils import wrap_angle


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

        # For adaptive tuners
        self.last_innovation: np.ndarray | None = None
        self.last_S: np.ndarray | None = None

    def predict(self, dt: float, Q_override: np.ndarray | None = None) -> None:
        """Predict step, with safe cropping for mixed-dimension IMMs."""
        # Crop state to match motion model dimension if needed
        state_dim = self.f.state_dim if hasattr(self.f, "state_dim") else None
        if state_dim is not None and self.x.size > state_dim:
            x_use = self.x[:state_dim]
        else:
            x_use = self.x

        F = self.f.F(x_use, dt)
        Q = Q_override if Q_override is not None else self.f.Q(dt, **self.q_params)

        # Nonlinear propagate:
        self.x = self.f.propagate(x_use, dt)

        # Use top-left block of P if needed
        P_reduced = self.P[: F.shape[0], : F.shape[0]]
        self.P = F @ P_reduced @ F.T + Q
        self.P_seq.append(self.P.copy())

    def update(
        self, z: np.ndarray, R_override: np.ndarray | None = None, return_likelihood: bool = False
    ):
        """
        Perform measurement update.
        If return_likelihood=True, returns measurement likelihood for IMM weighting.
        """
        H = self.h.H(self.x)
        R = R_override if R_override is not None else self.h.R(self.x, **self.r_params)
        z_pred = self.h.h(self.x)
        y = z - z_pred

        # Wrap bearing residual if applicable
        if y.size >= 2:
            y[1] = wrap_angle(float(y[1]))

        # Innovation covariance
        S = H @ self.P @ H.T + R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Joseph form covariance update (numerically stable)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        # Record for diagnostics
        self.innovations.append(y.copy())
        self.S_seq.append(S.copy())
        self.P_seq.append(self.P.copy())

        # Expose for tuners
        self.last_innovation = y.copy()
        self.last_S = S.copy()

        # If IMM requests likelihood, compute and return it
        if return_likelihood:
            m = z.shape[0]
            detS = np.linalg.det(S)
            if detS <= 0:
                detS = 1e-12
            norm_const = 1.0 / np.sqrt(((2 * np.pi) ** m) * detS)
            exponent = -0.5 * np.dot(y.T, np.linalg.inv(S) @ y)
            likelihood = float(norm_const * np.exp(exponent))
            return likelihood
