import numpy as np
from kf_lab.core.utils import wrap_angle


class CT2D:
    """
    Coordinated Turn model in 2D.
    State x = [px, py, v, psi, omega]
      px,py: position (m)
      v: speed (m/s)
      psi: heading (rad)
      omega: turn rate (rad/s)
    Process noise: white acceleration on v and white noise on omega via scalar qv, qw.
    """

    state_dim = 5  # <--- added for IMM/EKF compatibility

    def __init__(self, qv: float = 0.1, qw: float = 0.01):
        self.qv = qv
        self.qw = qw

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        px, py, v, psi, w = x
        if abs(w) < 1e-6:
            pxn = px + v * np.cos(psi) * dt
            pyn = py + v * np.sin(psi) * dt
        else:
            s = np.sin(psi + w * dt) - np.sin(psi)
            c = np.cos(psi + w * dt) - np.cos(psi)
            pxn = px + (v / w) * (s)
            pyn = py - (v / w) * (c)
        vn = v
        psin = wrap_angle(psi + w * dt)
        wn = w
        return np.array([pxn, pyn, vn, psin, wn], dtype=float)

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        px, py, v, psi, w = x
        spsi, cpsi = np.sin(psi), np.cos(psi)

        if abs(w) < 1e-6:
            # Linearized near zero-turn
            F = np.eye(5, dtype=float)
            F[0, 2] = cpsi * dt
            F[0, 3] = -v * spsi * dt
            F[1, 2] = spsi * dt
            F[1, 3] = v * cpsi * dt
            F[3, 4] = dt
            return F

        # General case
        swdt = np.sin(psi + w * dt)
        cwdt = np.cos(psi + w * dt)
        spsi, cpsi = np.sin(psi), np.cos(psi)

        dpx_dv = (1.0 / w) * (swdt - spsi)
        dpx_dpsi = (v / w) * (cwdt - cpsi)
        dpx_dw = (-v / (w * w)) * (swdt - spsi) + (v / w) * (cwdt * dt)

        dpy_dv = -(1.0 / w) * (cwdt - cpsi)
        dpy_dpsi = (v / w) * (swdt - spsi)
        dpy_dw = (v / (w * w)) * (cwdt - cpsi) + (v / w) * (swdt * dt)

        F = np.eye(5, dtype=float)
        F[0, 2] = dpx_dv
        F[0, 3] = dpx_dpsi
        F[0, 4] = dpx_dw
        F[1, 2] = dpy_dv
        F[1, 3] = dpy_dpsi
        F[1, 4] = dpy_dw
        F[3, 4] = dt
        return F

    def Q(self, dt: float, **params) -> np.ndarray:
        qv = params.get("qv", self.qv)
        qw = params.get("qw", self.qw)
        # simple diagonal on v and omega with dt scaling
        Q = np.zeros((5, 5), dtype=float)
        Q[2, 2] = qv * dt
        Q[4, 4] = qw * dt
        return Q
