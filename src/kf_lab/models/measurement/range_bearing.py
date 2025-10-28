import numpy as np
from kf_lab.core.utils import wrap_angle


class RangeBearing2D:
    """
    Range/Bearing from a fixed sensor platform at (xs, ys) with heading psi_s.
    Works with both CV/CA state [px,py,...] and CT state [px,py,v,psi,omega] (reads only px,py).
    z = [range, bearing], where bearing is relative to sensor heading (wrapped to [-pi,pi)).
    """

    def __init__(
        self,
        xs: float = 0.0,
        ys: float = 0.0,
        psi_s: float = 0.0,
        sigma_r: float = 5.0,
        sigma_b: float = np.deg2rad(2.0),
    ):
        self.xs = xs
        self.ys = ys
        self.psi_s = psi_s
        self.sigma_r = sigma_r
        self.sigma_b = sigma_b

    def h(self, x: np.ndarray) -> np.ndarray:
        px, py = x[0], x[1]
        dx, dy = px - self.xs, py - self.ys
        rng = np.hypot(dx, dy)
        bearing = wrap_angle(np.arctan2(dy, dx) - self.psi_s)
        return np.array([rng, bearing], dtype=float)

    def H(self, x: np.ndarray) -> np.ndarray:
        px, py = x[0], x[1]
        dx, dy = px - self.xs, py - self.ys
        eps = 1e-9
        r2 = dx * dx + dy * dy
        r2 = max(r2, eps)  # clamp to avoid divide-by-zero
        r = np.sqrt(r2)

        dr_dpx = dx / r
        dr_dpy = dy / r
        db_dpx = -dy / r2
        db_dpy = dx / r2

        H = np.zeros((2, x.size), dtype=float)
        H[0, 0] = dr_dpx
        H[0, 1] = dr_dpy
        H[1, 0] = db_dpx
        H[1, 1] = db_dpy
        return H

    def R(self, **params) -> np.ndarray:
        sr = params.get("sigma_r", self.sigma_r)
        sb = params.get("sigma_b", self.sigma_b)
        return np.diag([sr**2, sb**2])
