import numpy as np


def straight_cv(
    duration_s: float = 20.0, truth_dt: float = 0.01, v0: float = 10.0, heading_deg: float = 0.0
):
    n = int(np.round(duration_s / truth_dt)) + 1
    t = np.linspace(0.0, duration_s, n)
    heading = np.deg2rad(heading_deg)
    vx = v0 * np.cos(heading)
    vy = v0 * np.sin(heading)
    px = vx * t
    py = vy * t
    truth = np.stack([px, py, np.full_like(t, vx), np.full_like(t, vy)], axis=1)
    return truth, t
