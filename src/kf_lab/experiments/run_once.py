from __future__ import annotations
import os
import time
import json
import numpy as np
from pathlib import Path

from kf_lab.models.motion.cv import CV2D
from kf_lab.models.measurement.pos2d import Pos2D
from kf_lab.filters.kf import KalmanFilter
from kf_lab.sims.scenarios import straight_cv
from kf_lab.sensors.sampler import simulate_measurements
from kf_lab.core.state import nearest_on_times
from kf_lab.metrics.error import rmse_pos_vel
from kf_lab.metrics.consistency import nees, nis
from kf_lab.viz.plots import plot_xy


def run_once(
    duration_s: float = 30.0,
    truth_dt: float = 0.01,
    v0: float = 12.0,
    heading_deg: float = 20.0,
    sensor_hz: float = 5.0,
    q_process: float = 0.2,
    sigma_px: float = 5.0,
    sigma_py: float = 5.0,
    P0_diag: list[float] = [100.0, 100.0, 100.0, 100.0],
    seed: int = 123,
    save_plots: bool = True,
    runs_dir: str | Path = "runs",
):
    # 1) truth
    truth, t_truth = straight_cv(
        duration_s=duration_s, truth_dt=truth_dt, v0=v0, heading_deg=heading_deg
    )

    # 2) sensor model and measurements
    meas_model = Pos2D(sigma_px=sigma_px, sigma_py=sigma_py)
    rng = np.random.default_rng(seed)
    t_meas, Z = simulate_measurements(truth, t_truth, meas_model, sensor_hz=sensor_hz, rng=rng)

    # 3) filter
    x0 = truth[0].copy()
    P0 = np.diag(P0_diag)
    cv = CV2D(q=q_process)
    kf = KalmanFilter(x0=x0, P0=P0, motion_model=cv, meas_model=meas_model, q=q_process)

    # 4) run at measurement times
    est = []
    last_t = float(t_meas[0])
    for ti, zi in zip(t_meas, Z):
        dt = float(ti - last_t)
        kf.predict(dt)
        kf.update(zi)
        est.append(kf.x.copy())
        last_t = float(ti)
    est = np.asarray(est)

    # 5) align truth for metrics and plots
    truth_s = nearest_on_times(t_truth, truth, t_meas)

    # 6) metrics
    err = est - truth_s
    summary = rmse_pos_vel(truth_s, est)
    nees_vals = nees(err, kf.P_seq)
    nis_vals = nis(np.asarray(kf.innovations), kf.S_seq)
    summary.update(
        {
            "mean_nees": float(np.mean(nees_vals)),
            "mean_nis": float(np.mean(nis_vals)),
            "N": int(len(t_meas)),
        }
    )

    # 7) save artifacts
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(runs_dir) / f"cv_posonly_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "truth.npy", truth_s)
    np.save(out_dir / "est.npy", est)
    np.save(out_dir / "meas.npy", Z)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if save_plots:
        # show and also save
        from matplotlib import pyplot as plt

        plot_xy(truth_s, est, Z, title="CV + pos-only KF")
        plt.savefig(out_dir / "xy.png", dpi=160)
        plt.close()

    print("Run summary:", summary)
    print(f"Saved to: {out_dir.resolve()}")
    return summary, out_dir


if __name__ == "__main__":
    run_once()
