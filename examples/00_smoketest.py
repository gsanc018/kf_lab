import numpy as np
from kf_lab.models.motion.cv import CV2D
from kf_lab.models.measurement.pos2d import Pos2D
from kf_lab.filters.kf import KalmanFilter
from kf_lab.sims.scenarios import straight_cv
from kf_lab.sensors.sampler import simulate_measurements
from kf_lab.viz.plots import plot_xy


def main():
    truth, t = straight_cv(duration_s=30.0, truth_dt=0.01, v0=10.0, heading_deg=25.0)
    meas_model = Pos2D(sigma_px=5.0, sigma_py=5.0)
    t_meas, Z = simulate_measurements(truth, t, meas_model, sensor_hz=5.0)

    # Build a CV KF
    x0 = truth[0].copy()
    P0 = np.diag([100.0, 100.0, 100.0, 100.0])
    cv = CV2D(q=0.2)
    kf = KalmanFilter(x0=x0, P0=P0, motion_model=cv, meas_model=meas_model, q=0.2)

    # Run at measurement times
    est = []
    last_t = t_meas[0]
    for ti, zi in zip(t_meas, Z):
        dt = float(ti - last_t)
        kf.predict(dt)
        kf.update(zi)
        est.append(kf.x.copy())
        last_t = ti

    est = np.asarray(est)
    # Slice truth to the same times for plotting
    idx = np.searchsorted(t, t_meas, side="left")
    truth_s = truth[np.clip(idx, 0, len(truth) - 1)]
    plot_xy(truth_s, est, Z, title="CV + pos-only KF smoke test")


if __name__ == "__main__":
    main()
