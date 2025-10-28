import numpy as np

from kf_lab.models.motion.cv import CV2D
from kf_lab.models.motion.ct import CT2D
from kf_lab.models.measurement.range_bearing import RangeBearing2D
from kf_lab.filters.ekf import EKF
from kf_lab.sensors.sampler import simulate_measurements
from kf_lab.core.state import nearest_on_times
from kf_lab.core.utils import wrap_angle
from kf_lab.metrics.error import rmse_pos_vel
from kf_lab.metrics.consistency import nees, nis
from kf_lab.viz.plots import plot_xy


def ct_truth(duration_s=40.0, truth_dt=0.01, v0=12.0, psi0_deg=0.0, turn_deg_per_s=6.0):
    """
    Generate a Coordinated Turn (CT) kinematic truth by integrating CT forward.
    Returns:
      truth_4d: (N,4) [px,py,vx,vy] for easy plotting/RMSE
      t:        (N,)   time stamps
      Xct:      (N,5) [px,py,v,psi,omega] full CT state for NEES
    """
    ct = CT2D()
    psi0 = np.deg2rad(psi0_deg)
    w = np.deg2rad(turn_deg_per_s)
    x = np.array([0.0, 0.0, v0, psi0, w], dtype=float)

    t = np.arange(0.0, duration_s + 1e-12, truth_dt)
    X = np.zeros((t.size, 5))
    for i in range(t.size):
        X[i] = x
        if i + 1 < t.size:
            x = ct.propagate(x, truth_dt)

    vx = X[:, 2] * np.cos(X[:, 3])
    vy = X[:, 2] * np.sin(X[:, 3])
    truth_4d = np.stack([X[:, 0], X[:, 1], vx, vy], axis=1)
    return truth_4d, t, X


def main():
    # 1) Truth: CT trajectory
    truth4, t_truth, Xct = ct_truth(
        duration_s=40.0, truth_dt=0.01, v0=12.0, psi0_deg=15.0, turn_deg_per_s=6.0
    )

    # 2) Nonlinear sensor: range/bearing from origin (heading 0)
    meas_model = RangeBearing2D(xs=0.0, ys=0.0, psi_s=0.0, sigma_r=5.0, sigma_b=np.deg2rad(2.0))
    t_meas, Z = simulate_measurements(truth4, t_truth, meas_model, sensor_hz=5.0)

    # 3) Two EKFs: CV (model mismatch) vs CT (correct model)

    # EKF with CV motion model (state 4D)
    # Initialize from the first CT state by converting v,psi -> vx,vy
    vx0 = Xct[0, 2] * np.cos(Xct[0, 3])
    vy0 = Xct[0, 2] * np.sin(Xct[0, 3])
    x0_cv = np.array([Xct[0, 0], Xct[0, 1], vx0, vy0], dtype=float)
    P0_cv = np.diag([200.0, 200.0, 100.0, 100.0])
    ekf_cv = EKF(
        x0=x0_cv,
        P0=P0_cv,
        motion_model=CV2D(q=0.2),
        meas_model=meas_model,
        q_params={"q": 0.2},
        r_params={"sigma_r": 5.0, "sigma_b": np.deg2rad(2.0)},
    )

    # EKF with CT motion model (state 5D)
    x0_ct = Xct[0].copy()
    P0_ct = np.diag([200.0, 200.0, 50.0, np.deg2rad(10.0) ** 2, np.deg2rad(2.0) ** 2])
    ekf_ct = EKF(
        x0=x0_ct,
        P0=P0_ct,
        motion_model=CT2D(qv=0.2, qw=np.deg2rad(0.5)),
        meas_model=meas_model,
        q_params={"qv": 0.2, "qw": np.deg2rad(0.5)},
        r_params={"sigma_r": 5.0, "sigma_b": np.deg2rad(2.0)},
    )

    # 4) Run both filters on the same measurements
    est_cv = []
    est_ct = []
    last_t = float(t_meas[0])
    for ti, zi in zip(t_meas, Z):
        dt = float(ti - last_t)

        ekf_cv.predict(dt)
        ekf_cv.update(zi)
        est_cv.append(ekf_cv.x.copy())

        ekf_ct.predict(dt)
        ekf_ct.update(zi)
        est_ct.append(ekf_ct.x.copy())

        last_t = float(ti)

    est_cv = np.asarray(est_cv)  # shape (N, 4): [px,py,vx,vy]
    est_cts = np.asarray(est_ct)  # shape (N, 5): [px,py,v,psi,omega]

    # Convert CT estimate to Cartesian [px,py,vx,vy] for intuitive RMSE plots
    vx_ct = est_cts[:, 2] * np.cos(est_cts[:, 3])
    vy_ct = est_cts[:, 2] * np.sin(est_cts[:, 3])
    est_ct_xyv = np.stack([est_cts[:, 0], est_cts[:, 1], vx_ct, vy_ct], axis=1)

    # Align truth to measurement times
    truth_s = nearest_on_times(t_truth, truth4, t_meas)  # (N,4)
    Xct_s = nearest_on_times(t_truth, Xct, t_meas)  # (N,5)

    # 5) Metrics

    # CV EKF — RMSE in 4D, NEES/NIS in 4D
    s_cv = rmse_pos_vel(truth_s, est_cv)
    s_cv["NEES"] = float(np.mean(nees(est_cv - truth_s, ekf_cv.P_seq)))  # P_seq: 4x4
    s_cv["NIS"] = float(np.mean(nis(np.asarray(ekf_cv.innovations), ekf_cv.S_seq)))

    # CT EKF — RMSE in Cartesian 4D (for interpretability), NEES in native 5D
    s_ct = rmse_pos_vel(truth_s, est_ct_xyv)

    # NEES in 5D: error in [px,py,v,psi,omega] with wrapped heading
    err_ct_state = est_cts - Xct_s
    err_ct_state[:, 3] = np.array([wrap_angle(a) for a in err_ct_state[:, 3]])
    s_ct["NEES"] = float(np.mean(nees(err_ct_state, ekf_ct.P_seq)))  # P_seq: 5x5
    s_ct["NIS"] = float(np.mean(nis(np.asarray(ekf_ct.innovations), ekf_ct.S_seq)))

    print("EKF(CV + RB)  :", s_cv)
    print("EKF(CT + RB)  :", s_ct)

    # 6) Plots
    plot_xy(truth_s, est_cv, Z, title="EKF with CV model (RB sensor)")
    plot_xy(truth_s, est_ct_xyv, Z, title="EKF with CT model (RB sensor)")


if __name__ == "__main__":
    main()
