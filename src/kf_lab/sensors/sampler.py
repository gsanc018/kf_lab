import numpy as np


def simulate_measurements(
    truth: np.ndarray,
    t_truth: np.ndarray,
    meas_model,
    sensor_hz: float = 5.0,
    bias: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng(123)

    dt_meas = 1.0 / sensor_hz
    t0, t1 = t_truth[0], t_truth[-1]
    n_meas = int(np.floor((t1 - t0) / dt_meas)) + 1
    t_meas = t0 + np.arange(n_meas) * dt_meas

    # nearest neighbor sample from truth
    idx = np.searchsorted(t_truth, t_meas, side="left")
    idx = np.clip(idx, 0, len(t_truth) - 1)
    x_samp = truth[idx]

    z = []
    R = meas_model.R()
    for x in x_samp:
        z_pred = meas_model.h(x)
        noise = rng.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R)
        zb = z_pred + noise + (bias if bias is not None else 0.0)
        z.append(zb)
    return t_meas, np.asarray(z)
