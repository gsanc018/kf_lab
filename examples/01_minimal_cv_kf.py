from kf_lab.experiments.run_once import run_once

if __name__ == "__main__":
    # Feel free to change these
    run_once(
        duration_s=40.0,
        truth_dt=0.01,
        v0=10.0,
        heading_deg=25.0,
        sensor_hz=5.0,
        q_process=0.2,
        sigma_px=5.0,
        sigma_py=5.0,
        P0_diag=[100, 100, 100, 100],
        seed=123,
        save_plots=True,
    )
