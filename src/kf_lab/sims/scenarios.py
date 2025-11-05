import numpy as np
from kf_lab.models.motion.cv import CV2D
from kf_lab.models.motion.ca import CA2D
from kf_lab.models.motion.ct import CT2D


# 1) Straight CV
def straight_cv(duration_s=20.0, truth_dt=0.01, v0=10.0, heading_deg=0.0, offset_xy=(0.0, 0.0)):
    n = int(np.round(duration_s / truth_dt)) + 1
    t = np.linspace(0.0, duration_s, n)
    heading = np.deg2rad(heading_deg)
    vx = v0 * np.cos(heading)
    vy = v0 * np.sin(heading)
    px = offset_xy[0] + vx * t
    py = offset_xy[1] + vy * t
    truth = np.stack([px, py, np.full_like(t, vx), np.full_like(t, vy)], axis=1)
    return truth, t


# 2) Accelerating CA
def accelerating_ca(
    duration_s=30.0, truth_dt=0.01, v0=8.0, a=0.5, heading_deg=0.0, offset_xy=(0.0, 0.0)
):
    ca = CA2D(q=0.0)
    psi = np.deg2rad(heading_deg)
    ax = a * np.cos(psi)
    ay = a * np.sin(psi)
    vx0 = v0 * np.cos(psi)
    vy0 = v0 * np.sin(psi)
    x = np.array([offset_xy[0], offset_xy[1], vx0, vy0, ax, ay], dtype=float)

    t = np.arange(0.0, duration_s + 1e-12, truth_dt)
    X = np.zeros((t.size, 6))
    for i in range(t.size):
        X[i] = x
        if i + 1 < t.size:
            x = ca.propagate(x, truth_dt)

    truth = np.stack([X[:, 0], X[:, 1], X[:, 2], X[:, 3]], axis=1)
    return truth, t


# 3) Constant-radius CT
def ct_turn(
    duration_s=40.0, truth_dt=0.01, v0=12.0, psi0_deg=0.0, turn_deg_per_s=6.0, offset_xy=(0.0, 0.0)
):
    ct = CT2D()
    psi0 = np.deg2rad(psi0_deg)
    w = np.deg2rad(turn_deg_per_s)
    x = np.array([offset_xy[0], offset_xy[1], v0, psi0, w], dtype=float)

    t = np.arange(0.0, duration_s + 1e-12, truth_dt)
    X = np.zeros((t.size, 5))
    for i in range(t.size):
        X[i] = x
        if i + 1 < t.size:
            x = ct.propagate(x, truth_dt)

    vx = X[:, 2] * np.cos(X[:, 3])
    vy = X[:, 2] * np.sin(X[:, 3])
    truth = np.stack([X[:, 0], X[:, 1], vx, vy], axis=1)
    return truth, t, X


# 4) Zigzag
def zigzag_cv(
    duration_s=40.0,
    truth_dt=0.01,
    v=10.0,
    psi1_deg=20.0,
    psi2_deg=-20.0,
    T_seg=5.0,
    offset_xy=(0.0, 0.0),
):
    cv = CV2D()
    psi1 = np.deg2rad(psi1_deg)
    psi2 = np.deg2rad(psi2_deg)
    x = np.array([offset_xy[0], offset_xy[1], v * np.cos(psi1), v * np.sin(psi1)], dtype=float)

    t = np.arange(0.0, duration_s + 1e-12, truth_dt)
    X = np.zeros((t.size, 4))
    for i, ti in enumerate(t):
        seg = int(np.floor(ti / T_seg))
        psi = psi1 if (seg % 2 == 0) else psi2
        speed = np.hypot(x[2], x[3])
        x[2] = speed * np.cos(psi)
        x[3] = speed * np.sin(psi)

        X[i] = x
        if i + 1 < t.size:
            x = cv.propagate(x, truth_dt)

    truth = X
    return truth, t


# 5) Lawnmower
def lawnmower_cv(
    duration_s=60.0, truth_dt=0.01, v=8.0, leg_s=5.0, rows=4, spacing=50.0, offset_xy=(0.0, 0.0)
):
    cv = CV2D()
    psi_right = 0.0
    psi_left = np.pi
    x = np.array(
        [offset_xy[0], offset_xy[1], v * np.cos(psi_right), v * np.sin(psi_right)], dtype=float
    )

    t = np.arange(0.0, duration_s + 1e-12, truth_dt)
    X = np.zeros((t.size, 4))
    row = 0
    heading = psi_right

    for i, ti in enumerate(t):
        leg_idx = int(np.floor(ti / leg_s))
        if leg_idx != int(np.floor((ti - truth_dt) / leg_s)):
            heading = psi_left if (row % 2 == 0) else psi_right
            row += 1
            x[1] -= spacing

        x[2] = v * np.cos(heading)
        x[3] = v * np.sin(heading)

        X[i] = x
        if i + 1 < t.size:
            x = cv.propagate(x, truth_dt)

    return X, t


# 6) Spiral CT
def spiral_ct(
    duration_s=40.0,
    truth_dt=0.01,
    v0=10.0,
    psi0_deg=0.0,
    omega0_deg_s=8.0,
    omega_final_deg_s=1.0,
    offset_xy=(0.0, 0.0),
):
    ct = CT2D()
    psi0 = np.deg2rad(psi0_deg)
    w0 = np.deg2rad(omega0_deg_s)
    wf = np.deg2rad(omega_final_deg_s)

    x = np.array([offset_xy[0], offset_xy[1], v0, psi0, w0], dtype=float)
    t = np.arange(0.0, duration_s + 1e-12, truth_dt)
    X = np.zeros((t.size, 5))
    for i, ti in enumerate(t):
        X[i] = x
        if i + 1 < t.size:
            alpha = (ti + truth_dt) / max(duration_s, 1e-9)
            x[4] = (1 - alpha) * w0 + alpha * wf
            x = ct.propagate(x, truth_dt)

    vx = X[:, 2] * np.cos(X[:, 3])
    vy = X[:, 2] * np.sin(X[:, 3])
    truth = np.stack([X[:, 0], X[:, 1], vx, vy], axis=1)
    return truth, t, X


# 7) Composite mission: CV + CT U-turn + CA accel + CV zigzagz


import numpy as np
from kf_lab.models.motion.cv import CV2D
from kf_lab.models.motion.ca import CA2D
from kf_lab.models.motion.ct import CT2D


def mission_u_turn_accel_zigzag(
    truth_dt=0.01,
    # Option A: provide a total duration and an allocation (fractions sum to 1)
    duration_s=None,
    split=(0.2, 0.2, 0.6),  # fractions for (CV1, CA, Zigzag) of remaining time after U-turn
    # Option B: or specify explicit segment durations (used when duration_s is None)
    t_cv1=10.0,
    t_ca=8.0,
    t_zigzag=12.0,
    # Kinematics & behavior
    v0=10.0,
    heading0_deg=0.0,  # initial heading, 0° = +x
    turn_rate_deg_s=10.0,  # CT turn rate (controls U-turn duration)
    a_mag=0.5,  # acceleration magnitude during CA
    zigzag_deg=20.0,  # ± amplitude about base heading during zigzag (deg)
    zigzag_T_seg=3.0,  # seconds per zig or zag segment
    # Start position offset
    offset_xy=(1000.0, 800.0),
):
    """
    Composite scenario:
      1) CV straight (t_cv1) →
      2) CT U-turn (exact 180°, duration = pi / omega) →
      3) CA accelerate straight (t_ca) →
      4) CV zigzag about current heading (t_zigzag).

    If `duration_s` is provided, the U-turn time is fixed by `turn_rate_deg_s`,
    and the remaining time is split by `split=(f_cv1,f_ca,f_zigzag)`.
    Otherwise, explicit (t_cv1, t_ca, t_zigzag) are used.

    Returns:
        truth : [N,4] array of [px, py, vx, vy]
        t     : [N] time stamps
    """
    # ---- Resolve segment durations ----
    w = np.deg2rad(turn_rate_deg_s)
    t_turn = np.pi / max(w, 1e-12)  # exact 180° U-turn duration

    if duration_s is not None:
        # allocate remaining time after U-turn
        f_cv1, f_ca, f_zig = split
        f_sum = max(f_cv1 + f_ca + f_zig, 1e-12)
        f_cv1 /= f_sum
        f_ca /= f_sum
        f_zig /= f_sum
        remaining = max(duration_s - t_turn, 0.0)
        t_cv1 = remaining * f_cv1
        t_ca = remaining * f_ca
        t_zigzag = remaining * f_zig

    # Storage
    truth_blocks = []
    time_blocks = []

    # ----------------------------
    # Segment 1: Straight CV
    # ----------------------------
    cv = CV2D()
    psi0 = np.deg2rad(heading0_deg)
    vx0 = v0 * np.cos(psi0)
    vy0 = v0 * np.sin(psi0)
    x_cv = np.array([offset_xy[0], offset_xy[1], vx0, vy0], dtype=float)

    t1 = np.arange(0.0, t_cv1 + 1e-12, truth_dt)
    X1 = np.zeros((t1.size, 4))
    for i in range(t1.size):
        X1[i] = x_cv
        if i + 1 < t1.size:
            x_cv = cv.propagate(x_cv, truth_dt)

    truth_blocks.append(X1)
    time_blocks.append(t1)

    # ----------------------------
    # Segment 2: CT U-turn (exact 180°)
    # ----------------------------
    ct = CT2D()
    # Convert end-of-seg1 state to CT paramization
    px, py, vx, vy = X1[-1]
    v = float(np.hypot(vx, vy))
    psi = float(np.arctan2(vy, vx))
    x_ct = np.array([px, py, v, psi, w], dtype=float)

    # continue time without duplicating last sample
    t2 = np.arange(truth_dt, t_turn + truth_dt / 2, truth_dt)
    X2_ct = np.zeros((t2.size, 5))
    for i in range(t2.size):
        x_ct = ct.propagate(x_ct, truth_dt)
        X2_ct[i] = x_ct

    vx2 = X2_ct[:, 2] * np.cos(X2_ct[:, 3])
    vy2 = X2_ct[:, 2] * np.sin(X2_ct[:, 3])
    X2 = np.stack([X2_ct[:, 0], X2_ct[:, 1], vx2, vy2], axis=1)

    truth_blocks.append(X2)
    time_blocks.append(t2 + time_blocks[-1][-1])

    # ----------------------------
    # Segment 3: Accelerate straight (CA) along current heading
    # ----------------------------
    ca = CA2D(q=0.0)  # deterministic truth for scenario
    px, py, vx, vy = X2[-1]
    psi_after = float(np.arctan2(vy, vx))
    ax = a_mag * np.cos(psi_after)
    ay = a_mag * np.sin(psi_after)
    x_ca = np.array([px, py, vx, vy, ax, ay], dtype=float)

    t3 = np.arange(truth_dt, t_ca + truth_dt / 2, truth_dt)
    X3_full = np.zeros((t3.size, 6))
    for i in range(t3.size):
        x_ca = ca.propagate(x_ca, truth_dt)
        X3_full[i] = x_ca
    X3 = np.stack([X3_full[:, 0], X3_full[:, 1], X3_full[:, 2], X3_full[:, 3]], axis=1)

    truth_blocks.append(X3)
    time_blocks.append(t3 + time_blocks[-1][-1])

    # ----------------------------
    # Segment 4: Zigzag CV about current heading
    # ----------------------------
    cv = CV2D()
    px, py, vx, vy = X3[-1]
    speed = float(np.hypot(vx, vy))
    psi_base = float(np.arctan2(vy, vx))
    psi_plus = psi_base + np.deg2rad(zigzag_deg)
    psi_minus = psi_base - np.deg2rad(zigzag_deg)

    x_cv = np.array([px, py, speed * np.cos(psi_plus), speed * np.sin(psi_plus)], dtype=float)
    t4 = np.arange(truth_dt, t_zigzag + truth_dt / 2, truth_dt)
    X4 = np.zeros((t4.size, 4))
    for i in range(t4.size):
        # switch heading every zigzag_T_seg
        seg = int(np.floor((i + 1) * truth_dt / zigzag_T_seg))
        psi_now = psi_plus if (seg % 2 == 0) else psi_minus
        x_cv[2] = speed * np.cos(psi_now)
        x_cv[3] = speed * np.sin(psi_now)
        x_cv = cv.propagate(x_cv, truth_dt)
        X4[i] = x_cv

    truth_blocks.append(X4)
    time_blocks.append(t4 + time_blocks[-1][-1])

    # ----------------------------
    # Stitch results
    # ----------------------------
    truth = np.vstack(truth_blocks)
    t = np.concatenate([time_blocks[0]] + [tb for tb in time_blocks[1:]])
    return truth, t
