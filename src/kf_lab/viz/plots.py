import numpy as np
import matplotlib.pyplot as plt


def plot_xy(
    truth: np.ndarray,
    estimates: np.ndarray | None = None,
    meas: np.ndarray | None = None,
    title: str = "XY",
):
    """
    Plot truth, estimates, and (optionally) measurements.
    Automatically handles measurements in Cartesian (x,y) or polar (range,bearing).

    - truth: [N,2+] array with columns [px, py, ...]
    - estimates: [N,2+] array with columns [px, py, ...]
    - meas: [N,2] array, either [x,y] or [range,bearing]
    """

    plt.figure()
    plt.plot(truth[:, 0], truth[:, 1], label="truth", linewidth=1.5)

    if estimates is not None:
        plt.plot(estimates[:, 0], estimates[:, 1], label="estimate", linewidth=1.2)

    if meas is not None:
        # detect if it's polar or cartesian
        if np.any(meas[:, 0] > 1e3) or np.any(np.abs(meas[:, 1]) > 2 * np.pi):
            # clearly not polar
            plt.scatter(meas[:, 0], meas[:, 1], s=8, alpha=0.5, label="measurements")
        else:
            # likely polar (range,bearing)
            # assume sensor at origin (0,0) and heading 0 for display
            r, b = meas[:, 0], meas[:, 1]
            x = r * np.cos(b)
            y = r * np.sin(b)
            plt.scatter(x, y, s=8, alpha=0.5, label="measurements (RB->xy)")

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
