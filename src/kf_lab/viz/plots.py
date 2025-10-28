import matplotlib.pyplot as plt
import numpy as np


def plot_xy(
    truth: np.ndarray, estimates: np.ndarray, meas: np.ndarray | None = None, title: str = "XY"
):
    plt.figure()
    plt.plot(truth[:, 0], truth[:, 1], label="truth")
    if estimates is not None:
        plt.plot(estimates[:, 0], estimates[:, 1], label="estimate")
    if meas is not None:
        plt.scatter(meas[:, 0], meas[:, 1], s=8, alpha=0.6, label="meas")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.legend()
    plt.show()
