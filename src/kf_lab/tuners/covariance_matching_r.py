import numpy as np
from collections import deque


class CovarianceMatchingRTuner:
    """
    Innovation-based adaptive measurement noise tuner (R) using covariance matching.

    Idea:
        S_k (predicted) = H P_pred H^T + R_k
        S_emp (empirical) ≈ E[nu_k nu_k^T] over a window

    We adjust R_k so that S_k tracks S_emp:
        R_candidate = S_emp - (S_k - R_old)

    Then low-pass filter and project to positive-definite.
    """

    def __init__(
        self,
        R0: np.ndarray,
        window: int = 20,
        alpha: float = 0.05,
        min_eig: float = 1e-6,
    ):
        """
        Parameters
        ----------
        R0 : np.ndarray
            Initial measurement covariance (e.g., from sensor.R()).
        window : int
            Number of past innovations to use for empirical covariance.
        alpha : float
            Smoothing factor for R updates (0 < alpha <= 1).
        min_eig : float
            Minimum eigenvalue to enforce positive definiteness.
        """
        self.R = R0.astype(float).copy()
        self.window = int(window)
        self.alpha = float(alpha)
        self.min_eig = float(min_eig)

        self._buffer: deque[np.ndarray] = deque(maxlen=self.window)

    def current_R(self) -> np.ndarray:
        """Return the current adapted R."""
        return self.R

    def _empirical_S(self) -> np.ndarray:
        """Compute empirical innovation covariance from the buffer."""
        if not self._buffer:
            return self.R.copy()

        m = self._buffer[0].shape[0]
        S_emp = np.zeros((m, m), dtype=float)
        for nu in self._buffer:
            S_emp += nu @ nu.T
        S_emp /= len(self._buffer)
        return S_emp

    def _project_pd(self, M: np.ndarray) -> np.ndarray:
        """Symmetrize and project to positive-definite by clipping eigenvalues."""
        M = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals_clipped = np.clip(eigvals, self.min_eig, None)
        return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    def update_from_ekf(self, ekf) -> np.ndarray:
        """
        Update R using the latest EKF innovation and S.

        This should be called AFTER ekf.update(...).

        Parameters
        ----------
        ekf : EKF
            EKF instance, assumed to have `last_innovation` and `last_S`.

        Returns
        -------
        np.ndarray
            The updated R matrix (also stored internally).
        """
        nu = ekf.last_innovation
        S = ekf.last_S

        if nu is None or S is None:
            # Not enough info yet
            return self.R

        # Ensure column vector (m x 1)
        nu = np.atleast_1d(nu)
        if nu.ndim == 1:
            nu = nu[:, None]

        # Store residual
        self._buffer.append(nu)

        # Need at least 2 samples for a meaningful covariance
        if len(self._buffer) < 2:
            return self.R

        # Empirical innovation covariance
        S_emp = self._empirical_S()

        # Current R (previous step)
        R_old = self.R

        # From S_k = HPH^T + R_k and S_emp ≈ desired S_k:
        # HPH^T = S_k - R_k  => R_new ≈ S_emp - HPH^T = S_emp - (S_k - R_old)
        R_candidate = S_emp - (S - R_old)

        # Smooth update: R_new = (1-alpha)*R_old + alpha*R_candidate
        R_new = (1.0 - self.alpha) * R_old + self.alpha * R_candidate

        # Enforce symmetry + positive-definite
        R_new = self._project_pd(R_new)

        self.R = R_new
        return self.R
