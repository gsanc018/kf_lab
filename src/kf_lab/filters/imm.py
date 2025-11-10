import numpy as np


class IMM:
    """
    Generic Interacting Multiple Model (IMM) estimator.
    Works even when subfilters have different state dimensions (e.g., CV=4D, CA=6D, CT=5D).
    """

    def __init__(self, filters, PI, mu0=None):
        """
        Args:
            filters : list of filter objects, each exposing .x, .P, .predict(dt), .update(z)
            PI      : (M,M) transition probability matrix, PI[i,j] = P(model_i -> model_j)
            mu0     : (M,) initial model probabilities, defaults to uniform
        """
        self.filters = filters
        self.M = len(filters)
        self.PI = np.asarray(PI, dtype=float)
        self.mu = np.ones(self.M) / self.M if mu0 is None else np.asarray(mu0, dtype=float)

        self.x = self.combine_state()
        self.P = self.combine_covariance()

        # bookkeeping
        self.mix_probs = np.zeros((self.M, self.M))
        self.likelihoods = np.ones(self.M)
        self.history = {"mu": [self.mu.copy()]}

    # ----------------------------------------------------------------------
    # core IMM steps
    # ----------------------------------------------------------------------

    def predict(self, dt):
        """Performs mixing and prediction for all models."""
        # --- find max dimension across all subfilters ---
        max_dim = max(f.x.size for f in self.filters)

        def pad_state(x, P, n=max_dim):
            if x.size == n:
                return x, P
            pad = n - x.size
            x_full = np.concatenate([x, np.zeros(pad)])
            P_full = np.pad(P, ((0, pad), (0, pad)), mode="constant")
            return x_full, P_full

        # ensure all filters share the same padded shape
        for i in range(self.M):
            self.filters[i].x, self.filters[i].P = pad_state(self.filters[i].x, self.filters[i].P)

        # 1. Mixing probabilities
        c_j = self.PI.T @ self.mu  # normalization constants
        for j in range(self.M):
            for i in range(self.M):
                if c_j[j] > 0:
                    self.mix_probs[i, j] = (self.PI[i, j] * self.mu[i]) / c_j[j]
                else:
                    self.mix_probs[i, j] = 0.0

        # 2. Mixed initial conditions
        x_mix = []
        P_mix = []
        for j in range(self.M):
            xj = np.zeros_like(self.filters[j].x)
            for i in range(self.M):
                xj += self.mix_probs[i, j] * self.filters[i].x
            Pj = np.zeros_like(self.filters[j].P)
            for i in range(self.M):
                dx = self.filters[i].x - xj
                Pj += self.mix_probs[i, j] * (self.filters[i].P + np.outer(dx, dx))
            x_mix.append(xj)
            P_mix.append(Pj)

        # 3. Crop each mixed state back to model dimension before prediction
        for j in range(self.M):
            f = self.filters[j]
            state_dim = f.x.shape[0]

            if x_mix[j].size > state_dim:
                f.x = x_mix[j][:state_dim]
                f.P = P_mix[j][:state_dim, :state_dim]
            else:
                f.x = x_mix[j]
                f.P = P_mix[j]

            f.predict(dt)

    # ----------------------------------------------------------------------

    def update(self, z):
        """Update each model filter with measurement z and update model probabilities."""
        # 1. Model-specific updates + likelihoods
        for j in range(self.M):
            likelihood = self.filters[j].update(z, return_likelihood=True)
            self.likelihoods[j] = likelihood

        # 2. Model probability update (Bayes)
        mu_pred = self.PI.T @ self.mu
        mu_new = self.likelihoods * mu_pred
        total = np.sum(mu_new)
        self.mu = mu_new / total if total > 0 else mu_pred

        # 3. Combine state/covariance
        self.x = self.combine_state()
        self.P = self.combine_covariance()
        self.history["mu"].append(self.mu.copy())

    # ----------------------------------------------------------------------
    # helper functions
    # ----------------------------------------------------------------------

    def combine_state(self):
        """Combine model states weighted by probabilities (handles mixed dimensions)."""
        max_dim = max(f.x.size for f in self.filters)
        x_combined = np.zeros(max_dim)
        for i, f in enumerate(self.filters):
            xi = np.pad(f.x, (0, max_dim - f.x.size), mode="constant")
            x_combined += self.mu[i] * xi
        return x_combined

    def combine_covariance(self):
        """Combine model covariances + between-model spread (handles mixed dimensions)."""
        max_dim = max(f.x.size for f in self.filters)
        P_combined = np.zeros((max_dim, max_dim))
        for i, f in enumerate(self.filters):
            xi = np.pad(f.x, (0, max_dim - f.x.size), mode="constant")
            Pi = np.pad(
                f.P, ((0, max_dim - f.P.shape[0]), (0, max_dim - f.P.shape[1])), mode="constant"
            )
            dx = xi - self.x
            P_combined += self.mu[i] * (Pi + np.outer(dx, dx))
        return P_combined

    def step(self, z, dt):
        """Convenience wrapper: predict(dt) + update(z)."""
        self.predict(dt)
        self.update(z)
        return self.x, self.P, self.mu

    def reset(self, x0, P0, mu0=None):
        """Reset all subfilters and model probabilities."""
        for f in self.filters:
            f.x = x0.copy()
            f.P = P0.copy()
        self.mu = np.ones(self.M) / self.M if mu0 is None else mu0.copy()
        self.x = self.combine_state()
        self.P = self.combine_covariance()
        self.history = {"mu": [self.mu.copy()]}
