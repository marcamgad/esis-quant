"""
Gaussian Hidden Markov Model, implemented from scratch (no hmmlearn,
sklearn, or statsmodels). Generic K-state, D-feature HMM -- this file has
no notion of "BULL/BEAR/etc"; regime labeling is a separate concern (see
models/regime_labeler.py) applied after fitting.

All forward/backward/Viterbi recursions run in log-space to avoid
underflow over long sequences, per the standard formulation (Rabiner
1989). Covariances are regularized with a small diagonal epsilon since
per-state sample covariance can be near-singular, especially early in EM
when a state has few effective observations.
"""

import logging

import numpy as np

logger = logging.getLogger("esis_quant.hmm")


def _logsumexp(a, axis=None):
    """Manual log-sum-exp with the max-subtraction trick. Returns -inf
    (not NaN) when every element along `axis` is -inf, since that's a
    legitimate value here (e.g. an impossible transition), not an error.
    """
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max_safe = np.where(np.isfinite(a_max), a_max, 0.0)
    summed = np.sum(np.exp(a - a_max_safe), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        out = np.log(summed) + a_max_safe
    out = np.where(np.isfinite(a_max), out, -np.inf)
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    else:
        out = out.reshape(())
    return out


def _log_gaussian_pdf(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """log N(x | mu, sigma) for every row of X (T, D), vectorized over T.

    log N(x|mu,S) = -0.5*D*log(2*pi) - 0.5*log|S| - 0.5*(x-mu)'S^-1(x-mu)

    `reg` is added to the diagonal before inversion, both to keep S
    invertible and because the spec calls for +1e-6*I on every covariance
    matrix for numerical stability.
    """
    d = X.shape[1]
    sigma_reg = sigma + reg * np.eye(d)
    sign, logdet = np.linalg.slogdet(sigma_reg)
    if sign <= 0:
        # Should not happen once regularized, but fail loudly rather than
        # silently returning a wrong density if it ever does.
        raise np.linalg.LinAlgError("Non-positive-definite covariance after regularization")
    inv = np.linalg.inv(sigma_reg)
    diff = X - mu
    maha = np.einsum("ti,ij,tj->t", diff, inv, diff)
    return -0.5 * d * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * maha


class GaussianHMM:
    """K-state HMM with full-covariance Gaussian emissions, fit via
    Baum-Welch (EM) and decoded via Viterbi. Single-sequence only (no
    multi-sequence fitting), which is all Layer 2 needs (one daily time
    series of market observations).
    """

    def __init__(self, n_states: int, n_iter: int = 200, tol: float = 1e-6,
                 n_init: int = 5, cov_reg: float = 1e-6, self_transition_prior: float = 0.85,
                 random_state: int = None):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.cov_reg = cov_reg
        self.self_transition_prior = self_transition_prior
        self.random_state = random_state

        self.pi_ = None
        self.A_ = None
        self.mu_ = None
        self.sigma_ = None
        self.monitor_ = None  # log-likelihood trace of the winning init

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_params(self, X: np.ndarray, rng: np.random.Generator):
        T, d = X.shape
        K = self.n_states

        pi = np.full(K, 1.0 / K)

        off_diag = (1.0 - self.self_transition_prior) / (K - 1) if K > 1 else 0.0
        A = np.full((K, K), off_diag)
        np.fill_diagonal(A, self.self_transition_prior)

        # k-means-style init: K distinct random rows of X as starting means.
        idx = rng.choice(T, size=K, replace=False)
        mu = X[idx].copy()

        global_cov = np.cov(X, rowvar=False)
        if global_cov.ndim == 0:
            global_cov = global_cov.reshape(1, 1)
        sigma = np.stack([global_cov.copy() for _ in range(K)])

        return pi, A, mu, sigma

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------
    def _log_emissions(self, X, mu, sigma):
        T = X.shape[0]
        K = mu.shape[0]
        log_B = np.zeros((T, K))
        for k in range(K):
            log_B[:, k] = _log_gaussian_pdf(X, mu[k], sigma[k], reg=self.cov_reg)
        return log_B

    @staticmethod
    def _forward(log_pi, log_A, log_B):
        T, K = log_B.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            # log_alpha[t,j] = logsumexp_i(log_alpha[t-1,i] + log_A[i,j]) + log_B[t,j]
            temp = log_alpha[t - 1][:, None] + log_A
            log_alpha[t] = _logsumexp(temp, axis=0) + log_B[t]
        return log_alpha

    @staticmethod
    def _backward(log_A, log_B):
        T, K = log_B.shape
        log_beta = np.zeros((T, K))
        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            # log_beta[t,i] = logsumexp_j(log_A[i,j] + log_B[t+1,j] + log_beta[t+1,j])
            temp = log_A + (log_B[t + 1] + log_beta[t + 1])[None, :]
            log_beta[t] = _logsumexp(temp, axis=1)
        return log_beta

    # ------------------------------------------------------------------
    # EM (Baum-Welch)
    # ------------------------------------------------------------------
    def _fit_single_init(self, X, rng):
        pi, A, mu, sigma = self._init_params(X, rng)
        T, d = X.shape
        K = self.n_states
        prev_loglik = -np.inf
        history = []

        for iteration in range(self.n_iter):
            with np.errstate(divide="ignore"):
                log_pi = np.log(pi)
                log_A = np.log(A)
            log_B = self._log_emissions(X, mu, sigma)

            log_alpha = self._forward(log_pi, log_A, log_B)
            log_beta = self._backward(log_A, log_B)
            total_loglik = _logsumexp(log_alpha[-1], axis=0)
            history.append(float(total_loglik))

            # E-step: gamma_t(i), xi_t(i,j)
            log_gamma = log_alpha + log_beta - total_loglik
            gamma = np.exp(log_gamma)

            log_xi = (
                log_alpha[:-1, :, None]
                + log_A[None, :, :]
                + (log_B[1:] + log_beta[1:])[:, None, :]
                - total_loglik
            )
            xi = np.exp(log_xi)

            # M-step
            eps = 1e-12
            pi = gamma[0] / (gamma[0].sum() + eps)

            denom_A = gamma[:-1].sum(axis=0) + eps  # (K,)
            A = xi.sum(axis=0) / denom_A[:, None]
            # Renormalize each row to sum to 1. A state never transitioned
            # from (denom_A ~ eps) yields an all-~zero row whose sum is ~0;
            # guard the divisor so it stays row-stochastic instead of
            # producing NaNs that would poison the rest of EM (HMM-ROBUST-01).
            row_sums = A.sum(axis=1, keepdims=True)
            A = A / np.where(row_sums > eps, row_sums, 1.0)

            gamma_sum = gamma.sum(axis=0) + eps  # (K,)
            mu = (gamma.T @ X) / gamma_sum[:, None]

            sigma_new = np.zeros_like(sigma)
            for k in range(K):
                diff = X - mu[k]
                weighted = (gamma[:, k : k + 1] * diff).T @ diff
                sigma_new[k] = weighted / gamma_sum[k]
            sigma = sigma_new

            if total_loglik - prev_loglik < self.tol and iteration > 0:
                if total_loglik < prev_loglik - 1e-8:
                    logger.warning(
                        "EM log-likelihood decreased (%.6f -> %.6f) at iter %d; "
                        "should not happen with a correct E/M step",
                        prev_loglik, total_loglik, iteration,
                    )
                prev_loglik = total_loglik
                break
            prev_loglik = total_loglik

        return {"pi": pi, "A": A, "mu": mu, "sigma": sigma, "loglik": prev_loglik, "history": history}

    def fit(self, X: np.ndarray):
        """X: (T, D) array, a single time-ordered sequence of observations.
        Runs `n_init` random restarts and keeps the one with the highest
        final log-likelihood (EM only finds a local optimum).
        """
        X = np.asarray(X, dtype=float)
        base_seed = self.random_state if self.random_state is not None else np.random.SeedSequence().entropy
        best = None
        for i in range(self.n_init):
            rng = np.random.default_rng(int(base_seed) + i)
            result = self._fit_single_init(X, rng)
            if best is None or result["loglik"] > best["loglik"]:
                best = result

        self.pi_ = best["pi"]
        self.A_ = best["A"]
        self.mu_ = best["mu"]
        self.sigma_ = best["sigma"]
        self.monitor_ = best["history"]
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray) -> float:
        """Total log P(X | fitted params) via the forward algorithm."""
        X = np.asarray(X, dtype=float)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.pi_)
            log_A = np.log(self.A_)
        log_B = self._log_emissions(X, self.mu_, self.sigma_)
        log_alpha = self._forward(log_pi, log_A, log_B)
        return float(_logsumexp(log_alpha[-1], axis=0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior state probabilities gamma_t(i) = P(q_t=i | X, params)
        for every t, using forward-backward with the fitted parameters.
        Returns (T, K).
        """
        X = np.asarray(X, dtype=float)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.pi_)
            log_A = np.log(self.A_)
        log_B = self._log_emissions(X, self.mu_, self.sigma_)
        log_alpha = self._forward(log_pi, log_A, log_B)
        log_beta = self._backward(log_A, log_B)
        total_loglik = _logsumexp(log_alpha[-1], axis=0)
        return np.exp(log_alpha + log_beta - total_loglik)

    def filtered_proba(self, X: np.ndarray) -> np.ndarray:
        """CAUSAL state probabilities P(q_t=i | X_0..t, params) for every t,
        using the forward pass ONLY -- no backward pass, no future data.
        Returns (T, K).

        This is the point-in-time correct alternative to predict_proba
        (which is the SMOOTHED posterior gamma_t = P(q_t | X_0..T) and peeks
        at the whole sequence, including the future). For anything that feeds
        a trading signal, use this: at date t it conditions only on
        observations up to and including t, so a regime call made "on day t"
        could actually have been made on day t in real time.

        Normalizing the forward variable per step: filtered_t(i) =
        alpha_t(i) / sum_j alpha_t(j), computed in log space then exponentiated.
        """
        X = np.asarray(X, dtype=float)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.pi_)
            log_A = np.log(self.A_)
        log_B = self._log_emissions(X, self.mu_, self.sigma_)
        log_alpha = self._forward(log_pi, log_A, log_B)
        log_norm = _logsumexp(log_alpha, axis=1)  # (T,) per-step normalizer
        return np.exp(log_alpha - log_norm[:, None])

    def filtered_decode(self, X: np.ndarray) -> np.ndarray:
        """Causal hard-state sequence: argmax of the FILTERED probabilities at
        each t. Deliberately NOT Viterbi -- Viterbi maximizes the joint path
        over the WHOLE sequence, so the state it assigns to day t depends on
        observations after t and is therefore not causal. Returns an int
        array (T,). Use this (not decode) for live/trading regime labels.
        """
        return np.argmax(self.filtered_proba(X), axis=1)

    def decode(self, X: np.ndarray):
        """Viterbi: most likely single state sequence for X given the
        fitted parameters. Returns (path, best_loglik) where path is an
        int array of shape (T,).

        NOTE: Viterbi is a SMOOTHED (whole-sequence) decode -- the state at t
        depends on the entire path, including the future. Legitimate for
        DESCRIPTIVE use (regime statistics over a completed history), but NOT
        causal; use filtered_decode for anything that feeds a live signal.
        """
        X = np.asarray(X, dtype=float)
        with np.errstate(divide="ignore"):
            log_pi = np.log(self.pi_)
            log_A = np.log(self.A_)
        log_B = self._log_emissions(X, self.mu_, self.sigma_)

        T, K = log_B.shape
        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = log_pi + log_B[0]
        for t in range(1, T):
            temp = delta[t - 1][:, None] + log_A  # rows=from-state i, cols=to-state j
            psi[t] = np.argmax(temp, axis=0)
            delta[t] = np.max(temp, axis=0) + log_B[t]

        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        best_loglik = float(delta[T - 1, path[T - 1]])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, best_loglik
