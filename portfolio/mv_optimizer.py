"""
Mean-variance optimizer, from scratch (no scipy.optimize / cvxpy).

Solves the long-only mean-variance problem
    maximize_w   mu' w - (risk_aversion/2) * w' Sigma w
    subject to   sum(w) = 1,  0 <= w_i <= max_weight
via projected gradient ascent: step along the gradient
    g = mu - risk_aversion * Sigma w
then project back onto the capped simplex (portfolio.covariance.
project_capped_simplex). The objective is concave (Sigma is PSD -- we use the
Ledoit-Wolf-shrunk, strictly-PD estimator), so projected gradient converges
to the global optimum; the projection keeps every iterate feasible, so the
result is always a valid long-only portfolio. A backtracking-free fixed step
scaled by 1/largest-eigenvalue keeps it stable without a line search.
"""

import logging

import numpy as np

from portfolio.covariance import ledoit_wolf_shrinkage, project_capped_simplex

logger = logging.getLogger("esis_quant.portfolio.mv")


def mean_variance_weights(mu, sigma, risk_aversion: float = 5.0, max_weight: float = 0.15,
                          max_iter: int = 5000, tol: float = 1e-9) -> dict:
    """mu: (N,) expected returns. sigma: (N,N) covariance (should be PD, e.g.
    Ledoit-Wolf shrunk). Returns {'weights', 'n_iter', 'objective'}.

    Long-only, fully invested, capped at max_weight. Projected gradient with a
    step of 1/L where L is an upper bound on the curvature (risk_aversion times
    the largest eigenvalue of Sigma) -- guarantees the ascent step never
    overshoots the concave quadratic.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)
    # if the position cap is too tight to be fully invested (n*cap < 1, e.g.
    # early in the sample with few eligible names), drop it -- a cap that
    # makes the feasible set empty is worse than no cap.
    if n * max_weight < 1.0:
        max_weight = 1.0

    # Lipschitz constant of the gradient = risk_aversion * lambda_max(Sigma).
    lam_max = float(np.linalg.eigvalsh(sigma)[-1])
    L = max(risk_aversion * lam_max, 1e-8)
    step = 1.0 / L

    w = project_capped_simplex(np.full(n, 1.0 / n), max_weight)  # feasible start = equal weight
    for it in range(max_iter):
        grad = mu - risk_aversion * (sigma @ w)
        w_new = project_capped_simplex(w + step * grad, max_weight)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    obj = float(mu @ w - 0.5 * risk_aversion * (w @ sigma @ w))
    return {"weights": w, "n_iter": it + 1, "objective": obj}


def mean_variance_from_returns(mu, returns_matrix, **kwargs) -> dict:
    """Convenience: estimate Sigma from a (T,N) returns matrix via Ledoit-Wolf,
    then run the optimizer. Returns the optimizer dict plus 'shrinkage'.
    """
    lw = ledoit_wolf_shrinkage(returns_matrix)
    out = mean_variance_weights(mu, lw["sigma"], **kwargs)
    out["shrinkage"] = lw["delta"]
    return out
