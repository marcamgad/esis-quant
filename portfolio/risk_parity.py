"""
Equal-Risk-Contribution (risk-parity) optimizer, from scratch via Newton's
method (no scipy).

Each asset's risk contribution is RC_i = w_i * (Sigma w)_i; risk parity wants
all RC_i equal. Rather than minimize sum_i (RC_i - RC_bar)^2 directly (which
is non-convex and has a scale degeneracy), we use Spinu's (2013) convex
log-barrier reformulation: minimize
    f(w) = 0.5 * w' Sigma w  -  b' log(w),     b_i = 1/N,  w > 0
whose unique stationary point satisfies  Sigma w = b / w, i.e.
w_i (Sigma w)_i = b_i for all i -- exactly equal (1/N) risk contributions.
The un-normalized solution is then rescaled to sum to 1 (RC equality is
invariant to positive scaling). f is strictly convex on w>0, so Newton's
method converges quadratically; we keep iterates strictly positive with a
step-halving guard (a full Newton step could otherwise cross w_i = 0, where
the log barrier is undefined).
"""

import logging

import numpy as np

logger = logging.getLogger("esis_quant.portfolio.rp")


def risk_parity_weights(sigma, budget=None, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """sigma: (N,N) covariance (PD). budget: (N,) target risk budget summing to
    1 (defaults to equal 1/N -> ERC). Returns {'weights', 'risk_contrib',
    'n_iter'} with weights long-only and summing to 1.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.shape[0]
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, dtype=float)

    # start from inverse-vol weights (a good, always-positive initial guess)
    vol = np.sqrt(np.diag(sigma))
    w = (1.0 / vol)
    w = w / w.sum()

    for it in range(max_iter):
        Sw = sigma @ w
        grad = Sw - b / w
        hess = sigma + np.diag(b / (w ** 2))
        delta = np.linalg.solve(hess, grad)

        # step-halving to keep w strictly positive (log barrier domain)
        alpha = 1.0
        while np.any(w - alpha * delta <= 0) and alpha > 1e-12:
            alpha *= 0.5
        w_new = w - alpha * delta
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    w = w / w.sum()
    Sw = sigma @ w
    rc = w * Sw
    rc = rc / rc.sum()  # normalized risk contributions (should be ~ b)
    return {"weights": w, "risk_contrib": rc, "n_iter": it + 1}
