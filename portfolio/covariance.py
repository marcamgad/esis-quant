"""
Covariance estimation and simplex projection shared by the optimizers, from
scratch.

Ledoit-Wolf (2004, "Honey, I Shrunk the Sample Covariance Matrix") shrinkage
toward the scaled-identity target F = mu*I, mu = trace(S)/N (the average
sample variance). The sample covariance S is a terrible estimator when the
number of assets N is not small relative to the number of observations T --
its extreme eigenvalues are biased (too large / too small), which a mean-
variance optimizer then exploits catastrophically ("error maximization").
Shrinking S toward mu*I pulls those eigenvalues back and, crucially, makes
the result strictly positive-definite even when S is singular (T < N), which
is exactly the regime where a from-scratch optimizer would otherwise blow up.

The optimal shrinkage intensity delta* is estimated from the data (not tuned)
via Ledoit-Wolf's asymptotically-optimal formula, so there is no free knob.
"""

import numpy as np


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Sample covariance of a (T, N) returns matrix (rows = time, cols =
    assets), demeaned per column, normalized by T (the MLE, matching the
    Ledoit-Wolf derivation which assumes 1/T not 1/(T-1)).
    """
    R = np.asarray(returns, dtype=float)
    Rc = R - R.mean(axis=0, keepdims=True)
    T = R.shape[0]
    return (Rc.T @ Rc) / T


def ledoit_wolf_shrinkage(returns: np.ndarray) -> dict:
    """Ledoit-Wolf shrinkage of the sample covariance toward mu*I.

    Uses the inner product <A,B> = trace(A B') / N so that mu = <S, I> =
    trace(S)/N. With
        d^2   = || S - mu*I ||_F^2 / N                 (dispersion of S from target)
        b_bar^2 = (1/T^2) sum_t || x_t x_t' - S ||_F^2 / N   (estimation error of S)
        b^2   = min(b_bar^2, d^2)                      (can't shrink more than d^2)
        delta = b^2 / d^2                              (optimal intensity in [0,1])
    the shrunk estimator is  Sigma = delta*mu*I + (1-delta)*S.

    Returns {'sigma', 'delta', 'mu', 'sample'}.
    """
    R = np.asarray(returns, dtype=float)
    T, N = R.shape
    Rc = R - R.mean(axis=0, keepdims=True)
    S = (Rc.T @ Rc) / T

    mu = np.trace(S) / N
    target = mu * np.eye(N)
    d2 = np.sum((S - target) ** 2) / N

    # estimation error of S: average over t of ||x_t x_t' - S||_F^2
    b_bar2 = 0.0
    for t in range(T):
        x = Rc[t][:, None]
        b_bar2 += np.sum((x @ x.T - S) ** 2)
    b_bar2 = b_bar2 / (T ** 2) / N

    b2 = min(b_bar2, d2)
    delta = float(b2 / d2) if d2 > 0 else 0.0
    delta = min(max(delta, 0.0), 1.0)
    sigma = delta * target + (1 - delta) * S
    return {"sigma": sigma, "delta": delta, "mu": float(mu), "sample": S}


def project_capped_simplex(v: np.ndarray, cap: float, total: float = 1.0, n_iter: int = 100) -> np.ndarray:
    """Euclidean projection of v onto { w : 0 <= w_i <= cap, sum w = total }.

    The projection has the form w_i = clip(v_i - tau, 0, cap) for a single
    scalar tau; sum(w(tau)) is continuous and non-increasing in tau, so we
    bisect on tau until the sum equals `total`. Requires cap*len(v) >= total
    (enough capacity to reach the target sum); otherwise the constraint set is
    empty and we raise rather than silently returning something infeasible.
    """
    v = np.asarray(v, dtype=float)
    n = len(v)
    if cap * n < total - 1e-12:
        raise ValueError(f"infeasible: cap*{n}={cap*n:.3f} < total={total}")
    lo, hi = float(v.min() - cap), float(v.max())
    for _ in range(n_iter):
        tau = 0.5 * (lo + hi)
        s = np.clip(v - tau, 0.0, cap).sum()
        if s > total:
            lo = tau
        else:
            hi = tau
    return np.clip(v - 0.5 * (lo + hi), 0.0, cap)
