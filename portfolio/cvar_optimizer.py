"""
Minimum-CVaR (Conditional Value at Risk) portfolio via the Rockafellar-Uryasev
(2000) linear program, solved with the from-scratch simplex in simplex.py.

CVaR_alpha(w) = expected loss in the worst (1-alpha) tail of the return
distribution -- a coherent risk measure (unlike VaR it's convex in w and
penalizes the depth of tail losses, not just their frequency). The RU result
is that CVaR can be computed WITHOUT first knowing VaR, as

    CVaR_alpha(w) = min_zeta  zeta + 1/((1-alpha) T) * sum_t max(-r_t' w - zeta, 0)

which linearizes with auxiliary u_t = max(-r_t'w - zeta, 0). The portfolio
problem (minimize tail risk, optionally rewarding return) becomes the LP

    min_{w, zeta, u}  zeta + 1/((1-alpha)T) sum_t u_t  -  return_weight * mu' w
    s.t.  u_t >= -r_t' w - zeta,  u_t >= 0,  sum w = 1,  0 <= w <= max_weight.

zeta is a free variable, split into zeta+ - zeta- for the non-negative-variable
simplex. return_weight defaults to 0 (pure minimum-CVaR) -- the risk-focused
allocation that matches this system's measured edge; raise it to trade tail
risk for expected return.
"""

import logging

import numpy as np

from portfolio.simplex import linprog

logger = logging.getLogger("esis_quant.portfolio.cvar")


def min_cvar_weights(scenarios, mu=None, alpha: float = 0.95, max_weight: float = 0.15,
                     return_weight: float = 0.0) -> dict:
    """scenarios: (T, N) matrix of historical asset returns (rows = scenarios).
    mu: (N,) expected returns (only used if return_weight > 0). Returns
    {'weights', 'cvar', 'var', 'status'}.
    """
    R = np.asarray(scenarios, dtype=float)
    T, N = R.shape
    mu = np.zeros(N) if mu is None else np.asarray(mu, dtype=float)
    if N * max_weight < 1.0:   # infeasible box -> drop the cap (see mv_optimizer note)
        max_weight = 1.0
    coef = 1.0 / ((1.0 - alpha) * T)

    # variable layout: [w (N) | zeta+ (1) | zeta- (1) | u (T)]
    nz = N + 2 + T
    iw, izp, izn, iu = slice(0, N), N, N + 1, slice(N + 2, N + 2 + T)

    c = np.zeros(nz)
    c[iw] = -return_weight * mu
    c[izp] = 1.0
    c[izn] = -1.0
    c[iu] = coef

    # CVaR constraints: -r_t'w - zeta - u_t <= 0  (i.e. u_t + r_t'w + zeta >= 0)
    A_ub = np.zeros((T + N, nz))
    b_ub = np.zeros(T + N)
    A_ub[:T, iw] = -R
    A_ub[:T, izp] = -1.0
    A_ub[:T, izn] = 1.0
    for t in range(T):
        A_ub[t, N + 2 + t] = -1.0
    # box: w_i <= max_weight
    for i in range(N):
        A_ub[T + i, i] = 1.0
        b_ub[T + i] = max_weight

    A_eq = np.zeros((1, nz))
    A_eq[0, iw] = 1.0
    b_eq = np.array([1.0])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    if res["status"] != "optimal":
        logger.warning("min_cvar_weights: LP %s -- falling back to equal weight", res["status"])
        w = np.full(N, 1.0 / N)
        return {"weights": w, "cvar": np.nan, "var": np.nan, "status": res["status"]}

    x = res["x"]
    w = np.clip(x[iw], 0.0, None)
    w = w / w.sum()
    zeta = x[izp] - x[izn]
    port_losses = -(R @ w)
    cvar = float(zeta + coef * np.maximum(port_losses - zeta, 0).sum())
    return {"weights": w, "cvar": cvar, "var": float(zeta), "status": "optimal"}
