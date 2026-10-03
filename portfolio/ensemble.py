"""
Ensemble portfolio (Layer 7.5): blend the three from-scratch optimizers'
WEIGHT vectors (not their returns) and apply post-blend constraints.

    w_ensemble = 0.40 * w_MV + 0.35 * w_RP + 0.25 * w_CVaR

Blending weights (rather than averaging the strategies' realized returns)
produces a single, directly-tradable book whose turnover and constraints we
can control -- averaging returns would be a paper construct you can't hold.
The 0.40/0.35/0.25 split leans toward the risk-based members (RP+CVaR = 0.60)
because Phase A showed this system's edge is risk management, not the return
forecast that mean-variance leans on.

Post-blend constraints, applied in order then renormalized: max position
(max_weight), max sector exposure (max_sector, using the Phase-A3 sector map),
and a diversification floor (the optimizers already spread weight, so
min_names is a check, not a forced selection).
"""

import logging

import numpy as np
import pandas as pd

from portfolio.covariance import ledoit_wolf_shrinkage
from portfolio.mv_optimizer import mean_variance_weights
from portfolio.risk_parity import risk_parity_weights
from portfolio.cvar_optimizer import min_cvar_weights

logger = logging.getLogger("esis_quant.portfolio.ensemble")

DEFAULT_BLEND = {"mv": 0.40, "rp": 0.35, "cvar": 0.25}


def _enforce_sector_cap(w: pd.Series, sectors: pd.Series, max_sector: float, n_iter: int = 20) -> pd.Series:
    """Scale down any sector over `max_sector` and redistribute the excess to
    under-cap names, proportionally, iterating until all sectors comply (or
    the cap is mathematically unreachable). Names with unknown sector are
    treated as their own singleton sectors.
    """
    w = w.copy()
    sec = sectors.reindex(w.index).fillna("_UNKNOWN_")
    for _ in range(n_iter):
        sector_tot = w.groupby(sec).sum()
        over = sector_tot[sector_tot > max_sector + 1e-9]
        if over.empty:
            break
        excess = 0.0
        for s in over.index:
            names = sec[sec == s].index
            scale = max_sector / sector_tot[s]
            excess += float(w[names].sum() * (1 - scale))
            w[names] = w[names] * scale
        under = sec[~sec.isin(over.index)].index
        if len(under) == 0 or w[under].sum() <= 0:
            break
        w[under] += excess * (w[under] / w[under].sum())
    return w


def ensemble_weights(returns_window: np.ndarray, tickers, mu=None, sectors: pd.Series = None,
                     blend=None, max_weight: float = 0.15, max_sector: float = 0.35,
                     alpha: float = 0.95, risk_aversion: float = 5.0) -> dict:
    """Build the ensemble book from a trailing (T, N) returns window.

    returns_window: (T, N) asset returns (columns aligned to `tickers`).
    mu: (N,) expected returns for the MV member (0 if None -> MV becomes
        min-variance). Returns {'weights' (pd.Series), 'components' (dict of
        each optimizer's weights)}.
    """
    blend = blend or DEFAULT_BLEND
    tickers = list(tickers)
    N = len(tickers)
    mu = np.zeros(N) if mu is None else np.asarray(mu, dtype=float)

    lw = ledoit_wolf_shrinkage(returns_window)
    sigma = lw["sigma"]

    eff_cap = max_weight if N * max_weight >= 1.0 else 1.0  # feasibility with few names
    w_mv = mean_variance_weights(mu, sigma, risk_aversion=risk_aversion, max_weight=max_weight)["weights"]
    w_rp = risk_parity_weights(sigma)["weights"]
    # risk parity has no cap; clip + renormalize so it respects max_weight too
    w_rp = np.clip(w_rp, 0, eff_cap)
    w_rp = w_rp / w_rp.sum()
    w_cvar = min_cvar_weights(returns_window, mu=mu, alpha=alpha, max_weight=max_weight)["weights"]

    blended = blend["mv"] * w_mv + blend["rp"] * w_rp + blend["cvar"] * w_cvar
    w = pd.Series(blended, index=tickers)

    if sectors is not None:
        w = _enforce_sector_cap(w, sectors, max_sector)
    w = w.clip(upper=max_weight)
    w = w / w.sum()

    return {"weights": w, "components": {
        "mv": pd.Series(w_mv, index=tickers),
        "rp": pd.Series(w_rp, index=tickers),
        "cvar": pd.Series(w_cvar, index=tickers)}, "shrinkage": lw["delta"]}
