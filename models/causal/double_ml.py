"""
Double / Debiased Machine Learning (Chernozhukov et al. 2018) via the
Robinson (1988) partialling-out score, with cross-fitting -- implemented
from scratch (nuisance functions are the coordinate-descent Lasso in
lasso.py; no sklearn / econml / doubleml).

Model (spec 5.1):
    Y = theta * D + g(X) + eps        (outcome equation)
    D = m(X) + v                      (treatment equation)

We want the causal effect theta of treatment D (sentiment/momentum) on
outcome Y (forward return), free of confounding by X (factor exposures,
macro state). Naive OLS of Y on D is biased whenever X drives both.

Partialling-out estimator:
    1. Ytilde = Y - ghat(X)           (residualize the outcome)
    2. Dtilde = D - mhat(X)           (residualize the treatment)
    3. theta_hat = sum(Dtilde * Ytilde) / sum(Dtilde^2)   (OLS, no intercept)

Cross-fitting (K time-ordered folds): ghat and mhat for the rows in fold k
are trained ONLY on the other folds, so the residuals for each row use
nuisance estimates that never saw that row -- this removes the own-
observation overfitting bias that would otherwise contaminate theta.

Standard error (Neyman-orthogonal score psi):
    psi_i     = (Ytilde_i - theta_hat * Dtilde_i) * Dtilde_i
    Var(theta)= E[Dtilde^2]^{-2} * E[psi^2]
    SE        = sqrt(Var(theta) / N)
"""

import logging

import numpy as np

import config
from data.linalg import ols_fit
from models.causal.lasso import LassoCoordinateDescent, lasso_cv

logger = logging.getLogger("esis_quant.causal.double_ml")


def _cross_fitted_residuals(target: np.ndarray, X: np.ndarray, n_folds: int,
                            lam: float, purge: int = 0) -> np.ndarray:
    """Cross-fitted residuals of `target` on X: for each time-ordered fold,
    fit Lasso on the other folds and predict the held-out fold. Returns an
    array aligned to `target` (row i's prediction never used row i). `purge`
    drops training rows within that many steps of each validation block to
    prevent h-day-forward-return overlap leakage (see time_series_folds).
    """
    from models.causal.lasso import time_series_folds

    n = len(target)
    residuals = np.full(n, np.nan)
    for train, val in time_series_folds(n, n_folds, purge=purge):
        model = LassoCoordinateDescent().fit(X[train], target[train], lam)
        residuals[val] = target[val] - model.predict(X[val])
    # any fold that got skipped (degenerate) -> fall back to full-sample fit
    missing = np.isnan(residuals)
    if missing.any():
        model = LassoCoordinateDescent().fit(X, target, lam)
        residuals[missing] = target[missing] - model.predict(X[missing])
    return residuals


def naive_ols_effect(Y: np.ndarray, D: np.ndarray) -> dict:
    """Confounder-blind baseline: OLS of Y on [1, D]; the slope on D is the
    naive (generally biased) treatment effect. Reported alongside the DML
    estimate so the confounding bias (naive - causal) is visible.
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    Xmat = np.column_stack([np.ones(len(D)), D])
    beta = ols_fit(Xmat, Y)
    resid = Y - Xmat @ beta
    n, k = Xmat.shape
    sigma2 = resid @ resid / max(n - k, 1)
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat + 1e-12 * np.eye(k))
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    theta = float(beta[1])
    return {"theta": theta, "se": se, "t_stat": theta / se if se > 0 else np.nan}


def double_ml(Y, D, X, n_folds: int = None, lam: float = None, purge: int = None) -> dict:
    """Estimate the causal effect theta of D on Y controlling for X by DML
    partialling-out with cross-fitting.

    Y : (N,) outcome. D : (N,) treatment. X : (N, p) confounders.
    lam: L1 penalty for both nuisance Lassos; if None, chosen once by
         time-series CV on the outcome regression Y~X and reused for both
         (a single shared, data-driven penalty).
    purge: rows to drop next to each validation block in cross-fitting, to
         stop overlapping-forward-return leakage. Defaults to the forward
         horizon (config.CAUSAL_FORWARD_RETURN_HORIZON) since Y is that
         horizon's forward return; pass 0 for a contemporaneous outcome.

    We keep K-fold cross-fitting (Chernozhukov et al. 2018) rather than a
    single walk-forward split -- cross-fitting is the estimator that makes
    the DML score Neyman-orthogonal and efficient -- and add De Prado
    purging on top to remove the concrete time-series leak (overlapping
    return windows at fold boundaries) without discarding the estimator's
    efficiency.

    Returns dict with theta, se, t_stat, the cross-fitted residuals
    (Y_tilde, D_tilde), the naive OLS baseline, chosen lambda, and n.
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_folds = n_folds if n_folds is not None else config.CAUSAL_DML_N_FOLDS
    purge = config.CAUSAL_FORWARD_RETURN_HORIZON if purge is None else purge
    n = len(Y)

    if n < config.CAUSAL_MIN_OBS:
        logger.warning("double_ml: only %d obs (< CAUSAL_MIN_OBS=%d) -- theta unreliable",
                       n, config.CAUSAL_MIN_OBS)

    if lam is None:
        lam = lasso_cv(X, Y, k=n_folds)["best_lambda"]

    Y_tilde = _cross_fitted_residuals(Y, X, n_folds, lam, purge=purge)
    D_tilde = _cross_fitted_residuals(D, X, n_folds, lam, purge=purge)

    denom = float(np.sum(D_tilde ** 2))
    if denom <= 1e-12:
        logger.warning("double_ml: residualized treatment has ~zero variance -- theta undefined")
        return {"theta": 0.0, "se": np.nan, "t_stat": np.nan, "Y_tilde": Y_tilde,
                "D_tilde": D_tilde, "naive": naive_ols_effect(Y, D), "lambda": lam,
                "n": n, "insufficient": True}

    theta = float(np.sum(D_tilde * Y_tilde) / denom)

    psi = (Y_tilde - theta * D_tilde) * D_tilde
    mean_d2 = np.mean(D_tilde ** 2)
    var_theta = (mean_d2 ** -2) * np.mean(psi ** 2)
    se = float(np.sqrt(var_theta / n))

    return {
        "theta": theta, "se": se, "t_stat": theta / se if se > 0 else np.nan,
        "Y_tilde": Y_tilde, "D_tilde": D_tilde,
        "naive": naive_ols_effect(Y, D), "lambda": lam, "n": n, "insufficient": False,
    }
