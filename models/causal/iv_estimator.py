"""
Instrumental-Variables estimation by Two-Stage Least Squares (2SLS),
implemented from scratch (no statsmodels / linearmodels), with the two
standard instrument-validity diagnostics:

  * first-stage F-statistic on the excluded instruments (Stock-Yogo weak-
    instrument rule of thumb: F > 10), and
  * the Sargan (1958) overidentification test (only defined when there are
    more instruments than endogenous regressors).

Model (spec 5.2): one endogenous regressor D (e.g. momentum), exogenous
controls X (always includes an intercept), and instruments Z that are
correlated with D (relevance) but affect Y only through D (exclusion).

2SLS in projection form. Let W = [X, Z] be all exogenous variables and
M = [X, D] the second-stage regressors. With projection P = W (W'W)^{-1} W':
    beta_2sls = (M' P M)^{-1} M' P Y
The coefficient on the D column of M is the IV estimate theta_IV.
"""

import logging

import numpy as np

import config
from data.linalg import ols_fit

logger = logging.getLogger("esis_quant.causal.iv_estimator")


def _ensure_2d(X):
    """Coerce X to a 2-D float array (reshape a 1-D vector to a column).
    Named for what it does -- it does NOT add an intercept; intercepts are
    added explicitly at the call sites in two_stage_least_squares.
    """
    if X is None:
        return None
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares from OLS of y on X (X includes any intercept)."""
    beta = ols_fit(X, y)
    resid = y - X @ beta
    return float(resid @ resid)


def first_stage_f(D: np.ndarray, Z: np.ndarray, X: np.ndarray) -> dict:
    """First-stage F-statistic for the EXCLUDED instruments Z in the
    regression D ~ X + Z (X already includes an intercept).

    F = ((RSS_restricted - RSS_full) / q) / (RSS_full / (n - k_full))
    where the restricted model drops Z (D ~ X only), q = #instruments,
    k_full = #cols in [X, Z]. F > 10 is the usual weak-instrument threshold.
    """
    D = np.asarray(D, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    X = _ensure_2d(X)
    n = len(D)
    q = Z.shape[1]

    full = np.column_stack([X, Z])
    rss_full = _ols_rss(full, D)
    rss_restr = _ols_rss(X, D)
    k_full = full.shape[1]
    df_resid = max(n - k_full, 1)
    num = (rss_restr - rss_full) / q
    den = rss_full / df_resid
    f_stat = float(num / den) if den > 0 else np.inf
    return {"f_stat": f_stat, "q": q, "df_resid": df_resid, "weak": f_stat < config.CAUSAL_WEAK_IV_F_THRESHOLD}


def sargan_test(residuals: np.ndarray, Z: np.ndarray, X: np.ndarray, n_endog: int = 1) -> dict:
    """Sargan overidentification test. Regress the 2SLS residuals on all
    exogenous vars [X, Z]; S = N * R^2 ~ chi^2(#instruments - #endogenous)
    under H0 that the instruments are valid (uncorrelated with the error).

    Only defined when overidentified (#instruments > #endogenous); returns
    NaN statistic otherwise.
    """
    residuals = np.asarray(residuals, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    X = _ensure_2d(X)
    n = len(residuals)
    q = Z.shape[1]
    df = q - n_endog
    if df <= 0:
        return {"statistic": np.nan, "df": df, "overidentified": False}

    W = np.column_stack([X, Z])
    beta = ols_fit(W, residuals)
    fitted = W @ beta
    tss = float(np.sum((residuals - residuals.mean()) ** 2))
    rss = float(np.sum((residuals - fitted) ** 2))
    r2 = 1 - rss / tss if tss > 0 else 0.0
    stat = float(n * r2)
    return {"statistic": stat, "df": df, "overidentified": True, "r2": r2}


def two_stage_least_squares(Y, D, Z, X=None) -> dict:
    """2SLS estimate of the effect theta_IV of endogenous D on Y, using
    instruments Z and exogenous controls X (an intercept is added
    automatically). Returns theta_IV, its SE and t-stat, first-stage F,
    Sargan overID test, and the fitted first-stage D-hat.
    """
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = len(Y)

    if X is None:
        Xc = np.ones((n, 1))
    else:
        Xc = np.column_stack([np.ones(n), _ensure_2d(X)])

    W = np.column_stack([Xc, Z])           # all exogenous (instruments + controls)
    M = np.column_stack([Xc, D])           # second-stage regressors (D endogenous)
    d_col = M.shape[1] - 1                  # index of D in M

    WtW_inv = np.linalg.inv(W.T @ W + 1e-10 * np.eye(W.shape[1]))
    P_M = W @ (WtW_inv @ (W.T @ M))        # projection of M onto col-space of W
    A = M.T @ P_M
    b = M.T @ (W @ (WtW_inv @ (W.T @ Y)))
    beta = np.linalg.solve(A + 1e-12 * np.eye(A.shape[0]), b)
    theta = float(beta[d_col])

    resid = Y - M @ beta
    sigma2 = float(resid @ resid / max(n - M.shape[1], 1))
    cov = sigma2 * np.linalg.inv(A + 1e-12 * np.eye(A.shape[0]))
    se = float(np.sqrt(max(cov[d_col, d_col], 0.0)))

    fs = first_stage_f(D, Z, Xc)
    sargan = sargan_test(resid, Z, Xc, n_endog=1)
    d_hat = P_M[:, d_col]

    return {
        "theta": theta, "se": se, "t_stat": theta / se if se > 0 else np.nan,
        "first_stage_f": fs, "sargan": sargan, "d_hat": d_hat, "n": n,
    }
