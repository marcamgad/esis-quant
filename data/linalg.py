"""
Shared, manually-implemented linear algebra utilities (no sklearn/
statsmodels). Used by data/microstructure.py and models/factor_model.py,
both of which need the same "rolling single-regressor OLS beta" pattern.
"""

import numpy as np
import pandas as pd


def ols_fit(X: np.ndarray, y: np.ndarray, ridge_eps: float = 1e-8) -> np.ndarray:
    """Manual OLS: beta = (X'X + eps*I)^-1 X'y. X must already include an
    intercept column if one is wanted. The ridge_eps term is added purely
    for numerical stability (near-singular X'X), not for regularization.
    Returns beta as a 1-D array of length X.shape[1].
    """
    XtX = X.T @ X
    XtX += ridge_eps * np.eye(XtX.shape[0])
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int, min_obs: int = None) -> pd.Series:
    """Rolling single-regressor OLS with intercept: y_t = a + b*x_t + e_t,
    re-fit on every trailing `window`-length slice. Returns a Series of
    the slope b, indexed like y (NaN wherever a window has too few valid
    observations or x is constant over the window, which would make X'X
    singular).

    This is the exact pattern needed for market beta, Kyle's lambda,
    EGP/oil sensitivity, and the beta used in realized-variance
    decomposition -- all of them are "regress this stock's series on that
    one other series, over a rolling window, with an intercept".
    """
    min_obs = min_obs if min_obs is not None else max(5, window // 4)
    out = pd.Series(index=y.index, dtype=float)
    y_vals = y.to_numpy()
    x_vals = x.to_numpy()

    for i in range(window, len(y) + 1):
        y_win = y_vals[i - window:i]
        x_win = x_vals[i - window:i]
        mask = ~(np.isnan(y_win) | np.isnan(x_win))
        if mask.sum() < min_obs or np.allclose(x_win[mask], x_win[mask][0]):
            continue
        X = np.column_stack([np.ones(mask.sum()), x_win[mask]])
        beta = ols_fit(X, y_win[mask])
        out.iloc[i - 1] = beta[1]
    return out
