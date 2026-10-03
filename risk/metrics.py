"""
Risk metrics (Layer 8.1), from scratch -- no scipy.

Includes a normal inverse-CDF (Acklam's rational approximation, ~1e-9
accurate) so Cornish-Fisher VaR needs no scipy.stats, and drawdown analytics
(depth, duration, recovery time) that the plain max-drawdown in
backtest.analytics doesn't cover.

Cornish-Fisher VaR adjusts the Gaussian quantile for the return series' own
skewness and excess kurtosis:
    z_cf = z + (z^2-1) S/6 + (z^3-3z) K/24 - (2z^3-5z) S^2/36
    VaR  = -(mu + z_cf * sigma)
Financial returns are left-skewed and fat-tailed, so a plain-normal VaR
understates tail risk; the CF expansion pulls the quantile out toward the
true (heavier) tail without assuming a specific distribution.
"""

import numpy as np
import pandas as pd


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's algorithm (no scipy)."""
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _clean(returns) -> np.ndarray:
    return pd.Series(returns).dropna().to_numpy(dtype=float)


def skewness(returns) -> float:
    r = _clean(returns)
    if len(r) < 3:
        return 0.0
    m = r.mean()
    s = r.std()
    return float(np.mean((r - m) ** 3) / s ** 3) if s > 0 else 0.0


def excess_kurtosis(returns) -> float:
    r = _clean(returns)
    if len(r) < 4:
        return 0.0
    m = r.mean()
    s = r.std()
    return float(np.mean((r - m) ** 4) / s ** 4 - 3.0) if s > 0 else 0.0


def historical_var(returns, alpha: float = 0.95) -> float:
    """Historical VaR (positive number = loss magnitude at the alpha level)."""
    r = _clean(returns)
    return float(-np.quantile(r, 1 - alpha)) if len(r) else float("nan")


def cornish_fisher_var(returns, alpha: float = 0.95) -> float:
    """Cornish-Fisher (modified) VaR as a positive loss magnitude."""
    r = _clean(returns)
    if len(r) < 4:
        return historical_var(returns, alpha)
    mu, sigma = r.mean(), r.std()
    S, K = skewness(r), excess_kurtosis(r)
    z = norm_ppf(1 - alpha)
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * K / 24 - (2*z**3 - 5*z) * S**2 / 36
    return float(-(mu + z_cf * sigma))


def drawdown_series(returns) -> pd.Series:
    r = pd.Series(returns).dropna()
    nav = (1 + r).cumprod()
    return nav / nav.cummax() - 1


def drawdown_analytics(returns) -> dict:
    """Max drawdown depth, its duration (peak->trough, in periods), and the
    recovery time (trough->new-high; NaN if not yet recovered).
    """
    r = pd.Series(returns).dropna().reset_index(drop=True)
    if r.empty:
        return {"max_drawdown": float("nan"), "duration": 0, "recovery": float("nan"), "recovered": False}
    nav = (1 + r).cumprod().to_numpy()
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1
    trough = int(np.argmin(dd))
    max_dd = float(dd[trough])
    # peak index preceding the trough
    peak_idx = int(np.argmax(nav[:trough + 1])) if trough > 0 else 0
    duration = trough - peak_idx
    # recovery: first index after trough where nav regains the prior peak
    recovery, recovered = float("nan"), False
    target = peak[trough]
    after = np.where(nav[trough:] >= target)[0]
    if len(after):
        recovery = int(after[0])
        recovered = True
    return {"max_drawdown": max_dd, "duration": int(duration),
            "recovery": recovery, "recovered": recovered}
