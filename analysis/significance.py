"""
Statistical significance of a Sharpe ratio via bootstrap (audit Tier-1 #4).

WHY. A backtest Sharpe of 1.70 is a single number from a SINGLE realized path
of one 9-year history. On its own it carries no confidence interval -- you
cannot tell 1.70 +/- 0.3 (real) from 1.70 +/- 1.5 (noise). Sizing capital
against a point estimate with unknown error is how backtests kill accounts.

METHOD (from scratch, no scipy). Daily returns are autocorrelated and
volatility-clustered, so a naive iid bootstrap understates the error. We use
two block bootstraps that preserve local dependence:

  * moving-block bootstrap: resample contiguous blocks of length L with
    replacement until the series length is reached, recompute the annualized
    Sharpe, repeat B times.
  * stationary bootstrap (Politis-Romano 1994): block lengths are drawn
    Geometric(1/L) so the resampled series is strictly stationary and the
    result is not sensitive to one arbitrary L. This is the default reported.

From the B bootstrap Sharpes we report the mean, the [2.5%, 97.5%] percentile
CI, and a one-sided p-value for H0: Sharpe <= 0 (fraction of bootstrap Sharpes
at or below 0). A CI that comfortably excludes 0 is weak evidence the edge is
real; a CI that straddles 0 means the backtest cannot distinguish the strategy
from luck -- which is itself the honest, useful finding.

Block length L defaults to ~ n**(1/3) (a standard rule of thumb), floored at 5
so daily dependence is captured.
"""

import numpy as np
import pandas as pd


def _ann_sharpe(r: np.ndarray, ppy: int = 252) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 and np.isfinite(sd) else 0.0


def _default_block(n: int) -> int:
    return max(5, int(round(n ** (1.0 / 3.0))))


def moving_block_sharpe(returns, n_boot: int = 2000, block: int = None,
                        seed: int = 12345, ppy: int = 252) -> np.ndarray:
    """B bootstrap Sharpes via the moving-block bootstrap."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(r)
    if n < 30:
        raise ValueError("need >= 30 observations to bootstrap a Sharpe")
    L = block or _default_block(n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / L))
    max_start = n - L
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(L)[None, :]).reshape(-1)[:n]
        out[b] = _ann_sharpe(r[idx], ppy)
    return out


def stationary_bootstrap_sharpe(returns, n_boot: int = 2000, mean_block: int = None,
                                seed: int = 12345, ppy: int = 252) -> np.ndarray:
    """B bootstrap Sharpes via the Politis-Romano stationary bootstrap
    (geometric block lengths, wrap-around indexing)."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(r)
    if n < 30:
        raise ValueError("need >= 30 observations to bootstrap a Sharpe")
    L = mean_block or _default_block(n)
    p = 1.0 / L
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(0, n)          # start a new block
            else:
                i = (i + 1) % n                 # continue the current block (wrap)
        out[b] = _ann_sharpe(r[idx], ppy)
    return out


def sharpe_significance(returns, n_boot: int = 2000, block: int = None,
                        method: str = "stationary", seed: int = 12345, ppy: int = 252) -> dict:
    """Point Sharpe + bootstrap CI + one-sided p-value for Sharpe <= 0."""
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    point = _ann_sharpe(r, ppy)
    if method == "moving":
        boots = moving_block_sharpe(r, n_boot, block, seed, ppy)
    else:
        boots = stationary_bootstrap_sharpe(r, n_boot, block, seed, ppy)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_value = float(np.mean(boots <= 0.0))       # H0: Sharpe <= 0
    return {"sharpe": point, "ci95": (float(lo), float(hi)), "p_value_le_0": p_value,
            "n_obs": int(len(r)), "block": block or _default_block(len(r)),
            "method": method, "n_boot": n_boot,
            "significant_at_5pct": bool(lo > 0.0)}


def years_to_significance(sharpe: float, z: float = 1.96) -> float:
    """How many YEARS of live record are needed before an annual Sharpe of this
    size becomes statistically distinguishable from zero at the given z.

    Uses the Lo (2002) iid standard error of an annualized Sharpe:
        SE(SR) ~ sqrt((1 + 0.5*SR^2) / Y)
    Require SR / SE >= z  ->  Y >= z^2 * (1 + 0.5*SR^2) / SR^2.

    This is why a thin edge is effectively unprovable: the required record grows
    like 1/SR^2. A real Sharpe of ~0.5 needs ~17 years; ~1.0 needs ~6. It is the
    honest counterweight to any short 'good streak'.
    """
    s = abs(float(sharpe))
    if s <= 0:
        return float("inf")
    return float(z ** 2 * (1 + 0.5 * s ** 2) / s ** 2)


def format_significance(d: dict) -> str:
    lo, hi = d["ci95"]
    verdict = ("edge is statistically distinguishable from luck"
               if d["significant_at_5pct"] else
               "CANNOT distinguish the edge from luck at 5% -- treat Sharpe as unproven")
    return (f"SHARPE SIGNIFICANCE ({d['method']} bootstrap, {d['n_boot']} resamples, "
            f"block~{d['block']}, n={d['n_obs']})\n"
            f"  point Sharpe : {d['sharpe']:.2f}\n"
            f"  95% CI       : [{lo:.2f}, {hi:.2f}]\n"
            f"  p(Sharpe<=0) : {d['p_value_le_0']:.3f}\n"
            f"  -> {verdict}")
