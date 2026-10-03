"""
Performance analytics for the backtest, implemented from scratch (no
pyfolio / empyrical). Everything takes a daily-return Series and returns
plain floats so it is trivially testable.

Conventions: returns are simple daily returns; annualization uses 252
trading days; the risk-free rate defaults to 0 (EGX T-bill/CBE overnight
can be wired in later) so Sharpe here is an EXCESS-over-zero Sharpe --
stated plainly rather than silently assuming a rate.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean(returns) -> np.ndarray:
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    return r


def annualized_return(returns) -> float:
    """Geometric annualized return from a daily-return series."""
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    total = np.prod(1 + r)
    years = len(r) / TRADING_DAYS
    return float(total ** (1 / years) - 1) if years > 0 and total > 0 else float(total - 1)


def annualized_vol(returns) -> float:
    r = _clean(returns)
    return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")


def sharpe_ratio(returns, rf: float = 0.0) -> float:
    """(mean daily excess return / daily std) * sqrt(252). rf is annual."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    daily_rf = rf / TRADING_DAYS
    excess = r - daily_rf
    sd = np.std(excess, ddof=1)
    return float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def sortino_ratio(returns, rf: float = 0.0) -> float:
    """Like Sharpe but the denominator penalizes only downside deviation."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    daily_rf = rf / TRADING_DAYS
    excess = r - daily_rf
    downside = excess[excess < 0]
    dd = np.sqrt(np.mean(downside ** 2)) if len(downside) else 0.0
    return float(np.mean(excess) / dd * np.sqrt(TRADING_DAYS)) if dd > 0 else float("nan")


def max_drawdown(returns) -> float:
    """Most negative peak-to-trough decline of the cumulative NAV (<= 0)."""
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1).min())


def calmar_ratio(returns) -> float:
    mdd = max_drawdown(returns)
    ann = annualized_return(returns)
    return float(ann / abs(mdd)) if mdd and mdd < 0 else float("nan")


def hit_rate(returns) -> float:
    r = _clean(returns)
    return float((r > 0).mean()) if len(r) else float("nan")


def historical_var(returns, alpha: float = 0.95) -> float:
    """Historical VaR: the (1-alpha) quantile of returns (a negative number)."""
    r = _clean(returns)
    return float(np.quantile(r, 1 - alpha)) if len(r) else float("nan")


def historical_cvar(returns, alpha: float = 0.95) -> float:
    """Expected shortfall: mean of returns at or below the VaR threshold."""
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    var = np.quantile(r, 1 - alpha)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def turnover_stats(turnover: dict, periods_per_year: int = 252,
                   rebalance_days: int = 21) -> dict:
    """Turnover per rebalance and annualised, from the engine's per-date dict.

    WHY THIS IS REPORTED. The engine has always computed turnover and never
    surfaced it next to CAGR/Sharpe (audit 2026-08-05, Section 15). Turnover is
    the multiplier on every cost assumption: a result that needs 4x the turnover
    to earn the same return is a far more fragile result, and you cannot see
    that from the return series alone. It is also the only way to check that
    live trading frequency matches what was backtested.
    """
    if not turnover:
        return {"turnover_per_rebalance": float("nan"), "turnover_annual": float("nan"),
                "n_rebalances": 0}
    vals = np.asarray([float(v) for v in turnover.values()], dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {"turnover_per_rebalance": float("nan"), "turnover_annual": float("nan"),
                "n_rebalances": 0}
    per = float(vals.mean())
    rebals_per_year = periods_per_year / max(rebalance_days, 1)
    return {"turnover_per_rebalance": per,
            "turnover_annual": per * rebals_per_year,
            "n_rebalances": int(len(vals))}


def summary(returns, rf: float = 0.0, turnover: dict = None,
            rebalance_days: int = 21) -> dict:
    """All headline metrics for a daily-return series, as plain floats.

    Pass `turnover` (the engine's per-rebalance dict) to have turnover reported
    alongside return and risk, rather than computed and discarded.
    """
    out = {
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns, rf),
        "sortino": sortino_ratio(returns, rf),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "hit_rate": hit_rate(returns),
        "var_95": historical_var(returns, 0.95),
        "cvar_95": historical_cvar(returns, 0.95),
        "n_days": int(len(_clean(returns))),
    }
    out.update(turnover_stats(turnover or {}, rebalance_days=rebalance_days))
    return out
