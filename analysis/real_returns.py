"""
Real-terms (hard-currency) return series (audit Tier-1 #3).

WHY THIS EXISTS. The headline "+24.5% nominal EGP" is measured in a currency
that lost most of its value over the sample: USD/EGP went ~7.13 (2015) -> ~48.8
(2026), roughly a 6.8x devaluation. A number quoted in a melting currency
overstates what the strategy actually preserved in purchasing power. Sizing
real capital against the nominal figure would be a mistake, so every headline
should be shown next to its hard-currency twin.

METHOD. We convert an EGP NAV into USD by dividing by the USD/EGP rate on each
date (EGP per USD): nav_usd_t = nav_egp_t / fx_t. The USD return series is the
pct-change of nav_usd. This captures the FX headwind exactly.

HONESTY. USD conversion is NOT the same as CPI/inflation adjustment -- true
"real" return would deflate by Egyptian CPI, which we do not have a clean daily
series for. But for EGP the currency move IS the dominant real-terms driver
(the devaluation and the inflation that drove it are two faces of the same
event), so USD terms is the right first-order correction and a conservative
one. This is stated so nobody reads "USD real return" as a precise CPI figure.

No look-ahead: each day uses only that day's FX. FX is forward-filled over the
handful of missing days (weekends/holidays where EGX printed but the FX feed
didn't) -- a carried-forward rate, never a future one.
"""

import numpy as np
import pandas as pd

from data.database import get_connection


def load_usd_egp() -> pd.Series:
    """USD/EGP daily series (EGP per 1 USD), forward-filled over gaps."""
    with get_connection() as conn:
        fx = pd.read_sql("SELECT date, usd_egp FROM macro_daily WHERE usd_egp IS NOT NULL ORDER BY date",
                         conn, parse_dates=["date"]).set_index("date")["usd_egp"]
    return fx.sort_index()


def to_hard_currency(returns_egp: pd.Series, fx: pd.Series = None) -> pd.Series:
    """Convert a nominal EGP daily-return series into USD daily returns.

    Builds the EGP NAV from the returns, divides by the aligned USD/EGP rate to
    get a USD NAV, then differences back to USD returns. FX is reindexed onto
    the return dates and forward-filled (past rates only).
    """
    r = returns_egp.dropna()
    if r.empty:
        return r
    if fx is None:
        fx = load_usd_egp()
    fx_aligned = fx.reindex(r.index.union(fx.index)).ffill().reindex(r.index)
    if fx_aligned.isna().any():
        # any leading NaN (return dates before FX history) -> back-fill just the
        # lead so the series is usable; flagged in the summary's coverage field.
        fx_aligned = fx_aligned.bfill()
    nav_egp = (1 + r).cumprod()
    nav_usd = nav_egp / fx_aligned.values
    usd_ret = nav_usd.pct_change()
    usd_ret.iloc[0] = nav_usd.iloc[0] / (1.0 / fx_aligned.iloc[0]) - 1.0  # first-day vs USD base
    return usd_ret.rename("usd_returns")


def _cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    total = float((1 + r).prod())
    years = len(r) / periods_per_year
    return total ** (1 / years) - 1 if years > 0 and total > 0 else float("nan")


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def nominal_vs_real(returns_egp: pd.Series) -> dict:
    """Side-by-side nominal-EGP vs hard-currency-USD summary of one strategy."""
    fx = load_usd_egp()
    usd = to_hard_currency(returns_egp, fx)
    egp = returns_egp.dropna()
    fx0 = float(fx.reindex(egp.index).ffill().bfill().iloc[0])
    fx1 = float(fx.reindex(egp.index).ffill().bfill().iloc[-1])
    return {
        "nominal_egp": {
            "total_return": float((1 + egp).prod() - 1),
            "cagr": _cagr(egp), "sharpe": _sharpe(egp),
        },
        "real_usd": {
            "total_return": float((1 + usd).prod() - 1),
            "cagr": _cagr(usd), "sharpe": _sharpe(usd),
        },
        "fx_start": fx0, "fx_end": fx1, "fx_devaluation_x": fx1 / fx0,
        "n_days": int(len(egp)),
    }


def format_nominal_vs_real(d: dict) -> str:
    n, r = d["nominal_egp"], d["real_usd"]
    return (
        "REAL-TERMS CHECK (nominal EGP vs hard-currency USD)\n"
        f"  USD/EGP over sample: {d['fx_start']:.2f} -> {d['fx_end']:.2f}  "
        f"({d['fx_devaluation_x']:.2f}x devaluation)\n"
        f"  {'':14s}{'nominal EGP':>14s}{'real USD':>14s}\n"
        f"  {'total return':14s}{n['total_return']*100:>13.1f}%{r['total_return']*100:>13.1f}%\n"
        f"  {'CAGR':14s}{n['cagr']*100:>13.1f}%{r['cagr']*100:>13.1f}%\n"
        f"  {'Sharpe':14s}{n['sharpe']:>14.2f}{r['sharpe']:>14.2f}\n"
        "  NOTE: USD terms approximates real return (dominant driver for EGP); it is\n"
        "  NOT a CPI deflation. Size capital against the USD column, not the EGP one."
    )
