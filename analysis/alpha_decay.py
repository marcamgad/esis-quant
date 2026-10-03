"""
Signal half-life and IC decay  (deferred item P6.21).

WHAT THIS MEASURES. For a cross-sectional signal, the information coefficient
(IC) is the rank correlation between today's signal and the forward return over
horizon h. Plotting IC against h shows how fast the signal's information decays,
and fitting an exponential to it gives a HALF-LIFE: the horizon at which half
the information is gone. That number tells you how often you must trade to
capture the signal -- and therefore whether the signal can survive its own
transaction costs.

EXPECT NOTHING, AND SAY SO. The factor regression (analysis/factor_exposure.py)
already found system alpha of -0.86% with t = -0.23 after the survivorship
correction. If there is no alpha, there is nothing to decay, and this module
will measure the decay of noise. That is a legitimate and useful result: it
turns "we never checked" into "we checked and there is nothing there." A
half-life fitted to noise is not evidence of a signal, so the code refuses to
report one when the ICs are statistically indistinguishable from zero.

WHY MONITOR IT ANYWAY. The live book is min-variance with mu=0 -- it makes no
cross-sectional bet at all, so it has no IC by construction. The signals that DO
exist in the database (alpha_score, causal_signal) are research artifacts off
the live path. Tracking their IC over time is how you would notice a signal
decaying BEFORE trading it, which is the point of monitoring rather than
post-mortem.

METHOD NOTE. Overlapping forward windows make consecutive observations nearly
identical, so the naive t-stat on mean IC is inflated by roughly sqrt(h). The
same correction applied in analysis/overlay_tuning.py is applied here.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.alpha_decay")

HORIZONS = (1, 5, 10, 21, 42, 63)
MIN_NAMES = 5


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < MIN_NAMES:
        return np.nan
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def ic_by_horizon(signal_wide: pd.DataFrame, returns_wide: pd.DataFrame,
                  horizons=HORIZONS) -> pd.DataFrame:
    """Mean IC at each forward horizon, with an overlap-corrected t-stat."""
    rows = []
    common = signal_wide.index.intersection(returns_wide.index)
    sig = signal_wide.loc[common]
    ret = returns_wide.loc[common]
    for h in horizons:
        fwd = (1 + ret).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        ics = []
        for d in sig.index:
            s = sig.loc[d].dropna()
            f = fwd.loc[d].dropna() if d in fwd.index else pd.Series(dtype=float)
            names = s.index.intersection(f.index)
            if len(names) < MIN_NAMES:
                continue
            v = _spearman(s[names].values, f[names].values)
            if v == v:
                ics.append(v)
        if len(ics) < 20:
            continue
        arr = np.asarray(ics)
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        t_raw = float(arr.mean() / se) if se > 0 else np.nan
        # overlapping windows share h-1 days; treating them as independent
        # inflates the t-stat by ~sqrt(h)
        t_adj = t_raw / np.sqrt(h) if t_raw == t_raw else np.nan
        rows.append({"horizon": h, "n_periods": len(arr), "mean_ic": float(arr.mean()),
                     "ic_std": float(arr.std(ddof=1)), "t_raw": t_raw, "t_adj": t_adj,
                     "significant": bool(abs(t_adj) >= 2.0) if t_adj == t_adj else False})
    return pd.DataFrame(rows)


def half_life(ic: pd.DataFrame) -> dict:
    """Fit |IC(h)| = |IC(0)| * exp(-lambda*h) and report ln(2)/lambda.

    REFUSES to fit when no horizon has a significant IC. A half-life fitted to
    noise is a number with no referent, and reporting one would manufacture the
    impression of a signal out of sampling error.
    """
    if ic.empty:
        return {"ok": False, "reason": "no IC data"}
    if not ic["significant"].any():
        return {"ok": False, "significant_any": False,
                "max_abs_t": float(ic["t_adj"].abs().max()),
                "reason": ("no horizon has an IC distinguishable from zero after the "
                           "overlap correction — there is no signal whose decay could "
                           "be measured, so no half-life is reported")}
    y = np.log(np.maximum(ic["mean_ic"].abs().values, 1e-9))
    x = ic["horizon"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return {"ok": False, "significant_any": True, "slope": float(slope),
                "reason": "IC does not decay over the horizons measured"}
    hl = float(np.log(2) / -slope)
    return {"ok": True, "half_life_days": hl, "lambda": float(-slope),
            "ic0": float(np.exp(intercept))}


def load_signal(column: str = "alpha_score") -> pd.DataFrame:
    from data.database import get_connection
    with get_connection() as conn:
        df = pd.read_sql(f"SELECT ticker, date, {column} AS v FROM signals "
                         f"WHERE {column} IS NOT NULL", conn, parse_dates=["date"])
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="v").sort_index()


def run(column: str = "alpha_score") -> dict:
    from backtest.engine import load_price_returns
    sig = load_signal(column)
    if sig.empty:
        return {"ok": False, "column": column, "reason": f"no {column} rows in signals"}
    ic = ic_by_horizon(sig, load_price_returns())
    return {"ok": True, "column": column, "ic": ic, "half_life": half_life(ic)}


def format_run(r: dict) -> str:
    if not r.get("ok"):
        return f"ALPHA DECAY — {r.get('column')}: {r.get('reason')}"
    ic = r["ic"]
    lines = [f"ALPHA DECAY — signal '{r['column']}'",
             f"  {'horizon':>8}{'periods':>9}{'mean IC':>10}{'t(raw)':>9}{'t(adj)':>9}  verdict"]
    for x in ic.itertuples():
        v = "significant" if x.significant else "not significant"
        lines.append(f"  {x.horizon:>7}d{x.n_periods:>9}{x.mean_ic:>10.4f}"
                     f"{x.t_raw:>9.2f}{x.t_adj:>9.2f}  {v}")
    hl = r["half_life"]
    lines.append("")
    if hl.get("ok"):
        lines.append(f"  half-life {hl['half_life_days']:.1f} days (IC0 {hl['ic0']:.4f})")
    else:
        lines.append(f"  no half-life: {hl.get('reason')}")
    lines.append("  t(adj) divides by sqrt(horizon) — overlapping windows are not")
    lines.append("  independent observations and the raw figure double-counts.")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    for col in ("alpha_score", "causal_signal"):
        print(format_run(run(col)))
        print()
