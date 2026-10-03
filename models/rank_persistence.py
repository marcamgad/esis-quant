"""
Phase 2.2 -- cross-sectional RANK PERSISTENCE as a standalone selection signal.

THE IDEA, AND WHY IT IS NOT A RELABELLED FACTOR. Every existing signal here asks
"how much does this stock LOAD on momentum / liquidity / EGP / oil". Rank
persistence asks a different question: "does this stock STAY where it is in the
cross-section". It is computed from the ordering itself, not from any beta, so
it is mechanically closer to orthogonal to the 5-factor score than another beta
would be. With 27 names and no size/value/quality data (Phase 2.1 showed the
regime-conditional premia collapse to the global prior), a factor-model-free
signal is one of the few genuinely untried options.

CONSTRUCTION. For each date, rank every ticker cross-sectionally on a base
signal. Persistence over a trailing window is how little that rank MOVES:

    persistence_i,t = 1 - 2 * mean_{s in window} | rank_i,s - rank_i,s-lag |

(1 = the rank never moves, 0 = it moves half the range on average.) A rolling
CORRELATION of rank against its own lag was implemented first and discarded: a
stable name's rank has near-zero variance over 20 days, so the correlation
divides by ~0 and returns inf/NaN, which silently dropped ~28% of periods
(86 usable vs 119 with the stable form). The mean-absolute-change version is
bounded, needs no variance, and measures the same idea directly.

High persistence = a stable leader or a stable laggard. Because that is
sign-agnostic, the tradable signal is persistence SIGNED by current rank:

    signal_i,t = persistence_i,t * (2*pct_rank_i,t - 1)

so a persistently-top-ranked name scores positive and a persistently-bottom one
negative. Everything uses data through t only; the forward return being
predicted is strictly after t.

PRE-REGISTERED DECISION RULE (fixed before the result was seen; do not edit
after reading output):

  NULL -- if the Fama-MacBeth premium on the persistence signal has |t| < 2.0,
          OR the signal's information coefficient (mean cross-sectional Spearman
          correlation with next-month return) is <= 0.

  REAL -- only if |t| >= 2.0 AND the mean IC > 0 AND the sign is consistent
          (IC positive in a majority of periods).

A null is a useful, publishable result and must be stated as plainly as a
positive one.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.rank_persistence")

DEFAULT_WINDOW = 20          # trailing days over which rank stability is measured
DEFAULT_LAG = 10             # how far back the comparison rank is taken
MIN_NAMES = 8
T_STAT_THRESHOLD = 2.0


def load_base_panel(column: str = "alpha_score") -> pd.DataFrame:
    """Wide (date x ticker) panel of the base signal to rank on."""
    with get_connection() as conn:
        df = pd.read_sql(f"SELECT ticker, date, {column} FROM signals "
                         f"WHERE {column} IS NOT NULL", conn, parse_dates=["date"])
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values=column).sort_index()


def compute_persistence(panel: pd.DataFrame, window: int = DEFAULT_WINDOW,
                        lag: int = DEFAULT_LAG) -> pd.DataFrame:
    """Signed rank-persistence signal, (date x ticker). Causal by construction:
    only ranks at or before t are used."""
    if panel.empty:
        return pd.DataFrame()
    # cross-sectional percentile rank each day (0..1), NaN-safe
    ranks = panel.rank(axis=1, pct=True)

    # Persistence = how little the name's rank MOVES, measured as mean absolute
    # rank change over the window, mapped to [-1, 1]:
    #     1  -> rank never changes
    #     0  -> rank moves half the range on average
    # A rolling CORRELATION of rank against its own lag was tried first and is
    # numerically fragile here: over a 20-day window a stable name's rank has
    # near-zero variance, so the correlation divides by ~0 and returns inf/NaN.
    # Mean absolute change is bounded, needs no variance, and measures the same
    # idea more directly.
    drift = (ranks - ranks.shift(lag)).abs()
    persistence = 1.0 - 2.0 * drift.rolling(window, min_periods=max(5, window // 2)).mean()
    signed = persistence * (2.0 * ranks - 1.0)
    return signed


def _monthly_align(signal: pd.DataFrame) -> pd.DataFrame:
    """Month-end value of the signal, long format -- matches the monthly
    Fama-MacBeth cadence used elsewhere in the project."""
    m = signal.resample("ME").last()
    out = m.stack().reset_index()
    out.columns = ["date", "ticker", "persistence"]
    return out


def evaluate(column: str = "alpha_score", window: int = DEFAULT_WINDOW,
             lag: int = DEFAULT_LAG) -> dict:
    """Fama-MacBeth premium and information coefficient for the signal."""
    from models.fama_macbeth import compute_monthly_returns
    panel = load_base_panel(column)
    if panel.empty:
        return {"error": f"no {column} data"}
    sig = _monthly_align(compute_persistence(panel, window, lag))
    sig["period"] = pd.PeriodIndex(sig["date"], freq="M")

    rets = compute_monthly_returns()
    rets["period"] = pd.PeriodIndex(pd.to_datetime(rets["month"]), freq="M")
    # signal at month m predicts the return of month m+1 -- no look-ahead
    sig["period"] = sig["period"] + 1
    merged = rets.merge(sig[["ticker", "period", "persistence"]],
                        on=["ticker", "period"], how="inner").dropna()

    coefs, ics = [], []
    for period, g in merged.groupby("period"):
        g = g.dropna(subset=["persistence", "monthly_return"])
        if len(g) < MIN_NAMES:
            continue
        X = np.column_stack([np.ones(len(g)), g["persistence"].to_numpy(float)])
        y = g["monthly_return"].to_numpy(float)
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        coefs.append(beta[1])
        ic = g["persistence"].rank().corr(g["monthly_return"].rank())
        if ic == ic:
            ics.append(float(ic))
    if len(coefs) < 6:
        return {"error": f"only {len(coefs)} usable periods"}

    c = np.asarray(coefs, dtype=float)
    se = c.std(ddof=1) / np.sqrt(len(c))
    ic_arr = np.asarray(ics, dtype=float)
    return {"n_periods": len(c), "premium": float(c.mean()), "se": float(se),
            "t_stat": float(c.mean() / se) if se > 0 else float("nan"),
            "mean_ic": float(ic_arr.mean()) if len(ic_arr) else float("nan"),
            "ic_positive_frac": float((ic_arr > 0).mean()) if len(ic_arr) else float("nan"),
            "window": window, "lag": lag, "base": column}


def verdict(res: dict) -> dict:
    """Apply the PRE-REGISTERED rule."""
    if "error" in res:
        return {"verdict": "NULL", "reasons": [res["error"]]}
    reasons = []
    if abs(res["t_stat"]) < T_STAT_THRESHOLD:
        reasons.append(f"premium not significant (|t| = {abs(res['t_stat']):.2f} < {T_STAT_THRESHOLD})")
    if not (res["mean_ic"] > 0):
        reasons.append(f"mean IC not positive ({res['mean_ic']:.4f})")
    if res.get("ic_positive_frac", 0) <= 0.5:
        reasons.append(f"IC positive in only {res['ic_positive_frac']*100:.0f}% of periods")
    return {"verdict": "NULL" if reasons else "REAL", "reasons": reasons}


def format_result(res: dict, v: dict) -> str:
    if "error" in res:
        return f"RANK PERSISTENCE: {res['error']}\n  PRE-REGISTERED VERDICT: {v['verdict']}"
    lines = [f"RANK PERSISTENCE SIGNAL (base={res['base']}, window={res['window']}d, lag={res['lag']}d)",
             f"  periods           : {res['n_periods']}",
             f"  FM premium        : {res['premium']:+.5f}  (se {res['se']:.5f})",
             f"  t-stat            : {res['t_stat']:+.2f}",
             f"  mean IC           : {res['mean_ic']:+.4f}",
             f"  IC positive       : {res['ic_positive_frac']*100:.0f}% of periods",
             f"  PRE-REGISTERED VERDICT: {v['verdict']}"]
    for r in v["reasons"]:
        lines.append(f"     - {r}")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    r = evaluate()
    print(format_result(r, verdict(r)))
