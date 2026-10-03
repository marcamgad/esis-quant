"""
Corporate-action reconciliation (audit Tier-1 #5).

WHY. The whole backtest and every live signal are computed on ADJUSTED close.
If Yahoo's split/dividend adjustment is wrong or missing for an EGX name, the
returns are silently corrupted -- a 2:1 split that Yahoo missed shows up as a
-50% "return" that never happened, poisoning both the covariance estimate and
the regime model. Our own corporate_actions table has only a handful of rows,
so we cannot assume it is complete either. This module cross-checks the two
against each other and reports the discrepancies instead of trusting either.

METHOD. For a clean series, factor_t = adjusted_close_t / close_t is a step
function: it is constant between corporate actions and jumps only on an
ex-date (down as you go back in time, because historical prices are scaled to
be comparable with today). We reconstruct that factor per ticker, find the days
where log(factor) jumps by more than a threshold, and call those the IMPLIED
corporate actions. Then:

  * IMPLIED events with NO row in corporate_actions  -> Yahoo adjusted for
    something we never recorded (usually fine, but unverified -- these are the
    ones to eyeball).
  * corporate_actions rows with NO implied jump       -> we recorded an action
    Yahoo did not adjust for -> the adjusted series is likely WRONG for that
    name and its returns should be distrusted.

A large implied jump (|Δlog factor| > SPLIT_LOG_THRESHOLD, ~ a >5% single-day
factor change) is almost certainly a split/bonus; small ones are dividends and
are noisier, so we separate the two. No look-ahead -- this is pure post-hoc
data QA over stored history.
"""

import numpy as np
import pandas as pd

from data.database import get_connection

# a factor change above this in log terms (~5.1%) is split/bonus territory;
# below it is dividend-sized and much noisier to attribute.
SPLIT_LOG_THRESHOLD = 0.05
# ignore sub-0.5% factor wobble (rounding in Yahoo's adj close)
NOISE_LOG_THRESHOLD = 0.005


def implied_actions(ticker: str) -> pd.DataFrame:
    """Days where adjusted_close/close jumps -> an implied corporate action."""
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT date, close, adjusted_close FROM daily_ohlcv "
            "WHERE ticker=? AND close > 0 AND adjusted_close > 0 ORDER BY date",
            conn, params=(ticker,), parse_dates=["date"])
    if df.empty:
        return pd.DataFrame(columns=["date", "log_jump", "factor_ratio", "kind"])
    df["factor"] = df["adjusted_close"] / df["close"]
    df["log_jump"] = np.log(df["factor"]).diff()
    hits = df[df["log_jump"].abs() > NOISE_LOG_THRESHOLD].copy()
    hits["factor_ratio"] = np.exp(hits["log_jump"])
    hits["kind"] = np.where(hits["log_jump"].abs() > SPLIT_LOG_THRESHOLD, "split/bonus", "dividend")
    return hits[["date", "log_jump", "factor_ratio", "kind"]].reset_index(drop=True)


def _recorded_actions(ticker: str) -> pd.DataFrame:
    with get_connection() as conn:
        # tolerate schema variance: pull all columns, find a date-like one
        df = pd.read_sql("SELECT * FROM corporate_actions WHERE ticker=?", conn, params=(ticker,))
    if df.empty:
        return pd.DataFrame(columns=["date"])
    date_col = next((c for c in ("ex_date", "date", "action_date", "effective_date") if c in df.columns), None)
    if date_col is None:
        return pd.DataFrame(columns=["date"])
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=["date"])


def reconcile_ticker(ticker: str, match_window_days: int = 3) -> dict:
    """Match implied vs recorded actions for one ticker (within a few days,
    since ex-date conventions differ between sources)."""
    imp = implied_actions(ticker)
    rec = _recorded_actions(ticker)
    imp_splits = imp[imp["kind"] == "split/bonus"]

    unrecorded, matched = [], 0
    rec_dates = list(rec["date"]) if not rec.empty else []
    used = set()
    for _, row in imp_splits.iterrows():
        d = row["date"]
        hit = None
        for j, rd in enumerate(rec_dates):
            if j in used:
                continue
            if abs((rd - d).days) <= match_window_days:
                hit = j
                break
        if hit is not None:
            used.add(hit)
            matched += 1
        else:
            unrecorded.append({"date": str(d.date()), "factor_ratio": float(row["factor_ratio"])})

    recorded_no_implied = [str(rd.date()) for j, rd in enumerate(rec_dates) if j not in used]
    return {
        "ticker": ticker,
        "n_implied_splits": int(len(imp_splits)),
        "n_implied_dividends": int((imp["kind"] == "dividend").sum()),
        "n_recorded": int(len(rec_dates)),
        "matched_splits": matched,
        "implied_splits_unrecorded": unrecorded,          # Yahoo adjusted, we didn't log (verify)
        "recorded_without_price_adjustment": recorded_no_implied,  # we logged, Yahoo DIDN'T adjust (SUSPECT)
    }


def reconcile_all(tickers=None) -> pd.DataFrame:
    """One row per ticker summarizing the reconciliation, worst offenders first."""
    if tickers is None:
        with get_connection() as conn:
            tickers = [r[0] for r in conn.execute(
                "SELECT DISTINCT ticker FROM daily_ohlcv ORDER BY ticker").fetchall()]
    rows = []
    for t in tickers:
        r = reconcile_ticker(t)
        rows.append({
            "ticker": t,
            "implied_splits": r["n_implied_splits"],
            "recorded": r["n_recorded"],
            "matched": r["matched_splits"],
            "unrecorded_splits": len(r["implied_splits_unrecorded"]),
            "recorded_unadjusted": len(r["recorded_without_price_adjustment"]),
            "suspect": len(r["implied_splits_unrecorded"]) + len(r["recorded_without_price_adjustment"]),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("suspect", ascending=False).reset_index(drop=True) if not df.empty else df


def format_reconcile(df: pd.DataFrame) -> str:
    if df.empty:
        return "corporate-action reconciliation: no data."
    total_unrec = int(df["unrecorded_splits"].sum())
    total_unadj = int(df["recorded_unadjusted"].sum())
    lines = ["CORPORATE-ACTION RECONCILIATION (adjusted_close vs corporate_actions)",
             f"  {len(df)} tickers | implied splits Yahoo made but we never logged: {total_unrec}",
             f"  logged actions Yahoo did NOT adjust for (SUSPECT returns): {total_unadj}"]
    flagged = df[df["suspect"] > 0].head(12)
    if flagged.empty:
        lines.append("  no discrepancies -- adjusted series and action log agree.")
    else:
        lines.append(f"  {'ticker':8s}{'impl.spl':>9s}{'logged':>8s}{'unrec':>7s}{'unadj':>7s}")
        for r in flagged.itertuples():
            lines.append(f"  {r.ticker:8s}{r.implied_splits:>9d}{r.recorded:>8d}"
                         f"{r.unrecorded_splits:>7d}{r.recorded_unadjusted:>7d}")
    lines.append("  ACTION: 'unadj' > 0 means a split we recorded is NOT in the adjusted price -> "
                 "that ticker's returns are corrupted; fix before trusting it.")
    return "\n".join(lines)
