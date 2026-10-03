"""
Source-level correction of uncorrected split/data artifacts in
daily_ohlcv.adjusted_close (Phase A2).

HONESTY / PROVENANCE (house rule 7): the ideal input here is the real EGX
corporate-actions disclosure feed. That feed is not available to this code,
and fabricating split/dividend records would be worse than having none. So
this module does NOT claim to have authoritative corporate-action data.
Instead it DETECTS artifacts empirically -- single-day adjusted returns beyond
what EGX's daily price limits (~10-20%) permit are, with near-certainty,
uncorrected corporate actions that Yahoo's Adj Close missed (confirmed by
hand for MFPC's ~8:1 split on 2024-01-02). Every row it writes to
corporate_actions is stamped action_type='SPLIT_DETECTED' precisely so nobody
mistakes a heuristic detection for a verified filing.

Two artifact kinds are handled differently:
  * split_like  -- a large jump that PERSISTS. Treated as a corporate action:
    all adjusted prices BEFORE the ex-date are back-adjusted by the jump
    factor so the series is continuous and the ex-date return becomes ~0
    (the standard backward split adjustment; a split has ~0 economic return).
  * reversal_spike -- a large jump that REVERSES within a few days. That's a
    bad tick, not a corporate action; the offending day is replaced by the
    geometric interpolation of its neighbours.

The original Yahoo Adj Close is snapshotted to daily_ohlcv_adjclose_v0 before
anything is changed, so the correction is auditable and reversible.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.corporate_actions")

ARTIFACT_THRESHOLD = 0.30   # |daily adjusted return| beyond this = artifact (EGX limits ~10-20%)
REVERSAL_WINDOW = 3         # a jump reversed within this many days is a bad tick, not a split
REVERSAL_FRAC = 0.5         # reversal must undo at least this fraction of the jump (in log terms)


def _load_adjusted() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT ticker, date, close, adjusted_close FROM daily_ohlcv ORDER BY ticker, date",
                           conn, parse_dates=["date"])


def detect_artifacts(threshold: float = ARTIFACT_THRESHOLD) -> pd.DataFrame:
    """Return every artifact candidate: ticker, date, ret, log_ret, kind
    ('split_like' | 'reversal_spike'). A jump is a reversal_spike if an
    opposite jump of comparable size occurs within REVERSAL_WINDOW days.
    """
    df = _load_adjusted()
    rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        lr = np.log(g["adjusted_close"]).diff()
        big = np.where(lr.abs() > np.log1p(threshold))[0]
        for i in big:
            kind = "split_like"
            lo, hi = i + 1, min(i + REVERSAL_WINDOW, len(g) - 1)
            for j in range(lo, hi + 1):
                if np.sign(lr[j]) == -np.sign(lr[i]) and abs(lr[j]) >= REVERSAL_FRAC * abs(lr[i]):
                    kind = "reversal_spike"
                    break
            rows.append({"ticker": ticker, "date": g["date"].iloc[i],
                         "ret": float(np.expm1(lr[i])), "log_ret": float(lr[i]), "kind": kind})
    return pd.DataFrame(rows)


def apply_corrections(threshold: float = ARTIFACT_THRESHOLD, dry_run: bool = False) -> dict:
    """Detect artifacts, snapshot the original Adj Close, correct
    daily_ohlcv.adjusted_close at the source, and record detected splits in
    corporate_actions. Returns a summary dict. dry_run=True detects + reports
    without writing.
    """
    artifacts = detect_artifacts(threshold)
    if artifacts.empty:
        logger.info("apply_corrections: no artifacts detected at threshold %.0f%%", threshold * 100)
        return {"detected": 0, "split_like": 0, "reversal_spike": 0}

    df = _load_adjusted()
    corrected = {}   # (ticker, date_str) -> new adjusted_close
    ca_rows = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        adj = g["adjusted_close"].to_numpy(dtype=float).copy()
        tick_art = artifacts[artifacts["ticker"] == ticker].sort_values("date")

        # 1) reversal spikes -> interpolate the offending day (geometric mean of neighbours)
        date_to_i = {d: k for k, d in enumerate(g["date"])}
        for _, a in tick_art[tick_art["kind"] == "reversal_spike"].iterrows():
            i = date_to_i[a["date"]]
            if 0 < i < len(adj) - 1:
                adj[i] = float(np.sqrt(adj[i - 1] * adj[i + 1]))

        # 2) split-like -> back-adjust everything strictly before each ex-date
        #    by that day's jump factor (cumulative for multiple splits).
        splits = tick_art[tick_art["kind"] == "split_like"]
        for _, a in splits.iterrows():
            i = date_to_i[a["date"]]
            factor = adj[i] / adj[i - 1] if i > 0 and adj[i - 1] > 0 else 1.0
            adj[:i] = adj[:i] * factor   # bring pre-ex prices onto the post-ex scale
            ca_rows.append((ticker, pd.Timestamp(a["date"]).strftime("%Y-%m-%d"),
                            "SPLIT_DETECTED", float(1.0 / factor) if factor > 0 else None, None))

        for k in range(len(g)):
            corrected[(ticker, g["date"].iloc[k].strftime("%Y-%m-%d"))] = float(adj[k])

    n_split = int((artifacts["kind"] == "split_like").sum())
    n_rev = int((artifacts["kind"] == "reversal_spike").sum())
    logger.info("apply_corrections: %d artifacts (%d split_like, %d reversal_spike)",
                len(artifacts), n_split, n_rev)
    if dry_run:
        return {"detected": len(artifacts), "split_like": n_split, "reversal_spike": n_rev,
                "artifacts": artifacts}

    with get_connection() as conn:
        # freeze original Adj Close (once)
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_ohlcv_adjclose_v0'").fetchone()
        if not exists:
            conn.execute("CREATE TABLE daily_ohlcv_adjclose_v0 AS "
                         "SELECT ticker, date, adjusted_close FROM daily_ohlcv")
            logger.info("apply_corrections: froze original Adj Close -> daily_ohlcv_adjclose_v0")
        # write corrected adjusted_close
        conn.executemany("UPDATE daily_ohlcv SET adjusted_close=? WHERE ticker=? AND date=?",
                         [(v, t, d) for (t, d), v in corrected.items()])
        # record detected splits (idempotent on the PK)
        conn.executemany(
            "INSERT OR REPLACE INTO corporate_actions (ticker, date, action_type, ratio, amount_egp) "
            "VALUES (?, ?, ?, ?, ?)", ca_rows)
    return {"detected": len(artifacts), "split_like": n_split, "reversal_spike": n_rev,
            "corrected_rows": len(corrected), "artifacts": artifacts}


def reconcile_and_gate(sigma_threshold: float = 20.0) -> dict:
    """A2 gate: after correction, no adjusted daily return should exceed
    `sigma_threshold` sigmas per ticker except documented macro events. Also
    counts how many rows now diverge from the frozen Yahoo Adj Close.
    """
    df = _load_adjusted()
    max_sigmas = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date")
        r = np.log(g["adjusted_close"]).diff()
        rc = r.dropna()
        if len(rc) > 30 and rc.std() > 0:
            z = (r - rc.mean()).abs() / rc.std()
            worst_i = int(z.iloc[1:].idxmax())
            max_sigmas.append((ticker, float(z.max()), g["date"].loc[worst_i]))
    worst = sorted(max_sigmas, key=lambda x: -x[1])[:8]
    with get_connection() as conn:
        has_snap = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_ohlcv_adjclose_v0'").fetchone()
        diverged = 0
        if has_snap:
            diverged = conn.execute(
                "SELECT COUNT(*) FROM daily_ohlcv d JOIN daily_ohlcv_adjclose_v0 v "
                "ON d.ticker=v.ticker AND d.date=v.date "
                "WHERE ABS(d.adjusted_close - v.adjusted_close) > 0.01").fetchone()[0]
    return {"worst_sigma": worst, "n_diverged_from_yahoo": diverged,
            "gate_pass": all(s < sigma_threshold for _, s, _ in max_sigmas)}
