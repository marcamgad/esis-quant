"""
Independent FORWARD paper-trading record (Part 2.4).

WHY THIS EXISTS, STATED BLUNTLY: every performance number this project has
produced is a BACKTEST -- a statement about a history the model was built on
top of, with the benefit of a fixed universe of tickers that survived to today
(survivorship), a regime model whose feature design was chosen after seeing
that history, and a strategy selected because it looked good on that same
sample. None of that is evidence the strategy works going forward. The only
thing that is, is an out-of-sample record accumulated in real time.

So this module marks the LOGGED trade lists to real EGX closes, day by day,
forward, and stores the result in its own table. It is deliberately NOT
importable into any backtest statistic, and no function here blends the two.
Until this log is long enough to be meaningful (many months, and critically at
least one BEAR regime cycle -- the overlay is the whole edge and it only acts
in BEAR), the honest answer to "does this work?" remains "unproven live".
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection
from backtest.engine import load_price_returns

logger = logging.getLogger("esis_quant.ops.paper")

BEAR_ACTED_EXPOSURE = 0.5   # a BEAR day counts only if exposure was cut below this


def _logged_trade_lists() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT run_id, as_of_date, regime, exposure, ticker, weight "
                           "FROM trade_list_log ORDER BY as_of_date", conn, parse_dates=["as_of_date"])


def update_paper_nav(start_nav: float = 1.0) -> pd.DataFrame:
    """Mark every logged trade list forward to real closes and append any new
    days to paper_trading_nav.

    Method (no look-ahead, by construction): the book logged with as_of date D
    is held from the NEXT trading session onward, until a newer trade list's
    as_of date arrives. Each day's return is sum_i w_i * r_i on realized
    adjusted returns; days with no applicable book (or a fully-cash book) earn
    0. Idempotent: only days after the last stored NAV are appended.
    """
    lists = _logged_trade_lists()
    if lists.empty:
        logger.info("update_paper_nav: no trade lists logged yet -- nothing to mark")
        return pd.DataFrame()

    rets = load_price_returns()
    with get_connection() as conn:
        existing = pd.read_sql("SELECT date, nav FROM paper_trading_nav ORDER BY date",
                               conn, parse_dates=["date"])
    last_date = existing["date"].max() if not existing.empty else None
    nav = float(existing["nav"].iloc[-1]) if not existing.empty else start_nav

    # book effective from the session AFTER each as_of date
    books = {}
    for run_id, g in lists.groupby("run_id"):
        as_of = g["as_of_date"].iloc[0]
        w = g[g["ticker"] != "__CASH__"].set_index("ticker")["weight"]
        books[as_of] = {"run_id": run_id, "weights": w,
                        "regime": g["regime"].iloc[0], "exposure": float(g["exposure"].iloc[0])}
    book_dates = sorted(books)

    start_from = book_dates[0]
    candidate_days = rets.index[rets.index > start_from]
    if last_date is not None:
        candidate_days = candidate_days[candidate_days > last_date]
    if len(candidate_days) == 0:
        logger.info("update_paper_nav: already up to date (last %s)", last_date)
        return existing

    rows = []
    for d in candidate_days:
        applicable = [bd for bd in book_dates if bd < d]
        if not applicable:
            continue
        book = books[max(applicable)]
        w = book["weights"]
        if w.empty or w.sum() <= 0:
            r = 0.0                                    # fully in cash
        else:
            day = rets.loc[d].reindex(w.index)
            r = float((day.fillna(0.0) * w).sum())     # missing print = no move
        nav *= (1 + r)
        rows.append({"date": d, "nav": nav, "daily_return": r, "exposure": book["exposure"],
                     "regime": book["regime"], "run_id": book["run_id"],
                     "n_positions": int((w > 0).sum())})

    out = pd.DataFrame(rows)
    if not out.empty:
        with get_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO paper_trading_nav "
                "(date, nav, daily_return, exposure, regime, run_id, n_positions) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(r.date.strftime("%Y-%m-%d"), float(r.nav), float(r.daily_return),
                  float(r.exposure), r.regime, r.run_id, int(r.n_positions)) for r in out.itertuples()])
        logger.info("update_paper_nav: appended %d day(s), NAV=%.4f", len(out), nav)
    return pd.concat([existing, out], ignore_index=True) if not existing.empty else out


def paper_track_record() -> dict:
    """Summarize the FORWARD record only. Returns n_days and, only once there
    is enough of it, basic stats -- with an explicit `sufficient` flag so no
    caller can quietly present 3 days of noise as a track record.
    """
    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM paper_trading_nav ORDER BY date", conn, parse_dates=["date"])
    if df.empty:
        return {"n_days": 0, "sufficient": False, "note": "no forward record yet"}

    r = df["daily_return"]
    n = len(df)
    # A BEAR day counts as evidence for the overlay ONLY if the overlay acted on
    # it (Stage 0). All 14 BEAR days logged Aug 2026 ran at exposure 1.0 because
    # the live ensemble was frozen, yet the gate counted them toward "survived a
    # BEAR regime" -- crediting a test the overlay never sat.
    is_bear = df["regime"] == "BEAR"
    acted = is_bear & (df["exposure"].fillna(1.0) < BEAR_ACTED_EXPOSURE)
    bear_days = int(acted.sum())
    out = {"n_days": n, "start": str(df["date"].min().date()), "end": str(df["date"].max().date()),
           "cum_return": float(df["nav"].iloc[-1] / df["nav"].iloc[0] - 1) if n > 1 else 0.0,
           "bear_days_observed": bear_days,
           "bear_days_overlay_inactive": int((is_bear & ~acted).sum()),
           # The overlay is the entire edge and it only acts in BEAR: without
           # observed BEAR days the forward record cannot speak to it at all.
           "sufficient": n >= 252 and bear_days >= 20,
           "note": ("forward record is too short to be evidence" if n < 252 else
                    "no BEAR regime observed yet -- the overlay (the edge) is untested live"
                    if bear_days < 20 else "forward record has minimum viable length")}
    if n > 20 and r.std() > 0:
        out["ann_return"] = float((1 + r).prod() ** (252 / n) - 1)
        out["sharpe"] = float(r.mean() / r.std() * np.sqrt(252))
        nav = df["nav"]
        out["max_drawdown"] = float((nav / nav.cummax() - 1).min())
    return out
