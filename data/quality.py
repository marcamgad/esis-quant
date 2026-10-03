"""
Data-quality gate + daily report (Part 2.1 / 2.5 of the production audit).

The audit found the system's most dangerous failure mode is not a crash but a
SILENT DEGRADATION: yfinance returns an empty frame, a ticker quietly stops
updating, a stray non-session print lands in the panel -- and the optimizer
happily produces a confident-looking book from a gutted universe. Nothing in
the pipeline noticed. (Measured: the universe collapsed to 2 of 26 names for
~5 months of 2024 and the backtest reported it as a real strategy.)

So: check the data BEFORE it reaches the optimizer, return structured problems,
and let the caller REFUSE to trade rather than degrade. Nothing here silently
repairs anything -- it reports.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.quality")

MAX_STALE_DAYS = 5          # a ticker not printing for this long is stale
MAX_DB_STALE_DAYS = 5       # whole-DB staleness before we refuse to emit a trade list
MIN_UNIVERSE = 10           # fewer tradable names than this = do not trade
# Macro can rot while prices stay fresh, silently truncating the regime signal.
# Slightly looser than the price limit because macro feeds publish on their own
# calendar, but tight enough that a multi-week gap cannot pass unnoticed.
MAX_MACRO_STALE_DAYS = 7
MAX_SIGNAL_LAG_SESSIONS = 2  # regime_history / regime_ensemble vs the last session
ANOMALY_RET = 0.30         # |daily return| beyond EGX limits = anomaly (see corporate_actions)


def _load_prices() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT ticker, date, close, adjusted_close, value_traded_egp "
                           "FROM daily_ohlcv ORDER BY ticker, date", conn, parse_dates=["date"])


def data_quality_report(as_of: pd.Timestamp = None) -> dict:
    """Full data-quality snapshot. Returns a dict with per-problem lists and an
    overall `ok` flag. Never raises on bad data -- it REPORTS bad data; callers
    decide whether to proceed (see assert_data_fit_to_trade).
    """
    px = _load_prices()
    if px.empty:
        return {"ok": False, "fatal": ["daily_ohlcv is empty"], "as_of": None,
                "stale_tickers": [], "anomalies": [], "spurious_dates": [], "n_tickers": 0}

    # Anchor EVERYTHING to the last genuine SESSION, not the last raw date.
    # A phantom print on a closed-market day (yfinance emitted 2 of 27 tickers on
    # 2026-07-23, Revolution Day) otherwise (a) becomes the as-of date, so the
    # universe is read on a day almost nothing traded -> "0 tradable names", and
    # (b) makes a stale DB look fresh, masking the very staleness this gate
    # exists to catch. See data.calendar.coverage_by_date.
    from data.calendar import valid_trading_days
    wide_all = px.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    sessions = valid_trading_days(wide_all)
    last_session = sessions[-1] if len(sessions) else px["date"].max()

    as_of = pd.Timestamp(as_of) if as_of is not None else last_session
    problems, fatal = [], []

    # --- 1. whole-DB staleness -------------------------------------------------
    db_age = (pd.Timestamp.now().normalize() - pd.Timestamp(last_session).normalize()).days
    if db_age > MAX_DB_STALE_DAYS:
        fatal.append(f"database is {db_age} days stale (last session {pd.Timestamp(last_session).date()}, "
                     f"limit {MAX_DB_STALE_DAYS})")

    # --- 2. per-ticker staleness ----------------------------------------------
    last_seen = px.groupby("ticker")["date"].max()
    stale = last_seen[(as_of - last_seen).dt.days > MAX_STALE_DAYS]
    stale_tickers = [{"ticker": t, "last_date": str(d.date()), "days_stale": int((as_of - d).days)}
                     for t, d in stale.items()]

    # --- 3. price anomalies (post-correction residue) --------------------------
    anomalies = []
    for ticker, g in px.groupby("ticker"):
        g = g.sort_values("date")
        r = g["adjusted_close"].pct_change()
        bad = g.loc[r.abs() > ANOMALY_RET]
        for _, row in bad.iterrows():
            anomalies.append({"ticker": ticker, "date": str(row["date"].date()),
                              "ret": float(r.loc[row.name])})

    # --- 4. stray non-session dates -------------------------------------------
    from data.calendar import spurious_dates
    sp = spurious_dates(wide_all)
    spurious = [{"date": str(pd.Timestamp(d).date()), "coverage": float(c)}
                for d, c in zip(sp["date"], sp["coverage"])]

    # --- 4b. MACRO freshness ---------------------------------------------------
    # The regime observation matrix depends on macro_daily.usd_egp
    # (models/regime_features.build_observation_matrix), so stale macro silently
    # TRUNCATES the regime signal -- and the regime signal sets live exposure.
    # Measured failure: prices ran to 2026-07-30 while usd_egp stopped at
    # 2026-07-07, so the live book was sized on a 23-day-old regime read that was
    # merely forward-filled. The price-staleness check above cannot see this,
    # because prices were perfectly fresh. Checked separately for that reason.
    macro_age = None
    with get_connection() as conn:
        mrow = conn.execute("SELECT MAX(date) FROM macro_daily WHERE usd_egp IS NOT NULL").fetchone()
    if mrow and mrow[0]:
        macro_last = pd.Timestamp(mrow[0])
        macro_age = int((pd.Timestamp.now().normalize() - macro_last.normalize()).days)
        if macro_age > MAX_MACRO_STALE_DAYS:
            fatal.append(f"macro (usd_egp) is {macro_age} days stale (last {macro_last.date()}, "
                         f"limit {MAX_MACRO_STALE_DAYS}) — the regime signal is truncated")
    else:
        fatal.append("macro_daily has no usd_egp — the regime signal cannot be built")

    # --- 4c. REGIME SIGNAL freshness ------------------------------------------
    # Fresh prices and fresh macro are not enough: the exposure the ticket
    # prints comes from regime_ensemble, which nothing refreshed after
    # 2026-07-30. For two months every ticket said "exposure 100%" from a
    # signal two months old, including through an Aug 2026 BEAR call. Each
    # artifact the exposure depends on must reach the last session within
    # MAX_SIGNAL_LAG_SESSIONS, or no ticket is produced.
    signal_lag = {}
    with get_connection() as conn:
        for table in ("regime_history", "regime_ensemble"):
            try:
                r = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()
            except Exception:  # noqa: BLE001 -- table missing
                r = None
            last = pd.Timestamp(r[0]) if r and r[0] else None
            if last is None:
                fatal.append(f"{table} is empty — the exposure signal does not exist")
                continue
            lag = int((sessions > last).sum()) if len(sessions) else 0
            signal_lag[table] = {"last": str(last.date()), "sessions_behind": lag}
            if lag > MAX_SIGNAL_LAG_SESSIONS:
                fatal.append(f"{table} ends {last.date()}, {lag} sessions behind the last session "
                             f"{pd.Timestamp(last_session).date()} (limit {MAX_SIGNAL_LAG_SESSIONS}) "
                             "— exposure would come from a stale regime read")

    # --- 5. tradable universe size --------------------------------------------
    from data.universe import load_tradable
    uni = load_tradable()
    n_tradable = 0
    if not uni.empty:
        # read the universe on the last genuine session, never on a phantom date
        uni_valid = uni[uni["date"] <= pd.Timestamp(last_session)]
        uni_valid = uni_valid if not uni_valid.empty else uni
        latest_uni = uni_valid[uni_valid["date"] == uni_valid["date"].max()]
        n_tradable = int(latest_uni["is_tradable"].sum())
        if n_tradable < MIN_UNIVERSE:
            fatal.append(f"tradable universe is {n_tradable} names (< {MIN_UNIVERSE}) — refusing to trade")
    else:
        fatal.append("tradable_universe table is empty/missing — run data.universe.build_tradable_universe")

    if stale_tickers:
        problems.append(f"{len(stale_tickers)} ticker(s) stale > {MAX_STALE_DAYS}d")
    if anomalies:
        problems.append(f"{len(anomalies)} price anomal(ies) |ret|>{ANOMALY_RET:.0%}")

    report = {"ok": not fatal, "as_of": str(as_of.date()), "db_age_days": int(db_age),
              "macro_age_days": macro_age, "signal_lag": signal_lag,
              "n_tickers": int(px["ticker"].nunique()), "n_tradable": n_tradable,
              "stale_tickers": stale_tickers, "anomalies": anomalies,
              "spurious_dates": spurious, "problems": problems, "fatal": fatal}
    lvl = logger.error if fatal else (logger.warning if problems else logger.info)
    lvl("data_quality_report: ok=%s db_age=%dd tradable=%d stale=%d anomalies=%d fatal=%s",
        report["ok"], db_age, n_tradable, len(stale_tickers), len(anomalies), fatal or "none")
    return report


class DataQualityError(RuntimeError):
    """Raised when the data is unfit to produce a trade list."""


def assert_data_fit_to_trade(as_of: pd.Timestamp = None) -> dict:
    """Hard gate: raise DataQualityError if the data is stale/degenerate.
    Call this BEFORE emitting any trade list. This is the difference between
    'the system quietly traded garbage' and 'the system refused and told you'.
    """
    rep = data_quality_report(as_of)
    if not rep["ok"]:
        raise DataQualityError("; ".join(rep["fatal"]))
    return rep


def format_report(rep: dict) -> str:
    """Human-readable daily data-quality report."""
    lines = [f"DATA QUALITY — as of {rep['as_of']} (db age {rep.get('db_age_days','?')}d)",
             f"status: {'OK' if rep['ok'] else 'FAILED'}",
             f"tickers: {rep['n_tickers']} | tradable: {rep['n_tradable']}"
             + (f" | macro age {rep['macro_age_days']}d" if rep.get("macro_age_days") is not None else "")]
    if rep["fatal"]:
        lines.append("FATAL: " + "; ".join(rep["fatal"]))
    if rep["stale_tickers"]:
        lines.append("stale: " + ", ".join(f"{s['ticker']}({s['days_stale']}d)" for s in rep["stale_tickers"][:10]))
    if rep["anomalies"]:
        lines.append(f"anomalies: {len(rep['anomalies'])} (e.g. " +
                     ", ".join(f"{a['ticker']} {a['date']} {a['ret']:+.0%}" for a in rep["anomalies"][:3]) + ")")
    if rep["spurious_dates"]:
        lines.append(f"non-session dates filtered: {len(rep['spurious_dates'])}")
    return "\n".join(lines)
