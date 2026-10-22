"""
Read-time split / bonus adjustment from CORROBORATED corporate actions.

WHY READ-TIME, NOT IN PLACE. The daily fetch re-downloads the last 30 days from
Yahoo and replaces those rows, so an adjustment written into daily_ohlcv is
silently undone. And when Yahoo later back-adjusts a split itself, an in-place
correction would apply it twice. So corrections are stored as EVENTS
(corporate_actions, action_type 'SPLIT_CORROBORATED') and applied to the price
panel as it is read -- and only while the stored series still shows the jump.

THE SELF-DISABLING RULE. For an event (ticker, ex_date, ratio R), the prices
before ex_date are divided by R only if the last price before ex_date divided
by the first price on/after it is within 12% of R. If the source has already
adjusted (no jump left), the factor is skipped. This makes the correction safe
to keep forever: it can fix a missing adjustment but can never create one.

Only events with two agreeing sources reach this table as SPLIT_CORROBORATED
(ops.reference_librarian). Raw `close` is never touched.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.ca_factors")

MATCH_TOL = 0.12


def corroborated_events() -> pd.DataFrame:
    try:
        with get_connection() as conn:
            return pd.read_sql("SELECT ticker, date, ratio FROM corporate_actions "
                               "WHERE action_type = 'SPLIT_CORROBORATED' AND ratio > 0",
                               conn, parse_dates=["date"])
    except Exception:  # noqa: BLE001 -- table missing in a fresh DB
        return pd.DataFrame(columns=["ticker", "date", "ratio"])


def adjust_series(s: pd.Series, events: pd.DataFrame) -> tuple:
    """Return (adjusted series, list of applied event dates) for one ticker."""
    s = s.copy()
    applied = []
    for e in events.sort_values("date", ascending=False).itertuples():
        before = s[s.index < e.date].dropna()
        after = s[s.index >= e.date].dropna()
        if before.empty or after.empty:
            continue
        jump = float(before.iloc[-1]) / float(after.iloc[0])
        if abs(jump / e.ratio - 1.0) <= MATCH_TOL:
            s.loc[s.index < e.date] = s.loc[s.index < e.date] / e.ratio
            applied.append(e.date)
    return s, applied


SPIKE = 0.40     # a bad tick sits >40% away from BOTH neighbours


def bad_ticks() -> pd.DataFrame:
    """Single-day prints confirmed wrong by hand (corporate_actions 'BAD_TICK')."""
    try:
        with get_connection() as conn:
            return pd.read_sql("SELECT ticker, date FROM corporate_actions WHERE action_type = 'BAD_TICK'",
                               conn, parse_dates=["date"])
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["ticker", "date"])


def drop_bad_ticks(s: pd.Series, dates) -> tuple:
    """Replace each listed print with the previous close -- but only while it is
    still a spike (self-disabling, like the split rule: once the vendor fixes the
    row there is nothing to replace)."""
    s = s.copy()
    fixed = []
    for d in dates:
        if d not in s.index:
            continue
        i = s.index.get_loc(d)
        prev, nxt = s.iloc[:i].dropna(), s.iloc[i + 1:].dropna()
        if prev.empty or nxt.empty:
            continue
        v, p, n = float(s.iloc[i]), float(prev.iloc[-1]), float(nxt.iloc[0])
        if abs(v / p - 1) > SPIKE and abs(v / n - 1) > SPIKE:
            s.iloc[i] = p
            fixed.append(d)
    return s, fixed


def adjust_panel(wide: pd.DataFrame, events: pd.DataFrame = None, ticks: pd.DataFrame = None) -> pd.DataFrame:
    """Wide (date x ticker) PRICE panel: confirmed bad ticks removed, then
    corroborated splits / rights issues back-adjusted."""
    events = corroborated_events() if events is None else events
    ticks = bad_ticks() if ticks is None else ticks
    if wide.empty or (events.empty and ticks.empty):
        return wide
    out = wide.copy()
    for t, g in ticks.groupby("ticker"):
        if t in out.columns:
            out[t], _ = drop_bad_ticks(out[t], list(g["date"]))
    for t, ev in events.groupby("ticker"):
        if t in out.columns:
            out[t], _ = adjust_series(out[t], ev)
    return out
