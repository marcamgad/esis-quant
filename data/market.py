"""
The market, as the EGX30 index -- not a homemade average.

WHY. Until 2026-10-04 there was no EGX30 history (Yahoo has none, and a storage
bug blanked what little there was), so every "market" in the system was the
equal-weight mean of ~200 names. That is a different animal: it weights a
small illiquid stock the same as CIB (EGX30's largest member at ~25-45%), and
nobody can buy it. Mubasher's daily EGX30 history (data/collectors.
fetch_egx30_history) now fills macro_daily.egx30_level from 1998.

CAVEATS, stated once:
  * PRICE index: dividends are not included (EGX30 yields ~3-5%), so it
    understates what an index fund earns by roughly that much a year;
  * Mubasher's file skips some days; returns across a gap are compounded into
    the next stored day (`returns(index=...)` forward-fills the LEVEL, never the
    return), so nothing is lost, only re-dated.
"""

import numpy as np
import pandas as pd

from data.database import get_connection

DIVIDEND_YIELD_NOTE = 0.04      # rough EGX30 yield, for display only; never added to returns


def level() -> pd.Series:
    with get_connection() as conn:
        d = pd.read_sql("SELECT date, egx30_level FROM macro_daily WHERE egx30_level IS NOT NULL "
                        "ORDER BY date", conn, parse_dates=["date"])
    return d.set_index("date")["egx30_level"].astype(float).rename("egx30")


def returns(index: pd.DatetimeIndex = None) -> pd.Series:
    """Daily EGX30 returns, optionally on another date index (e.g. the price panel's)."""
    lv = level()
    if index is not None:
        index = pd.DatetimeIndex(index)
        lv = lv.reindex(lv.index.union(index)).ffill().reindex(index)
    r = lv.pct_change()
    return r.replace([np.inf, -np.inf], np.nan).rename("egx30")


def window_return(start, end) -> float:
    """EGX30 return over (start, end], from the last stored level on/before each date."""
    lv = level()
    a, b = lv[lv.index <= pd.Timestamp(start)], lv[lv.index <= pd.Timestamp(end)]
    if a.empty or b.empty:
        return np.nan
    return float(b.iloc[-1] / a.iloc[-1] - 1)
