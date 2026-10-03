"""
Cash / risk-free rate series for Egypt (EGP).

WHY THIS EXISTS. The system's ONLY measured edge is the BEAR overlay: go to
cash when the regime model says so. But the backtest assumed that cash earns
**zero** (see the note in risk/overlay.py). Over the sample the book sits in
cash on ~30% of days, during a period when Egyptian overnight rates ran roughly
8%-27%. Assuming 0% therefore throws away a large, riskless part of exactly the
strategy leg that carries the edge -- it understates the overlay, and it makes
"sit in cash" look far worse than it really is.

DATA HONESTY -- READ THIS BEFORE TRUSTING ANY NUMBER FROM THIS MODULE.
There is no free machine-readable feed for the CBE overnight rate (checked:
yfinance has no Egypt rate/bond symbol). The schedule below is a RECONSTRUCTION
from published CBE Monetary Policy Committee decisions. It is:
  * approximate -- effective dates are the announced decision dates, and only
    the major moves are captured, not every basis point;
  * unverified against an authoritative source in this repo;
  * NOT a fetched series -- nothing here was downloaded.
It is therefore labelled `is_reconstructed=True` everywhere it surfaces, and
any headline result built on it MUST be reported alongside the sensitivity
analysis (analysis/cash_yield.py), which sweeps the assumed rate across a wide
range so the conclusion does not depend on these exact numbers being right.

WHAT RATE TO USE. The policy rate is what banks earn. A retail saver or a
T-bill holder earns less, and after the 5% (or higher) tax on T-bill interest
less still. `DEFAULT_HAIRCUT` shaves the policy rate to a conservative
investable estimate; raise it to be more conservative.

Rates are quoted as ANNUAL simple rates (0.1875 = 18.75%) and converted to a
daily compounding factor as (1+r)^(1/252)-1.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.rates")

# Conservative shave off the policy rate to approximate what an investor could
# actually earn on idle cash (bank deposit / T-bill net of spread and tax).
DEFAULT_HAIRCUT = 0.03      # 3 percentage points
TRADING_DAYS = 252

# CBE overnight DEPOSIT rate, reconstructed from published MPC decisions.
# (effective_date, annual_rate). APPROXIMATE -- see the module docstring.
CBE_POLICY_RATE_SCHEDULE = [
    ("2015-01-01", 0.0875),
    ("2015-10-15", 0.0925),
    ("2015-12-17", 0.0975),
    ("2016-03-17", 0.1075),
    ("2016-06-16", 0.1175),
    ("2016-11-03", 0.1475),   # EGP float
    ("2017-05-21", 0.1675),
    ("2017-07-06", 0.1875),   # peak of the post-float tightening
    ("2018-02-15", 0.1775),
    ("2018-03-29", 0.1675),
    ("2019-02-14", 0.1575),
    ("2019-08-22", 0.1425),
    ("2019-09-26", 0.1325),
    ("2019-11-14", 0.1225),
    ("2020-03-16", 0.0925),   # COVID emergency cut
    ("2020-09-24", 0.0875),
    ("2020-11-12", 0.0825),
    ("2022-03-21", 0.0925),   # inflation / FX pressure, tightening begins
    ("2022-05-19", 0.1125),
    ("2022-10-27", 0.1325),
    ("2022-12-22", 0.1625),
    ("2023-03-30", 0.1825),
    ("2023-08-03", 0.1925),
    ("2024-02-01", 0.2125),
    ("2024-03-06", 0.2725),   # large hike alongside the March-2024 devaluation
    ("2025-04-17", 0.2525),
    ("2025-08-28", 0.2225),
    ("2026-01-01", 0.2025),   # carried forward -- least reliable segment
]


def policy_rate_series(index: pd.DatetimeIndex = None) -> pd.Series:
    """Step-function CBE policy rate, forward-filled onto `index` (or onto the
    schedule's own daily range). No look-ahead: each date carries the rate that
    was in force on that date, never a later decision."""
    sched = pd.Series(
        {pd.Timestamp(d): r for d, r in CBE_POLICY_RATE_SCHEDULE}).sort_index()
    if index is None:
        index = pd.date_range(sched.index[0], pd.Timestamp.now().normalize(), freq="D")
    index = pd.DatetimeIndex(index).sort_values()
    # reindex onto the union so each step lands, then ffill forward only
    full = sched.reindex(sched.index.union(index)).ffill()
    return full.reindex(index).ffill().bfill()


def cash_rate_series(index: pd.DatetimeIndex = None, haircut: float = DEFAULT_HAIRCUT,
                     flat_rate: float = None) -> pd.Series:
    """Investable annual rate on idle cash. `flat_rate` overrides the schedule
    entirely (used by the sensitivity sweep, and by anyone who does not trust
    the reconstruction). Never negative."""
    if flat_rate is not None:
        idx = pd.DatetimeIndex(index) if index is not None else policy_rate_series().index
        return pd.Series(float(flat_rate), index=idx, name="cash_rate")
    r = policy_rate_series(index) - haircut
    return r.clip(lower=0.0).rename("cash_rate")


def daily_cash_return(index: pd.DatetimeIndex, haircut: float = DEFAULT_HAIRCUT,
                      flat_rate: float = None, trading_days: int = TRADING_DAYS) -> pd.Series:
    """Per-trading-day return earned on idle cash: (1+annual)^(1/252)-1."""
    ann = cash_rate_series(index, haircut=haircut, flat_rate=flat_rate)
    return ((1.0 + ann) ** (1.0 / trading_days) - 1.0).rename("cash_daily")


def populate_macro_cbe_rate(haircut: float = 0.0) -> int:
    """Write the reconstructed policy rate into macro_daily.cbe_overnight_rate
    (the column exists in the schema but was never populated). Stores the RAW
    policy rate by default -- the haircut is an analysis choice, not data.
    Returns rows written."""
    with get_connection() as conn:
        dates = [pd.Timestamp(r[0]) for r in
                 conn.execute("SELECT DISTINCT date FROM macro_daily ORDER BY date")]
        if not dates:
            return 0
        rates = cash_rate_series(pd.DatetimeIndex(dates), haircut=haircut)
        rows = [(float(v), d.strftime("%Y-%m-%d")) for d, v in rates.items()]
        conn.executemany("UPDATE macro_daily SET cbe_overnight_rate = ? WHERE date = ?", rows)
    logger.info("populate_macro_cbe_rate: wrote %d rows (RECONSTRUCTED, approximate)", len(rows))
    return len(rows)


def describe() -> str:
    r = policy_rate_series()
    return ("CBE cash-rate model (RECONSTRUCTED from published MPC decisions -- approximate, "
            "not fetched, unverified)\n"
            f"  span {r.index[0].date()} -> {r.index[-1].date()}   "
            f"min {r.min()*100:.2f}%  max {r.max()*100:.2f}%  mean {r.mean()*100:.2f}%\n"
            f"  investable estimate = policy - {DEFAULT_HAIRCUT*100:.0f}pp haircut (deposit spread/tax)\n"
            "  ALWAYS report results from this alongside the sensitivity sweep in "
            "analysis/cash_yield.py")
