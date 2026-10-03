"""
Trading-calendar integrity (audit fix: CAL-01).

THE BUG THIS FIXES: daily_ohlcv is built by unioning each ticker's yfinance
history. yfinance occasionally emits a stray print for 1-3 tickers on a day
the EGX was actually CLOSED (e.g. 2024-05-01 Labour Day, 2026-07-02). Pivoting
that into a (date x ticker) panel manufactures a "trading day" on which ~89% of
the universe is NaN. Two consequences, both real and both silently corrupting:

  1. A return computed across such a date spans a holiday gap -- garbage.
  2. Any rule requiring complete data over a lookback window sees that one bad
     date and excludes almost every ticker for the WHOLE window behind it. On
     this database that collapsed the investable universe to 2 names out of 26
     for ~5 months of 2024 (measured), and the backtest happily reported the
     result of a 2-stock book as if it were a diversified strategy.

THE RULE: a date is a real trading day only if at least `min_coverage` of the
tickers that have ANY price history around that date actually printed. EGX
trades as a market -- either most of the universe prints or the exchange was
shut. A 50% floor is far below any plausible real session and far above the
~11% these stray dates show, so it separates them cleanly.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.calendar")

MIN_COVERAGE = 0.50

# THE SECOND PHANTOM-SESSION MODE (found 2026-08-18). Coverage alone is not
# enough. On 2026-08-16 yfinance returned a row for 198 of 198 tickers -- 100%
# coverage, so the coverage rule accepted it as a session -- but EVERY row had
# volume 0 and 189 of them were bit-for-bit copies of the 2026-08-13 close. It
# was a carried-forward day, not a traded one. The consequence is worse than the
# stray-print mode this module was written for: a fabricated session becomes the
# newest date, which RESETS the whole-DB staleness clock in data.quality, so a
# five-day-old signal presents itself as two days old and the gate that exists to
# catch staleness reports everything healthy.
#
# A real EGX session has 80-85% of listed names printing non-zero volume (median
# 0.855 over 3,205 dates here); the phantom days sit at 0.000-0.010. A 20% floor
# separates them with a wide margin in both directions.
MIN_TRADED_FRACTION = 0.20

# A ticker whose last print is within this many sessions of the panel's end is
# treated as STILL ACTIVE (listed through the end), not delisted. See the
# right-edge bug in coverage_by_date below.
DELIST_GRACE_SESSIONS = 10


def coverage_by_date(price_panel: pd.DataFrame = None,
                     delist_grace: int = DELIST_GRACE_SESSIONS) -> pd.Series:
    """Fraction of LISTED tickers that printed a price on each date. A ticker
    counts as listed on date d only between its own first and last print, so a
    stock that IPO'd in 2020 doesn't drag down 2016's coverage.

    RIGHT-EDGE BUG (fixed here): using each ticker's LAST print as its delisting
    date is correct for a stock that genuinely stopped trading mid-sample, but
    catastrophically wrong at the newest end of the panel. If yfinance emits a
    phantom print for 2 tickers on a day the EGX was shut (e.g. 2026-07-23,
    Revolution Day), every OTHER ticker's last print is the day before -- so all
    of them get marked "delisted" on the phantom date, leaving coverage =
    2/2 = 100%. The stray date then passes the filter, and because it is the
    LATEST date it becomes the as-of date for live trading: the tradable
    universe collapses to those 2 zero-volume names and the daily job dies at
    the quality gate ("tradable universe is 0 names").

    The fix: a ticker still printing within `delist_grace` sessions of the end
    of the panel is considered active through the panel's end, so a phantom
    final date is measured against the FULL universe (2/27 = 7%) and rejected.
    """
    if price_panel is None:
        with get_connection() as conn:
            px = pd.read_sql("SELECT ticker, date, adjusted_close FROM daily_ohlcv",
                             conn, parse_dates=["date"])
        price_panel = px.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()

    if price_panel.empty:
        return pd.Series(dtype=float)

    idx = price_panel.index
    # the date beyond which a final print means "still active", not "delisted"
    grace_cutoff = idx[max(0, len(idx) - 1 - delist_grace)]

    listed = pd.DataFrame(False, index=idx, columns=price_panel.columns)
    for c in price_panel.columns:
        s = price_panel[c]
        valid = s.notna()
        if not valid.any():
            continue
        first, last = valid.idxmax(), valid[::-1].idxmax()
        if last >= grace_cutoff:
            last = idx[-1]      # still active -> listed through the end of the panel
        listed.loc[first:last, c] = True

    n_listed = listed.sum(axis=1).replace(0, np.nan)
    n_traded = (price_panel.notna() & listed).sum(axis=1)
    return (n_traded / n_listed).fillna(0.0)


def traded_fraction_by_date(tickers=None) -> pd.Series:
    """Fraction of rows on each date carrying NON-ZERO volume.

    Read straight from daily_ohlcv rather than from a passed-in panel, because
    the panel callers hand around holds adjusted_close only -- and a carried-
    forward bar is indistinguishable from a traded one on price alone. Volume is
    the field that gives it away, so it has to come from the table.

    A row counts as traded if EITHER volume or value_traded_egp is positive.
    Both must be zero to call a bar untraded. On live data the two agree exactly
    (checked across every session since 2026-08-05), but they are populated
    independently and a source that fills only one of them would otherwise have
    its entire history condemned as phantom.

    `tickers` MUST be the same set the coverage was computed over. Scoring the
    traded fraction on the whole table while coverage was scored on a caller's
    panel compares two different universes, and the mismatch is not theoretical:
    it made a synthetic-panel test fail because a date its panel considered a
    normal session (2026-07-23) is a real phantom in the production database.
    Passing the panel's own columns keeps both tests on one universe, and a
    panel of names absent from the table simply yields no rows -- no judgement
    rather than a wrong one.
    """
    q = ("SELECT date, COUNT(*) AS n, "
         "SUM(CASE WHEN volume > 0 OR value_traded_egp > 0 THEN 1 ELSE 0 END) AS traded "
         "FROM daily_ohlcv")
    params = ()
    if tickers is not None:
        tickers = list(tickers)
        if not tickers:
            return pd.Series(dtype=float)
        q += " WHERE ticker IN (%s)" % ",".join("?" * len(tickers))
        params = tuple(tickers)
    q += " GROUP BY date"
    with get_connection() as conn:
        df = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    if df.empty:
        return pd.Series(dtype=float)
    return (df["traded"] / df["n"]).set_axis(pd.DatetimeIndex(df["date"])).sort_index()


def valid_trading_days(price_panel: pd.DataFrame = None, min_coverage: float = MIN_COVERAGE,
                       min_traded: float = MIN_TRADED_FRACTION) -> pd.DatetimeIndex:
    """Dates that look like genuine EGX sessions.

    TWO independent tests, because there are two ways a non-session sneaks in:
      - coverage: too FEW names printed (stray prints on a closed day)
      - traded:   enough names printed but NONE of them actually traded
                  (a carried-forward day -- see MIN_TRADED_FRACTION)
    A date must pass both.
    """
    cov = coverage_by_date(price_panel)
    good = cov.index[cov >= min_coverage]
    dropped = len(cov) - len(good)
    if dropped:
        logger.info("valid_trading_days: dropped %d/%d dates with <%.0f%% universe coverage "
                    "(stray non-session prints)", dropped, len(cov), min_coverage * 100)

    traded = traded_fraction_by_date(
        tickers=list(price_panel.columns) if price_panel is not None else None)
    if len(traded):
        # Only judge dates we actually have volume for. A date absent from the
        # volume series is left alone rather than dropped: silently discarding
        # real sessions is a worse failure than keeping a suspect one, and
        # callers passing synthetic panels have no volume to be judged on.
        judged = traded.reindex(good)
        untraded = judged[judged.notna() & (judged < min_traded)]
        if len(untraded):
            logger.warning("valid_trading_days: dropped %d date(s) with <%.0f%% of rows carrying "
                           "volume -- carried-forward, not traded: %s", len(untraded),
                           min_traded * 100,
                           ", ".join(f"{d.date()} ({f:.1%})" for d, f in untraded.tail(5).items()))
            good = good.difference(untraded.index)
    return pd.DatetimeIndex(good)


def spurious_dates(price_panel: pd.DataFrame = None, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """The dates being rejected, with their coverage -- for the data-quality report."""
    cov = coverage_by_date(price_panel)
    bad = cov[cov < min_coverage]
    return pd.DataFrame({"date": bad.index, "coverage": bad.values}).sort_values("date")
