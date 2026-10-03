"""
POINT-IN-TIME universe construction  (Tier 1, item 1 — selection bias).

THE BIAS THIS EXISTS TO REMOVE. `config.EGX_SEED_TICKERS` is a hand-maintained
list of 47 names. Twenty of them were added on 2026-07-28, explicitly ranked by
liquidity AS OF 2026, and then used across a backtest starting in 2015. That is
selection bias in its purest form: the 2015 portfolio was chosen partly because
of what those companies went on to become. Every figure in the three prior
audits — CAGR, Sharpe, the factor regression, the cost sweep, even the
survivorship correction — inherits it.

`data/universe.py:build_tradable_universe` already applies a point-in-time
LIQUIDITY filter, which is good, but it can only filter names that were fetched
in the first place. The bias lives one level up, in which tickers exist at all.

WHAT THIS DOES INSTEAD. At each date, rank EVERY ticker with price history by
its trailing median value traded, and take the top N. Membership is recomputed
per date from data available at that date only, so a name enters when it becomes
liquid and leaves when it stops — including by delisting. No 2026 information
reaches a 2015 decision.

WHAT IT STILL DOES NOT FIX, and this must not be glossed:

  * It can only rank tickers Yahoo still resolves. Companies that delisted and
    whose symbols were recycled or dropped remain invisible. Four were recovered
    by hand (ESRS, UASG, SUCE, OCIC); the true number is unknown. So this
    removes SELECTION bias substantially and SURVIVORSHIP bias only partially.
  * Yahoo's EGX coverage itself may be non-random — names that were always thin
    may be absent for reasons correlated with their returns.

So the honest claim is: this makes the universe a function of the past instead
of a function of the answer. It does not make it complete.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.pit_universe")

# Breadth of the point-in-time universe. Chosen to match the order of magnitude
# of the old hand-picked list (47) so the comparison isolates SELECTION rather
# than confounding it with a change in breadth. Not tuned against any result.
PIT_TOP_N = 50
PIT_LIQUIDITY_WINDOW = 60      # trailing sessions for the liquidity median
PIT_MIN_HISTORY = 180          # need at least this much history to be rankable
PIT_MIN_VALUE_EGP = 50_000.0   # floor: below this a name is not really tradable


def _load_value_traded() -> pd.DataFrame:
    with get_connection() as conn:
        vt = pd.read_sql("SELECT ticker, date, value_traded_egp FROM daily_ohlcv",
                         conn, parse_dates=["date"])
    if vt.empty:
        return pd.DataFrame()
    return vt.pivot(index="date", columns="ticker",
                    values="value_traded_egp").sort_index()


def build_pit_universe(top_n: int = PIT_TOP_N, window: int = PIT_LIQUIDITY_WINDOW,
                       min_history: int = PIT_MIN_HISTORY,
                       min_value: float = PIT_MIN_VALUE_EGP) -> pd.DataFrame:
    """Long (date, ticker, in_universe, rank, median_value) frame.

    The rank at date D uses only data STRICTLY UP TO D. `.rolling(window)` on a
    date-sorted frame is inclusive of D, which is correct: value traded on D is
    known at D's close, and the book is set from D's close for D+1 onward.
    """
    vt = _load_value_traded()
    if vt.empty:
        return pd.DataFrame()
    from data.calendar import valid_trading_days
    vt = vt.loc[vt.index.isin(valid_trading_days(vt))]

    med = vt.rolling(window, min_periods=max(window // 2, 5)).median()
    # a name must also have enough history to be rankable at all
    seen = vt.notna().cumsum()
    eligible = (med >= min_value) & (seen >= min_history)
    ranked = med.where(eligible).rank(axis=1, ascending=False, method="first")
    inuni = ranked <= top_n

    rows = []
    for d in vt.index:
        r = ranked.loc[d]
        keep = r[inuni.loc[d].fillna(False)]
        for t, rk in keep.items():
            rows.append({"date": d, "ticker": t, "in_universe": 1,
                         "rank": float(rk), "median_value": float(med.loc[d, t])})
    out = pd.DataFrame(rows)
    logger.info("build_pit_universe: %d (date,ticker) memberships across %d dates",
                len(out), out["date"].nunique() if not out.empty else 0)
    return out


def store_pit_universe(df: pd.DataFrame = None) -> int:
    """Materialise into `pit_universe`. Kept in its OWN table, deliberately not
    overwriting `tradable_universe`: the two must be comparable side by side,
    and silently replacing the live universe definition mid-audit would make
    every before/after number incomparable."""
    df = build_pit_universe() if df is None else df
    if df.empty:
        return 0
    with get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS pit_universe ("
                     "date TEXT, ticker TEXT, in_universe INTEGER, rank REAL, "
                     "median_value REAL, PRIMARY KEY (date, ticker))")
        conn.execute("DELETE FROM pit_universe")
        conn.executemany(
            "INSERT OR REPLACE INTO pit_universe (date, ticker, in_universe, rank, "
            "median_value) VALUES (?,?,?,?,?)",
            [(d.strftime("%Y-%m-%d"), t, int(u), float(r), float(m))
             for d, t, u, r, m in df[["date", "ticker", "in_universe", "rank",
                                      "median_value"]].itertuples(index=False)])
    return len(df)


def load_pit_tradable() -> pd.DataFrame:
    """In the shape BacktestEngine expects from data.universe.load_tradable:
    columns (date, ticker, is_tradable)."""
    with get_connection() as conn:
        try:
            df = pd.read_sql("SELECT date, ticker, in_universe AS is_tradable "
                             "FROM pit_universe", conn, parse_dates=["date"])
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return df


def turnover_report(df: pd.DataFrame = None) -> dict:
    """How much does membership actually churn? A point-in-time universe that
    never changes is the hand-picked list wearing a disguise, so this is the
    check that the fix did something."""
    df = build_pit_universe() if df is None else df
    if df.empty:
        return {"ok": False}
    by_date = df.groupby("date")["ticker"].apply(set)
    dates = sorted(by_date.index)
    entries, exits = 0, 0
    for a, b in zip(dates, dates[1:]):
        entries += len(by_date[b] - by_date[a])
        exits += len(by_date[a] - by_date[b])
    ever = set(df["ticker"])
    always = set.intersection(*by_date.values) if len(by_date) else set()
    return {"ok": True, "n_dates": len(dates), "names_ever": len(ever),
            "names_always": len(always), "entries": entries, "exits": exits,
            "mean_size": float(by_date.apply(len).mean())}


def format_comparison(pit: dict, seed_n: int) -> str:
    lines = ["POINT-IN-TIME UNIVERSE vs the hand-picked seed list",
             f"  hand-picked seed list        : {seed_n} names, fixed, chosen in 2026",
             f"  point-in-time universe       : {pit['mean_size']:.0f} names on an average date",
             f"  distinct names ever included : {pit['names_ever']}",
             f"  names present on EVERY date  : {pit['names_always']}",
             f"  membership changes           : {pit['entries']} entries / {pit['exits']} exits",
             f"  dates covered                : {pit['n_dates']}"]
    if pit["names_always"] >= pit["mean_size"] * 0.9:
        lines.append("  WARNING: membership barely changes — this is close to a fixed list")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    df = build_pit_universe()
    n = store_pit_universe(df)
    print(f"stored {n} memberships\n")
    print(format_comparison(turnover_report(df), len(config.EGX_SEED_TICKERS)))
