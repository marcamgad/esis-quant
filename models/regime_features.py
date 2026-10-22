"""
Builds the observation matrix Layer 2's HMM trains on, sourced entirely
from Layer 1's SQLite database (daily_ohlcv, macro_daily).

REDUCED FROM THE ORIGINAL 7-FEATURE SPEC TO 6, DOCUMENTED HERE RATHER THAN
FAKED:
  o1: market-proxy 20-day return
  o2: market-proxy 20-day realized volatility (annualized)
  o3: market breadth (% of universe above its 50-day moving average)
  o4: USD/EGP 20-day change
  o5: average Amihud illiquidity across the universe
  o6: market-proxy 60-day realized volatility (annualized) -- VIX-proxy
      fallback per the spec's own "else realized vol" clause

Dropped: "yield spread (Egypt 10Y - CBE rate)". This was actually
investigated, not just assumed unavailable:
  - cbe.org.eg: confirmed WAF-blocked (see data/collectors.py
    scrape_cbe_decisions), returns a challenge page, not data.
  - FRED (fred.stlouisfed.org/graph/fredgraph.csv, no API key needed):
    tried ~12 plausible IMF-IFS/OECD-style series IDs for an Egypt
    discount/interbank rate; all 404 except one World-Bank-mirrored
    annual deposit-rate series (DDDI06EGA156NWDB) that stops at 2021 --
    four years stale and annual-frequency, useless for a daily HMM
    feature.
  - No Egypt 10Y bond yield series found on FRED at all.
  There is a real, fresher proxy for short-term Egyptian rates --
  Treasury-bill discount rates announced in scraped Arabic news (see
  data/collectors.py extract_tbill_discount_rate) -- but those are sparse
  and event-driven (only a handful of dates have a mention at all), not a
  continuous multi-year daily series, so they can't backfill 2015-2026
  for HMM training. Fabricating either series would be worse than
  omitting them; this is a genuine, confirmed data-availability gap.

The "EGX30" referenced throughout is a market-proxy index built from
config.EGX_SEED_TICKERS, weighted by each stock's own prior-day EGP value
traded (not equal-weighted, and not a real free-float market-cap
replication either -- Layer 1 doesn't have reliable market-cap data, see
config.EGX_SEED_TICKERS docstring). This is NOT the real EGX30. Confirmed
via the raw Yahoo Finance chart API directly (bypassing yfinance's
wrapper) that Yahoo's backend has exactly one data point for ^CASE30 --
today's snapshot, zero history -- so this isn't a yfinance library
limitation. stooq.com, historically a free CSV source for world indices,
now serves a JS challenge page instead of CSV. No working free historical
source for the real index was found after checking both.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection
from data.microstructure import MicrostructureAnalyzer

logger = logging.getLogger("esis_quant.regime_features")

MIN_TICKER_COVERAGE = 10  # below this many tickers reporting on a date, breadth/Amihud avg -> NaN for that date


def _load_price_panel(tickers=None) -> dict:
    tickers = tickers or [t["ticker"] for t in config.EGX_SEED_TICKERS]
    panel = {}
    with get_connection() as conn:
        for ticker in tickers:
            # adjusted_close (split/dividend back-adjusted) aliased to `close`
            # so every return/momentum/volatility/breadth computation
            # downstream is on the adjusted series -- unadjusted close has
            # artificial jumps on ex-div/split dates that would register as
            # fake returns and volatility spikes. volume/value_traded stay raw
            # (they must not be back-adjusted).
            df = pd.read_sql(
                "SELECT date, adjusted_close AS close, volume, value_traded_egp FROM daily_ohlcv "
                "WHERE ticker = ? ORDER BY date",
                conn, params=(ticker,), parse_dates=["date"], index_col="date",
            )
            if not df.empty:
                panel[ticker] = df
    if not panel:
        raise ValueError("No price data in daily_ohlcv -- run Layer 1 collectors first")
    # corroborated, still-unadjusted splits (data/ca_factors.py) -- a fake -50%
    # day would otherwise read as a crash in breadth and volatility
    from data.ca_factors import corroborated_events, adjust_series
    ev = corroborated_events()
    for t, e in ev.groupby("ticker"):
        if t in panel:
            panel[t]["close"], _ = adjust_series(panel[t]["close"], e)
    return panel


def load_macro(columns=("usd_egp",)) -> pd.DataFrame:
    """Public loader for macro_daily columns, reused by models/factor_returns.py
    (which also needs brent_oil_usd) so both layers query the DB the same way.
    """
    col_list = ", ".join(columns)
    with get_connection() as conn:
        df = pd.read_sql(
            f"SELECT date, {col_list} FROM macro_daily ORDER BY date",
            conn, parse_dates=["date"], index_col="date",
        )
    return df


def load_returns_panel(tickers=None) -> pd.DataFrame:
    """Public loader for per-ticker daily returns (T x tickers), used both
    to build the market proxy and for cross-sectional stats (e.g. average
    pairwise correlation per regime) that need per-ticker data directly.
    """
    panel = _load_price_panel(tickers)
    return pd.DataFrame({t: df["close"].pct_change() for t, df in panel.items()})


def build_market_proxy(panel: dict) -> pd.DataFrame:
    """Market-proxy index, weighted by each stock's own PRIOR-DAY EGP value
    traded (not equal-weighted). This is closer to how a real index
    behaves than equal-weighting -- a real EGX30 is float-market-cap
    weighted, and Layer 1 doesn't have reliable market-cap/float data (see
    config.EGX_SEED_TICKERS), so value-traded is the best available proxy
    for "how much this name should matter", not a true replication.
    Weights use t-1 traded value specifically so a stock's return on day t
    isn't weighted by information only available at the close of day t
    (that would leak same-day volume/price co-movement into the weight).

    Falls back to equal-weighting on any date where liquidity weighting
    isn't usable (no valid t-1 weights, or fewer than
    MIN_TICKER_COVERAGE names have one) rather than dropping the date.
    Returns a DataFrame with columns: return, level, n_tickers.
    """
    returns = pd.DataFrame({t: df["close"].pct_change() for t, df in panel.items()})
    value_traded = pd.DataFrame({t: df["value_traded_egp"] for t, df in panel.items()})
    lagged_weights = value_traded.shift(1)

    coverage = returns.notna().sum(axis=1)

    valid = returns.notna() & lagged_weights.notna() & (lagged_weights > 0)
    masked_weights = lagged_weights.where(valid)
    weight_sum = masked_weights.sum(axis=1)
    normalized_weights = masked_weights.div(weight_sum, axis=0)
    weighted_return = (returns.where(valid) * normalized_weights).sum(axis=1, skipna=True, min_count=1)

    equal_weight_return = returns.mean(axis=1, skipna=True)
    n_valid_weighted = valid.sum(axis=1)
    use_fallback = weight_sum.isna() | (weight_sum <= 0) | (n_valid_weighted < MIN_TICKER_COVERAGE)
    combined_return = weighted_return.where(~use_fallback, equal_weight_return)

    combined_return = combined_return.where(coverage >= MIN_TICKER_COVERAGE).dropna()
    level = 100.0 * (1.0 + combined_return).cumprod()
    return pd.DataFrame({"return": combined_return, "level": level, "n_tickers": coverage.loc[combined_return.index]})


def build_breadth(panel: dict, ma_window: int = 50) -> pd.Series:
    """Fraction of the universe trading above its own `ma_window`-day
    moving average, per date. NaN where fewer than MIN_TICKER_COVERAGE
    tickers have a valid signal that day.
    """
    above_ma = {}
    for ticker, df in panel.items():
        ma = df["close"].rolling(ma_window, min_periods=ma_window).mean()
        # .astype(float) before .where() -- otherwise the bool/NaN mix leaves
        # the column as dtype=object, which silently poisons the downstream
        # .mean() result (also object dtype) and breaks matplotlib later.
        above_ma[ticker] = (df["close"] > ma).astype(float).where(ma.notna())

    frame = pd.DataFrame(above_ma)
    coverage = frame.notna().sum(axis=1)
    breadth = frame.mean(axis=1, skipna=True)
    return breadth.where(coverage >= MIN_TICKER_COVERAGE).dropna()


def build_avg_amihud(panel: dict, window: int = 20) -> pd.Series:
    """Cross-sectional average of MicrostructureAnalyzer.amihud_illiquidity
    across the universe, per date, requiring MIN_TICKER_COVERAGE names.
    """
    illiq = {}
    for ticker, df in panel.items():
        returns = df["close"].pct_change()
        illiq[ticker] = MicrostructureAnalyzer.amihud_illiquidity(returns, df["value_traded_egp"], window=window)

    frame = pd.DataFrame(illiq)
    coverage = frame.notna().sum(axis=1)
    avg_illiq = frame.mean(axis=1, skipna=True)

    n_panel = len(panel)
    n_contributing = frame.notna().any(axis=0).sum()
    if n_contributing < n_panel:
        logger.info(
            "build_avg_amihud: %d/%d tickers had any valid Amihud data "
            "(remaining %d were all-NaN, likely zero volume throughout)",
            n_contributing, n_panel, n_panel - n_contributing,
        )

    return avg_illiq.where(coverage >= MIN_TICKER_COVERAGE).dropna()


def build_observation_matrix(tickers=None) -> pd.DataFrame:
    """Assemble o1..o6 (see module docstring) plus supporting raw series
    (market_proxy_level, breadth, usd_egp, realized_vol_20d/60d) needed
    for the regime chart, aligned on their common date index.

    Rows with any NaN are dropped -- mostly the first ~60 trading days of
    whatever history is loaded (60-day realized-vol warm-up dominates the
    other windows). With a multi-year backfill this is a small fraction
    of the sample; with only the ~90-day default Layer 1 pull it can wipe
    out nearly everything, which is why Layer 2's orchestration backfills
    a longer history before calling this.
    """
    panel = _load_price_panel(tickers)
    macro = load_macro()

    proxy = build_market_proxy(panel)
    breadth = build_breadth(panel, ma_window=50)
    avg_illiq = build_avg_amihud(panel, window=20)

    realized_vol_20d = proxy["return"].rolling(20).std() * np.sqrt(252)
    realized_vol_60d = proxy["return"].rolling(60).std() * np.sqrt(252)
    ret_20d = proxy["level"].pct_change(20)
    usd_egp_change_20d = macro["usd_egp"].pct_change(20)

    combined = pd.DataFrame(
        {
            "o1_market_return_20d": ret_20d,
            "o2_realized_vol_20d": realized_vol_20d,
            "o3_breadth": breadth,
            "o4_usd_egp_change_20d": usd_egp_change_20d,
            "o5_avg_amihud": avg_illiq,
            "o6_realized_vol_60d": realized_vol_60d,
            "market_proxy_level": proxy["level"],
            "market_proxy_daily_return": proxy["return"],
            "usd_egp": macro["usd_egp"],
        }
    )

    before = len(combined)
    combined = combined.dropna()
    logger.info(
        "build_observation_matrix: %d/%d dates retained after warm-up/coverage dropna (%s to %s)",
        len(combined), before,
        combined.index.min().date() if len(combined) else None,
        combined.index.max().date() if len(combined) else None,
    )
    return combined


OBSERVATION_COLUMNS = [
    "o1_market_return_20d",
    "o2_realized_vol_20d",
    "o3_breadth",
    "o4_usd_egp_change_20d",
    "o5_avg_amihud",
    "o6_realized_vol_60d",
]
