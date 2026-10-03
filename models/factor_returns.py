"""
Daily factor return series, built entirely from real Layer 1 data.

REDUCED FROM THE ORIGINAL 9-FACTOR SPEC TO 5, DOCUMENTED HERE RATHER THAN
FAKED. Implementable with what Layer 1 actually collects:
  MKT: market-proxy daily return (the same liquidity-weighted proxy built
       for Layer 2 -- see models/regime_features.build_market_proxy)
  MOM: WML, 12-1 month momentum long-short, monthly rebalance
  LIQ: IML, Amihud-illiquidity long-short, monthly rebalance
  EGP: raw daily USD/EGP return (not a sorted portfolio -- the spec
       describes EGP sensitivity as a regression on the raw FX return,
       not a long-short factor construction)
  OIL: raw daily Brent return (same rationale as EGP)

Dropped, because Layer 1 has no fundamental-data collector at all (no
earnings, book value, revenue, COGS, assets, ROE/ROA anywhere in the
schema -- not a missing field, a missing entire data source):
  SMB (needs market cap -- no shares-outstanding/float data, see
       config.EGX_SEED_TICKERS docstring)
  HML (needs Price-to-Book -- no book value data)
  QMJ (needs ROE/ROA/gross profitability -- no financial-statement data)
  CBE-surprise/EVENT (needs CBE decision dates -- confirmed WAF-blocked,
       see data/collectors.py scrape_cbe_decisions; the T-bill discount
       rate scan in data/collectors.py is a real but sparse/event-driven
       proxy, not a substitute for continuous CBE event data)
Building any of these properly would mean standing up a whole new
fundamentals-scraping pipeline (e.g. EGX disclosures or mubasher.info
financial statements) -- a substantial separate effort, not something to
fake with placeholder numbers in a factor model that's meant to inform
real trading decisions.
"""

import logging

import numpy as np
import pandas as pd

from models.regime_features import _load_price_panel, load_macro, build_market_proxy
from data.microstructure import MicrostructureAnalyzer

logger = logging.getLogger("esis_quant.factor_returns")

MOMENTUM_LOOKBACK_DAYS = 252  # ~12 months of trading days
MOMENTUM_SKIP_DAYS = 21  # ~1 month, skipped to avoid short-term reversal (Jegadeesh-Titman)
ILLIQUIDITY_WINDOW_DAYS = 20
LONG_SHORT_QUANTILE = 0.30  # top/bottom 30% per the spec
MIN_NAMES_PER_SIDE = 3  # below this many names on a side, skip that month's rebalance


def _monthly_rebalance_dates(index: pd.DatetimeIndex) -> list:
    """First observed trading date of each calendar month in `index`."""
    seen = set()
    dates = []
    for date in index:
        key = (date.year, date.month)
        if key not in seen:
            seen.add(key)
            dates.append(date)
    return dates


def _build_long_short_factor(score_panel: pd.DataFrame, returns_panel: pd.DataFrame,
                              quantile: float = LONG_SHORT_QUANTILE,
                              min_names: int = MIN_NAMES_PER_SIDE) -> pd.Series:
    """Monthly-rebalanced long-short portfolio return: at the first trading
    day of each month, rank tickers by that day's score (already only
    using information available up to that day -- no lookahead as long as
    the caller's score_panel is itself lookahead-free, e.g. trailing
    momentum/illiquidity), go long the top `quantile` and short the bottom
    `quantile`, hold equal-weight until the next rebalance date.

    If a rebalance date doesn't have enough valid scores on both sides,
    the previous month's winners/losers carry over rather than producing
    a gap (a real portfolio wouldn't unwind to cash just because one
    month's ranking was under-populated).
    """
    rebalance_dates = _monthly_rebalance_dates(score_panel.index)
    factor_return = pd.Series(index=returns_panel.index, dtype=float)
    winners, losers = [], []

    for i, rdate in enumerate(rebalance_dates):
        scores_today = score_panel.loc[rdate].dropna()
        n_side = max(1, int(len(scores_today) * quantile))
        if len(scores_today) >= min_names * 2:
            ranked = scores_today.sort_values(ascending=False)
            winners = ranked.index[:n_side].tolist()
            losers = ranked.index[-n_side:].tolist()

        period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if period_end is not None:
            mask = (returns_panel.index > rdate) & (returns_panel.index < period_end)
        else:
            mask = returns_panel.index > rdate
        period_dates = returns_panel.index[mask]

        if winners and losers:
            winners_ret = returns_panel.loc[period_dates, winners].mean(axis=1, skipna=True)
            losers_ret = returns_panel.loc[period_dates, losers].mean(axis=1, skipna=True)
            factor_return.loc[period_dates] = (winners_ret - losers_ret).to_numpy()

    return factor_return.dropna()


def build_momentum_scores(panel: dict) -> pd.DataFrame:
    """mom_score_t = close_{t-skip} / close_{t-skip-lookback} - 1, per
    ticker. Uses only data available as of t (no lookahead).
    """
    scores = {}
    for ticker, df in panel.items():
        close = df["close"]
        scores[ticker] = close.shift(MOMENTUM_SKIP_DAYS) / close.shift(MOMENTUM_SKIP_DAYS + MOMENTUM_LOOKBACK_DAYS) - 1
    return pd.DataFrame(scores)


def build_illiquidity_scores(panel: dict, window: int = ILLIQUIDITY_WINDOW_DAYS) -> pd.DataFrame:
    """Rolling Amihud illiquidity per ticker, reusing the Layer 1
    microstructure implementation (already tested against synthetic
    ground truth in tests/test_microstructure.py).
    """
    scores = {}
    for ticker, df in panel.items():
        returns = df["close"].pct_change()
        scores[ticker] = MicrostructureAnalyzer.amihud_illiquidity(returns, df["value_traded_egp"], window=window)
    return pd.DataFrame(scores)


def build_factor_returns(tickers=None) -> pd.DataFrame:
    """Assemble the 5 daily factor return series on their common date
    index. Returns a DataFrame with columns MKT, MOM, LIQ, EGP, OIL.
    """
    panel = _load_price_panel(tickers)
    returns_panel = pd.DataFrame({t: df["close"].pct_change() for t, df in panel.items()})

    mkt = build_market_proxy(panel)["return"].rename("MKT")

    mom_scores = build_momentum_scores(panel)
    mom = _build_long_short_factor(mom_scores, returns_panel).rename("MOM")

    illiq_scores = build_illiquidity_scores(panel)
    liq = _build_long_short_factor(illiq_scores, returns_panel).rename("LIQ")

    macro = load_macro(columns=("usd_egp", "brent_oil_usd"))
    egp = macro["usd_egp"].pct_change().rename("EGP")
    oil = macro["brent_oil_usd"].pct_change().rename("OIL")

    factors = pd.concat([mkt, mom, liq, egp, oil], axis=1, join="inner").dropna()
    logger.info("build_factor_returns: %d days, %s to %s", len(factors),
                factors.index.min().date() if len(factors) else None,
                factors.index.max().date() if len(factors) else None)
    return factors


FACTOR_NAMES = ["MKT", "MOM", "LIQ", "EGP", "OIL"]
