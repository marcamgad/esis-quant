"""
Composite signal (Layer 7.1): a single cross-sectional score per (ticker,
date) blending the layers that produce a real, dense signal -- factor alpha
(Layer 3) and the causal signal (Layer 5) -- with regime-probability-dependent
weights from the point-in-time filtered regimes (Layer 2, Phase A).

IMPORTANT / HONEST DESIGN NOTE: the backtest (Phase A + the B6 gate) measured
NO stock-selection alpha in this system -- signal-ranked portfolios do not
beat equal-weight risk-adjusted, and min-variance (which ignores expected
returns entirely) was the best single optimizer. Consequently the portfolio
layer runs mean-variance with mu = 0 (i.e. minimum-variance), and this
composite is NOT fed in as an expected-return tilt -- doing so would let the
optimizer "error-maximize" on a signal with no measured edge. The composite is
kept for monitoring / research and for the day a genuinely predictive signal
exists; its weighting is regime-aware so it is ready to be used the moment
selection alpha is demonstrated, not before.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.portfolio.signals")

# regime -> (alpha_weight, causal_weight). In calmer regimes lean on factor
# alpha; in stressed regimes lean on the causal (momentum-derived) signal.
REGIME_BLEND = {
    "BULL": (0.6, 0.4), "RECOVERY": (0.5, 0.5),
    "HIGH_VOL": (0.4, 0.6), "BEAR": (0.4, 0.6), "UNKNOWN": (0.5, 0.5),
}


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)


def compute_composite_signal(tickers=None) -> pd.DataFrame:
    """Cross-sectional composite score per (ticker, date): regime-weighted
    blend of z-scored alpha_score and causal_signal. Returns columns
    [ticker, date, composite_score]. Empty if the signals table is unpopulated.
    """
    with get_connection() as conn:
        sig = pd.read_sql("SELECT ticker, date, alpha_score, causal_signal FROM signals "
                          "WHERE alpha_score IS NOT NULL", conn, parse_dates=["date"])
        reg = pd.read_sql("SELECT date, regime FROM regime_history", conn, parse_dates=["date"])
    if sig.empty:
        return pd.DataFrame(columns=["ticker", "date", "composite_score"])
    if tickers is not None:
        sig = sig[sig["ticker"].isin(tickers)]

    sig["causal_signal"] = sig["causal_signal"].fillna(0.0)
    sig["z_alpha"] = _zscore_by_date(sig, "alpha_score")
    sig["z_causal"] = _zscore_by_date(sig, "causal_signal")
    sig = sig.merge(reg, on="date", how="left")
    sig["regime"] = sig["regime"].fillna("UNKNOWN")

    wa = sig["regime"].map(lambda r: REGIME_BLEND.get(r, REGIME_BLEND["UNKNOWN"])[0]).to_numpy()
    wc = sig["regime"].map(lambda r: REGIME_BLEND.get(r, REGIME_BLEND["UNKNOWN"])[1]).to_numpy()
    sig["composite_score"] = wa * sig["z_alpha"].to_numpy() + wc * sig["z_causal"].to_numpy()
    return sig[["ticker", "date", "composite_score"]]


def store_composite_signal(df: pd.DataFrame) -> int:
    """Upsert composite_score into the signals table."""
    rows = [(r.ticker, r.date.strftime("%Y-%m-%d"), float(r.composite_score)) for r in df.itertuples()]
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO signals (ticker, date, composite_score) VALUES (?, ?, ?) "
            "ON CONFLICT(ticker, date) DO UPDATE SET composite_score=excluded.composite_score", rows)
    return len(rows)
