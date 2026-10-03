"""
Per-stock rolling factor exposures (betas against each factor_returns
series), persisted into the factor_exposures table.

REDUCED FACTOR SET: only beta_market, beta_momentum, beta_liquidity,
beta_egp_sensitivity, beta_oil_sensitivity are computed -- see
models/factor_returns module docstring for why beta_size/beta_value/
beta_quality have no real data to compute from (no market cap, book
value, or ROE/ROA/profitability data anywhere in Layer 1). Those three
columns are left NULL, not fabricated.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection
from data.linalg import rolling_ols_beta
from data.microstructure import MicrostructureAnalyzer
from models.factor_returns import build_factor_returns
from models.regime_features import _load_price_panel

logger = logging.getLogger("esis_quant.factor_model")

ROLLING_WINDOW_DAYS = 60

FACTOR_TO_COLUMN = {
    "MKT": "beta_market",
    "MOM": "beta_momentum",
    "LIQ": "beta_liquidity",
    "EGP": "beta_egp_sensitivity",
    "OIL": "beta_oil_sensitivity",
}
BETA_COLUMNS = list(FACTOR_TO_COLUMN.values())
UNAVAILABLE_BETA_COLUMNS = ["beta_size", "beta_value", "beta_quality"]


def compute_factor_exposures(tickers=None, window: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
    """Long-format DataFrame: ticker, date, beta_market, beta_momentum,
    beta_liquidity, beta_egp_sensitivity, beta_oil_sensitivity,
    idiosyncratic_vol -- one row per (ticker, date) with at least one
    valid rolling estimate.
    """
    panel = _load_price_panel(tickers)
    factors = build_factor_returns(tickers)

    per_ticker_frames = []
    for ticker, df in panel.items():
        stock_return = df["close"].pct_change().rename("stock")
        aligned = pd.concat([stock_return, factors], axis=1, join="inner")
        if aligned["stock"].notna().sum() < window:
            continue

        betas = {
            col: rolling_ols_beta(aligned["stock"], aligned[factor_name], window=window)
            for factor_name, col in FACTOR_TO_COLUMN.items()
        }
        variance_components = MicrostructureAnalyzer.realized_variance_components(
            aligned["stock"], aligned["MKT"], window=window
        )
        betas["idiosyncratic_vol"] = np.sqrt(variance_components["idiosyncratic_var"] * 252)

        ticker_df = pd.DataFrame(betas)
        ticker_df = ticker_df.dropna(how="all")
        if ticker_df.empty:
            continue
        ticker_df["ticker"] = ticker
        per_ticker_frames.append(ticker_df.reset_index().rename(columns={"index": "date"}))

    if not per_ticker_frames:
        logger.warning("compute_factor_exposures: no ticker produced any valid rolling estimate")
        return pd.DataFrame(columns=["ticker", "date"] + BETA_COLUMNS + ["idiosyncratic_vol"])

    result = pd.concat(per_ticker_frames, ignore_index=True)
    logger.info("compute_factor_exposures: %d (ticker, date) rows across %d tickers",
                len(result), result["ticker"].nunique())
    return result


def _none_if_nan(value):
    return None if pd.isna(value) else float(value)


def store_factor_exposures(exposures_df: pd.DataFrame):
    rows = []
    for _, row in exposures_df.iterrows():
        date_str = (row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"]))
        rows.append((
            row["ticker"],
            date_str,
            _none_if_nan(row.get("beta_market")),
            _none_if_nan(row.get("beta_momentum")),
            _none_if_nan(row.get("beta_liquidity")),
            _none_if_nan(row.get("beta_egp_sensitivity")),
            _none_if_nan(row.get("beta_oil_sensitivity")),
            _none_if_nan(row.get("idiosyncratic_vol")),
        ))
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO factor_exposures
               (ticker, date, beta_market, beta_size, beta_value,
                beta_momentum, beta_quality, beta_liquidity,
                beta_egp_sensitivity, beta_oil_sensitivity,
                idiosyncratic_vol)
               VALUES (?, ?, ?, NULL, NULL, ?, NULL, ?, ?, ?, ?)
               ON CONFLICT(ticker, date) DO UPDATE SET
                 beta_market=excluded.beta_market,
                 beta_momentum=excluded.beta_momentum,
                 beta_liquidity=excluded.beta_liquidity,
                 beta_egp_sensitivity=excluded.beta_egp_sensitivity,
                 beta_oil_sensitivity=excluded.beta_oil_sensitivity,
                 idiosyncratic_vol=excluded.idiosyncratic_vol""",
            rows,
        )
    logger.info("store_factor_exposures: stored %d rows", len(exposures_df))
