"""
Regime-conditioned composite alpha score -- the point where Layer 3
(factor exposures + Fama-MacBeth premia) combines with Layer 2 (the
current market regime) to produce one score per stock per day. This is
the actual Layer 1 + 2 + 3 wiring: factor_exposures and regime_history
are both read straight from the database Layer 1 populates and Layer
2/3 write into.

alpha_score_i,t = sum_k w_k,t * standardized_beta_ik,t
w_k,t = fama_macbeth_premium_k * regime_adjustment[regime_t][k]

REGIME ADJUSTMENTS -- ADAPTED FROM THE ORIGINAL 9-FACTOR SPEC, since only
5 of the 9 factors exist here (see models/factor_returns module docstring
for why SMB/HML/QMJ/CBE-surprise aren't available):
  BULL:     upweight MOM -- spec's other instruction ("downweight VALUE")
            is dropped since VALUE/HML isn't a factor we have.
  BEAR:     upweight LIQUIDITY -- spec's "upweight QUALITY" dropped,
            QUALITY/QMJ isn't a factor we have.
  HIGH_VOL: upweight EGP_SENSITIVITY -- spec's "upweight QUALITY" dropped,
            same reason.
  RECOVERY: the spec calls for upweighting VALUE and SIZE -- NEITHER is
            available (no book-value or market-cap data anywhere in
            Layer 1; see config.EGX_SEED_TICKERS docstring). Rather than
            invent a substitute rationale for which of the 5 available
            factors should stand in for two completely different missing
            factors, RECOVERY gets neutral (1.0x) weights on all 5. This
            is a real gap, not a considered design choice -- flagged here
            so the neutral weighting isn't mistaken for "recovery doesn't
            need factor tilting".
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection
from models.factor_model import BETA_COLUMNS

logger = logging.getLogger("esis_quant.alpha_score")

STANDARDIZE_CLIP = 2.5
ALPHA_CLIP = 3.0
MIN_TICKERS_FOR_ZSCORE = 5

REGIME_ADJUSTMENTS = {
    "BULL": {"beta_market": 1.0, "beta_momentum": 1.5, "beta_liquidity": 1.0,
             "beta_egp_sensitivity": 1.0, "beta_oil_sensitivity": 1.0},
    "BEAR": {"beta_market": 1.0, "beta_momentum": 1.0, "beta_liquidity": 1.5,
             "beta_egp_sensitivity": 1.0, "beta_oil_sensitivity": 1.0},
    "HIGH_VOL": {"beta_market": 1.0, "beta_momentum": 1.0, "beta_liquidity": 1.0,
                 "beta_egp_sensitivity": 1.5, "beta_oil_sensitivity": 1.0},
    "RECOVERY": {"beta_market": 1.0, "beta_momentum": 1.0, "beta_liquidity": 1.0,
                 "beta_egp_sensitivity": 1.0, "beta_oil_sensitivity": 1.0},
}


def _load_exposures_and_regimes(tickers=None):
    with get_connection() as conn:
        exposures = pd.read_sql(
            "SELECT ticker, date, " + ", ".join(BETA_COLUMNS) + " FROM factor_exposures",
            conn, parse_dates=["date"],
        )
        regimes = pd.read_sql("SELECT date, regime FROM regime_history", conn, parse_dates=["date"])
    if tickers:
        exposures = exposures[exposures["ticker"].isin(tickers)]
    return exposures, regimes


def _cross_sectional_zscore(df: pd.DataFrame, columns: list, clip: float) -> pd.DataFrame:
    """Per-date z-score across tickers for each column, clipped at
    +/-clip. A date with fewer than MIN_TICKERS_FOR_ZSCORE valid values
    for a column gets NaN for that column that date -- a z-score against
    a handful of names isn't meaningful. Vectorized via groupby.transform
    (not groupby.apply -- that has a pandas edge-case bug with a single
    group that raised a spurious KeyError on the "date" column).
    """
    out = df.copy()
    for col in columns:
        grp = df.groupby("date")[col]
        count = grp.transform("count")
        mean = grp.transform("mean")
        std = grp.transform("std")

        with np.errstate(divide="ignore", invalid="ignore"):
            z = (df[col] - mean) / std
        z = z.mask(std == 0, 0.0)  # constant cross-section that date -> z=0, not inf/NaN from a 0/0 divide
        z = z.clip(-clip, clip)
        z = z.mask(count < MIN_TICKERS_FOR_ZSCORE, np.nan)  # too few names that date -> not meaningful
        out[col] = z
    return out


def compute_alpha_scores(premia: pd.Series, tickers=None) -> pd.DataFrame:
    """premia: Series indexed by factor name (e.g. from
    fama_macbeth.run_fama_macbeth()['premia']['premium']), giving each
    factor's average Fama-MacBeth risk premium.
    Returns long-format DataFrame: ticker, date, regime, alpha_score.
    """
    exposures, regimes = _load_exposures_and_regimes(tickers)
    if exposures.empty:
        raise ValueError("No factor_exposures found -- run models.factor_model first")
    if regimes.empty:
        raise ValueError("No regime_history found -- run models.regime_pipeline first")

    standardized = _cross_sectional_zscore(exposures, BETA_COLUMNS, STANDARDIZE_CLIP)
    merged = standardized.merge(regimes, on="date", how="inner")

    alpha = pd.Series(0.0, index=merged.index)
    covered = pd.Series(False, index=merged.index)
    for regime, adjustments in REGIME_ADJUSTMENTS.items():
        mask = merged["regime"] == regime
        for col in BETA_COLUMNS:
            weight = premia.get(col, 0.0) * adjustments[col]
            alpha.loc[mask] += weight * merged.loc[mask, col].fillna(0.0)
        covered |= mask

    merged["alpha_score"] = alpha
    merged.loc[~covered, "alpha_score"] = np.nan  # unrecognized regime label -- don't silently score it as 0

    final = _cross_sectional_zscore(merged, ["alpha_score"], ALPHA_CLIP)

    result = final[["ticker", "date", "regime", "alpha_score"]].dropna(subset=["alpha_score"])
    logger.info("compute_alpha_scores: %d (ticker, date) rows scored", len(result))
    return result


def store_alpha_scores(scores_df: pd.DataFrame):
    """Upserts into the `signals` table (alpha_score, regime columns --
    sentiment_score/causal_signal/tfm_probability/etc. belong to Layers
    4-6, not built yet, left NULL). Uses executemany for performance on
    large DataFrames."""
    rows = []
    for _, row in scores_df.iterrows():
        date_str = (row["date"].strftime("%Y-%m-%d")
                    if hasattr(row["date"], "strftime")
                    else str(row["date"]))
        rows.append((
            row["ticker"],
            date_str,
            row["regime"],
            float(row["alpha_score"]),
        ))
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO signals (ticker, date, regime, alpha_score)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(ticker, date) DO UPDATE SET
                 regime=excluded.regime,
                 alpha_score=excluded.alpha_score""",
            rows,
        )
    logger.info("store_alpha_scores: stored %d rows", len(scores_df))
