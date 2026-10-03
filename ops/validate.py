"""
Startup validation (Part 2.5): fail LOUDLY on a misconfigured or degraded
install rather than proceeding with quietly-wrong behavior.

The audit's recurring theme is that this system's failures are silent: an
empty table, a missing checkpoint, a stale DB, a two-name universe -- none of
them raised, all of them produced confident-looking output. Every check here
turns one of those into an exception at startup.
"""

import logging
import os

import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.ops.validate")


class ConfigError(RuntimeError):
    """Configuration/environment is unfit to run."""


def validate_startup(require_trading: bool = True) -> dict:
    """Check config + database invariants. Raises ConfigError listing EVERY
    problem found (not just the first), so one run surfaces all of them.
    `require_trading=False` relaxes the checks that only matter for producing
    a live trade list (useful for research/backtest-only sessions).
    """
    problems, warnings_ = [], []

    # --- config sanity ---------------------------------------------------
    if not os.path.exists(config.DB_PATH):
        problems.append(f"DB_PATH does not exist: {config.DB_PATH}")
    if len(config.EGX_SEED_TICKERS) < 10:
        problems.append(f"EGX_SEED_TICKERS has {len(config.EGX_SEED_TICKERS)} tickers (< 10): "
                        "universe too small to diversify")
    from backtest.fees import book_pct_per_side
    cost = book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)
    if not (0 < cost < 0.05):
        problems.append(f"implausible transaction cost per turnover: {cost}")
    for name in ("TFT_ZSCORE_CLIP", "TFT_MAX_STALE_FRAC", "CAUSAL_MIN_OBS"):
        if not hasattr(config, name):
            problems.append(f"config.{name} is missing")

    if not os.path.exists(config.DB_PATH):
        raise ConfigError("; ".join(problems))

    # --- database invariants ---------------------------------------------
    with get_connection() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"daily_ohlcv", "regime_history", "stocks"}
        missing = required - tables
        if missing:
            problems.append(f"missing required tables: {sorted(missing)}")

        if "daily_ohlcv" in tables:
            n = conn.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0]
            if n == 0:
                problems.append("daily_ohlcv is empty -- run the Layer 1 collectors")
        if "regime_history" in tables:
            n = conn.execute("SELECT COUNT(*) FROM regime_history").fetchone()[0]
            if n == 0 and require_trading:
                problems.append("regime_history is empty -- the BEAR overlay (the system's only "
                                "measured edge) cannot run")
        if "tradable_universe" not in tables and require_trading:
            problems.append("tradable_universe missing -- run data.universe.build_tradable_universe "
                            "(without it the tradability filter silently passes everything through)")

    # --- research-only assets (warn, never block) -------------------------
    if not os.path.exists(config.SENTIMENT_MODEL_CHECKPOINT_PATH):
        warnings_.append("sentiment checkpoint absent (Layer 4 is research-only; not on the live path)")
    if not os.path.exists(config.TFT_MODEL_CHECKPOINT_PATH):
        warnings_.append("TFT checkpoint absent (Layer 6 is research-only; not on the live path)")

    for w in warnings_:
        logger.warning("validate_startup: %s", w)
    if problems:
        raise ConfigError("startup validation FAILED: " + "; ".join(problems))
    logger.info("validate_startup: OK (%d warnings)", len(warnings_))
    return {"ok": True, "warnings": warnings_}
