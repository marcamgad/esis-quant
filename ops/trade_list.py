"""
Live trade-list generation + audit trail (Part 2.2 / 2.3).

The audit found the system had NO live decision path: run_full_system produces
a BACKTEST, not a book you could trade tomorrow. This module closes that gap
and does so behind a hard data-quality gate, because the whole point of the
audit was that this system's failure mode is confident output from bad data.

Order of operations (the gate comes FIRST, deliberately):
  1. assert_data_fit_to_trade()  -- raises if the DB is stale, the universe is
     degenerate, or the tradable table is missing. NO trade list is produced
     from data we don't trust.
  2. Build the eligible universe exactly as the backtest does (same coverage
     rule, same tradability table) so the live book and the backtested book
     are the same construction -- not two subtly different code paths.
  3. Min-variance weights (Ledoit-Wolf shrunk covariance), mu = 0: Phase A/B
     measured NO selection alpha, so no return forecast is used.
  4. Apply the regime exposure: 0 in BEAR (the only measured edge), else 1.
  5. Log every line to trade_list_log with the regime/exposure it was based on.
"""

import logging
import uuid

import numpy as np
import pandas as pd

import config
from data.database import get_connection
from data.quality import assert_data_fit_to_trade
from data.universe import load_tradable
from backtest.engine import BacktestEngine, load_price_returns
from portfolio.covariance import ledoit_wolf_shrinkage
from portfolio.mv_optimizer import mean_variance_weights

logger = logging.getLogger("esis_quant.ops.trade_list")


def generate_trade_list(lookback: int = 180, max_weight: float = 0.15,
                        min_names: int = 5, skip_quality_gate: bool = False) -> dict:
    """Produce today's target book. Returns
    {run_id, as_of, regime, exposure, weights (Series), cash, n_positions,
     quality}. Raises DataQualityError if the data is unfit (unless
    skip_quality_gate, which exists ONLY for tests -- never use it live).
    """
    # Tier-1 #1: the live book is min-variance core + regime overlay ONLY. The
    # signal-selection path is proven to hurt (~-18% selection), so it must
    # never reach a live trade list. This function contains no call into the
    # selection path; the flag makes that guarantee explicit and fails loudly if
    # someone flips it off thinking selection is safe to trade.
    if not getattr(config, "LIVE_MIN_VAR_ONLY", True):
        raise RuntimeError("generate_trade_list: LIVE_MIN_VAR_ONLY is False. Signal selection "
                           "measured NEGATIVE (-18% Brinson selection) and must not trade live.")

    quality = None if skip_quality_gate else assert_data_fit_to_trade()

    rets = load_price_returns()          # calendar-filtered (CAL-01)
    trad = load_tradable()
    if rets.empty:
        raise RuntimeError("generate_trade_list: no price returns available")

    as_of = rets.index[-1]
    window = rets.loc[:as_of].tail(lookback)

    eng = BacktestEngine(min_names=min_names, max_weight=max_weight)
    trad_map = None
    if not trad.empty:
        trad_map = trad[trad["is_tradable"] == 1].groupby("date")["ticker"].apply(set)
    cols = eng._eligible_names(window, as_of, trad_map)
    if len(cols) < min_names:
        raise RuntimeError(f"generate_trade_list: only {len(cols)} eligible names on {as_of.date()} "
                           f"(need >= {min_names}) -- refusing to emit a trade list")

    lw = ledoit_wolf_shrinkage(window[cols].fillna(0.0).values)
    w = mean_variance_weights(np.zeros(len(cols)), lw["sigma"], max_weight=max_weight)["weights"]
    weights = pd.Series(w, index=cols)
    weights = weights[weights > 1e-6]
    weights = weights / weights.sum()

    # Sector limit. The optimiser only sees the correlation matrix, and
    # same-sector names co-move more in a sector shock than a quiet-period
    # correlation says (see portfolio/sector_cap.py for the flour-mill case).
    # Removes concentration only; adds no return view.
    from portfolio.sector_cap import apply_sector_cap, load_sectors
    weights, sector_info = apply_sector_cap(weights, load_sectors(), max_weight=max_weight)

    # regime exposure: the ONLY measured edge. Regime as of the latest stored
    # date <= as_of (never a future label).
    with get_connection() as conn:
        reg = pd.read_sql("SELECT date, regime FROM regime_history WHERE date <= ? ORDER BY date",
                          conn, params=(as_of.strftime("%Y-%m-%d"),), parse_dates=["date"])
    regime = str(reg["regime"].iloc[-1]) if not reg.empty else "UNKNOWN"

    # Exposure comes from the ENSEMBLE regime signal, not a single HMM fit.
    # Measured: one fit's BEAR label moves with the EM random seed alone (478-790
    # BEAR days, ~10pp of CAGR), so trading a single fit means trading estimation
    # noise. The ensemble averages independent refits and gives a CONTINUOUS
    # exposure (1 - p_bear), so disagreement between fits produces partial
    # de-risking instead of an arbitrary on/off. Falls back to the old hard label
    # if the ensemble is missing, and records which path was used.
    from models.regime_ensemble import exposure_for
    exp_info = exposure_for(as_of, fallback_regime=regime)
    exposure = exp_info["exposure"]

    # HARD KILL-SWITCH, now on the LIVE path (it was backtest-only until 2026-08).
    # This is deliberately INDEPENDENT of the regime model: the regime overlay is
    # the thing that might be wrong, so the backstop cannot be derived from it.
    # It reads the forward paper record's realized losses and, if a hard limit is
    # breached, forces exposure to zero regardless of what the regime believes.
    # Applied multiplicatively so it can only ever REDUCE exposure, never raise it.
    ks_info = {"tripped": False, "multiplier": 1.0, "reason": "not evaluated"}
    try:
        from risk.kill_switch import live_state
        ks_info = live_state()
        if ks_info["tripped"]:
            exposure *= ks_info["multiplier"]
            logger.error("KILL-SWITCH TRIPPED: forcing exposure to 0 -- %s", ks_info["reason"])
            from ops.health import alert
            alert("ESIS-QUANT — KILL-SWITCH TRIPPED",
                  f"Realized losses breached a hard limit:\n\n{ks_info['reason']}\n\n"
                  "Today's target book is FLAT regardless of the regime signal. "
                  "This is the model-independent backstop, not a regime call.")
    except Exception as exc:  # noqa: BLE001 -- a backstop failure must be loud, not fatal
        logger.warning("kill-switch could not be evaluated: %s", exc)

    weights = weights * exposure

    run_id = uuid.uuid4().hex[:12]
    result = {"run_id": run_id, "as_of": as_of, "regime": regime, "exposure": exposure,
              "weights": weights, "cash": float(1.0 - weights.sum()),
              "n_positions": int((weights > 0).sum()), "quality": quality,
              "p_bear": exp_info["p_bear"], "regime_source": exp_info["source"],
              "kill_switch_tripped": ks_info["tripped"],
              "kill_switch_reason": ks_info["reason"]}
    _log_trade_list(result)
    logger.info("generate_trade_list: run=%s as_of=%s regime=%s exposure=%.0f%% positions=%d "
                "cash=%.0f%% source=%s p_bear=%s",
                run_id, as_of.date(), regime, exposure * 100, result["n_positions"],
                result["cash"] * 100, exp_info["source"], exp_info["p_bear"])
    return result


def _log_trade_list(result: dict) -> int:
    """Append the trade list to the audit trail (Part 2.3)."""
    now = pd.Timestamp.now()
    rows = [(result["run_id"], now.isoformat(timespec="seconds"),
             pd.Timestamp(result["as_of"]).strftime("%Y-%m-%d"), result["regime"],
             float(result["exposure"]), t, float(v))
            for t, v in result["weights"].items()]
    if not rows:  # fully in cash: still log the decision, with a CASH marker
        rows = [(result["run_id"], now.isoformat(timespec="seconds"),
                 pd.Timestamp(result["as_of"]).strftime("%Y-%m-%d"), result["regime"],
                 float(result["exposure"]), "__CASH__", 1.0)]
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO trade_list_log "
                         "(run_id, generated_at, as_of_date, regime, exposure, ticker, weight) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def latest_trade_list() -> pd.DataFrame:
    """The most recently logged trade list (audit trail read-back)."""
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT * FROM trade_list_log WHERE run_id = "
            "(SELECT run_id FROM trade_list_log ORDER BY generated_at DESC LIMIT 1)",
            conn, parse_dates=["as_of_date"])


def format_trade_list(result: dict) -> str:
    """Human-readable trade list for the brief/log."""
    pb = result.get("p_bear")
    src = result.get("regime_source", "?")
    lines = [f"TRADE LIST {result['run_id']} — as of {pd.Timestamp(result['as_of']).date()}",
             f"regime={result['regime']}  exposure={result['exposure']*100:.0f}%  "
             f"positions={result['n_positions']}  cash={result['cash']*100:.0f}%",
             f"regime source={src}" + (f"  P(BEAR)={pb:.2f}" if pb is not None else "")]
    if result.get("kill_switch_tripped"):
        lines.append(f"*** KILL-SWITCH TRIPPED — book forced FLAT: {result.get('kill_switch_reason')}")
    if result["n_positions"] == 0:
        lines.append("  (fully in cash — regime de-risk)")
    else:
        for t, v in result["weights"].sort_values(ascending=False).items():
            if v > 0:
                lines.append(f"  {t:6s} {v*100:5.2f}%")
    return "\n".join(lines)


def latest_session() -> "pd.Timestamp":
    """Newest session for which prices exist. This is the as-of date a freshly
    generated trade list would carry, so it is what staleness is measured
    against."""
    from backtest.engine import load_price_returns
    rets = load_price_returns()
    return rets.index[-1] if not rets.empty else None


def target_is_stale(tl: "pd.DataFrame" = None) -> dict:
    """Is the stored target built on an older session than the data allows?

    WHY THIS EXISTS. reconcile() used to regenerate the target ONLY when none
    was stored, and otherwise served whatever was in the log. So a ticket
    produced without running the daily job first silently re-issued a previous
    day's book -- same weights, same regime call, dressed up with today's date
    on the filename. A stale BEAR/BULL call is the single most dangerous thing
    this system can serve, because the whole (fragile) edge is regime timing.

    Returns {} when fresh, otherwise the detail needed to say so out loud.
    """
    tl = latest_trade_list() if tl is None else tl
    if tl is None or tl.empty:
        return {"reason": "no target stored", "stored": None, "latest": None}
    try:
        latest = latest_session()
    except Exception as exc:  # noqa: BLE001
        logger.warning("target_is_stale: cannot determine latest session: %s", exc)
        return {}
    if latest is None:
        return {}
    stored = pd.Timestamp(tl["as_of_date"].iloc[0]).normalize()
    if stored >= pd.Timestamp(latest).normalize():
        return {}
    return {"reason": "a newer session is available", "stored": stored,
            "latest": pd.Timestamp(latest).normalize()}
