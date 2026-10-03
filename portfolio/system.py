"""
The full ESIS-QUANT system, assembled end-to-end (the Phase C5 pipeline as a
reusable function) so the dashboard, alerts, and report all read ONE honest
source of truth rather than re-deriving it.

System = min-variance CORE (Phase B, the best single optimizer; mu=0 because
Phase A/B measured no selection alpha) + a BEAR regime de-risking OVERLAY
(Phase C, the actual edge). The drawdown circuit breaker is left OFF by
default -- the C5 gate showed it whipsaws and the regime overlay alone cuts
drawdown more effectively. Everything is net of the 0.55% round-trip cost,
including the overlay's cash-in/cash-out trades.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection
from data.universe import load_tradable
from backtest.engine import BacktestEngine, load_price_returns
from backtest import analytics as an
from portfolio.covariance import ledoit_wolf_shrinkage
from portfolio.mv_optimizer import mean_variance_weights
from risk.overlay import apply_overlay
from risk import metrics as rm
from risk.attribution import brinson_attribution, timing_effect

logger = logging.getLogger("esis_quant.portfolio.system")

# Percentage-equivalent cost DERIVED from the one real schedule at the
# default book size -- not a hardcoded constant. Used only where a scalar
# turnover multiplier is unavoidable; the engine itself charges per order.
def _cost_scalar() -> float:
    from backtest.fees import book_pct_per_side
    return book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)


COST = _cost_scalar()


def _min_variance_fn(window: pd.DataFrame) -> pd.Series:
    lw = ledoit_wolf_shrinkage(window.values)
    w = mean_variance_weights(np.zeros(window.shape[1]), lw["sigma"], max_weight=0.15)["weights"]
    return pd.Series(w, index=window.columns)


def _equal_weight_fn(window: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / window.shape[1], index=window.columns)


def run_full_system(rebalance_days: int = 21, lookback: int = 180, bear_scale: float = 0.0,
                    aum_egp: float = None, cash_yield: bool = False,
                    cash_flat_rate: float = None) -> dict:
    """Run core + overlay + benchmarks and compute metrics + attribution.
    Returns a dict with the return series, exposure, weights, per-strategy
    metric summaries, the BHB/timing attribution, and the latest live state.

    `aum_egp`: if set, the backtest charges LIQUIDITY-AWARE costs sized to this
    capital (Tier-2 #7) instead of the flat 0.55%. This is how you see what the
    edge looks like net of what it would really cost YOU to trade at your size.

    `cash_yield`: credit the un-invested fraction with the Egyptian cash rate
    instead of 0%. DEFAULT OFF, deliberately: it materially raises the headline
    numbers, and it is only real if the idle cash is ACTUALLY placed somewhere
    earning interest. Cash sitting in a broker account (or a simulator) earns
    nothing, and turning this on would then be self-deception. See
    analysis/cash_yield.py for the sensitivity sweep and data/rates.py for the
    (reconstructed, unverified) rate series.
    """
    rets = load_price_returns()
    trad = load_tradable()
    with get_connection() as conn:
        sectors = pd.read_sql("SELECT ticker, sector FROM stocks", conn).set_index("ticker")["sector"]
        reg = pd.read_sql("SELECT date, regime, param_vintage_date FROM regime_history", conn,
                          parse_dates=["date"]).set_index("date")

    eng = BacktestEngine(rebalance_days=rebalance_days, cost_per_turnover=COST, aum_egp=aum_egp)
    core = eng.run_portfolio(_min_variance_fn, rets, lookback=lookback, tradable=trad)
    eqw = eng.run_portfolio(_equal_weight_fn, rets, lookback=lookback, tradable=trad)

    core_ret = core["net_returns"]
    eqw_ret = eqw["net_returns"]
    regime_series = reg["regime"].reindex(core_ret.index).ffill()
    cash_daily = None
    if cash_yield:
        from data.rates import daily_cash_return
        cash_daily = daily_cash_return(core_ret.index, flat_rate=cash_flat_rate)
    overlay = apply_overlay(core_ret, regime_series, bear_scale=bear_scale,
                            cost_per_turnover=COST, cash_daily=cash_daily)
    managed = overlay["managed_returns"]

    def summ(r, turnover=None):
        # turnover is passed through so it is reported NEXT TO return and risk.
        # It was computed by the engine and discarded before the 2026-08-05
        # audit; without it you cannot see how much trading a result required,
        # which is the multiplier on every cost assumption.
        s = an.summary(r, turnover=turnover, rebalance_days=rebalance_days)
        s["cornish_fisher_var95"] = rm.cornish_fisher_var(r)
        s.update({f"dd_{k}": v for k, v in rm.drawdown_analytics(r).items()})
        return s

    metrics = {"full_system": summ(managed, core.get("turnover")),
               "core": summ(core_ret, core.get("turnover")),
               "equal_weight": summ(eqw_ret, eqw.get("turnover"))}
    bh = brinson_attribution(core["weights"], core["rebalance_dates"], rets, sectors)
    te = timing_effect(managed, core_ret)

    latest_date = regime_series.index[-1]
    latest = {
        "date": latest_date,
        "regime": str(regime_series.iloc[-1]),
        "param_vintage": reg["param_vintage_date"].reindex([latest_date]).iloc[0]
        if latest_date in reg.index else None,
        "exposure": float(overlay["exposure"].iloc[-1]),
        "breaker_active": bool(overlay["breaker_active"].iloc[-1]),
        "current_drawdown": float(rm.drawdown_series(managed).iloc[-1]),
    }

    return {"returns": {"full_system": managed, "core": core_ret, "equal_weight": eqw_ret},
            "exposure": overlay["exposure"], "weights": core["weights"], "regime": regime_series,
            "metrics": metrics, "attribution": {"brinson": bh, "timing": te}, "latest": latest}
