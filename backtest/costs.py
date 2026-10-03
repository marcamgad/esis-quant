"""
Liquidity-aware transaction costs (audit Tier-2 #7).

WHY. The backtest charges a flat 0.55% round-trip. That is fine for a liquid
name traded small, but EGX liquidity varies by orders of magnitude across the
27-name universe; an illiquid ticker traded at size can slip far more than
0.10%. A flat cost flatters exactly the trades that would hurt most in reality.

MODEL. cost_fraction = commission_per_side + slippage(participation), where
participation = trade_value_EGP / ADV_EGP (ADV = trailing average value traded)
and slippage follows a square-root market-impact law:

    slippage = SLIPPAGE_PCT + COST_IMPACT_COEF * sqrt(participation)

The base SLIPPAGE_PCT is the small-trade floor; the sqrt term is the size
penalty (standard Almgren-style impact -- impact grows sub-linearly in size).
COST_IMPACT_COEF is a conservative literature default, NOT EGX-calibrated (we
have no execution/tick data), so this is a STRESS assumption, deliberately
pessimistic, not a measurement. Stated so it is never mistaken for ground truth.

This module gives (a) a per-trade cost from a name's own ADV and the trade
size, and (b) a portfolio-level effective round-trip cost for a target book at
a given capital, so a backtest can replace the flat number with a book- and
size-aware one.
"""

import numpy as np
import pandas as pd

import config
from data.database import get_connection


def average_daily_value(tickers=None, lookback: int = None, as_of: str = None) -> pd.Series:
    """Trailing average value-traded (EGP) per ticker -- the liquidity measure."""
    lookback = lookback or config.COST_ADV_LOOKBACK_DAYS
    with get_connection() as conn:
        q = "SELECT ticker, date, value_traded_egp FROM daily_ohlcv WHERE value_traded_egp > 0"
        params = []
        if as_of:
            q += " AND date <= ?"
            params.append(as_of)
        if tickers:
            q += " AND ticker IN (" + ",".join("?" * len(tickers)) + ")"
            params += list(tickers)
        df = pd.read_sql(q + " ORDER BY date", conn, params=params, parse_dates=["date"])
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby("ticker").apply(lambda g: g.sort_values("date")["value_traded_egp"].tail(lookback).mean())


def _flat_reference() -> float:
    """The old flat round-trip figure, now DERIVED from the real schedule at the
    default book size instead of being a config constant. Reported next to the
    liquidity-aware number so the gap between them stays visible."""
    from backtest.fees import book_pct_per_side
    return 2.0 * book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)


def trade_cost_fraction(trade_value_egp: float, adv_egp: float,
                        commission: float = None, base_slippage: float = None,
                        impact_coef: float = None) -> dict:
    """Cost of ONE trade as a fraction of its notional, given the name's ADV."""
    # Commission defaults to ZERO here: the broker leg is charged per ORDER by
    # backtest/fees.py. Defaulting it to a percentage was how a second cost
    # model survived, and adding it alongside the schedule double-counts.
    commission = 0.0 if commission is None else commission
    base_slippage = config.SLIPPAGE_PCT if base_slippage is None else base_slippage
    impact_coef = config.COST_IMPACT_COEF if impact_coef is None else impact_coef
    if adv_egp is None or adv_egp <= 0 or not np.isfinite(adv_egp):
        participation = float("inf")           # unknown liquidity -> treat as worst case
    else:
        participation = trade_value_egp / adv_egp
    slippage = base_slippage + impact_coef * np.sqrt(min(participation, 4.0)) if np.isfinite(participation) \
        else base_slippage + impact_coef * 2.0
    return {"participation": participation, "commission": commission,
            "slippage": float(slippage), "one_way": float(commission + slippage)}


def book_roundtrip_cost(weights: pd.Series, capital_egp: float, adv: pd.Series = None,
                        as_of=None) -> dict:
    """Effective ROUND-TRIP cost (fraction of capital) to fully build then fully
    unwind a target book of `weights` at `capital_egp`, size-aware per name.
    This is the number a backtest should use instead of the flat 0.55%.

    TWO COST REGIMES, AND THEY DOMINATE AT OPPOSITE ENDS OF THE SIZE RANGE:

      * At LARGE size, market impact dominates -- it scales with sqrt of
        participation, so it is a percentage and grows with the order.
      * At SMALL size, the FLAT per-order fee dominates -- EGP 2 on an 8 EGP
        order is 25%. It does not shrink with the order; it shrinks with the
        NUMBER of orders. Splitting 1,000 EGP across 18 names costs 18x the
        flat fee for the same total notional.

    Modelling only the percentage leg made this function understate the true
    cost of a small, widely-split book by more than 10x, which is precisely
    the regime the live book is in. Both legs are now charged.
    """
    w = weights[weights > 0]
    if w.empty:
        return {"roundtrip_cost": 0.0, "per_name": pd.DataFrame(), "flagged_high_impact": [],
                "flat_fee_share": 0.0, "n_orders": 0,
                "flat_cost_reference": _flat_reference()}
    if adv is None:
        adv = average_daily_value(list(w.index))
    from backtest.fees import one_side_cost
    rows = []
    for t, wi in w.items():
        notional = float(wi) * capital_egp
        c = trade_cost_fraction(notional, float(adv.get(t, np.nan)))
        # impact/commission leg is already a fraction of THIS order's notional
        pct_leg = c["one_way"] * notional
        # flat + stamp leg, charged per order regardless of how small the order is
        fee = one_side_cost(notional, as_of=as_of, spread_per_side=0.0)
        one_way_egp = pct_leg + fee["broker_flat"] + fee["stamp_egp"]
        rows.append({"ticker": t, "weight": float(wi), "notional_egp": notional,
                     "adv_egp": float(adv.get(t, np.nan)), "participation": c["participation"],
                     "one_way_cost": (one_way_egp / notional) if notional > 0 else np.nan,
                     "one_way_egp": one_way_egp, "flat_egp": fee["broker_flat"]})
    per = pd.DataFrame(rows)
    total_egp = 2.0 * float(per["one_way_egp"].sum())          # build + unwind
    flat_egp = 2.0 * float(per["flat_egp"].sum())
    roundtrip = total_egp / capital_egp if capital_egp > 0 else float("inf")
    flagged = per[per["participation"] > config.COST_MAX_PARTICIPATION]["ticker"].tolist()
    return {"roundtrip_cost": roundtrip, "per_name": per.sort_values("participation", ascending=False),
            "flagged_high_impact": flagged, "flat_cost_reference": _flat_reference(),
            "total_egp": total_egp, "n_orders": int(len(per)),
            "flat_fee_share": (flat_egp / total_egp) if total_egp > 0 else 0.0}


def format_book_cost(d: dict, capital_egp: float) -> str:
    lines = [f"LIQUIDITY-AWARE COST for a {capital_egp:,.0f} EGP book",
             f"  effective round-trip: {d['roundtrip_cost']*100:.2f}%   "
             f"(flat-model reference: {d['flat_cost_reference']*100:.2f}%)"]
    if d.get("n_orders"):
        lines.append(f"  {d['n_orders']} orders per side; the flat per-order fee is "
                     f"{d['flat_fee_share']*100:.0f}% of that cost "
                     f"({d.get('total_egp', 0):,.0f} EGP round trip)")
    if d["flagged_high_impact"]:
        lines.append(f"  HIGH-IMPACT names (>{config.COST_MAX_PARTICIPATION*100:.0f}% of ADV): "
                     f"{', '.join(d['flagged_high_impact'])}")
    per = d["per_name"]
    if not per.empty:
        lines.append(f"  {'ticker':8s}{'weight':>8s}{'partic.':>9s}{'1-way':>8s}")
        for r in per.head(6).itertuples():
            part = f"{r.participation*100:.1f}%" if np.isfinite(r.participation) else "n/a"
            lines.append(f"  {r.ticker:8s}{r.weight*100:>7.1f}%{part:>9s}{r.one_way_cost*100:>7.2f}%")
    return "\n".join(lines)
