"""
Brinson-Hood-Beebower performance attribution (Layer 8.2), from scratch, plus
a regime-timing term for this system's cash overlay.

For each holding period, the portfolio's active return over an equal-weight
benchmark of the same investable names is decomposed, per sector s, into:
    allocation_s  = (wp_s - wb_s) * (Rb_s - Rb)     -- sector over/under-weight
    selection_s   =  wb_s        * (Rp_s - Rb_s)    -- picking within a sector
    interaction_s = (wp_s - wb_s) * (Rp_s - Rb_s)   -- the cross term
where wp_s/wb_s are portfolio/benchmark sector weights, Rp_s/Rb_s the within-
sector returns, Rb the total benchmark return. By construction
    sum_s (allocation_s + selection_s + interaction_s) = Rp - Rb
exactly each period -- this identity is the correctness check (if it doesn't
hold, the attribution math is wrong, not the portfolio).

The regime overlay is a TIME-SERIES bet, not a cross-sectional one, so it gets
its own term: timing = r_managed - r_core, i.e. the return the exposure
multiplier added or gave up versus staying fully invested in the core book.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.risk.attribution")


def _period_returns(returns_wide: pd.DataFrame, start, end, names) -> pd.Series:
    """Compound each name's daily returns over (start, end]."""
    win = returns_wide.loc[(returns_wide.index > start) & (returns_wide.index <= end), names]
    return (1 + win).prod() - 1


def brinson_attribution(weights_by_period: dict, rebalance_dates, returns_wide: pd.DataFrame,
                        sectors: pd.Series) -> dict:
    """weights_by_period: {rebalance_date: pd.Series of portfolio weights}.
    rebalance_dates: ordered list; each period runs to the next rebalance.
    Returns per-period and aggregate allocation/selection/interaction, plus the
    realized active return, and the max per-period reconciliation error.
    """
    rows = []
    recon_err = 0.0
    for i, T in enumerate(rebalance_dates):
        if T not in weights_by_period:
            continue
        wp = weights_by_period[T]
        names = list(wp.index)
        if len(names) < 2:
            continue
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns_wide.index[-1]
        ret = _period_returns(returns_wide, T, end, names).reindex(names).fillna(0.0)
        wp = wp / wp.sum()
        wb = pd.Series(1.0 / len(names), index=names)   # equal-weight benchmark, same universe
        sec = sectors.reindex(names).fillna("_UNKNOWN_")

        Rp = float((wp * ret).sum())
        Rb = float((wb * ret).sum())
        alloc = sel = inter = 0.0
        for s in sec.unique():
            mask = sec == s
            wp_s, wb_s = float(wp[mask].sum()), float(wb[mask].sum())
            Rp_s = float((wp[mask] * ret[mask]).sum() / wp_s) if wp_s > 0 else 0.0
            Rb_s = float((wb[mask] * ret[mask]).sum() / wb_s) if wb_s > 0 else 0.0
            alloc += (wp_s - wb_s) * (Rb_s - Rb)
            sel += wb_s * (Rp_s - Rb_s)
            inter += (wp_s - wb_s) * (Rp_s - Rb_s)
        recon_err = max(recon_err, abs((alloc + sel + inter) - (Rp - Rb)))
        rows.append({"date": T, "Rp": Rp, "Rb": Rb, "active": Rp - Rb,
                     "allocation": alloc, "selection": sel, "interaction": inter})

    df = pd.DataFrame(rows)
    total = {k: float(df[k].sum()) for k in ["active", "allocation", "selection", "interaction"]} if not df.empty else {}
    return {"per_period": df, "total": total, "max_reconciliation_error": float(recon_err)}


def timing_effect(managed_returns: pd.Series, core_returns: pd.Series) -> dict:
    """Regime-overlay contribution = managed - core, summed and annualized-ish
    as a cumulative log difference. Positive => the exposure overlay ADDED
    value (dodged losses) vs staying fully invested.
    """
    m = managed_returns.dropna()
    c = core_returns.reindex(m.index).fillna(0.0)
    cum_managed = float((1 + m).prod() - 1)
    cum_core = float((1 + c).prod() - 1)
    return {"cum_managed": cum_managed, "cum_core": cum_core, "timing_contrib": cum_managed - cum_core}
