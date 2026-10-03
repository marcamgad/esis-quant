"""
Integer-aware share allocation (whole shares, real budget).

THE PROBLEM THIS SOLVES. The reconciler sizes each name independently: it
computes that name's EGP gap, divides by price, and floors. On a small book
every gap is a fraction of a share, so every name floors to zero and NOTHING is
ever bought -- the measured outcome on a 1,000 EGP book was 34% of capital
stranded in cash with "nothing actionable" reported indefinitely.

WHY PER-NAME FLOORING IS THE WRONG RULE. It treats idle cash as neutral. It is
not. When the target book is 100% invested, cash is a position with a target
weight of ZERO, so holding 34% cash is a 34-point tracking error all by itself.
Overshooting one name by half a share is a far smaller error than leaving 96 EGP
uninvested -- but a per-name rule can never see that, because it never compares
the two.

THE OBJECTIVE. Choose an integer share vector n minimizing squared tracking
error against the target weights,

    TE(n) = sum_i ( n_i * p_i / V  -  w_target_i )^2 ,   V = sum_i n_i p_i + cash

subject to not overspending. Cash drag is captured automatically: if cash is
held, every realized weight sits below its target and each shortfall is charged.

THE ALGORITHM. Integer-constrained least squares is NP-hard in general, but this
instance is tiny (tens of names, hundreds of shares) and the objective is convex
and separable enough that a greedy marginal search is effectively optimal: while
some affordable share purchase reduces TE, buy the single share that reduces it
most. Each step is the best local move and re-evaluates all names, so it can
buy a name whose own gap is under one share whenever doing so beats holding
cash. Terminates when no affordable purchase helps (or cash runs out).

TRANSACTION COSTS. A purchase that barely improves TE is not worth its
commission and spread. `min_improvement_bps` requires each buy to reduce TE by a
minimum amount, so the allocator does not churn for cosmetic precision.

This module only ADDS shares (deploying available cash). It deliberately does
not sell to fund rebalancing -- selling to chase tracking error incurs real cost
and tax on a small book; the reconciler still handles genuine SELL/TRIM signals.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.integer_allocator")

DEFAULT_MIN_IMPROVEMENT_BPS = 1.0    # a buy must cut TE by >= 1bp (in weight^2 terms, x1e4)


def tracking_error(shares: pd.Series, prices: pd.Series, target_w: pd.Series,
                   cash: float) -> float:
    """Sum of squared weight deviations from target, over total portfolio value
    (positions + cash). Idle cash shows up as a shortfall on every name."""
    names = target_w.index.union(shares.index)
    sh = shares.reindex(names, fill_value=0.0)
    px = prices.reindex(names)
    value = (sh * px).fillna(0.0)
    total = float(value.sum() + cash)
    if total <= 0:
        return float("inf")
    w = value / total
    tw = target_w.reindex(names, fill_value=0.0)
    return float(((w - tw) ** 2).sum())


def greedy_integer_allocation(target_w: pd.Series, prices: pd.Series, cash: float,
                              current_shares: pd.Series = None,
                              min_improvement_bps: float = DEFAULT_MIN_IMPROVEMENT_BPS,
                              max_iter: int = 10_000, exclude=None) -> dict:
    """Buy whole shares, one at a time, always taking the purchase that most
    reduces tracking error, until nothing affordable helps.

    `exclude`: names you refuse to buy. They are removed from the BUY candidate
    set but KEPT in the target weights, so their shortfall still counts as
    tracking error -- the allocator then redeploys that cash into whatever names
    you do allow, rather than silently pretending the target changed.

    Returns {orders (Series of shares to BUY), te_before, te_after, cash_before,
    cash_after, deployed, n_buys}.
    """
    target_w = target_w[target_w > 0].astype(float)
    if target_w.empty:
        return {"orders": pd.Series(dtype=float), "te_before": float("nan"),
                "te_after": float("nan"), "cash_before": cash, "cash_after": cash,
                "deployed": 0.0, "n_buys": 0}
    target_w = target_w / target_w.sum()

    names = list(target_w.index)
    px = prices.reindex(names).astype(float)
    banned = set(exclude or ())
    tradable = [n for n in names
                if np.isfinite(px.get(n, np.nan)) and px[n] > 0 and n not in banned]
    if not tradable:
        raise ValueError("greedy_integer_allocation: no priced names in the target")

    held = (current_shares if current_shares is not None else pd.Series(dtype=float))
    shares = held.reindex(sorted(set(names) | set(held.index)), fill_value=0.0).astype(float)

    te_before = tracking_error(shares, prices, target_w, cash)
    remaining = float(cash)

    # SCALE-INVARIANT churn threshold. The improvement from one more share is
    # measured in squared WEIGHT units, so it shrinks as the book grows: a share
    # that moves a 1,000 EGP book by 3% moves a 10,000 EGP book by 0.3%, and the
    # squared term falls ~100x. A fixed absolute floor therefore stops the
    # allocator earlier the MORE capital you have -- measured: 21% cash left
    # idle at 1,000 EGP but 46% at 10,000, which is backwards. Scaling the
    # threshold by the typical squared weight of one position keeps "is this
    # trade worth its commission?" meaning the same at every size.
    n_target = max(len(tradable), 1)
    scale = (1.0 / n_target) ** 2
    thresh = float(min_improvement_bps) / 1e4 * scale
    bought = {}

    for _ in range(max_iter):
        best_name, best_te = None, tracking_error(shares, prices, target_w, remaining)
        for n in tradable:
            p = float(px[n])
            if p > remaining:
                continue                      # cannot afford one share
            trial = shares.copy()
            trial[n] = trial.get(n, 0.0) + 1.0
            te = tracking_error(trial, prices, target_w, remaining - p)
            if te < best_te - thresh:
                best_name, best_te = n, te
        if best_name is None:
            break                             # no affordable buy improves things
        p = float(px[best_name])
        shares[best_name] = shares.get(best_name, 0.0) + 1.0
        remaining -= p
        bought[best_name] = bought.get(best_name, 0) + 1

    orders = pd.Series(bought, dtype=float).sort_values(ascending=False)
    te_after = tracking_error(shares, prices, target_w, remaining)
    logger.info("greedy_integer_allocation: %d buys, deployed %.2f, TE %.5f -> %.5f",
                int(orders.sum()) if len(orders) else 0, cash - remaining, te_before, te_after)
    return {"orders": orders, "te_before": te_before, "te_after": te_after,
            "cash_before": float(cash), "cash_after": float(remaining),
            "deployed": float(cash - remaining), "n_buys": int(orders.sum()) if len(orders) else 0,
            "final_shares": shares}


def format_allocation(res: dict, prices: pd.Series) -> str:
    o = res["orders"]
    if o.empty:
        return ("  integer allocator: no purchase reduces tracking error -- "
                "holding cash is currently the best achievable book.")
    lines = [f"  integer allocator: {res['n_buys']} share(s) across {len(o)} name(s), "
             f"deploying {res['deployed']:,.0f} of {res['cash_before']:,.0f} EGP cash",
             f"  tracking error {res['te_before']:.5f} -> {res['te_after']:.5f} "
             f"({(1 - res['te_after']/res['te_before'])*100:.0f}% better)" if res["te_before"] > 0 else ""]
    for t, n in o.items():
        p = float(prices.get(t, float("nan")))
        lines.append(f"     BUY {t:6s} {int(n):3d} sh @ {p:8.2f} = {int(n)*p:8.0f} EGP")
    lines.append(f"  cash left: {res['cash_after']:,.0f} EGP")
    return "\n".join(l for l in lines if l)
