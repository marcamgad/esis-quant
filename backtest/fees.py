"""
Date-aware EGX / Thndr fee schedule  (Phase 0.1 -- cost-model staleness fix).

WHY THIS EXISTS. The cost model in use was BOTH stale and structurally wrong:

  1. STALE: a stamp duty took effect on EGX trades from the 2026-07-29 session
     (Law No. 153 of 2026) -- 0.05% per side on standard transactions, 0.025%
     per side on same-day/intraday, paid by buyer AND seller, owed regardless of
     profit. Order files exist dated after that session priced without it.

  2. STRUCTURALLY WRONG, and this matters more: the model charged a flat
     percentage (config.THNDR_COMMISSION_PCT = 0.175%/side). Thndr actually
     charges **EGP 2 + 0.1% of order value, PER ORDER**. A fixed EGP 2 is not a
     rounding difference on a small book -- on a 40 EGP order it is a 5% fee.
     A percentage-only model cannot express that, so it systematically
     understated the cost of exactly the small orders this book places, and the
     error grows as the book is split across more names.

DATE-AWARENESS IS DELIBERATE. Rates are a lookup keyed by effective date, not
constants, so the next change is a one-line addition rather than another silent
staleness. Costing a 2024 backtest trade with a 2026 stamp duty would be
look-ahead in reverse -- the schedule returns what was actually payable on the
date of the trade.

SOURCE AND VERIFICATION. The stamp-duty terms and the Thndr fee structure here
were supplied by the operator, not independently verified from a primary source
by this code. They are recorded with their effective dates so they can be
checked against a real contract note; `analysis/day_trade.implied_cost_from_log`
measures realised all-in cost from actual fills and is the authority if the two
disagree.
"""

import logging
from bisect import bisect_right

import pandas as pd

logger = logging.getLogger("esis_quant.fees")

# ---------------------------------------------------------------------------
# Broker schedule: (effective_date, flat EGP per order, fraction of order value)
# ---------------------------------------------------------------------------
# MEASURED 2026-08-04 from 8 real Thndr contract notes, not operator-supplied.
# Least-squares fit over notionals 100-155 EGP gave EGP 4.952 + 0.1624% ALL-IN,
# max residual 0.039 EGP across all 8 orders -- a very tight fit. Subtracting the
# 0.05%/side stamp duty (which one_side_cost adds separately) leaves the broker
# leg at EGP 4.95 + 0.112%.
#
# This REPLACES the previous "EGP 2 + 0.10%" entry rather than being appended as
# a later change, because there is no evidence Thndr changed anything: the EGP 2
# was operator-supplied and explicitly flagged unverified in this module's
# docstring, and 4.95 is the first number measured from actual fills. Treating a
# corrected guess as a rate change would silently keep the wrong figure in every
# pre-2026-08 backtest. If a real schedule change is ever confirmed, split this
# row by its effective date.
#
# The correction is not small: EGP 4.95 vs 2.00 is 2.36x the modelled fee on the
# live book, and on a 1,000 EGP book split 8 ways it is 4.1% per side.
THNDR_SCHEDULE = [
    ("1900-01-01", 0.0, 0.00175),   # legacy assumption kept for pre-2026 backtests
    ("2026-01-01", 4.95, 0.00112),  # EGP 4.95 + 0.112% per order (measured 2026-08-04)
]

# Stamp duty: (effective_date, standard per-side rate, same-day/intraday rate)
STAMP_DUTY_SCHEDULE = [
    ("1900-01-01", 0.0, 0.0),
    ("2026-07-29", 0.0005, 0.00025),   # Law No. 153 of 2026
]

DEFAULT_SPREAD_PER_SIDE = 0.0010     # half-spread crossed; not a fee but costs like one


def _lookup(schedule, as_of):
    dates = [pd.Timestamp(d) for d, *_ in schedule]
    i = bisect_right(dates, pd.Timestamp(as_of)) - 1
    return schedule[max(i, 0)][1:]


def broker_fee(order_value_egp: float, as_of=None) -> dict:
    """Thndr's charge on ONE order: a flat EGP amount plus a percentage.

    The flat part is what makes small orders expensive, so it is returned
    separately -- `pct_of_order` is the number that actually matters and it
    rises without limit as the order shrinks.
    """
    as_of = pd.Timestamp.now() if as_of is None else as_of
    flat, pct = _lookup(THNDR_SCHEDULE, as_of)
    v = max(float(order_value_egp), 0.0)
    fee = flat + pct * v
    return {"flat": flat, "pct_rate": pct, "fee_egp": fee,
            "pct_of_order": (fee / v) if v > 0 else float("inf")}


def stamp_duty(order_value_egp: float, as_of=None, same_day: bool = False) -> dict:
    """Stamp duty on ONE side. Owed on the transaction, not on any profit."""
    as_of = pd.Timestamp.now() if as_of is None else as_of
    std, intraday = _lookup(STAMP_DUTY_SCHEDULE, as_of)
    rate = intraday if same_day else std
    return {"rate": rate, "fee_egp": rate * max(float(order_value_egp), 0.0)}


def one_side_cost(order_value_egp: float, as_of=None, same_day: bool = False,
                  spread_per_side: float = DEFAULT_SPREAD_PER_SIDE) -> dict:
    """All-in cost of ONE order: broker + stamp duty + half-spread."""
    b = broker_fee(order_value_egp, as_of)
    s = stamp_duty(order_value_egp, as_of, same_day)
    v = max(float(order_value_egp), 0.0)
    spread = spread_per_side * v
    total = b["fee_egp"] + s["fee_egp"] + spread
    return {"broker_egp": b["fee_egp"], "stamp_egp": s["fee_egp"], "spread_egp": spread,
            "total_egp": total, "pct_of_order": (total / v) if v > 0 else float("inf"),
            "broker_flat": b["flat"]}


def round_trip_cost(order_value_egp: float, as_of=None, same_day: bool = False,
                    spread_per_side: float = DEFAULT_SPREAD_PER_SIDE) -> dict:
    """Buy then sell the same position: two orders, two stamp duties."""
    buy = one_side_cost(order_value_egp, as_of, same_day, spread_per_side)
    sell = one_side_cost(order_value_egp, as_of, same_day, spread_per_side)
    total = buy["total_egp"] + sell["total_egp"]
    v = max(float(order_value_egp), 0.0)
    return {"total_egp": total, "pct_of_order": (total / v) if v > 0 else float("inf"),
            "buy": buy, "sell": sell}


def book_round_trip_cost(position_values, as_of=None, same_day: bool = False,
                         spread_per_side: float = DEFAULT_SPREAD_PER_SIDE) -> dict:
    """Cost of round-tripping a WHOLE BOOK, order by order.

    Charging per order is the whole point: splitting 1,000 EGP across 7 names
    pays the flat fee seven times on each side. A percentage-only model shows
    no difference between 1 order and 7; this does.
    """
    vals = [float(v) for v in position_values if float(v) > 0]
    if not vals:
        return {"total_egp": 0.0, "pct_of_book": 0.0, "n_orders": 0, "book_value": 0.0}
    total = sum(round_trip_cost(v, as_of, same_day, spread_per_side)["total_egp"] for v in vals)
    book = sum(vals)
    flat_part = sum(round_trip_cost(v, as_of, same_day, spread_per_side)["buy"]["broker_flat"]
                    for v in vals) * 2
    return {"total_egp": total, "pct_of_book": total / book, "n_orders": len(vals) * 2,
            "book_value": book, "flat_fee_egp": flat_part,
            "flat_share_of_cost": (flat_part / total) if total > 0 else 0.0}


def format_book_cost(d: dict) -> str:
    return (f"round-trip on {d['book_value']:,.0f} EGP across {d['n_orders']//2} name(s): "
            f"{d['total_egp']:,.2f} EGP = {d['pct_of_book']*100:.2f}%\n"
            f"   of which flat per-order fees: {d.get('flat_fee_egp',0):,.2f} EGP "
            f"({d.get('flat_share_of_cost',0)*100:.0f}% of the total cost)")


def book_pct_per_side(aum_egp: float = None, n_names: int = None, as_of=None,
                      spread_per_side: float = 0.0) -> float:
    """Percentage-equivalent ONE-SIDE cost for a whole book, derived from the
    real schedule at a stated size and breadth.

    WHY THIS EXISTS. Several modules legitimately need a single scalar cost
    (a turnover multiplier) rather than a per-order charge. Before the
    2026-08-05 audit they each hardcoded config.THNDR_COMMISSION_PCT, which
    meant a second, wrong cost model lived alongside this one. Deriving the
    scalar HERE keeps a single source of truth: the scalar changes when the
    schedule changes, and it is explicitly a function of book size and name
    count, which a constant can never be.

    The number is only meaningful WITH its (aum, n_names) -- quoting it bare is
    how the old constant became misleading in the first place.
    """
    import config as _cfg
    aum = float(aum_egp if aum_egp is not None else _cfg.BACKTEST_DEFAULT_AUM_EGP)
    n = max(int(n_names if n_names is not None else DEFAULT_BOOK_NAMES), 1)
    if aum <= 0:
        return float("inf")
    per_order = aum / n
    total = n * one_side_cost(per_order, as_of=as_of,
                              spread_per_side=spread_per_side)["total_egp"]
    return total / aum


# Typical live breadth (the name cap in ops/order_ticket at the default book
# size). Used only to derive the scalar above when a caller does not say.
DEFAULT_BOOK_NAMES = 12
