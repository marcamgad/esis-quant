"""
Idle-cash yield reminder.

THE GAP THIS CLOSES. When the regime overlay de-risks, the freed capital is
logged as "idle cash ~X EGP" and then forgotten -- it sits in the brokerage
account earning approximately nothing, while data/rates.py shows Egyptian short
rates running roughly 8-27% over the sample. About 26% of backtest days are
spent in cash, so the finding and the action were never connected: the system
proved cash should be earning something and then did nothing with that.

WHAT THIS IS AND IS NOT. It is a REMINDER with the magnitude computed, so the
number is visible rather than assumed. It is not advice, and it does not tell
you which instrument to buy -- it states how much cash is idle, what that is
plausibly costing per year at the prevailing short rate, and whether the amount
clears the real product minimums.

WHY THRESHOLDS, AND WHY THEY APPLY TO IDLE CASH RATHER THAN BOOK SIZE:

  * Egyptian T-BILLS have a minimum subscription of EGP 25,000 and multiples.
    Below that the instrument is simply unavailable -- a hard product
    constraint, not a preference.
  * MONEY MARKET FUNDS hold T-bills for you with daily liquidity and much lower
    minimums, at a slightly lower net yield after fund fees. That makes them the
    sensible route between the two tiers.
  * The test is on IDLE CASH, not on the book: a large book that de-risks has
    enough idle cash to qualify, a small fully-invested one does not.

WHY IT STAYS SILENT BELOW THE LOWER TIER. At 100 EGP idle, a 20% yield is 20 EGP
a year -- less than the friction of moving it, and far less than one round-trip
fee. A reminder that fires on sums too small to act on trains the reader to
ignore it, which is worse than not having it.

DURATION MATTERS TOO. Moving cash out and back has friction, and the regime can
flip within days. `IDLE_CASH_MIN_DAYS` requires the cash to have been sitting
for a while before the reminder fires, so a brief de-risk does not trigger a
suggestion to go and buy a 3-month bill.
"""

import logging

import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.idle_cash")


def current_short_rate(as_of=None) -> float:
    """Prevailing investable short rate, from the reconstructed CBE series.

    APPROXIMATE by construction (see data/rates.py) -- used to size the
    opportunity, never quoted as a rate you can actually get.
    """
    from data.rates import cash_rate_series
    idx = pd.DatetimeIndex([pd.Timestamp(as_of or pd.Timestamp.now()).normalize()])
    try:
        return float(cash_rate_series(idx).iloc[-1])
    except Exception:  # noqa: BLE001
        return float("nan")


def days_idle(min_cash: float = 1.0) -> int:
    """How long cash has been sitting, from the daily snapshot history.

    Counts consecutive recent snapshots whose cash exceeded `min_cash`. Returns
    0 when there is no history -- an unknown duration must not be reported as a
    long one.
    """
    with get_connection() as conn:
        df = pd.read_sql("SELECT date, cash FROM user_daily_snapshot "
                         "WHERE cash IS NOT NULL ORDER BY date DESC", conn)
    if df.empty:
        return 0
    n = 0
    for v in df["cash"]:
        if float(v or 0.0) > min_cash:
            n += 1
        else:
            break
    return n


def idle_cash_state(as_of=None) -> dict:
    """Idle cash, what it is plausibly costing, and which route (if any) applies."""
    from ops import portfolio_tracker as pt
    pos, cash = pt.holdings()
    invested = float(pos["value"].sum()) if not pos.empty else 0.0
    total = invested + float(cash)
    rate = current_short_rate(as_of)
    idle = float(cash)

    forgone_year = idle * rate if rate == rate else float("nan")
    held_days = days_idle()

    if idle >= config.IDLE_CASH_TBILL_MIN:
        tier = "tbill"
    elif idle >= config.IDLE_CASH_MMF_MIN:
        tier = "mmf"
    else:
        tier = "below_minimums"

    # MATERIALITY IS RELATIVE, THE PRODUCT MINIMUMS ARE ABSOLUTE, AND THE TWO CAN
    # DISAGREE. The tiers above answer "which instrument can I even buy". They do
    # NOT answer "does this matter". Observed 2026-08-04: 3,167 EGP idle sat below
    # the 5,000 MMF minimum and the reminder stayed silent -- while being 32% of
    # the book and costing ~546 EGP a year, which is larger than the entire
    # expected annual return of the strategy the other 68% is invested in.
    #
    # A reminder that goes quiet exactly when the drag exceeds the edge is worse
    # than useless. So a SECOND, relative test runs alongside the tiers: a large
    # share of the book, idle, whose annual cost is a meaningful fraction of the
    # book, is reported whatever instrument is or is not available at that size.
    idle_pct = (idle / total) if total > 0 else 0.0
    cost_pct = (forgone_year / total) if (total > 0 and forgone_year == forgone_year) else 0.0
    material = (idle_pct >= config.IDLE_CASH_MATERIAL_PCT
                and cost_pct >= config.IDLE_CASH_MATERIAL_COST_PCT)

    return {"idle": idle, "invested": invested, "total": total,
            "idle_pct": idle_pct,
            "short_rate": rate, "forgone_per_year": forgone_year,
            "forgone_per_month": forgone_year / 12 if forgone_year == forgone_year else float("nan"),
            "days_idle": held_days, "tier": tier,
            "material": bool(material), "cost_pct_of_book": cost_pct,
            "actionable": ((tier != "below_minimums" or material)
                           and held_days >= config.IDLE_CASH_MIN_DAYS),
            "tbill_min": config.IDLE_CASH_TBILL_MIN, "mmf_min": config.IDLE_CASH_MMF_MIN}


def format_note(s: dict) -> str:
    """Report line. Returns '' when there is nothing worth saying -- silence is
    the correct output below the minimums."""
    if not s or (s["tier"] == "below_minimums" and not s.get("material")):
        return ""
    if s["days_idle"] < config.IDLE_CASH_MIN_DAYS:
        return (f"    {s['idle']:,.0f} EGP idle ({s['idle_pct']*100:.0f}% of book) — only "
                f"{s['days_idle']} session(s) so far; no action while it may be short-lived.")
    lines = [f"    {s['idle']:,.0f} EGP idle ({s['idle_pct']*100:.0f}% of book) for "
             f"{s['days_idle']} sessions, earning ~0% in the broker account.",
             f"    At ~{s['short_rate']*100:.1f}% short rates that is ~"
             f"{s['forgone_per_year']:,.0f} EGP/year ({s['forgone_per_month']:,.0f}/month) foregone."]
    if s["tier"] == "below_minimums":
        lines.append(f"    Below the {s['mmf_min']:,.0f} EGP money-market minimum, so no obvious "
                     "product fits — but this is "
                     f"{s['idle_pct']*100:.0f}% of your book costing "
                     f"{s['cost_pct_of_book']*100:.1f}%/yr, which is why it is being said anyway.")
        lines.append("    The cheapest fix is usually to put the cash to work in the book itself, "
                     "not to go shopping for a cash product at this size.")
    elif s["tier"] == "tbill":
        lines.append(f"    Above the {s['tbill_min']:,.0f} EGP T-bill minimum — a direct T-bill "
                     "is available, as are money market funds (daily liquidity).")
    else:
        lines.append(f"    Below the {s['tbill_min']:,.0f} EGP T-bill minimum, but above "
                     f"{s['mmf_min']:,.0f} — a money market fund (daily liquidity) is the "
                     "route at this size.")
    lines.append("    Not advice, and not free: moving cash has friction, and the regime can "
                 "re-invest it. Yield figures come from a RECONSTRUCTED rate series.")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    s = idle_cash_state()
    note = format_note(s)
    print(f"idle {s['idle']:,.2f} EGP ({s['idle_pct']*100:.0f}%), tier={s['tier']}, "
          f"days_idle={s['days_idle']}, actionable={s['actionable']}")
    print(note or "  (silent: below the minimums where acting is worthwhile)")
