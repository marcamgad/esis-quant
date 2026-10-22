"""
Daily ORDER TICKET file (the "open a file, do it, fill it in, save it" flow).

The workflow you asked for:
  1. write_ticket()  -> creates orders_<date>.csv listing exactly what to buy
     and how much, sized to your book, capped at MAX_TICKET_NAMES names.
  2. You open it, place those trades, and type TWO numbers per row: PAID_EGP
     (what the shares cost) and FEE_EGP (Thndr's commission). Rows you leave
     blank are skipped. The fill price is derived as PAID_EGP / shares.
  3. apply_ticket() -> reads your fills and records them in the DB (your real
     holdings + cash + ledger), then logs the day's decision. The fee is
     deducted from cash and folded into the cost basis, and compared against
     what backtest/fees.py predicted.

Sizing is MECHANICAL and never overspends cash. SELL/TRIM lines come from the
reconciler (genuine exit signals); BUY/ADD sizing comes from the integer
allocator (portfolio/integer_allocator.py), which picks whole shares that most
reduce tracking error instead of flooring each name's gap independently -- the
old per-name rule bought nothing on a small book and stranded the cash. These
are a starting point, not advice; you decide what to place, and only your
filled numbers are saved.

EXECUTION IS MANUAL BY DESIGN -- A PERMANENT DECISION, NOT AN OVERSIGHT
(audit deferred item P6.19).

There is no broker API, no automated order placement, no partial-fill callback
and no rejection handler, and there will not be. Reasons, in order of weight:

  1. Thndr exposes no programmatic order interface to retail. Any "integration"
     would be UI scraping, which breaks silently and places real orders when it
     breaks. Inventing a broker API this project cannot actually call would be
     worse than having none.
  2. The go-live gate has never passed (0 of 189 required forward sessions with
     a bear regime observed). Automating execution for a strategy that has not
     earned the right to trade automatically inverts the safety ordering.
  3. Human-in-the-loop is the last control that catches what the code cannot:
     a stale price, a halted name, a suspended stock, a fat-fingered quantity.

PARTIAL FILLS AND REJECTIONS ARE THEREFORE HANDLED BY THE OPERATOR, and the
system's job is to make that safe rather than to pretend it is automated:
  * a partial fill is entered as the quantity ACTUALLY filled (edit `shares`);
  * a rejected or skipped order is left blank and is simply not recorded;
  * ops/reconcile.py then checks the resulting book against the broker, which
    catches anything mis-entered.

If Thndr ever ships a real API, this note is the thing to revisit first.

CLI:
  py -3.11 -m ops.order_ticket write        # make today's ticket (prints path)
  py -3.11 -m ops.order_ticket apply        # save your filled ticket to the DB
"""

import csv
import logging
import os

import pandas as pd

import config
from data.database import get_connection
from ops import portfolio_tracker as pt
from risk.pretrade import screen_order, format_screen

logger = logging.getLogger("esis_quant.ops.order_ticket")

TICKET_DIR = os.path.join(config.BASE_DIR, "orders")

# You fill TWO columns, not four. PAID_EGP is the cash that left your account
# for the shares (e.g. 350 for 22 shares) and FEE_EGP is Thndr's commission on
# that order. Everything else is derived: shares come from the `shares` column
# (edit it if you bought a different quantity) and the fill price is
# PAID_EGP / shares. This is both less to type and less to get wrong -- the old
# FILLED_price asked you to reverse-engineer a per-share average from a total
# your broker already shows you.
#
# TRADE-OFF, STATED: dropping FILLED_time means execution quality can no longer
# measure intraday timing. It still measures fill-vs-signal-close, which is the
# part that validates the cost model, so the loss is real but small.
HEADER = ["action", "ticker", "target_pct", "ref_price", "max_price", "shares",
          "suggested_egp", "adv_pct", "flags", "PAID_EGP", "FEE_EGP"]

# NAME-COUNT CAP -- the single biggest cost lever on a small book.
#
# Thndr charges a FLAT EGP 4.95 per order on top of the percentage fee (MEASURED
# 2026-08-04 from 8 real contract notes; see backtest/fees.py). That flat
# component does not shrink with order size, only with the NUMBER of orders, so
# each extra name costs about another 1pp of round-trip cost on a 1,000 EGP book
# regardless of how the money is split.
#
# Measured on the live book (1,000 EGP, real fee schedule):
#
#   names:   3      4      5      6      8     10     12     15     20     24
#   cost:  3.7%   4.7%   5.7%   6.7%   8.6%  10.6%  12.6%  15.5%  20.5%  24.4%
#   vol:  14.8%  13.9%  13.4%  12.8%  12.1%  11.8%  11.7%  11.7%  11.7%  11.8%
#
# Two facts fall out. Risk stops falling past ~10-12 names, so names beyond that
# are pure cost. And cost per name is ROUGHLY CONSTANT (~1pp) while the risk
# benefit decays fast -- past ~5 names each name buys under 0.5pp of volatility
# for a full 1pp of certain cost.
#
# THE CAP IS THEREFORE SIZE-DEPENDENT, not a constant. The same 8 names that are
# extravagant at 1,000 EGP are conservative at 25,000, because the flat fee is
# amortised over 25x the notional. A fixed number would be wrong at one end or
# the other, so the cap is solved from the measured cost curve against a budget.
#
# The cap applies to the TICKET, not the optimizer -- the target book stays the
# honest min-variance solution, and the cap is an explicit, visible concession
# to the fee schedule rather than a hidden change to the model.
NAME_CAP_FLOOR = 5          # below this the book is too concentrated to call diversified
NAME_CAP_CEILING = 12       # above this measured risk no longer falls -- pure cost
# Share of the book's expected ANNUAL real return (~3.7%, and itself optimistic --
# see the real-terms caveats in the daily report) that may be spent on a full
# build-and-unwind. 0.5 means: do not spend more than half a year's expected
# return on the round trip. This is a budget, not a prediction.
NAME_CAP_COST_BUDGET = 0.5 * 0.037
MAX_TICKET_NAMES = None     # None -> solve from capital; an int forces that count


def _fee_share(notional_egp: float) -> float:
    """Commission as a fraction of the order it is charged on. Unknown reads as
    infinite, so an order we cannot price is treated as too expensive rather
    than as free."""
    if not notional_egp or notional_egp <= 0:
        return float("inf")
    try:
        from backtest.fees import one_side_cost
        return one_side_cost(notional_egp, spread_per_side=0.0)["total_egp"] / notional_egp
    except Exception:  # noqa: BLE001
        return float("inf")


def _uneconomic(notional_egp: float) -> bool:
    """Can this order clear its own commission? Uses the SAME bound the
    reconciler applies, so the two paths into the ticket cannot disagree."""
    return _fee_share(notional_egp) > pt.MAX_FEE_SHARE_OF_TRADE


def cap_target_names(target: "pd.Series", held=None, max_names: int = None,
                     capital_egp: float = None) -> tuple:
    """Cut the target book to `max_names` and renormalise, keeping held names.

    THIS MUST BE THE ONLY PLACE THE CAP IS APPLIED. It is used by the ticket
    (what to buy) AND by the reconciler (what to hold), and those two MUST agree.
    When only the ticket capped, the reconciler kept comparing an 8-name book
    against a 24-name target: every held name read overweight -> TRIM, and every
    excluded name read BUY. That is a permanent false rebalance signal which, if
    followed, rebuilds the very book the cap exists to prevent -- at ~EGP 5 per
    order. A cap applied in one place and not the other is worse than no cap.

    Held names are never evicted: dropping one would emit a SELL because a
    threshold moved rather than because the model changed its mind.
    """
    if target is None or len(target) == 0:
        return target, None
    if max_names is None:
        max_names = (int(MAX_TICKET_NAMES) if MAX_TICKET_NAMES
                     else suggested_name_cap(capital_egp or 0.0))
    max_names = int(max_names)
    if max_names <= 0 or len(target) <= max_names:
        return target, None
    held = set(held or ())
    # Top-N by weight is exactly the selection that can return two names from one
    # sector (the 2026-09-04 flour-mill pair). select_diverse keeps the weight
    # ordering but limits names per sector; with no sector data it reduces to
    # plain top-N plus held names, i.e. the previous behaviour.
    from portfolio.sector_cap import select_diverse, apply_sector_cap, load_sectors
    sectors = load_sectors()
    keep = select_diverse(target, sectors, max_names, held=held) | (set(target.index) & held)
    dropped = sorted(set(target.index) - keep)
    out = target[target.index.isin(keep)]
    total = float(out.sum())
    if total > 0:
        out = out / total
    out, sector_info = apply_sector_cap(out, sectors)
    return out, {"cap": max_names, "kept": len(out), "dropped": dropped,
                 "sector": sector_info}


def suggested_name_cap(capital_egp: float, budget: float = None) -> int:
    """Largest name count whose measured round-trip cost fits the budget.

    Falls back to the floor when the cost model cannot be evaluated -- on a small
    book, erring toward FEWER names is the safe direction, because the failure
    mode being guarded against is paying more in fees than the book can earn.
    """
    budget = NAME_CAP_COST_BUDGET if budget is None else budget
    cap = float(capital_egp or 0.0)
    if cap <= 0:
        return NAME_CAP_FLOOR
    try:
        from backtest.fees import one_side_cost
        best = NAME_CAP_FLOOR
        for n in range(NAME_CAP_FLOOR, NAME_CAP_CEILING + 1):
            # equal-notional split is the right approximation here: the flat fee
            # is per ORDER and does not care how the money is distributed
            per = cap / n
            rt = 2.0 * n * one_side_cost(per, spread_per_side=0.0)["total_egp"] / cap
            if rt <= budget:
                best = n
        return best
    except Exception:  # noqa: BLE001
        logger.warning("suggested_name_cap: cost model unavailable, using floor")
        return NAME_CAP_FLOOR

# --- LIVE market-mechanics gates (Part 1.3 / Part 2) -------------------------
# config.COST_MAX_PARTICIPATION already existed but was applied ONLY in
# backtest/costs.py -- the live ticket never checked whether an order it was
# printing was large relative to the name's ADV. Applied here so the check
# reaches the account, not just the simulation.
#
# EGX price limits -- VERIFIED 2026-08-04, no longer a guess.
# Source: EFG Hermes EGX trading specifications
# (https://tg.efghermes.com/Market/GetById/1103), corroborated by EGX rules:
#   * regular trading band: +/-20% from the previous close
#   * a 10% move from the previous close triggers a ~10-minute suspension,
#     after which trading resumes to the permitted ceiling/floor
#   * auction matching: +/-10% from the MVWAP of the last EGP 300k traded
# So 10% is the level where FILLABILITY actually degrades (suspension), and 20%
# is the hard band. The warning fires at 9% -- just below the suspension
# trigger -- so the flag appears BEFORE the name halts, not after.
# (data/corporate_actions.ARTIFACT_THRESHOLD = 0.30 is a DATA-cleaning
# threshold for split artifacts and is deliberately NOT reused here.)
PRICE_LIMIT_WARN = 0.09          # just under the 10% suspension trigger
PRICE_LIMIT_HARD = 0.20          # EGX regular-trading band
PRICE_SUSPEND_TRIGGER = 0.10     # 10% move -> ~10-min suspension

# EGX has NO BOARD LOTS -- verified from the same source ("No board Lots").
# Whole-share rounding in portfolio/integer_allocator.py is therefore correct
# and complete; no minimum-lot constraint needs to be encoded.
HALT_LOOKBACK_DAYS = 3           # no volume in this many sessions -> possible halt

# ref_price is the last CLOSE. By the time you trade, the live price may have
# gapped away from it -- and a name that jumped is no longer the trade the
# optimizer sized. max_price is the ceiling above which the suggestion should be
# SKIPPED rather than chased (measured case: CLHO/CIRA opened ~3 EGP above their
# ref price and were correctly skipped by hand; this makes that rule explicit).
PRICE_TOLERANCE = 0.03   # skip a buy if the live price is >3% above ref


# Only the current ticket is kept. Old tickets are working files, not records
# (the DB ledger is the permanent audit trail), and a stale one lying around is
# actively misleading -- it lists prices and sizes that no longer apply.
KEEP_TICKETS = 1


def _ticket_path(generated_on: str = None) -> str:
    """Ticket files are named by the day they were GENERATED, not by the signal's
    as-of date. Those differ whenever the market is shut (a ticket written on
    Sunday 27/7 is built on Thursday 26/7's close), and the filename you look for
    should be today's date. The as-of date is recorded in the file header."""
    os.makedirs(TICKET_DIR, exist_ok=True)
    day = generated_on or pd.Timestamp.now().strftime("%Y-%m-%d")
    return os.path.join(TICKET_DIR, f"orders_{day}.csv")


def latest_ticket_path() -> str:
    """Newest ticket file on disk, or None. Used by apply() so it works
    regardless of which day the file is named for."""
    if not os.path.isdir(TICKET_DIR):
        return None
    files = sorted(f for f in os.listdir(TICKET_DIR)
                   if f.startswith("orders_") and f.endswith(".csv"))
    return os.path.join(TICKET_DIR, files[-1]) if files else None


def prune_old_tickets(keep: int = KEEP_TICKETS, protect: str = None) -> list:
    """Delete all but the `keep` newest ticket files. Tickets are working files
    (the DB ledger is the permanent record -- applied fills live there), so old
    unapplied tickets are just clutter that can mislead. Returns deleted names.

    ORDERED BY MODIFICATION TIME, NOT BY NAME. The locked-file fallback writes
    `orders_<date>_<HHMMSS>.csv`, which sorts AFTER the plain `orders_<date>.csv`
    for the same day. Name-ordering therefore judged a fresh write to be the
    older file and deleted the ticket it had just produced -- observed live on
    2026-08-04, where the scale-in plan vanished the moment it was written.

    `protect` is belt-and-braces: the file just written is never a prune
    candidate, whatever the timestamps say.
    """
    if not os.path.isdir(TICKET_DIR):
        return []
    protect = os.path.basename(protect) if protect else None
    files = [f for f in os.listdir(TICKET_DIR)
             if f.startswith("orders_") and f.endswith(".csv") and f != protect]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(TICKET_DIR, f)))
    keep = max(keep - 1, 0) if protect else keep      # the protected file counts
    doomed = files[:-keep] if keep > 0 else files
    deleted = []
    for f in doomed:
        try:
            os.remove(os.path.join(TICKET_DIR, f))
            deleted.append(f)
        except OSError:
            # locked (probably open in Excel/Notepad) -- leave it; it'll be
            # swept on a later run once closed. Never fail the ticket write.
            logger.warning("prune_old_tickets: %s is in use, skipped", f)
    return deleted


def _order_flags(ticker: str, notional_egp: float) -> tuple:
    """Pre-trade sanity flags for one order: participation vs ADV, possible
    price-limit proximity, and possible halt.

    Reuses backtest.costs.average_daily_value rather than recomputing ADV, and
    reads today's move straight from daily_ohlcv. Returns (adv_pct, flags_str).
    Never raises -- a flag we cannot compute must not stop the ticket, but its
    absence is reported as "?" rather than silently as "fine".
    """
    flags = []
    adv_pct = None
    try:
        from backtest.costs import average_daily_value
        adv = average_daily_value([ticker])
        a = float(adv.get(ticker, float("nan")))
        if a == a and a > 0:
            adv_pct = round(100.0 * notional_egp / a, 2)
            if notional_egp / a > config.COST_MAX_PARTICIPATION:
                flags.append(f"SIZE>{config.COST_MAX_PARTICIPATION*100:.0f}%ADV")
    except Exception:  # noqa: BLE001
        flags.append("ADV?")
    # The flag above is a LABEL. It never stopped anything, which is what the
    # 2026-08-05 audit scored 🔴 on Section 14. The screen below is the control:
    # it returns the size actually permitted, and write_ticket drops or resizes
    # the row accordingly. Kept separate from the flag so the printed reason and
    # the enforced decision come from one place and cannot drift apart.

    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT date, close, adjusted_close, volume FROM daily_ohlcv "
                "WHERE ticker=? ORDER BY date DESC LIMIT ?",
                (ticker, HALT_LOOKBACK_DAYS + 1)).fetchall()
        if len(rows) >= 2:
            last, prev = rows[0], rows[1]
            if last["adjusted_close"] and prev["adjusted_close"]:
                mv = last["adjusted_close"] / prev["adjusted_close"] - 1.0
                if abs(mv) >= PRICE_LIMIT_HARD:
                    flags.append(f"AT-BAND{mv*100:+.0f}%")      # at/through the 20% limit
                elif abs(mv) >= PRICE_SUSPEND_TRIGGER:
                    flags.append(f"SUSPEND-RISK{mv*100:+.0f}%")  # 10% -> 10-min halt
                elif abs(mv) >= PRICE_LIMIT_WARN:
                    flags.append(f"MOVED{mv*100:+.0f}%")
        # a name with no traded volume across the lookback may be halted
        vols = [r["volume"] or 0 for r in rows[:HALT_LOOKBACK_DAYS]]
        if vols and all(v <= 0 for v in vols):
            flags.append("NO-VOLUME/HALT?")
    except Exception:  # noqa: BLE001
        flags.append("MKT?")

    return adv_pct, "|".join(flags)


def write_ticket(path: str = None, use_allocator: bool = True, exclude=None,
                 max_names: int = None, deposit: float = 0.0) -> dict:
    """Generate today's order ticket sized to your current book. Returns
    {path, as_of, rows, suggested_total, cash, idle, capped}.

    `max_names` caps how many names the ticket will BUY. Left unset it is solved
    from the book's size against the measured fee schedule (see
    suggested_name_cap), because the right number at 1,000 EGP and at 25,000 EGP
    are not the same. Pass 0 to disable the cap. Exits are never capped -- if the
    system says get out of a name, that line always prints.
    """
    from ops.trade_list import latest_trade_list
    from ops.agent_guard import assert_not_halted
    assert_not_halted()                     # no ticket while the Sentinel has halted the system
    tl = latest_trade_list()
    if tl.empty:
        raise RuntimeError("no suggestion logged yet -- run the daily job first")
    as_of = str(tl["as_of_date"].iloc[0].date())

    deposit = max(float(deposit or 0.0), 0.0)
    if max_names is None and deposit > 0:
        # size the cap to the POST-deposit book: a 10,000 EGP book affords more
        # names than a 1,000 EGP one, and planning against the old cap would
        # under-diversify the book being built
        max_names = suggested_name_cap(pt.reconcile()["portfolio_value"] + deposit)
    rec = pt.reconcile(max_names=max_names, extra_cash=deposit)
    capped = rec.get("capped")
    pos, cash = pt.holdings()

    # T+2: size BUYs against SETTLED cash only. The balance shown by the broker
    # includes sale proceeds that have not settled yet; spending them can place
    # an order the account cannot fund. pt.available_cash() applies the
    # conservative default and reports the unsettled amount so the ticket can
    # say so rather than silently sizing smaller.
    cash_state = pt.available_cash()
    # the deposit is spendable for PLANNING only -- it is not in the account yet
    avail = min(rec["cash"], cash_state["available"]) + deposit
    rows, suggested_total = [], 0.0
    blocked = []   # orders the risk screen rejected or resized

    # SELL/TRIM come from the reconciler -- those are genuine exit signals.
    for a in rec["actions"]:
        if a["action"] not in ("SELL", "TRIM"):
            continue
        price = a["price"]
        if not (price == price and price > 0):
            continue
        shares = int(round(a["shares"]))
        if shares <= 0:
            continue
        avail += shares * price
        advp, flags = _order_flags(a["ticker"], shares * price)
        rows.append([a["action"], a["ticker"], round(a["target_w"] * 100, 1),
                     round(price, 2), round(price * (1 + PRICE_TOLERANCE), 2),
                     shares, round(shares * price, 0), advp, flags, "", ""])

    # BUY sizing comes from the INTEGER ALLOCATOR, not per-name flooring. On a
    # small book every per-name gap is a fraction of a share, so flooring buys
    # nothing and strands the cash; the allocator prices idle cash as the
    # tracking error it is and buys the whole shares that most reduce it.
    if use_allocator and not rec["de_risked"]:
        from portfolio.integer_allocator import greedy_integer_allocation
        target = pd.Series({a["ticker"]: a["target_w"] for a in rec["actions"]
                            if a["target_w"] > 0}, dtype=float)
        # Apply the name cap to the TARGET before allocating, and renormalise so
        # the kept names absorb the full book rather than leaving the tail in
        # cash. Names already held are kept regardless of rank -- dropping one
        # would print a SELL purely because the cap moved, which is churn, not a
        # signal, and it would cost a fee to express.
        # already capped by reconcile() -- the single capping point
        prices = pt._latest_prices()
        cur = pos.set_index("ticker")["shares"] if not pos.empty else pd.Series(dtype=float)
        if not target.empty and avail > 0:
            alloc = greedy_integer_allocation(target, prices, avail, current_shares=cur,
                                             exclude=exclude)
            for t, n in alloc["orders"].items():
                price = float(prices.get(t, float("nan")))
                if not (price == price and price > 0):
                    continue
                shares = int(n)
                egp = shares * price
                # THE ALLOCATOR DOES NOT KNOW ABOUT FEES. It minimises tracking
                # error, so it happily spends the last 9 EGP of residual cash on
                # one share -- an order whose flat ~EGP 5 commission is 55% of
                # its own notional. reconcile() screens its own actions for this,
                # but allocator output reaches the ticket by a different path and
                # was never checked. Printing such a line is worse than leaving
                # the cash idle: idle cash costs the short rate, this costs half
                # the trade on day one.
                if _uneconomic(egp):
                    logger.info("write_ticket: dropping %s %.0f EGP -- fee is %.0f%% of the trade",
                                t, egp, _fee_share(egp) * 100)
                    continue
                # PRE-TRADE RISK SCREEN -- this can REJECT or RESIZE the order.
                # Before the 2026-08-05 audit, participation was printed as a
                # flag and the order went out anyway.
                held_val = float(cur.get(t, 0.0)) * price
                scr = screen_order(t, egp, book_egp=rec["portfolio_value"],
                                   current_value_egp=held_val, side="buy")
                if scr["verdict"] == "reject":
                    blocked.append(scr)
                    logger.warning("write_ticket: REJECTED %s -- %s", t, "; ".join(scr["reasons"]))
                    continue
                if scr["verdict"] == "resize" and scr["allowed_egp"] < egp:
                    blocked.append(scr)
                    shares = int(scr["allowed_egp"] // price)
                    if shares <= 0:
                        continue
                    egp = shares * price
                suggested_total += egp
                avail -= egp
                held_now = float(cur.get(t, 0.0))
                advp, flags = _order_flags(t, egp)
                rows.append(["ADD" if held_now > 0 else "BUY", t,
                             round(float(target.get(t, 0.0)) * 100, 1),
                             round(price, 2), round(price * (1 + PRICE_TOLERANCE), 2),
                             shares, round(egp, 0), advp, flags, "", ""])

    path = path or _ticket_path()   # named for TODAY, not the signal's as-of

    # If the ticket is open in Excel/Notepad the write raises PermissionError.
    # The daily popup calls this every morning, so a file the user forgot to
    # close must NOT crash the run -- fall back to a suffixed name and report it.
    try:
        with open(path, "a", encoding="utf-8-sig"):
            pass
    except OSError:
        stamp = pd.Timestamp.now().strftime("%H%M%S")
        day = pd.Timestamp.now().strftime("%Y-%m-%d")
        alt = os.path.join(TICKET_DIR, f"orders_{day}_{stamp}.csv")
        logger.warning("write_ticket: %s is locked (open in another program); writing %s",
                       os.path.basename(path), os.path.basename(alt))
        path = alt

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        # comment lines written raw (not via csv) so commas in them don't get
        # quoted -- the reader filters any line whose first real char is '#'.
        f.write(f"# ESIS-QUANT order ticket  as-of {as_of}  regime={rec['regime']}  "
                f"exposure={rec['exposure']*100:.0f}%\n")
        if cash_state["unsettled"] > 0:
            f.write(f"# T+2: {cash_state['unsettled']:,.2f} EGP of your balance is UNSETTLED "
                    f"sale proceeds and is NOT used for sizing ({cash_state['policy']}).\n")
        f.write(f"# your book {rec['portfolio_value']:,.0f} EGP (cash {cash:,.0f}).\n")
        if rec.get("signal_note"):
            f.write(f"# *** STALE SIGNAL: {rec['signal_note']}\n")
            f.write("#     The regime call below is NOT from the newest session. Regime timing is\n")
            f.write("#     the only thing this system claims to do, so treat this ticket as\n")
            f.write("#     unverified until the daily job runs cleanly.\n")
        if deposit > 0:
            f.write(f"# *** PLAN ONLY -- assumes a DEPOSIT of {deposit:,.2f} EGP that is NOT yet in\n")
            f.write(f"#     the account. Your RECORDED cash is still {cash:,.2f} EGP.\n")
            f.write(f"#     AFTER the money actually lands in Thndr, record the real balance:\n")
            f.write(f"#        py -3.11 -m ops.portfolio_tracker cash <your real Thndr cash>\n")
            f.write(f"#     Do that BEFORE 'apply', or the book will go cash-negative.\n")
        f.write("# FILL IN TWO COLUMNS ONLY, then run 'apply':\n")
        f.write("#   PAID_EGP = total cash that left your account for the SHARES on that row\n")
        f.write("#              (e.g. 350 for 22 shares). Exclude the fee.\n")
        f.write("#   FEE_EGP  = Thndr's commission on that one order (e.g. 2.09).\n")
        f.write("# The fill price is derived as PAID_EGP / shares -- do not compute it yourself.\n")
        f.write("# If you bought a DIFFERENT quantity than suggested, edit the `shares` column too.\n")
        f.write("# Leave a row's PAID_EGP blank and that row is skipped entirely.\n")
        f.write("# suggested sizes are the system's target for your capital -- a starting point, not advice.\n")
        if capped:
            f.write(f"# NAME CAP: showing the top {capped['kept']} names (cap {capped['cap']}). "
                    f"Thndr's flat EGP 2/order means each extra name costs ~0.4% of a 1,000 EGP\n")
            f.write(f"#   book per round trip, and measured risk stops falling past ~10 names. "
                    f"Not shown: {', '.join(capped['dropped'][:12])}"
                    f"{' ...' if len(capped['dropped']) > 12 else ''}\n")
        f.write("# flags: SIZE>..%ADV = order is large vs the name's daily volume; MOVED..% = already\n")
        f.write("#   moved sharply today (may be at its price band, may not fill); NO-VOLUME/HALT? = not trading.\n")
        f.write(f"# generated_at={pd.Timestamp.now().isoformat(timespec='seconds')}\n")
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    # prune AFTER writing: only now is the previous ticket genuinely superseded
    removed = prune_old_tickets(protect=path)
    if removed:
        logger.info("write_ticket: removed superseded ticket(s): %s", ", ".join(removed))
    return {"path": path, "removed": removed, "as_of": as_of, "rows": len(rows),
            "capped": capped, "deposit": deposit, "book": rec["portfolio_value"],
            "blocked": blocked,
            "suggested_total": suggested_total, "cash": cash,
            # cash left AFTER the suggested buys -- measured against spendable
            # cash, not against the whole book. Subtracting from portfolio_value
            # counts money already invested in existing positions as "idle",
            # which on a top-up ticket overstates it by the value of the book.
            "idle": max(0.0, avail)}


def refresh_after_change(exclude=None) -> str:
    """Rewrite the ticket because the BOOK changed (a fill was recorded, cash was
    adjusted). The suggestion is a function of your holdings, so a ticket written
    before a trade is stale the moment that trade lands -- it would tell you to
    buy something you already own.

    Best-effort by design: never raises. Recording a trade must always succeed
    even if the ticket cannot be rebuilt (no signal logged yet, file locked in
    Excel, etc.) -- the ledger is the record, the ticket is a convenience.
    """
    try:
        out = write_ticket(exclude=exclude)
        logger.info("refresh_after_change: rewrote %s (%d lines)",
                    os.path.basename(out["path"]), out["rows"])
        return out["path"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("refresh_after_change: could not rebuild the ticket: %s", exc)
        return None


def _read_ticket(path: str) -> pd.DataFrame:
    # skip comment lines (first real char '#', tolerating a stray leading quote)
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln for ln in f if not ln.lstrip().lstrip('"').startswith("#")]
    from io import StringIO
    return pd.read_csv(StringIO("".join(lines)))


def already_decided(as_of: str) -> dict:
    """Has this as-of suggestion ALREADY been booked as a real position change?

    THE FAILURE THIS PREVENTS, observed live on 2026-08-04. There are two ways to
    turn one day's suggestion into holdings -- `portfolio_tracker accept` (books
    the model book at reference prices, no fees) and `order_ticket apply` (books
    the fills you actually got). They are alternatives, never both. Running both
    for the same as-of date booked the same intent twice: 13 phantom positions at
    reference prices on top of 8 real ones, 1,804 EGP of buys against a 1,000 EGP
    account, and every downstream number wrong until the ledger was rebuilt by
    hand.

    Nothing prevented it, because each command was individually valid. This makes
    the conflict explicit at the only point where it can still be caught cheaply.
    """
    if not as_of:
        return {}
    try:
        h = pt.decision_history()
    except Exception:  # noqa: BLE001
        return {}
    if h.empty or "as_of_date" not in h:
        return {}
    prior = h[(h["as_of_date"].astype(str) == str(as_of))
              & (h["decision"].isin(["ACCEPTED", "OWN"]))]
    if prior.empty:
        return {}
    last = prior.iloc[-1]
    return {"decision": str(last["decision"]), "decided_at": str(last["decided_at"]),
            "note": str(last.get("note", ""))}


def apply_ticket(path: str = None, as_of: str = None, force: bool = False) -> dict:
    """Read the filled ticket and record the real trades in the DB. Only rows
    with a positive PAID_EGP are saved; the fill price is derived as
    PAID_EGP / shares and FEE_EGP is charged to cash. Logs the day's decision as
    OWN (you executed your own fills, guided by the ticket)."""
    if path is None:
        from ops.trade_list import latest_trade_list
        tl = latest_trade_list()
        as_of = as_of or (str(tl["as_of_date"].iloc[0].date()) if not tl.empty else None)
        # find the newest ticket on disk -- the filename is the generation date,
        # which need not match the signal's as-of date
        path = latest_ticket_path() or _ticket_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"no ticket at {path} -- run 'write' first")
    prior = already_decided(as_of)
    if prior and not force:
        # Refuse rather than warn. A warning printed into a scrolling popup is
        # not a control -- the double-book already happened once that way.
        return {"applied": [], "skipped": [], "path": path, "n": 0, "fee_check": {},
                "blocked": prior,
                "reason": (f"the {as_of} suggestion was already booked as "
                           f"{prior['decision']} at {prior['decided_at']}. Applying "
                           f"again would double-count it. Reverse that decision "
                           f"first, or pass force=True if you are certain these "
                           f"are additional real fills.")}
    df = _read_ticket(path)
    ticket_generated_at = None
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                if line.startswith("# generated_at="):
                    ticket_generated_at = line.split("=", 1)[1].strip()
                    break
    except OSError:
        pass
    applied, skipped = [], []
    for r in df.itertuples():
        # You type the TOTAL you paid and the fee; the per-share price is
        # derived. Shares come from the `shares` column, which you may edit if
        # you traded a different quantity than suggested.
        paid = _num(getattr(r, "PAID_EGP", None))
        fee = _num(getattr(r, "FEE_EGP", None))
        fs = _num(getattr(r, "shares", None))
        if paid <= 0:
            continue                      # blank row = not traded, by design
        if fs <= 0:
            # A total with no quantity cannot be turned into a price. Refusing is
            # correct: guessing the quantity would silently invent a cost basis.
            skipped.append({"ticker": str(r.ticker),
                            "why": "PAID_EGP given but `shares` is blank or zero"})
            continue
        fp = paid / fs
        side = "sell" if str(r.action).upper() in ("SELL", "TRIM") else "buy"
        # A blank fee is filled from the calibrated schedule rather than treated
        # as zero. Zero is never true -- the broker always charges -- and booking
        # it would understate the cost basis and overstate cash, which is the
        # bug that made the DB drift from the account in the first place. The
        # schedule reproduces measured contract notes to within ~0.04 EGP, so
        # this is an accurate stand-in, but it IS modelled and is reported as such.
        fee_modelled = False
        if fee <= 0:
            try:
                from backtest.fees import one_side_cost
                fee = one_side_cost(paid, spread_per_side=0.0)["total_egp"]
                fee_modelled = True
            except Exception:  # noqa: BLE001
                fee = 0.0
        pt.record_trade(side, str(r.ticker), fs, fp, fee=fee)

        # Phase 0.2: measure the fill against the SIGNAL price the optimizer
        # sized on (ref_price = close of the as-of session). Without this the
        # modelled slippage in backtest/costs.py can never be checked against
        # what the account actually experiences. Best-effort: a failure to
        # record the measurement must never lose the trade itself.
        try:
            from ops.execution_quality import record_fill
            sig_px = float(getattr(r, "ref_price", float("nan")))
            record_fill(str(r.ticker), side, fs, sig_px, fp,
                        ticket_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                        as_of_date=as_of, ticket_generated_at=ticket_generated_at,
                        filled_at=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("apply_ticket: could not record execution quality: %s", exc)

        applied.append({"side": side, "ticker": str(r.ticker), "shares": fs,
                        "price": fp, "paid": paid, "fee": fee,
                        "fee_modelled": fee_modelled})
    if applied:
        pt.log_decision("OWN", note=f"executed {len(applied)} fills from order ticket {os.path.basename(path)}",
                        as_of=as_of)
        refresh_after_change()   # the book moved -> the old ticket is now stale
    return {"applied": applied, "skipped": skipped, "path": path, "n": len(applied),
            "fee_check": fee_model_check(applied)}


def _num(v) -> float:
    """Parse a user-typed cell. Tolerates blanks, 'nan', thousands separators and
    a stray currency word, because this file is filled in by hand. Anything that
    is not a number reads as 0, which means 'not traded' -- never as a guess."""
    s = str(v if v is not None else "").strip().replace(",", "")
    for junk in ("EGP", "egp", "L.E.", "LE"):
        s = s.replace(junk, "")
    s = s.strip()
    if s in ("", "nan", "None", "-"):
        return 0.0
    try:
        return max(float(s), 0.0)
    except ValueError:
        return 0.0


def fee_model_check(applied: list) -> dict:
    """Compare the fee you were ACTUALLY charged against what backtest/fees.py
    predicts for the same orders.

    This is the first time the cost model can be checked against reality rather
    than against its own assumptions. A persistent gap here means every backtest
    number in the system is biased, so it is worth surfacing on day one instead
    of after twenty fills.
    """
    rows = [a for a in applied if a.get("fee", 0) > 0 and not a.get("fee_modelled")]
    if not rows:
        return {}
    from backtest.fees import one_side_cost
    today = pd.Timestamp.now()
    actual = sum(a["fee"] for a in rows)
    predicted = sum(one_side_cost(a["paid"], as_of=today,
                                  spread_per_side=0.0)["total_egp"] for a in rows)
    return {"n": len(rows), "actual_egp": actual, "predicted_egp": predicted,
            "error_egp": actual - predicted,
            "error_pct": ((actual - predicted) / predicted) if predicted > 0 else float("nan")}


def _main(argv=None):
    import sys
    logging.basicConfig(level=logging.WARNING)
    cmd = (argv or sys.argv[1:] or ["write"])[0]
    if cmd == "write":
        rest = (argv or sys.argv[1:])[1:]
        dep = 0.0
        if "--deposit" in rest:
            i = rest.index("--deposit")
            dep = float(rest[i + 1])
            rest = rest[:i] + rest[i + 2:]
        out = write_ticket(max_names=int(rest[0]) if rest else None, deposit=dep)
        if out["deposit"]:
            print(f"  PLAN assuming a {out['deposit']:,.2f} EGP deposit -> book "
                  f"{out['book']:,.0f} EGP. Not yet recorded as cash.")
        print(f"wrote order ticket: {out['path']}")
        print(f"  as-of {out['as_of']}  |  {out['rows']} lines  |  suggested buys "
              f"{out['suggested_total']:,.0f} EGP  |  idle cash ~{out['idle']:,.0f} EGP")
        if out["capped"]:
            c = out["capped"]
            print(f"  name cap {c['cap']}: {len(c['dropped'])} lower-weight name(s) not shown "
                  f"(each extra name is another flat EGP 2 per side).")
        print("  open it and fill TWO columns per row you traded:")
        print("     PAID_EGP = total paid for the shares (exclude fee)")
        print("     FEE_EGP  = Thndr's commission on that order")
        print("  then:  py -3.11 -m ops.order_ticket apply")
    elif cmd == "apply":
        rest = (argv or sys.argv[1:])[1:]
        out = apply_ticket(force="--force" in rest)
        if out.get("blocked"):
            print("REFUSED -- " + out["reason"])
            return
        for s in out.get("skipped", []):
            print(f"  SKIPPED {s['ticker']}: {s['why']}")
        if out["n"] == 0:
            print("no filled rows found -- nothing saved. (fill PAID_EGP + FEE_EGP first)")
        else:
            print(f"saved {out['n']} trade(s) to the DB:")
            for a in out["applied"]:
                tag = " (fee modelled)" if a.get("fee_modelled") else ""
                print(f"   {a['side'].upper():4s} {a['ticker']:6s} {a['shares']:g} sh  "
                      f"paid {a['paid']:,.2f} + fee {a['fee']:,.2f}  -> "
                      f"{a['price']:.4f}/sh{tag}")
            fc = out.get("fee_check") or {}
            if fc:
                print(f"  FEE MODEL CHECK: you paid {fc['actual_egp']:,.2f} EGP, the model "
                      f"predicted {fc['predicted_egp']:,.2f} ({fc['error_pct']*100:+.0f}%).")
                if abs(fc.get("error_pct", 0)) > 0.20:
                    print("    >20% off — backtest/fees.py needs recalibrating to your real schedule.")
            pos, cash = pt.holdings()
            print(f"  your book now: {len(pos)} positions, cash {cash:,.0f} EGP")
    else:
        print(f"unknown command '{cmd}' -- use 'write' or 'apply'")


if __name__ == "__main__":
    _main()
