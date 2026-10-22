"""
Interactive portfolio tracker + reconciler.

The workflow this implements:
  1. The system proposes a target book (ops.trade_list.generate_trade_list).
  2. YOU tell it what you actually did -- "I bought 100 COMI at 80.5",
     "I sold 50 SWDY at 41" -- and it keeps your real holdings + cash.
  3. It compares YOUR holdings to the system's CURRENT target and tells you,
     per stock, whether to BUY / ADD / HOLD / TRIM / SELL to align, with the
     approximate EGP amount and share count for each move.

The advice is a MECHANICAL reconciliation of your positions against the
system's own target book -- it is the system talking, not a human
recommendation. If the system is de-risked (BEAR regime -> 100% cash target),
the reconciler says to sell down to cash. Nothing here places orders; it tells
you what orders would align you with the model.

CLI:
  py -3.11 -m ops.portfolio_tracker cash 100000
  py -3.11 -m ops.portfolio_tracker buy COMI 100 80.5
  py -3.11 -m ops.portfolio_tracker sell COMI 40 82
  py -3.11 -m ops.portfolio_tracker show
  py -3.11 -m ops.portfolio_tracker advice
"""

import argparse
import logging

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.ops.tracker")

REBALANCE_THRESHOLD = 0.02   # ignore weight gaps smaller than this (avoid churn)

# A weight-gap threshold alone cannot stop churn, because it is scale-free while
# the fee is not: a 4% drift on a 1,000 EGP book is a 42 EGP trade carrying a
# flat ~EGP 5 commission -- 12% of the trade. The gap test says "rebalance", the
# arithmetic says the correction costs more than the drift. So a second,
# ABSOLUTE test is applied: a suggested trade must be able to clear its own fee.
#
# 2% is deliberately loose. It still permits genuinely useful trades (270 EGP and
# up at the measured schedule) while blocking the small ones that can only lose
# money. Tightening it to 1% would demand 591 EGP per order, which on a 1,000 EGP
# book means no trade is ever economic -- true, but it would silence the system
# entirely rather than informing the operator, so the looser bound is used and
# the fee share is reported alongside every action.
MAX_FEE_SHARE_OF_TRADE = 0.02


# --------------------------------------------------------------------------
# state: holdings + cash
# --------------------------------------------------------------------------
def set_cash(amount: float):
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("INSERT INTO user_account (id, cash, updated_at) VALUES (1, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET cash=excluded.cash, updated_at=excluded.updated_at",
                     (float(amount), now))
    logger.info("set_cash: %.2f EGP", amount)


def record_cash_flow(amount: float, date=None, note: str = "") -> None:
    """Record money moving INTO (+) or OUT OF (-) the account.

    Updates the cash balance and the cash_flows table, which the book guard uses
    to separate deposits and withdrawals from profit and loss.
    """
    from risk.kill_switch import ensure_cash_flows_table
    d = pd.Timestamp(date or pd.Timestamp.now()).strftime("%Y-%m-%d")
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        ensure_cash_flows_table(conn)
        conn.execute("INSERT INTO cash_flows (date, amount, note, recorded_at) VALUES (?,?,?,?)",
                     (d, float(amount), note or None, now))
        acc = conn.execute("SELECT cash FROM user_account WHERE id=1").fetchone()
        cash = (float(acc["cash"]) if acc else 0.0) + float(amount)
        conn.execute("INSERT INTO user_account (id, cash, updated_at) VALUES (1, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET cash=excluded.cash, updated_at=excluded.updated_at",
                     (cash, now))
    logger.info("record_cash_flow: %+.2f EGP on %s (%s)", amount, d, note)


def get_cash() -> float:
    with get_connection() as conn:
        row = conn.execute("SELECT cash FROM user_account WHERE id=1").fetchone()
    return float(row["cash"]) if row else 0.0


def _ensure_fee_column(conn):
    """Add ledger.fee in place for books created before fees were tracked.
    Existing rows get 0.0, which is the honest value: their fee was never
    recorded, so it must not be guessed retrospectively."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_trade_ledger)")}
    if "fee" not in cols:
        conn.execute("ALTER TABLE user_trade_ledger ADD COLUMN fee REAL DEFAULT 0.0")
    if "price_source" not in cols:
        conn.execute("ALTER TABLE user_trade_ledger ADD COLUMN price_source TEXT DEFAULT 'broker'")


def record_trade(action: str, ticker: str, shares: float, price: float, fee: float = 0.0,
                 price_source: str = "broker"):
    """Apply a buy/sell to your holdings + cash and append to the ledger.
    Buying spends cash; selling adds cash. Average cost is updated on buys.

    `fee` is the REAL commission actually charged by the broker, in EGP, and it
    always LEAVES the account: a buy costs shares*price + fee, a sell credits
    shares*price - fee. This used to be omitted entirely, which made the DB cash
    balance drift above the real one by the accumulated commission -- roughly
    EGP 2 per order, which on a small book is the dominant cost. Fees also enter
    the average cost on a buy, because the cost basis of a position is what it
    actually cost to acquire, not the quoted price.

    `price_source` records whether `price` is a REAL fill ("broker") or an
    estimate standing in for one ("assumed"). Estimates enter the ledger when a
    trade is reported without its execution price and the last mark is used
    instead -- four exits on 2026-09-04 were booked that way, carrying an
    estimated -179.59 of realised P&L that was then reported alongside genuine
    figures with nothing distinguishing them. Realised P&L is the one number a
    backtest cannot flatter, so it is the last place an unlabelled guess should
    sit.
    """
    action = action.lower()
    if action not in ("buy", "sell"):
        raise ValueError("action must be 'buy' or 'sell'")
    shares, price = float(shares), float(price)
    fee = max(float(fee or 0.0), 0.0)
    if shares <= 0:
        return   # no-op: don't log zero-share "trades" to the ledger
    now = pd.Timestamp.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        row = conn.execute("SELECT shares, avg_cost FROM user_positions WHERE ticker=?", (ticker,)).fetchone()
        cur_sh = float(row["shares"]) if row else 0.0
        cur_cost = float(row["avg_cost"]) if row and row["avg_cost"] is not None else 0.0

        if action == "buy":
            new_sh = cur_sh + shares
            # fee is part of what the shares cost you, so it belongs in the basis
            new_cost = ((cur_sh * cur_cost + shares * price + fee) / new_sh
                        if new_sh > 0 else price)
            cash_delta = -(shares * price + fee)
        else:  # sell
            if shares > cur_sh + 1e-9:
                raise ValueError(f"cannot sell {shares} {ticker}: you hold {cur_sh}")
            new_sh = cur_sh - shares
            new_cost = cur_cost  # realized P&L not tracked here; cost basis unchanged
            cash_delta = shares * price - fee

        if new_sh <= 1e-9:
            conn.execute("DELETE FROM user_positions WHERE ticker=?", (ticker,))
        else:
            conn.execute("INSERT INTO user_positions (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?) "
                         "ON CONFLICT(ticker) DO UPDATE SET shares=excluded.shares, avg_cost=excluded.avg_cost, "
                         "updated_at=excluded.updated_at", (ticker, new_sh, new_cost, now))
        _ensure_fee_column(conn)
        conn.execute("INSERT INTO user_trade_ledger (traded_at, action, ticker, shares, price, fee, "
                     "price_source) VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (now, action, ticker, shares, price, fee, price_source))
        # A SELL credits the balance now but settles T+2. Record it as pending so
        # available_cash() can exclude it; buys settle the other way and are
        # already paid for out of settled funds.
        if action == "sell":
            trade_day = pd.Timestamp(now).normalize()
            conn.execute("INSERT INTO pending_settlements (trade_date, settles_on, ticker, "
                         "side, amount, settled) VALUES (?, ?, ?, 'sell', ?, 0)",
                         (str(trade_day.date()), _settlement_date(trade_day), ticker,
                          float(shares * price)))
        # cash
        acc = conn.execute("SELECT cash FROM user_account WHERE id=1").fetchone()
        cash = (float(acc["cash"]) if acc else 0.0) + cash_delta
        conn.execute("INSERT INTO user_account (id, cash, updated_at) VALUES (1, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET cash=excluded.cash, updated_at=excluded.updated_at", (cash, now))
    logger.info("record_trade: %s %s %s @ %.4f", action, shares, ticker, price)


def _settlement_date(trade_date, days: int = None) -> str:
    """T+N in TRADING sessions, not calendar days -- a Thursday sale on EGX
    settles the following Monday, not Saturday. Uses the same session list the
    rest of the system trusts (data.calendar), falling back to calendar days
    only if no session data is available."""
    import config as _cfg
    days = _cfg.SETTLEMENT_DAYS if days is None else days
    td = pd.Timestamp(trade_date).normalize()
    try:
        from data.calendar import valid_trading_days
        sessions = valid_trading_days()
        future = sessions[sessions > td]
        if len(future) >= days:
            return str(future[days - 1].date())
    except Exception:  # noqa: BLE001
        pass
    # FALLBACK = EGX WEEK, not calendar days (Stage 0). Future sessions never
    # exist in the price data yet, so this branch ran for EVERY new trade and
    # settled a Thursday sale on Saturday. EGX trades Sunday-Thursday; Friday and
    # Saturday are skipped. Public holidays are NOT known here, so a settlement
    # across a holiday can still read one session early.
    d, left = td, days
    while left > 0:
        d += pd.Timedelta(days=1)
        if d.dayofweek not in (4, 5):        # Fri=4, Sat=5
            left -= 1
    return str(d.date())


def settle_due(as_of=None) -> int:
    """Mark matured settlements as settled. Idempotent; safe to call often."""
    as_of = str(pd.Timestamp(as_of or pd.Timestamp.now()).normalize().date())
    with get_connection() as conn:
        cur = conn.execute("UPDATE pending_settlements SET settled=1 "
                           "WHERE settled=0 AND settles_on <= ?", (as_of,))
        return int(cur.rowcount or 0)


def unsettled_cash(as_of=None) -> float:
    """Sale proceeds credited to the balance but not yet settled."""
    settle_due(as_of)
    with get_connection() as conn:
        row = conn.execute("SELECT COALESCE(SUM(amount),0) FROM pending_settlements "
                           "WHERE settled=0").fetchone()
    return float(row[0] or 0.0)


def available_cash(as_of=None) -> dict:
    """Cash the allocator may actually SPEND today.

    EGX settles T+2. The tracker credits sale proceeds to the balance
    immediately (that is what the broker shows), but those funds are not
    settled, so treating them as spendable can size a buy against money the
    account does not have yet. Default is conservative -- unsettled proceeds
    are excluded -- and switchable via config.ALLOW_UNSETTLED_REINVESTMENT
    once the broker's actual reinvestment policy is confirmed.
    """
    import config as _cfg
    total = get_cash()
    unsettled = unsettled_cash(as_of)
    if getattr(_cfg, "ALLOW_UNSETTLED_REINVESTMENT", False):
        return {"total": total, "unsettled": unsettled, "available": total,
                "policy": "unsettled reinvestment ALLOWED (config)"}
    return {"total": total, "unsettled": unsettled,
            "available": max(0.0, total - unsettled),
            "policy": f"unsettled proceeds NOT spendable (T+{_cfg.SETTLEMENT_DAYS})"}


def rebuild_cost_basis() -> pd.DataFrame:
    """Recompute every position's average cost by replaying the whole ledger.

    Needed whenever a recorded fill price is corrected after the fact (the
    tracker books trades at the last close when the real broker fill isn't known
    yet, then the fill is supplied later). Replaying is the only safe way to fix
    the basis: avg_cost is path-dependent, so patching one trade's price cannot
    be undone arithmetically without re-running the sequence.

    Share counts and cash are NOT touched -- those are already reconciled
    against the broker; only avg_cost is rewritten.
    """
    with get_connection() as conn:
        led = pd.read_sql("SELECT action, ticker, shares, price FROM user_trade_ledger ORDER BY id", conn)
    basis = {}
    for r in led.itertuples():
        sh, cost = basis.get(r.ticker, (0.0, 0.0))
        if str(r.action).lower() == "buy":
            new_sh = sh + r.shares
            cost = (sh * cost + r.shares * r.price) / new_sh if new_sh > 0 else r.price
            sh = new_sh
        else:                       # sells reduce the position, cost basis unchanged
            sh = max(0.0, sh - r.shares)
        basis[r.ticker] = (sh, cost)

    rows = []
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        for t, (sh, cost) in basis.items():
            cur = conn.execute("SELECT shares, avg_cost FROM user_positions WHERE ticker=?", (t,)).fetchone()
            if not cur:
                continue            # position closed / not held -- nothing to update
            conn.execute("UPDATE user_positions SET avg_cost=?, updated_at=? WHERE ticker=?",
                         (cost, now, t))
            rows.append({"ticker": t, "shares": float(cur["shares"]),
                         "old_avg_cost": float(cur["avg_cost"]), "new_avg_cost": float(cost)})
    logger.info("rebuild_cost_basis: rewrote %d position(s) from the ledger", len(rows))
    return pd.DataFrame(rows)


def _latest_prices() -> pd.Series:
    with get_connection() as conn:
        df = pd.read_sql("SELECT ticker, adjusted_close, date FROM daily_ohlcv d "
                         "WHERE date = (SELECT MAX(date) FROM daily_ohlcv d2 WHERE d2.ticker=d.ticker)",
                         conn)
    return df.set_index("ticker")["adjusted_close"]


def holdings() -> pd.DataFrame:
    """Current holdings valued at the latest available price, with weights over
    total portfolio value (positions + cash)."""
    with get_connection() as conn:
        pos = pd.read_sql("SELECT ticker, shares, avg_cost FROM user_positions", conn)
    prices = _latest_prices()
    cash = get_cash()
    if pos.empty:
        return pd.DataFrame(columns=["ticker", "shares", "price", "value", "weight", "avg_cost", "unrealized_pct"]), cash
    pos["price"] = pos["ticker"].map(prices)
    pos["value"] = pos["shares"] * pos["price"]
    total = pos["value"].sum() + cash
    pos["weight"] = pos["value"] / total if total > 0 else 0.0
    pos["unrealized_pct"] = (pos["price"] / pos["avg_cost"] - 1).where(pos["avg_cost"] > 0)
    return pos.sort_values("weight", ascending=False), cash


def reconcile(max_names: int = None, extra_cash: float = 0.0) -> dict:
    """Compare your holdings to the system's CURRENT target book and produce
    per-ticker actions (BUY / ADD / HOLD / TRIM / SELL / EXIT). Returns a dict
    with the actions list and context (regime, exposure, portfolio value).

    THIS IS THE SINGLE PLACE THE NAME CAP IS APPLIED. Callers that want a
    different cap pass `max_names` here rather than trimming the result, so the
    reported `capped["dropped"]` list is always complete relative to the full
    target. Capping twice produced a truthful-looking but partial list.

    `extra_cash` sizes the plan against a deposit that HAS NOT LANDED YET. It is
    used for planning a scale-in and deliberately does NOT touch the stored cash
    balance: recording money that is not in the account is exactly how the book
    stops matching the broker. Current weights are computed against the
    post-deposit total, because that is what the holdings will be worth a
    fraction of once the money arrives -- otherwise an already-invested book
    looks fully weighted and the plan comes back empty.
    """
    from ops.trade_list import latest_trade_list, generate_trade_list, target_is_stale

    # RECOMPUTE WHENEVER THE DATA HAS MOVED ON, not only when nothing is stored.
    # Serving the previous session's target with today's date on it re-issues a
    # stale regime call, and regime timing is the only thing this system claims
    # to do. Freshness is judged against the newest session that has prices, so
    # the target always reflects every session the model could have seen.
    tl = latest_trade_list()
    stale = target_is_stale(tl)
    signal_note = None
    if tl.empty or stale:
        try:
            res = generate_trade_list()
            target = res["weights"]
            regime, exposure = res["regime"], res["exposure"]
            if stale and stale.get("stored") is not None:
                logger.info("reconcile: target regenerated (%s -> %s)",
                            stale["stored"].date(), stale["latest"].date())
            tl = None                      # regenerated; the stored copy is superseded
        except Exception as exc:  # noqa: BLE001
            # Falling back to the stale target is survivable; doing it SILENTLY
            # is not. The caller must be able to say the signal is old and why.
            if tl is None or tl.empty:
                raise
            signal_note = (f"could not regenerate the target ({exc}); still using "
                           f"{pd.Timestamp(tl['as_of_date'].iloc[0]).date()}")
            logger.warning("reconcile: %s", signal_note)
    if tl is not None and not tl.empty:
        hold = tl[tl["ticker"] != "__CASH__"]
        target = hold.set_index("ticker")["weight"] if not hold.empty else pd.Series(dtype=float)
        regime = tl["regime"].iloc[0]
        exposure = float(tl["exposure"].iloc[0])

    pos, cash = holdings()
    prices = _latest_prices()
    extra_cash = max(float(extra_cash or 0.0), 0.0)
    invested = float(pos["value"].sum()) if not pos.empty else 0.0
    total = invested + cash + extra_cash
    # recompute weights against the post-deposit total rather than reusing the
    # stored ones, which are a fraction of the CURRENT book
    cur_w = ((pos.set_index("ticker")["value"] / total) if (not pos.empty and total > 0)
             else pd.Series(dtype=float))

    # Apply the SAME name cap the ticket applies. Comparing an 8-name book
    # against the uncapped 24-name target made every held name read overweight
    # (TRIM) and every excluded name read BUY -- a permanent false rebalance
    # signal that, if followed, rebuilds the book the cap exists to prevent.
    # Lazy import: order_ticket imports this module at load time.
    capped = None
    try:
        from ops.order_ticket import cap_target_names
        target, capped = cap_target_names(target, held=set(cur_w.index),
                                          max_names=max_names, capital_egp=total)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconcile: could not apply the name cap: %s", exc)

    names = sorted(set(cur_w.index) | set(target.index))
    _holds = deliberate_holds()
    actions = []
    for t in names:
        cw = float(cur_w.get(t, 0.0))
        tw = float(target.get(t, 0.0))
        gap = tw - cw
        price = float(prices.get(t, float("nan")))
        egp = abs(gap) * total
        shares = egp / price if price and price == price and price > 0 else float("nan")

        if tw == 0 and cw > 0:
            act = "SELL"   # not in target -> exit fully
        elif cw == 0 and tw > 0:
            act = "BUY"    # new position
        elif abs(gap) < REBALANCE_THRESHOLD:
            act = "HOLD"
        elif gap > 0:
            act = "ADD"
        else:
            act = "TRIM"
        # IS THIS TRADE WORTH ITS OWN FEE? A flat ~EGP 5 per order means a 42 EGP
        # rebalance pays 12% in commission to correct a few percent of drift --
        # value destroyed with certainty to chase a benefit that is not even
        # measurable. Actions that cannot clear their own cost are downgraded to
        # HOLD and marked, rather than being printed as if they were free.
        # NaN, not inf, when there is nothing to price: a name already HOLD has no
        # trade to be uneconomic about, and inf would flag every one of them.
        fee_pct = float("nan")
        if act != "HOLD" and egp > 0:
            try:
                from backtest.fees import one_side_cost
                fee_pct = one_side_cost(egp)["total_egp"] / egp
            except Exception:  # noqa: BLE001
                fee_pct = float("nan")
        uneconomic = fee_pct == fee_pct and fee_pct > MAX_FEE_SHARE_OF_TRADE
        # A position the operator has DECIDED to keep is not re-proposed for sale
        # every morning. The gap is real and stays visible in `deliberate` and in
        # the risk diagnostic -- it simply stops being re-asked, because a
        # question already answered crowds out the ones that are not.
        held_by_choice = t in _holds and act in ("SELL", "TRIM")
        if uneconomic or held_by_choice:
            act = "HOLD"
        actions.append({"ticker": t, "action": act, "current_w": cw, "target_w": tw,
                        "gap": gap, "egp": egp if act != "HOLD" else 0.0,
                        "shares": shares if act != "HOLD" else 0.0, "price": price,
                        "fee_share": fee_pct, "uneconomic": bool(uneconomic),
                        "deliberate": bool(t in _holds),
                        "hold_reason": _holds.get(t, "")})

    order = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4}
    actions.sort(key=lambda a: (order[a["action"]], -a["egp"]))
    return {"regime": regime, "exposure": exposure, "portfolio_value": total,
            "cash": cash, "extra_cash": extra_cash, "invested": invested,
            "actions": actions, "capped": capped, "signal_note": signal_note,
            "deliberate_holds": _holds,
            "de_risked": exposure <= 1e-6}


def format_advice(rec: dict) -> str:
    lines = [f"SYSTEM TARGET — regime={rec['regime']}  exposure={rec['exposure']*100:.0f}%"
             f"  |  your portfolio ~{rec['portfolio_value']:,.0f} EGP (cash {rec['cash']:,.0f})"]
    if rec["de_risked"]:
        lines.append(">> System is DE-RISKED (BEAR): target is 100% CASH. Suggested action: SELL all positions.")
    lines.append("-" * 66)
    verb = {"SELL": "SELL (exit)", "TRIM": "TRIM", "BUY": "BUY", "ADD": "ADD", "HOLD": "HOLD/keep"}
    any_action = False
    for a in rec["actions"]:
        if a["action"] == "HOLD":
            lines.append(f"  HOLD/keep {a['ticker']:6s}  (you {a['current_w']*100:4.1f}% ~ target {a['target_w']*100:4.1f}%)")
        else:
            any_action = True
            sh = f"~{a['shares']:.0f} sh" if a["shares"] == a["shares"] else "?"
            lines.append(f"  {verb[a['action']]:11s} {a['ticker']:6s}  you {a['current_w']*100:4.1f}% -> "
                         f"target {a['target_w']*100:4.1f}%   (~{a['egp']:,.0f} EGP, {sh} @ {a['price']:.2f})")
    if not any_action and not rec["de_risked"]:
        lines.append("  You are already aligned with the system's target — nothing to do.")
    lines.append("-" * 66)
    lines.append("This is a mechanical reconciliation vs the system's target book, not advice. "
                 "The system's edge is unproven live — see ops.status.")
    return "\n".join(lines)


def log_decision(decision: str, note: str = "", run_id: str = None, as_of=None):
    """Record what you decided about the current suggestion:
    ACCEPTED (took the system's book), DECLINED (ignored it), or OWN (you're
    trading on your own view). Stored so we can later compare the two paths."""
    decision = decision.upper()
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    if run_id is None or as_of is None:
        from ops.trade_list import latest_trade_list
        tl = latest_trade_list()
        if not tl.empty:
            run_id = run_id or tl["run_id"].iloc[0]
            as_of = as_of or str(tl["as_of_date"].iloc[0].date())
    with get_connection() as conn:
        conn.execute("INSERT INTO user_decisions (decided_at, run_id, as_of_date, decision, note) "
                     "VALUES (?, ?, ?, ?, ?)", (now, run_id, str(as_of) if as_of else None, decision, note))
    logger.info("log_decision: %s (%s)", decision, note or "")


def accept_suggestion(dry_run: bool = False) -> dict:
    """TAKE THE SYSTEM'S ADVICE: rebalance your paper portfolio to match the
    system's current target book exactly -- selling what's not in it and
    buying/adjusting the rest to target weights at current prices. Records the
    trades and logs the decision as ACCEPTED. If the system is de-risked
    (BEAR), this sells everything to cash.
    """
    rec = reconcile()
    executed = []
    avail = rec["cash"]      # running cash, so buys never exceed what's on hand
    # reconcile already orders SELL/TRIM before BUY/ADD so cash is freed first.
    # Sells/trims execute at their computed size; BUYS are floored to whole
    # shares AND capped to the cash available at that moment, so an accept can
    # never overspend the book (the old round() drove cash negative on a small
    # book). Zero-share moves are skipped, not logged.
    for a in rec["actions"]:
        if a["action"] == "HOLD" or a["shares"] != a["shares"] or a["shares"] <= 0:
            continue
        price = a["price"]
        if not (price == price and price > 0):
            continue
        if a["action"] in ("SELL", "TRIM"):
            side, shares = "sell", int(round(a["shares"]))
            avail += shares * price
        else:
            side = "buy"
            shares = min(int(a["shares"]), int(avail // price))   # floor + affordable
            if shares > 0:
                avail -= shares * price
        if shares <= 0:
            continue
        executed.append({**a, "side": side, "shares": shares})
        if not dry_run:
            record_trade(side, a["ticker"], shares, price)
    if not dry_run:
        log_decision("ACCEPTED", note=f"rebalanced to system target ({rec['regime']}, "
                                       f"{rec['exposure']*100:.0f}% exposure)")
    return {"executed": executed, "regime": rec["regime"], "exposure": rec["exposure"],
            "de_risked": rec["de_risked"]}


def decision_history() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT decided_at, as_of_date, decision, note FROM user_decisions "
                           "ORDER BY decided_at DESC", conn)


def pending_decisions() -> pd.DataFrame:
    """Suggestions the system logged that you never answered (missed/forgot the
    popup). One row per as-of date with no decision recorded, newest first.
    Nothing is lost when you skip a day -- the system still logs its book and
    snapshots yours; this just lists the gaps so you can see them."""
    with get_connection() as conn:
        return pd.read_sql(
            "SELECT DISTINCT as_of_date, regime, exposure FROM trade_list_log t "
            "WHERE NOT EXISTS (SELECT 1 FROM user_decisions d "
            "WHERE d.as_of_date = t.as_of_date) ORDER BY as_of_date DESC",
            conn, parse_dates=["as_of_date"])


def catchup(decision: str = "MISSED", note: str = "no action taken (missed popup)") -> int:
    """Backfill every unanswered suggestion with one decision. Default 'MISSED'
    means you did nothing that day -- your book stayed as it was, which is the
    honest record of a forgotten popup. Pass decision='DECLINED' if you want
    those days counted as deliberate skips instead. Returns how many were filled."""
    pend = pending_decisions()
    for r in pend.itertuples():
        log_decision(decision, note=note, as_of=str(r.as_of_date.date()))
    return int(len(pend))


def set_invested_capital(amount: float, note: str = "") -> None:
    """Total money you have put into the account, net of withdrawals.

    WHY. The first snapshot (7,555 EGP in August) was not the money invested:
    more went in later without being recorded, so "since start" read +63% on an
    account that was actually down about 100 EGP. Profit is measured against
    this figure when it is set. Deposits recorded later via record_cash_flow are
    added on top.
    """
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS account_capital (as_of TEXT PRIMARY KEY, "
                     "invested REAL NOT NULL, note TEXT)")
        conn.execute("INSERT OR REPLACE INTO account_capital (as_of, invested, note) VALUES (?,?,?)",
                     (now, float(amount), note or None))


def invested_capital():
    """Latest invested-capital figure plus any deposits/withdrawals recorded after it."""
    with get_connection() as conn:
        try:
            r = conn.execute("SELECT as_of, invested FROM account_capital ORDER BY as_of DESC LIMIT 1").fetchone()
        except Exception:  # noqa: BLE001
            return None
        if not r:
            return None
        try:
            later = conn.execute("SELECT COALESCE(SUM(amount), 0) FROM cash_flows WHERE recorded_at > ?",
                                 (r["as_of"],)).fetchone()[0]
        except Exception:  # noqa: BLE001
            later = 0.0
    return float(r["invested"]) + float(later or 0.0)


def record_daily_total(total_egp: float, note: str = "") -> dict:
    """Record the day's ENDING total book value, typed in by hand.

    Exists because the broker is the source of truth for what you actually have,
    and the system's own mark (positions x last close) drifts from it: fills
    differ from closes, and Yahoo's EGX close arrives a day late. One number a
    day, entered from the app, gives an exact NAV series to measure against --
    independent of every assumption in the tracker.

    Returns the running performance so far so the number means something the
    moment you enter it.
    """
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        first = conn.execute("SELECT user_value FROM user_daily_snapshot "
                             "WHERE user_value > 0 ORDER BY date LIMIT 1").fetchone()
        prev = conn.execute("SELECT date, user_value FROM user_daily_snapshot "
                            "WHERE user_value > 0 AND date < ? ORDER BY date DESC LIMIT 1",
                            (today,)).fetchone()
        conn.execute(
            "INSERT INTO user_daily_snapshot (date, user_value, cash, n_positions, decision, "
            "regime, exposure, system_nav) VALUES (?, ?, NULL, NULL, ?, NULL, NULL, NULL) "
            "ON CONFLICT(date) DO UPDATE SET user_value=excluded.user_value, "
            "decision=COALESCE(excluded.decision, user_daily_snapshot.decision)",
            (today, float(total_egp), note or None))
    cap = invested_capital()
    start = cap if cap else (float(first["user_value"]) if first else float(total_egp))
    out = {"date": today, "total": float(total_egp), "start": start,
           "basis": "invested capital" if cap else "first snapshot",
           "since_start_pct": (total_egp / start - 1) if start > 0 else 0.0,
           "prev": float(prev["user_value"]) if prev else None}
    out["day_pct"] = (total_egp / out["prev"] - 1) if out["prev"] else None
    logger.info("record_daily_total: %s = %.2f EGP", today, total_egp)
    return out


def _refresh_ticket():
    """Rebuild the order ticket after the book changed. The suggestion is a
    function of your holdings, so a ticket written before a fill is stale the
    moment it lands. Best-effort: recording a trade must never fail because the
    ticket could not be rewritten."""
    try:
        from ops.order_ticket import refresh_after_change
        path = refresh_after_change()
        if path:
            print(f"   ticket refreshed: {path}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("_refresh_ticket: %s", exc)


def _main(argv=None):
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(prog="portfolio_tracker")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    sub.add_parser("advice")
    sub.add_parser("accept")                       # take the system's book in full
    sub.add_parser("decline")                      # ignore this suggestion
    sub.add_parser("decisions")                    # your decision history
    sub.add_parser("catchup")                      # backfill missed/forgotten days
    tot = sub.add_parser("total"); tot.add_argument("amount", type=float)
    tot.add_argument("note", nargs="?", default="")
    o = sub.add_parser("own"); o.add_argument("note", nargs="?", default="")  # going my own way
    c = sub.add_parser("cash"); c.add_argument("amount", type=float)
    for a in ("deposit", "withdraw"):
        f = sub.add_parser(a)
        f.add_argument("amount", type=float); f.add_argument("date", nargs="?", default=None)
        f.add_argument("note", nargs="?", default="")
    for a in ("buy", "sell"):
        s = sub.add_parser(a)
        s.add_argument("ticker"); s.add_argument("shares", type=float); s.add_argument("price", type=float)
    args = p.parse_args(argv)

    if args.cmd == "accept":
        out = accept_suggestion()
        if out["de_risked"]:
            print(f"ACCEPTED — system is de-risked ({out['regime']}); sold everything to cash.")
        else:
            print(f"ACCEPTED — rebalanced to the system's target book ({out['regime']}, "
                  f"{out['exposure']*100:.0f}% invested):")
            for e in out["executed"]:
                print(f"   {e['side'].upper():4s} {e['ticker']:6s} ~{e['shares']:.0f} sh @ {e['price']:.2f}")
        print("\n" + format_advice(reconcile()))    # should now say 'already aligned'
    elif args.cmd == "decline":
        log_decision("DECLINED", note="ignored this suggestion")
        print("DECLINED — logged. No trades made; your portfolio is unchanged.")
    elif args.cmd == "own":
        log_decision("OWN", note=args.note or "trading on my own view")
        print("OWN — logged. Use buy/sell to record your own trades; 'advice' still "
              "shows how they compare to the system's book.")
    elif args.cmd == "total":
        r = record_daily_total(args.amount, args.note)
        print(f"recorded {r['date']}: {r['total']:,.2f} EGP")
        if r["day_pct"] is not None:
            print(f"   vs previous entry ({r['prev']:,.2f}): {r['day_pct']*100:+.2f}%")
        print(f"   since start ({r['start']:,.2f}): {r['since_start_pct']*100:+.2f}%")
    elif args.cmd == "catchup":
        pend = pending_decisions()
        if pend.empty:
            print("nothing to catch up — every suggestion has a decision logged.")
        else:
            n = catchup()
            print(f"caught up {n} missed day(s), marked MISSED (no action taken):")
            for r in pend.itertuples():
                print(f"   {r.as_of_date.date()}  regime={r.regime}  exposure={r.exposure*100:.0f}%")
    elif args.cmd == "decisions":
        dh = decision_history()
        if dh.empty:
            print("no decisions logged yet.")
        else:
            for r in dh.itertuples():
                print(f"  {str(r.decided_at)[:16]}  {r.decision:9s} (as-of {r.as_of_date})  {r.note}")
    elif args.cmd in ("deposit", "withdraw"):
        amt = abs(args.amount) * (1 if args.cmd == "deposit" else -1)
        record_cash_flow(amt, args.date, args.note)
        print(f"recorded {args.cmd} {abs(amt):,.2f} EGP on {args.date or 'today'}; "
              f"cash now {get_cash():,.2f}")
    elif args.cmd == "cash":
        set_cash(args.amount); print(f"cash set to {args.amount:,.2f} EGP")
        _refresh_ticket()
    elif args.cmd in ("buy", "sell"):
        record_trade(args.cmd, args.ticker, args.shares, args.price)
        print(f"recorded: {args.cmd} {args.shares} {args.ticker} @ {args.price}")
        _refresh_ticket()
    elif args.cmd == "show":
        pos, cash = holdings()
        if pos.empty:
            print(f"no positions. cash {cash:,.2f} EGP")
        else:
            for r in pos.itertuples():
                up = f"{r.unrealized_pct*100:+.1f}%" if r.unrealized_pct == r.unrealized_pct else "n/a"
                print(f"  {r.ticker:6s} {r.shares:8.0f} sh @ {r.price:7.2f} = {r.value:12,.0f} EGP  "
                      f"({r.weight*100:4.1f}%, unreal {up})")
            print(f"  cash: {cash:,.0f} EGP")
    elif args.cmd == "advice":
        print(format_advice(reconcile()))


if __name__ == "__main__":
    _main()


# --- deliberate holds: positions the operator has decided to keep ------------
#
# WHY THIS EXISTS. reconcile() compares holdings to the model target and lists
# every gap as an action. When the operator has DECIDED to keep a position the
# model does not want, that gap is permanent, and re-printing it as "SELL OIH"
# every morning is noise -- it trains the reader to skip the actions list, which
# is exactly where a genuine new signal would appear.
#
# A deliberate hold is NOT the system agreeing. The position still appears in
# the risk diagnostic, still counts toward concentration, and the kill switch
# and book guard still apply to it. All that changes is that a decision already
# made stops being re-asked.

def set_broker_mark(ticker: str, date, market_value: float, shares: float) -> float:
    """Record a price read off the operator's actual broker account.

    Marks written here are tagged source='broker' and are PROTECTED from the
    price feed: DataCollector._store_ohlcv refuses to delete or replace them.
    Before that protection existed, a single vendor refresh on 2026-08-18 wiped
    nine broker marks for 2026-08-16 and put stale Aug-13 prices back in their
    place, which silently un-did a day of verified valuation.

    Takes market VALUE and SHARES rather than a price, because that is what a
    broker screen actually shows -- deriving share counts from a value and a
    guessed price is what previously put OIH at 1327 shares against a true 1335.
    Volume is stored as 0: a broker mark is a valuation, not evidence of a
    session, and data.calendar.MIN_TRADED_FRACTION relies on that being honest.
    """
    shares = float(shares)
    if shares <= 0:
        raise ValueError(f"set_broker_mark({ticker}): shares must be positive, got {shares}")
    px = float(market_value) / shares
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    with get_connection() as conn:
        DataCollector = None  # imported lazily to keep this module import-light
        from data.collectors import DataCollector as _DC
        _DC._ensure_source_column(conn)
        conn.execute(
            "INSERT OR REPLACE INTO daily_ohlcv "
            "(ticker, date, open, high, low, close, volume, value_traded_egp, "
            " adjusted_close, source) VALUES (?,?,?,?,?,?,0,0.0,?,'broker')",
            (ticker, d, px, px, px, px, px))
    logger.info("set_broker_mark: %s %s = %.4f (%.2f / %g shares)", ticker, d, px, market_value, shares)
    return px


def set_deliberate_hold(ticker: str, reason: str = "", on: bool = True) -> None:
    """Mark (or unmark) a ticker as held by choice, against the model."""
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS deliberate_holds ("
                     "ticker TEXT PRIMARY KEY, reason TEXT, set_at TEXT)")
        if on:
            conn.execute("INSERT OR REPLACE INTO deliberate_holds (ticker, reason, set_at) "
                         "VALUES (?,?,?)", (ticker, reason, now))
        else:
            conn.execute("DELETE FROM deliberate_holds WHERE ticker=?", (ticker,))
    logger.info("deliberate hold %s: %s (%s)", "SET" if on else "CLEARED", ticker, reason)


def deliberate_holds() -> dict:
    """{ticker: reason} for positions the operator has chosen to keep."""
    with get_connection() as conn:
        try:
            rows = conn.execute("SELECT ticker, reason FROM deliberate_holds").fetchall()
        except Exception:  # noqa: BLE001
            return {}
    return {r["ticker"]: (r["reason"] or "") for r in rows}
