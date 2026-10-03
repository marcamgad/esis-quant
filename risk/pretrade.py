"""
Pre-trade risk checks that can actually STOP an order  (audit 2026-08-05, P3).

THE GAP THIS CLOSES. The audit scored Section 14 🔴 with this evidence: risk/
computed numbers for reports, and nothing in the order path consulted them.
ADV participation was rendered as a string in the order ticket's `flags`
column -- an order 5x the name's entire daily volume printed with a warning and
was otherwise indistinguishable from a good one. The only control that could
change a decision was the kill switch, and it operates on the whole book.

A limit that cannot reject is not a limit, it is a comment.

THREE CHECKS, DELIBERATELY DIFFERENT IN KIND:

  1. PARTICIPATION (per order, liquidity). Can this order be filled without
     moving the name? Compares notional to trailing ADV. Rejects by default;
     `config.ADV_LIMIT_ACTION` can resize instead.
  2. POSITION WEIGHT (per order, concentration). Would this leave one name above
     MAX_POSITION_WEIGHT of the book? The optimizer already caps weights, but
     the optimizer is not the only thing that generates orders -- the integer
     allocator and manual edits both bypass it.
  3. DAILY LOSS (book-level, but blocks NEW risk only). Distinct from the kill
     switch on purpose: the kill switch flattens everything and imposes a
     cooldown, which is the right response to a catastrophic day and the wrong
     response to a merely bad one. This just declines to add.

WHAT THIS IS NOT. It is not a risk model, and it does not size positions. It is
a set of hard bounds applied after sizing, so that a sizing bug, a stale price
or a fat-fingered manual edit cannot reach the account unchallenged.

FAIL-CLOSED, WITH ONE EXCEPTION. An order whose liquidity cannot be evaluated is
REJECTED, not passed -- unknown liquidity is the case most likely to be
catastrophic. The exception is the daily-loss check: if the paper record cannot
be read the check abstains rather than blocking everything, because an
unreadable NAV file must not halt trading on its own.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("esis_quant.risk.pretrade")


def check_participation(ticker: str, notional_egp: float, adv_egp: float = None,
                        limit: float = None, action: str = None) -> dict:
    """Is this order too large relative to the name's daily volume?"""
    limit = config.COST_MAX_PARTICIPATION if limit is None else limit
    action = (config.ADV_LIMIT_ACTION if action is None else action).lower()
    if adv_egp is None:
        try:
            from backtest.costs import average_daily_value
            adv_egp = float(average_daily_value([ticker]).get(ticker, np.nan))
        except Exception:  # noqa: BLE001
            adv_egp = float("nan")
    notional = max(float(notional_egp or 0.0), 0.0)

    if not (adv_egp == adv_egp and adv_egp > 0):
        # Unknown liquidity is the dangerous case, not the benign one.
        return {"check": "participation", "ok": action == "warn", "ticker": ticker,
                "participation": float("nan"), "limit": limit, "action": action,
                "allowed_egp": 0.0 if action != "warn" else notional,
                "reason": "liquidity unknown (no ADV) — cannot size this order safely"}

    part = notional / adv_egp
    if part <= limit:
        return {"check": "participation", "ok": True, "ticker": ticker,
                "participation": part, "limit": limit, "action": action,
                "allowed_egp": notional, "reason": ""}
    allowed = limit * adv_egp
    return {"check": "participation", "ok": action == "warn", "ticker": ticker,
            "participation": part, "limit": limit, "action": action,
            "allowed_egp": (allowed if action == "resize" else
                            (notional if action == "warn" else 0.0)),
            "reason": (f"order is {part*100:.0f}% of {ticker}'s daily volume "
                       f"(limit {limit*100:.0f}%); "
                       f"{'resized to' if action == 'resize' else 'max'} "
                       f"{allowed:,.0f} EGP")}


def check_position_weight(ticker: str, post_trade_value_egp: float, book_egp: float,
                          limit: float = None) -> dict:
    """Would this order leave one name too large a share of the book?"""
    limit = config.MAX_POSITION_WEIGHT if limit is None else limit
    book = float(book_egp or 0.0)
    if book <= 0:
        return {"check": "position_weight", "ok": False, "ticker": ticker,
                "weight": float("nan"), "limit": limit, "allowed_egp": 0.0,
                "reason": "book value unknown — cannot evaluate concentration"}
    w = float(post_trade_value_egp) / book
    if w <= limit:
        return {"check": "position_weight", "ok": True, "ticker": ticker,
                "weight": w, "limit": limit, "allowed_egp": float(post_trade_value_egp),
                "reason": ""}
    return {"check": "position_weight", "ok": False, "ticker": ticker,
            "weight": w, "limit": limit, "allowed_egp": limit * book,
            "reason": (f"would leave {ticker} at {w*100:.0f}% of the book "
                       f"(limit {limit*100:.0f}%); max {limit*book:,.0f} EGP")}


def check_daily_loss(limit: float = None) -> dict:
    """Is the book already down enough today that adding risk is unwise?

    Reads the SAME forward paper NAV the kill switch reads, so the two cannot
    disagree about what happened. Abstains (ok=True) when the record is missing
    or too short -- an unreadable file must not halt trading by itself, and the
    kill switch remains the hard backstop.
    """
    limit = config.MAX_DAILY_LOSS_BLOCK_NEW if limit is None else limit
    try:
        from risk.kill_switch import live_returns
        r = live_returns()
    except Exception as exc:  # noqa: BLE001
        return {"check": "daily_loss", "ok": True, "today": float("nan"), "limit": limit,
                "conclusive": False, "reason": f"paper record unreadable ({exc}) — abstaining"}
    if r is None or len(r) == 0:
        return {"check": "daily_loss", "ok": True, "today": float("nan"), "limit": limit,
                "conclusive": False, "reason": "no forward record yet — abstaining"}
    today = float(pd.Series(r).dropna().iloc[-1])
    ok = today > limit
    return {"check": "daily_loss", "ok": ok, "today": today, "limit": limit,
            "conclusive": True,
            "reason": "" if ok else (f"book is {today*100:.1f}% today (limit "
                                     f"{limit*100:.0f}%) — not adding new risk")}


def check_book_drawdown() -> dict:
    """Is YOUR ACCOUNT (not the model's paper track) far enough below its peak
    that adding risk is unwise?

    The model kill switch reads the strategy's uninterrupted paper NAV so it can
    judge the MODEL. That is the wrong series for protecting the ACCOUNT: the
    live book deviates from the model deliberately, so the paper series can look
    healthy while real money is down badly. This check closes that gap, and it
    became material the moment real money was involved.
    """
    try:
        from risk.kill_switch import book_state
        b = book_state()
    except Exception as exc:  # noqa: BLE001
        return {"check": "book_drawdown", "ok": True, "conclusive": False,
                "reason": f"book history unreadable ({exc}) — abstaining"}
    return {"check": "book_drawdown", "ok": not b["block_new_risk"],
            "conclusive": b["conclusive"], "drawdown": b["drawdown"],
            "level": b["level"], "allowed_egp": 0.0 if b["block_new_risk"] else None,
            "reason": b["reason"] if b["block_new_risk"] else ""}


def screen_order(ticker: str, notional_egp: float, book_egp: float,
                 current_value_egp: float = 0.0, side: str = "buy",
                 adv_egp: float = None) -> dict:
    """Run every applicable check on one order. Returns the verdict plus the
    permitted size, which may be smaller than requested.

    SELLS are exempt from the concentration and daily-loss checks: refusing to
    let someone REDUCE risk because risk is high would be perverse. Liquidity
    still applies -- a sell too large for the market is just as unfillable.
    """
    checks = [check_participation(ticker, notional_egp, adv_egp=adv_egp)]
    if side.lower() == "buy":
        checks.append(check_position_weight(ticker, float(current_value_egp) + float(notional_egp),
                                            book_egp))
        checks.append(check_daily_loss())
        checks.append(check_book_drawdown())
    failed = [c for c in checks if not c["ok"]]
    allowed = min([c.get("allowed_egp", notional_egp) for c in checks
                   if c.get("allowed_egp") is not None] + [float(notional_egp)])
    verdict = "allow" if not failed else (
        "resize" if allowed > 0 and all(c.get("action") == "resize" or c["ok"] for c in checks)
        else "reject")
    if failed:
        logger.warning("pre-trade %s %s %s: %s", verdict.upper(), side, ticker,
                       "; ".join(c["reason"] for c in failed))
    return {"ticker": ticker, "side": side, "requested_egp": float(notional_egp),
            "allowed_egp": float(max(allowed, 0.0)), "verdict": verdict,
            "checks": checks, "reasons": [c["reason"] for c in failed]}


def format_screen(s: dict) -> str:
    if s["verdict"] == "allow":
        return ""
    head = f"  {s['verdict'].upper()} {s['side'].upper()} {s['ticker']} " \
           f"{s['requested_egp']:,.0f} EGP"
    if s["verdict"] == "resize":
        head += f" -> {s['allowed_egp']:,.0f} EGP"
    return head + "\n" + "\n".join(f"      {r}" for r in s["reasons"] if r)
