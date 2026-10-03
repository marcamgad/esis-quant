"""
Pre-trade risk limits must actually STOP orders.

The 2026-08-05 audit scored Section 14 red because every limit in risk/ was
advisory: an order 5x a name's daily volume printed a warning string and went
out anyway. These tests exist to make "the limit binds" a property that fails
loudly if anyone weakens it back to a comment.
"""
import pandas as pd
import pytest

import config
from risk import pretrade


def test_oversized_order_is_rejected_not_flagged():
    """The core regression: too big vs ADV must return a verdict that stops the
    order, not a string that decorates it."""
    s = pretrade.screen_order("AAA", notional_egp=500_000, book_egp=1_000_000,
                              adv_egp=100_000)          # 5x the daily volume
    assert s["verdict"] == "reject"
    assert s["allowed_egp"] == 0.0
    assert any("daily volume" in r for r in s["reasons"])


def test_normal_order_passes_untouched():
    s = pretrade.screen_order("AAA", notional_egp=1_000, book_egp=100_000,
                              adv_egp=10_000_000)
    assert s["verdict"] == "allow"
    assert s["allowed_egp"] == pytest.approx(1_000)
    assert s["reasons"] == []


def test_unknown_liquidity_fails_closed(monkeypatch):
    """An order we cannot evaluate is the MOST dangerous case, not the safest.
    It must reject rather than sail through."""
    c = pretrade.check_participation("ZZZ", 10_000, adv_egp=float("nan"))
    assert c["ok"] is False
    assert c["allowed_egp"] == 0.0
    assert "unknown" in c["reason"]


def test_resize_mode_trims_instead_of_rejecting(monkeypatch):
    monkeypatch.setattr(config, "ADV_LIMIT_ACTION", "resize")
    s = pretrade.screen_order("AAA", notional_egp=500_000, book_egp=10_000_000,
                              adv_egp=100_000)
    assert s["verdict"] == "resize"
    # trimmed to exactly the participation limit
    assert s["allowed_egp"] == pytest.approx(config.COST_MAX_PARTICIPATION * 100_000)
    assert s["allowed_egp"] < s["requested_egp"]


def test_warn_mode_restores_advisory_behaviour(monkeypatch):
    """The old behaviour must still be reachable, but only by explicitly asking
    for it -- so that choosing it is a visible decision."""
    monkeypatch.setattr(config, "ADV_LIMIT_ACTION", "warn")
    s = pretrade.screen_order("AAA", notional_egp=500_000, book_egp=10_000_000,
                              adv_egp=100_000)
    assert s["verdict"] == "allow"


def test_concentration_limit_binds():
    """The optimizer caps weights, but the integer allocator and manual edits
    both bypass it, so the limit is re-checked at the order."""
    s = pretrade.screen_order("AAA", notional_egp=5_000, book_egp=10_000,
                              current_value_egp=0.0, adv_egp=10_000_000)
    assert s["verdict"] == "reject"                     # 50% of the book
    assert any("of the book" in r for r in s["reasons"])


def test_concentration_counts_what_is_already_held():
    """Adding to an existing position must be judged on the POST-trade weight,
    not on the increment alone -- otherwise a limit is evaded by buying twice."""
    ok = pretrade.check_position_weight("AAA", 1_500, 10_000)
    breach = pretrade.check_position_weight("AAA", 2_500, 10_000)
    assert ok["ok"] is True
    assert breach["ok"] is False


def test_sells_are_exempt_from_concentration_and_daily_loss():
    """Refusing to let someone REDUCE risk because risk is high would be
    perverse. Liquidity still applies to a sell."""
    s = pretrade.screen_order("AAA", notional_egp=5_000, book_egp=10_000,
                              side="sell", adv_egp=10_000_000)
    assert s["verdict"] == "allow"
    assert [c["check"] for c in s["checks"]] == ["participation"]


def test_daily_loss_blocks_new_risk(monkeypatch):
    monkeypatch.setattr(pretrade, "check_daily_loss",
                        lambda limit=None: {"check": "daily_loss", "ok": False,
                                            "today": -0.08, "limit": -0.05,
                                            "conclusive": True, "allowed_egp": 0.0,
                                            "reason": "book is -8.0% today"})
    s = pretrade.screen_order("AAA", 100, book_egp=10_000, adv_egp=10_000_000)
    assert s["verdict"] == "reject"


def test_daily_loss_abstains_when_the_record_is_unreadable(monkeypatch):
    """An unreadable NAV file must not halt trading on its own -- the kill
    switch remains the hard backstop."""
    import risk.kill_switch as ks
    monkeypatch.setattr(ks, "live_returns", lambda: (_ for _ in ()).throw(OSError("gone")))
    d = pretrade.check_daily_loss()
    assert d["ok"] is True
    assert d["conclusive"] is False


def test_limits_are_read_from_config_not_hardcoded(monkeypatch):
    """A limit that ignores its config entry is a limit nobody can change."""
    monkeypatch.setattr(config, "MAX_POSITION_WEIGHT", 0.90)
    c = pretrade.check_position_weight("AAA", 5_000, 10_000)
    assert c["ok"] is True
    monkeypatch.setattr(config, "MAX_POSITION_WEIGHT", 0.10)
    c = pretrade.check_position_weight("AAA", 5_000, 10_000)
    assert c["ok"] is False


# --- book guard: protects the ACCOUNT, not the model ------------------------

def test_book_guard_blocks_new_risk_in_drawdown(monkeypatch):
    """The model kill switch reads the strategy's paper NAV, so a healthy model
    with a badly-down account would pass. This is the check that doesn't."""
    import risk.kill_switch as ks
    monkeypatch.setattr(ks, "book_state", lambda: {
        "level": "block", "conclusive": True, "n_days": 30, "drawdown": -0.18,
        "peak": 10000.0, "now": 8200.0, "block_new_risk": True,
        "reason": "account is -18.0% below its peak"})
    c = pretrade.check_book_drawdown()
    assert c["ok"] is False
    s = pretrade.screen_order("AAA", 100, book_egp=10_000, adv_egp=1e9)
    assert s["verdict"] == "reject"


def test_book_guard_is_independent_of_the_model_kill_switch(monkeypatch):
    """A healthy paper record must NOT excuse a drawn-down account."""
    import risk.kill_switch as ks
    monkeypatch.setattr(pretrade, "check_daily_loss",
                        lambda limit=None: {"check": "daily_loss", "ok": True,
                                            "conclusive": True, "reason": ""})
    monkeypatch.setattr(ks, "book_state", lambda: {
        "level": "flatten", "conclusive": True, "n_days": 40, "drawdown": -0.30,
        "peak": 10000.0, "now": 7000.0, "block_new_risk": True,
        "reason": "account is -30.0% below its peak"})
    s = pretrade.screen_order("AAA", 100, book_egp=7_000, adv_egp=1e9)
    assert s["verdict"] == "reject"


def test_book_nav_does_not_double_count_cash():
    """user_value already includes cash; adding it again showed a 13,119 peak on
    a 9,963 book and would have delayed every drawdown alert."""
    import inspect
    from risk import kill_switch as ks
    src = inspect.getsource(ks.book_nav)
    assert "user_value" in src
    assert 'df["cash"]' not in src


def test_book_guard_escalates_with_drawdown():
    from risk import kill_switch as ks
    assert ks.BOOK_WARN_DRAWDOWN > ks.BOOK_BLOCK_DRAWDOWN > ks.BOOK_FLATTEN_DRAWDOWN
    assert ks.BOOK_FLATTEN_DRAWDOWN == pytest.approx(config.GO_LIVE_MAX_DRAWDOWN)


def test_sells_are_never_blocked_by_the_book_guard(monkeypatch):
    """Blocking a SELL because the account is down would trap you in the loss."""
    import risk.kill_switch as ks
    monkeypatch.setattr(ks, "book_state", lambda: {
        "level": "flatten", "conclusive": True, "n_days": 40, "drawdown": -0.30,
        "peak": 10000.0, "now": 7000.0, "block_new_risk": True, "reason": "down 30%"})
    s = pretrade.screen_order("AAA", 500, book_egp=7_000, side="sell", adv_egp=1e9)
    assert s["verdict"] == "allow"
