"""Watchlist rules: expiry, publication dates, small bases, guard, size arithmetic."""

import pandas as pd
import pytest

from ops import watchlist as w


def test_writer_cannot_touch_other_tables():
    w.ensure_tables()
    con = w._writer()
    try:
        with pytest.raises(Exception, match="not authorized"):
            con.execute("UPDATE user_positions SET shares = shares")
        con.execute("INSERT INTO research_notes (ticker, author, note, created_at) VALUES ('X','t','n','now')")
        con.execute("DELETE FROM research_notes WHERE ticker='X' AND author='t'")
    finally:
        con.close()


def test_min_position_meets_the_cost_cap_and_is_minimal():
    from backtest.fees import round_trip_cost
    m = w.min_position("COMI", 0.03)
    assert m["possible"] and m["round_trip_pct"] <= 0.03
    one_less = (m["shares"] - 1) * m["price"]
    if one_less > 0:
        assert round_trip_cost(one_less)["pct_of_order"] + 2 * w.SLIPPAGE > 0.03


def test_entries_expire_without_a_fresh_trigger():
    """An analyst trigger 10 days old is outside the 7-day window: no entry."""
    w.ensure_tables()
    con = w._writer()
    old = str((pd.Timestamp.now() - pd.Timedelta(days=10)).date())
    con.execute("INSERT OR REPLACE INTO watchlist_triggers VALUES ('COMI','analyst','analyst_note',?,"
                "'old reason','manual:test',?)", (old, w._now()))
    con.commit()
    con.close()
    w.refresh()
    act = w.active()
    reasons = [d for r in act.itertuples() if r.ticker == "COMI" for _, d in r.reasons]
    assert "old reason" not in reasons


def test_fresh_manual_trigger_enters_and_has_expiry():
    r = w.add_manual("AMOC", "test reason: watch the November EGM", author="test")
    w.refresh()
    a = w.active().set_index("ticker")
    assert "AMOC" in a.index
    assert a.loc["AMOC", "expires"] == r["expires_unless_renewed"]


def test_growth_rejects_stale_and_labels_small_bases(monkeypatch):
    import data.fundamentals as f
    now = pd.Timestamp("2026-10-17")
    fake = {"period": "Q2 2026", "period_end": "2026-06-30", "value": 73.0, "prior_value": 4.0, "change": 17.25}
    monkeypatch.setattr(f, "quarterly_yoy", lambda t, d: dict(fake))
    g = w._growth("X", now)
    assert "small base" in g["label"] and "4 -> 73" in g["label"]
    monkeypatch.setattr(f, "quarterly_yoy", lambda t, d: dict(fake, change=0.25, value=5.0))
    assert w._growth("X", now)["label"] == "+25% y/y"
    monkeypatch.setattr(f, "quarterly_yoy", lambda t, d: dict(fake, period_end="2024-12-31"))
    assert w._growth("X", now) == {}                                  # two years old: not current
    monkeypatch.setattr(f, "quarterly_yoy", lambda t, d: dict(fake, prior_value=-3.0))
    assert w._growth("X", now) == {}                                  # from a loss: no growth rate
