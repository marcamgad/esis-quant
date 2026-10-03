"""
Idle-cash reminder. The behaviour that matters is WHEN IT STAYS SILENT: a
reminder that fires on sums too small to act on trains the reader to ignore it.
"""
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import idle_cash as ic
from ops import portfolio_tracker as pt


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "ic.db"))
    database.init_db()
    with get_connection() as conn:   # rates for the reconstructed short-rate lookup
        conn.executemany("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, 48.0)",
                         [(str(d.date()),) for d in pd.bdate_range("2026-01-01", periods=200)])
    return config.DB_PATH


def _snapshots(cash_by_day):
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO user_daily_snapshot (date, user_value, cash, "
                         "n_positions) VALUES (?, ?, ?, 0)",
                         [(d, float(c), float(c)) for d, c in cash_by_day])


def _note_for(idle, days, tier):
    return ic.format_note({"idle": idle, "invested": 0.0, "total": idle, "idle_pct": 1.0,
                           "short_rate": 0.17, "forgone_per_year": idle * 0.17,
                           "forgone_per_month": idle * 0.17 / 12, "days_idle": days,
                           "tier": tier, "actionable": True,
                           "tbill_min": config.IDLE_CASH_TBILL_MIN,
                           "mmf_min": config.IDLE_CASH_MMF_MIN})


def test_silent_below_the_minimums(temp_db):
    """100 EGP at 17% is ~17 EGP/year -- less than one round-trip fee."""
    assert _note_for(100.0, 30, "below_minimums") == ""


def test_mmf_tier_names_the_fund_route_not_tbills(temp_db):
    note = _note_for(6_000.0, 30, "mmf")
    assert "money market fund" in note
    assert "Below the 25,000" in note


def test_tbill_tier_offered_only_above_the_minimum(temp_db):
    note = _note_for(40_000.0, 30, "tbill")
    assert "Above the 25,000" in note
    assert "direct T-bill" in note


def test_short_idle_period_does_not_suggest_moving_cash(temp_db):
    """The regime can flip back within days; moving cash has friction."""
    note = _note_for(40_000.0, 2, "tbill")
    assert "no action while it may be short-lived" in note
    assert "T-bill minimum" not in note


def test_tiering_is_driven_by_idle_cash_not_book_size(temp_db):
    """A big fully-invested book must NOT trigger; a de-risked one must."""
    with get_connection() as conn:   # 100k book, fully invested, only 1k cash
        conn.execute("INSERT OR REPLACE INTO daily_ohlcv (ticker,date,close,adjusted_close,"
                     "volume,value_traded_egp) VALUES ('AAA','2026-06-01',100,100,1,1000)")
    pt.set_cash(1_000.0)
    pt.record_trade("buy", "AAA", 990, 100.0)   # spends ~99k -> tiny cash left
    pt.set_cash(1_000.0)
    s = ic.idle_cash_state()
    assert s["tier"] == "below_minimums"        # big book, but idle cash is small
    pt.set_cash(40_000.0)
    assert ic.idle_cash_state()["tier"] == "tbill"


def test_days_idle_counts_consecutive_recent_sessions(temp_db):
    _snapshots([("2026-06-01", 5000), ("2026-06-02", 5000), ("2026-06-03", 0), ("2026-06-04", 5000)])
    # newest-first: 06-04 has cash, 06-03 does not -> streak stops at 1
    assert ic.days_idle() == 1


def test_days_idle_zero_without_history(temp_db):
    """Unknown duration must report 0, never a plausible-looking number."""
    assert ic.days_idle() == 0


# --- relative materiality, not just product minimums ------------------------

def _book(monkeypatch, idle, invested, rate=0.172, days=30):
    import pandas as pd
    from ops import idle_cash as ic
    monkeypatch.setattr(ic, "current_short_rate", lambda as_of=None: rate)
    monkeypatch.setattr(ic, "days_idle", lambda min_cash=1.0: days)
    pos = pd.DataFrame({"ticker": ["AAA"], "value": [invested]})
    import ops.portfolio_tracker as pt
    monkeypatch.setattr(pt, "holdings", lambda: (pos, idle))
    return ic


def test_small_idle_cash_that_is_a_big_share_of_the_book_is_reported(monkeypatch):
    """3,167 EGP is below every cash-product minimum, yet it was 32% of a ~10k
    book costing 5.5%/yr -- more than the strategy's whole expected return. The
    absolute tiers stayed silent about the largest drag on the book."""
    ic = _book(monkeypatch, idle=3167.0, invested=6820.0)
    s = ic.idle_cash_state()
    assert s["tier"] == "below_minimums"       # still no product fits
    assert s["material"] is True               # but it matters anyway
    assert s["actionable"] is True
    assert ic.format_note(s) != ""


def test_genuinely_trivial_idle_cash_stays_silent(monkeypatch):
    """A reminder that fires on sums too small to act on trains you to ignore it."""
    ic = _book(monkeypatch, idle=50.0, invested=9950.0)
    s = ic.idle_cash_state()
    assert s["material"] is False
    assert ic.format_note(s) == ""


def test_large_share_but_negligible_cost_stays_silent(monkeypatch):
    """Both tests must pass: a big share of a tiny book costs almost nothing in
    EGP, and nagging about it is noise."""
    ic = _book(monkeypatch, idle=30.0, invested=40.0, rate=0.0001)
    s = ic.idle_cash_state()
    assert s["idle_pct"] > 0.20                # share is large
    assert s["material"] is False              # but the cost is not
    assert ic.format_note(s) == ""


def test_brief_idleness_still_defers(monkeypatch):
    """Materiality must not override the duration guard -- the regime can
    reinvest the cash within days and moving it has friction."""
    ic = _book(monkeypatch, idle=3167.0, invested=6820.0, days=1)
    s = ic.idle_cash_state()
    assert s["material"] is True
    assert s["actionable"] is False
    assert "only 1 session" in ic.format_note(s)
