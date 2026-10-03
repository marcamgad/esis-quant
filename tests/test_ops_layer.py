"""
Operations layer: execution quality (Phase 0.2) and broker reconciliation.
These protect the INTEGRITY OF THE TRADE RECORD, so their failure modes matter
as much as their happy paths.
"""
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import execution_quality as eq
from ops import reconcile as rc
from ops import portfolio_tracker as pt


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "ops.db"))
    database.init_db()
    return config.DB_PATH


def _price(t, p, date="2026-08-03"):
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO daily_ohlcv (ticker,date,close,adjusted_close,"
                     "volume,value_traded_egp) VALUES (?,?,?,?,1000,5000000)", (t, date, p, p))


# --------------------------------------------------------------------------
# execution quality
# --------------------------------------------------------------------------
def test_shortfall_sign_is_unfavourable_positive_for_both_sides():
    """Positive must ALWAYS mean 'worse than the decision price', or buys and
    sells would cancel each other in the average."""
    assert eq.implementation_shortfall("buy", 100.0, 102.0) == pytest.approx(0.02)
    assert eq.implementation_shortfall("sell", 100.0, 98.0) == pytest.approx(0.02)
    assert eq.implementation_shortfall("buy", 100.0, 98.0) == pytest.approx(-0.02)
    assert eq.implementation_shortfall("sell", 100.0, 102.0) == pytest.approx(-0.02)


def test_record_fill_computes_delay_and_stores(temp_db):
    r = eq.record_fill("AAA", "buy", 10, 100.0, 101.0,
                       ticket_generated_at="2026-08-03 09:00:00",
                       filled_at="2026-08-03 10:30:00")
    assert r["shortfall"] == pytest.approx(0.01)
    assert r["delay_minutes"] == pytest.approx(90.0)
    assert len(eq.load_fills()) == 1


def test_missing_fill_time_leaves_delay_null_not_guessed(temp_db):
    """A missing measurement must never be silently replaced by a plausible one."""
    r = eq.record_fill("AAA", "buy", 5, 100.0, 100.5, filled_at=None)
    assert r["delay_minutes"] is None
    assert eq.load_fills()["delay_minutes"].isna().all()


def test_summary_is_notional_weighted(temp_db):
    """A bad fill on a tiny line must not dominate the cost estimate."""
    eq.record_fill("BIG", "buy", 100, 100.0, 100.1)    # +0.1% on 10,010 EGP
    eq.record_fill("TINY", "buy", 1, 10.0, 11.0)        # +10%  on 11 EGP
    s = eq.summary()
    assert s["mean_shortfall"] > s["weighted_mean_shortfall"]   # weighting tames the tiny line
    assert s["weighted_mean_shortfall"] < 0.02


def test_zero_signal_price_does_not_crash(temp_db):
    assert eq.implementation_shortfall("buy", 0.0, 10.0) != eq.implementation_shortfall("buy", 0.0, 10.0) \
        or True   # NaN != NaN; the point is it returns rather than raising


# --------------------------------------------------------------------------
# broker reconciliation
# --------------------------------------------------------------------------
def test_reconcile_passes_when_book_matches(temp_db, monkeypatch):
    monkeypatch.setattr(rc, "_alert", lambda r: None)
    _price("AAA", 50.0)
    pt.set_cash(1000.0)
    pt.record_trade("buy", "AAA", 4, 50.0)          # 200 spent -> cash 800
    r = rc.reconcile_positions({"AAA": 4}, 800.0)
    assert r["ok"] and r["n_breaks"] == 0


def test_reconcile_detects_share_break_and_alerts(temp_db, monkeypatch):
    fired = {}
    monkeypatch.setattr(rc, "_alert", lambda r: fired.update(r))
    _price("AAA", 50.0)
    pt.set_cash(1000.0)
    pt.record_trade("buy", "AAA", 4, 50.0)
    r = rc.reconcile_positions({"AAA": 3}, 800.0)     # broker says 3, book says 4
    assert not r["ok"]
    assert r["breaks"][0]["ticker"] == "AAA"
    assert r["breaks"][0]["diff"] == pytest.approx(1.0)
    assert fired, "a break must raise the loud alert"


def test_reconcile_detects_cash_break(temp_db, monkeypatch):
    monkeypatch.setattr(rc, "_alert", lambda r: None)
    _price("AAA", 50.0)
    pt.set_cash(1000.0)
    r = rc.reconcile_positions({}, 900.0)             # 100 EGP unexplained
    assert not r["ok"]
    assert r["cash_diff"] == pytest.approx(100.0)


def test_small_cash_difference_is_tolerated(temp_db, monkeypatch):
    monkeypatch.setattr(rc, "_alert", lambda r: None)
    pt.set_cash(1000.0)
    assert rc.reconcile_positions({}, 999.50)["ok"]   # rounding, not a break


def test_missing_position_on_either_side_is_a_break(temp_db, monkeypatch):
    monkeypatch.setattr(rc, "_alert", lambda r: None)
    _price("AAA", 50.0)
    pt.set_cash(1000.0)
    r = rc.reconcile_positions({"ZZZ": 10}, 1000.0)   # broker holds one we don't know about
    assert not r["ok"]
    assert any(b["ticker"] == "ZZZ" for b in r["breaks"])


def test_staleness_flags_an_unchecked_book(temp_db):
    assert rc.is_stale()                              # never run
    with get_connection() as conn:
        conn.execute("INSERT INTO broker_reconciliation (recon_date, checked_at, ok, "
                     "n_position_breaks, cash_internal, cash_broker, cash_diff, detail) "
                     "VALUES (?,?,1,0,0,0,0,'ok')",
                     ("2026-08-03", pd.Timestamp.now().isoformat(timespec="seconds")))
    assert not rc.is_stale(max_age_days=3)


# --------------------------------------------------------------------------
# T+2 settlement
# --------------------------------------------------------------------------
def test_sale_proceeds_are_unsettled_immediately(temp_db):
    """The core fix: selling credits the balance but NOT spendable cash."""
    _price("AAA", 50.0)
    pt.set_cash(100.0)
    pt.record_trade("buy", "AAA", 2, 50.0)      # spends all settled cash
    pt.record_trade("sell", "AAA", 2, 60.0)     # +120 credited, unsettled
    st = pt.available_cash()
    assert st["total"] == pytest.approx(120.0)          # broker balance
    assert st["unsettled"] == pytest.approx(120.0)      # none of it settled
    assert st["available"] == pytest.approx(0.0)        # nothing spendable today


def test_settlement_matures_and_frees_cash(temp_db):
    _price("AAA", 50.0)
    pt.set_cash(100.0)
    pt.record_trade("buy", "AAA", 2, 50.0)
    pt.record_trade("sell", "AAA", 2, 60.0)
    assert pt.available_cash()["available"] == pytest.approx(0.0)
    # force the pending row to have matured
    with get_connection() as conn:
        conn.execute("UPDATE pending_settlements SET settles_on='2000-01-01'")
    st = pt.available_cash()
    assert st["unsettled"] == pytest.approx(0.0)
    assert st["available"] == pytest.approx(120.0)      # now spendable


def test_buys_do_not_create_pending_settlements(temp_db):
    _price("AAA", 50.0)
    pt.set_cash(500.0)
    pt.record_trade("buy", "AAA", 2, 50.0)
    assert pt.unsettled_cash() == pytest.approx(0.0)
    assert pt.available_cash()["available"] == pytest.approx(400.0)


def test_reinvestment_policy_flag_is_respected(temp_db, monkeypatch):
    """If the broker DOES allow reinvesting unsettled proceeds, the switch must
    restore the permissive behaviour rather than requiring a code change."""
    import config as cfg
    _price("AAA", 50.0)
    pt.set_cash(100.0)
    pt.record_trade("buy", "AAA", 2, 50.0)
    pt.record_trade("sell", "AAA", 2, 60.0)
    monkeypatch.setattr(cfg, "ALLOW_UNSETTLED_REINVESTMENT", True)
    st = pt.available_cash()
    assert st["available"] == pytest.approx(120.0)
    assert "ALLOWED" in st["policy"]


def test_settlement_date_uses_trading_sessions_not_calendar(temp_db):
    """T+2 must skip non-sessions -- a Thursday sale does not settle Saturday."""
    import pandas as pd
    idx = pd.bdate_range("2026-08-02", periods=10)   # Sun-start business days
    with get_connection() as conn:
        for t_ in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"):
            conn.executemany("INSERT OR REPLACE INTO daily_ohlcv (ticker,date,close,"
                             "adjusted_close,volume,value_traded_egp) VALUES (?,?,10,10,1,1000)",
                             [(t_, str(d.date())) for d in idx])
    got = pd.Timestamp(pt._settlement_date(idx[0]))
    assert got > idx[0]
    assert got in set(idx)             # lands on a real session, not a weekend


# --------------------------------------------------------------------------
# kill-switch on the LIVE path
# --------------------------------------------------------------------------
def _seed_nav(returns):
    import pandas as pd
    idx = pd.bdate_range("2026-07-01", periods=len(returns))
    nav = 1.0
    rows = []
    for d, r in zip(idx, returns):
        nav *= (1 + r)
        rows.append((str(d.date()), nav, float(r), 1.0, "BULL", "r", 5))
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO paper_trading_nav (date,nav,daily_return,"
                         "exposure,regime,run_id,n_positions) VALUES (?,?,?,?,?,?,?)", rows)


def test_kill_switch_fails_open_with_too_little_history(temp_db):
    """A brand-new record must not be read as 'flat', but must say it's unsure."""
    from risk import kill_switch as ks
    _seed_nav([0.001, 0.002])
    s = ks.live_state()
    assert s["tripped"] is False and s["multiplier"] == 1.0
    assert s["conclusive"] is False


def test_kill_switch_trips_live_on_a_big_daily_loss(temp_db):
    from risk import kill_switch as ks
    _seed_nav([0.001] * 8 + [-0.09])        # -9% breaches the -7% daily limit
    s = ks.live_state()
    assert s["tripped"] is True
    assert s["multiplier"] == 0.0
    assert "breached" in s["reason"]


def test_kill_switch_quiet_when_within_limits(temp_db):
    from risk import kill_switch as ks
    _seed_nav([0.002, -0.003] * 6)
    s = ks.live_state()
    assert s["tripped"] is False and s["conclusive"] is True
    assert s["reason"] == "within limits"
