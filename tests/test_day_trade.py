"""
Day-trade experiment tracker: the fee arithmetic must be exact, and logged
results must be measured (from real fills) rather than modelled.
"""
import pandas as pd
import pytest

import config
from data import database
from analysis import day_trade as dt


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "dt.db"))
    database.init_db()
    return config.DB_PATH


def test_round_trip_is_twice_per_side():
    c = dt.round_trip_cost(commission_per_side=0.00175, other_fees_per_side=0.0005,
                           spread_per_side=0.0010)
    assert c["per_side"] == pytest.approx(0.00325)
    assert c["round_trip"] == pytest.approx(0.0065)


def test_required_gross_adds_the_hurdle():
    assert dt.required_gross(0.0035, cost=0.0059) == pytest.approx(0.0094)


def test_cost_drag_compounds_destructively():
    """Flat picks + daily fees must compound to a large loss -- the core point."""
    drag = dt.annualize(-0.0059)
    assert drag < -0.7                      # -77% or worse over 252 days
    assert dt.annualize(0.0035) > 1.3       # +0.35%/day nets >130%/yr IF achievable


def test_log_day_measures_net_from_fills(temp_db):
    r = dt.log_day(1000.0, 1004.0)
    assert r["net_egp"] == pytest.approx(4.0)
    assert r["net_pct"] == pytest.approx(0.004)
    s = dt.summarize()
    assert s["n_days"] == 1 and s["win_rate"] == 1.0


def test_log_is_idempotent_per_day(temp_db):
    dt.log_day(1000.0, 1004.0, date="2026-07-28")
    dt.log_day(1000.0, 990.0, date="2026-07-28")     # correction, same day
    df = dt.load_log()
    assert len(df) == 1
    assert df["sell_total"].iloc[0] == pytest.approx(990.0)


def test_summary_reports_losses_honestly(temp_db):
    for i, (b, s) in enumerate([(1000, 995), (1000, 1002), (1000, 993)]):
        dt.log_day(b, s, date=f"2026-07-{20+i:02d}")
    summ = dt.summarize()
    assert summ["n_days"] == 3
    assert summ["cum_net_pct"] < 0
    assert summ["win_rate"] == pytest.approx(1/3)
    txt = dt.format_summary(summ)
    assert "proves nothing" in txt          # honest small-sample warning


def test_implied_cost_recovers_the_fee(temp_db):
    """If gross is ~0 and you net -0.6%/day, the implied round trip is 0.6%."""
    for i in range(5):
        dt.log_day(1000.0, 994.0, date=f"2026-07-{10+i:02d}")
    out = dt.implied_cost_from_log(assumed_gross=0.0)
    assert out["implied_round_trip"] == pytest.approx(0.006, abs=1e-4)


def test_day_ticket_deploys_nearly_all_capital(temp_db, monkeypatch, tmp_path):
    """Day-trade capital is round-tripped the same day, so cash left behind is
    capital not working. The ticket must deploy essentially all of it, unlike
    the tracking-error allocator which stops at the target weights."""
    from data.database import get_connection
    from analysis import day_trade as d

    dates = ["2026-07-27"]
    px = {"AAA": 9.29, "BBB": 26.81, "CCC": 5.37}
    with get_connection() as conn:
        for t, p in px.items():
            conn.execute("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, adjusted_close, "
                         "volume, value_traded_egp) VALUES (?, ?, ?, ?, 1000000, 50000000)",
                         (t, dates[0], p, p))
        rows = [("r1", "2026-07-28T09:00:00", "2026-07-27", "RECOVERY", 1.0, t, w)
                for t, w in [("AAA", 0.5), ("BBB", 0.3), ("CCC", 0.2)]]
        conn.executemany("INSERT OR REPLACE INTO trade_list_log (run_id, generated_at, as_of_date, "
                         "regime, exposure, ticker, weight) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    monkeypatch.setattr(d, "DAYTRADE_DIR", str(tmp_path))
    monkeypatch.setattr(d, "MIN_ADV_EGP", 0)
    out = d.write_day_ticket(1000.0)
    # leftover must be smaller than the cheapest share -- nothing affordable left
    assert out["left"] < min(px.values())
    assert out["spent"] > 985
    assert out["spent"] <= 1000.0 + 1e-9        # never overspends


def test_capital_compounds_from_last_sell(temp_db):
    """Each morning deploys whatever yesterday's sell returned -- the strategy
    compounds, so the base must not be a fixed number."""
    from analysis import day_trade as d
    assert d.current_capital()["capital"] == pytest.approx(1000.0)   # before any log
    d.log_day(1000.0, 1004.0, date="2026-07-29")
    c = d.current_capital()
    assert c["capital"] == pytest.approx(1004.0)          # tomorrow deploys 1004
    assert c["cum_pct"] == pytest.approx(0.004)
    d.log_day(1004.0, 998.0, date="2026-07-30")
    c2 = d.current_capital()
    assert c2["capital"] == pytest.approx(998.0)          # losses compound too
    assert c2["cum_pct"] == pytest.approx(-0.002)
