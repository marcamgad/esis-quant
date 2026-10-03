"""
Tests for the daily report: snapshotting your book, the decision scorecard
(forward system return after each decision), and the learning summary that
compares your book to the system's. Uses a temp DB seeded with a trade list,
a paper-NAV curve, holdings, and decisions -- no network, no real data.
"""

import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import portfolio_tracker as pt
from ops import daily_report as dr


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "rep.db"))
    database.init_db()
    return config.DB_PATH


def _price(t, p, date="2026-07-16"):
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, value_traded_egp) "
                     "VALUES (?, ?, ?, ?, 0, 0)", (t, date, p, p))


def _target(regime, weights, exposure=1.0, run="r1", as_of="2026-07-16"):
    now = pd.Timestamp.now().isoformat()
    rows = [(run, now, as_of, regime, exposure, t, w) for t, w in weights.items()] or \
           [(run, now, as_of, regime, exposure, "__CASH__", 1.0)]
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO trade_list_log "
                         "(run_id, generated_at, as_of_date, regime, exposure, ticker, weight) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def _paper(navs):
    """navs: list of (date, nav)."""
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO paper_trading_nav "
                         "(date, nav, daily_return, exposure, regime, run_id, n_positions) "
                         "VALUES (?, ?, 0, 1, 'BULL', 'r', 1)", navs)


def test_snapshot_is_idempotent_per_day(temp_db):
    _price("AAA", 100)
    pt.set_cash(100_000)
    dr.snapshot_today("2026-07-16", "BULL", 1.0, "ACCEPTED")
    dr.snapshot_today("2026-07-16", "BULL", 1.0, "ACCEPTED")   # same day again
    snaps = dr._snapshots()
    assert len(snaps) == 1                       # one row per date
    assert snaps["user_value"].iloc[0] == pytest.approx(100_000)


def test_scorecard_forward_return_after_decline(temp_db):
    _target("BULL", {"AAA": 1.0}, as_of="2026-07-16")
    # system book rose 10% over the sessions after the decision
    _paper([("2026-07-16", 100.0), ("2026-07-17", 105.0), ("2026-07-20", 110.0)])
    pt.log_decision("DECLINED", "sat out", run_id="r1", as_of="2026-07-16")
    sc = dr.decision_scorecard()
    row = sc.iloc[0]
    assert row["decision"] == "DECLINED"
    assert row["system_fwd"] == pytest.approx(0.10, abs=1e-6)   # missed +10%


def test_learning_summary_your_book_vs_system(temp_db):
    # two snapshots: your book flat, system up 10% -> you are BEHIND
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO user_daily_snapshot (date, user_value, cash, n_positions, decision, regime, exposure, system_nav) "
            "VALUES (?, ?, 0, 0, ?, 'BULL', 1.0, ?)",
            [("2026-07-15", 100_000, "DECLINED", 100.0),
             ("2026-07-16", 100_000, "DECLINED", 110.0)])
    summ = dr.learning_summary()
    assert summ["your_return"] == pytest.approx(0.0, abs=1e-9)
    assert summ["system_return"] == pytest.approx(0.10, abs=1e-6)


def test_report_renders_all_sections(temp_db):
    _price("AAA", 100); _price("BBB", 50)
    pt.set_cash(100_000)
    _target("BULL", {"AAA": 0.6, "BBB": 0.4}, as_of="2026-07-16")
    text = dr.daily_report(write=True)
    assert "SYSTEM SUGGESTION" in text
    assert "YOUR BOOK" in text
    assert "YOUR DECISION" in text
    assert "WHAT WE'VE LEARNED" in text
    # snapshot got written
    assert len(dr._snapshots()) == 1


def test_report_flags_accepted_but_not_aligned(temp_db):
    _price("AAA", 100)
    pt.set_cash(100_000)                       # all cash, not aligned to a 100% AAA target
    _target("BULL", {"AAA": 1.0}, as_of="2026-07-16")
    pt.log_decision("ACCEPTED", "took it", run_id="r1", as_of="2026-07-16")
    text = dr.daily_report(write=True)
    assert "not aligned" in text               # caught the drift between decision and book


def test_profit_estimate_scales_to_book():
    from analysis import profit_estimate as pe
    d = pe.estimate(1_000)
    if not d:                                    # cache may be absent in CI
        pytest.skip("profit estimate cache not built")
    for r in d["rows"]:
        assert r["median_egp"] == pytest.approx(r["median_pct"] * 1_000, rel=1e-6)
        assert r["lo_egp"] < r["hi_egp"]
        assert 0.0 <= r["p_profit"] <= 1.0


def test_news_brief_matches_holdings(temp_db):
    from ops import news_brief as nb
    with get_connection() as conn:
        conn.execute("INSERT INTO news_articles (source, headline_arabic, url, published_at, "
                     "tickers_mentioned, sentiment_score) VALUES "
                     "('almalnews', 'خبر عن الشركة', 'http://x', '2026-07-10T10:00:00', 'COMI,HRHO', 0.4)")
        conn.execute("INSERT INTO news_articles (source, headline_arabic, url, published_at, "
                     "tickers_mentioned, sentiment_score) VALUES "
                     "('almalnews', 'خبر اخر', 'http://y', '2026-07-09T10:00:00', 'ZZZZ', NULL)")
    df = nb.recent_news(["COMI"], limit=5)
    assert len(df) == 1 and "COMI" in str(df.iloc[0]["tickers_mentioned"])
    txt = nb.format_news_brief(["COMI"])
    assert "CONTEXT ONLY" in txt                 # honest label always present
