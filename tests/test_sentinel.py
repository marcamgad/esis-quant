"""Agent 1 (Sentinel): invariants, HALT lifecycle, write boundary. Temp DB only."""

import sqlite3

import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import agent_guard as ag
from ops import sentinel as sn


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "sn.db"))
    database.init_db()
    monkeypatch.setattr(sn, "task_results", lambda: {})
    monkeypatch.setattr(sn, "diagnose_log", lambda *a, **k: "")
    return config.DB_PATH


def _prices(days, n=12):
    rows = [(f"T{i}", str(d.date()), 10.0, 10.0, 1000.0, 1e6) for d in days for i in range(n)]
    with get_connection() as conn:
        conn.executemany("INSERT INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, "
                         "value_traded_egp) VALUES (?,?,?,?,?,?)", rows)
        conn.execute("CREATE TABLE IF NOT EXISTS tradable_universe "
                     "(ticker TEXT, date DATE, is_tradable INTEGER, PRIMARY KEY (ticker, date))")
        conn.executemany("INSERT INTO tradable_universe (date, ticker, is_tradable) VALUES (?,?,1)",
                         [(str(days[-1].date()), f"T{i}") for i in range(n)])


def _artifacts(last):
    d = str(pd.Timestamp(last).date())
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, 48.0)", (d,))
        conn.execute("INSERT OR REPLACE INTO regime_history (date, regime) VALUES (?, 'BULL')", (d,))
        conn.execute("INSERT OR REPLACE INTO regime_ensemble (date, p_bear, n_fits, computed_at) VALUES (?,0,5,?)", (d, d))


NOW = pd.Timestamp("2026-10-01 20:00")          # Thursday evening: 1 Oct should be in the DB


def test_expected_session_respects_the_egx_week_and_job_time():
    assert sn.expected_session(pd.Timestamp("2026-10-01 20:00")) == pd.Timestamp("2026-10-01")
    assert sn.expected_session(pd.Timestamp("2026-10-01 12:00")) == pd.Timestamp("2026-09-30")
    assert sn.expected_session(pd.Timestamp("2026-10-03 12:00")) == pd.Timestamp("2026-10-01")   # Saturday


def test_fresh_system_is_ok(temp_db):
    days = pd.bdate_range(end="2026-10-01", periods=30, freq="C", weekmask="Sun Mon Tue Wed Thu")
    _prices(days); _artifacts(days[-1])
    res = sn.run(now=NOW, with_tasks=False)
    assert res["state"]["state"] == "OK"
    assert not [f for f in res["findings"] if f["severity"] == "critical"]


def test_frozen_ensemble_halts_then_recovery_clears(temp_db):
    """The audit's main finding: the exposure signal froze and nothing stopped the ticket."""
    days = pd.bdate_range(end="2026-10-01", periods=30, freq="C", weekmask="Sun Mon Tue Wed Thu")
    _prices(days); _artifacts(days[-1])
    with get_connection() as conn:
        conn.execute("DELETE FROM regime_ensemble")
        conn.execute("INSERT OR REPLACE INTO regime_ensemble (date, p_bear, n_fits, computed_at) VALUES (?,0,5,?)",
                     (str(days[-10].date()), str(days[-10].date())))
    res = sn.run(now=NOW, with_tasks=False)
    assert res["state"]["state"] == "HALT" and "regime_ensemble" in res["state"]["reason"]
    with pytest.raises(ag.SystemHalted):
        ag.assert_not_halted()
    _artifacts(days[-1])                                      # the daily job catches up
    res = sn.run(now=NOW, with_tasks=False)
    assert res["state"]["state"] == "OK"
    ag.assert_not_halted()
    assert res["resolved"] >= 1                               # the finding closed itself


def test_killed_run_is_closed_and_reported(temp_db):
    days = pd.bdate_range(end="2026-10-01", periods=30, freq="C", weekmask="Sun Mon Tue Wed Thu")
    _prices(days); _artifacts(days[-1])
    with get_connection() as conn:
        conn.execute("INSERT INTO job_runs (job, started_at) VALUES ('daily_run', '2026-10-01T17:00:00')")
    res = sn.run(now=NOW, with_tasks=False)
    assert any(f["kind"] == "killed_run" for f in res["findings"])
    with get_connection() as conn:
        assert conn.execute("SELECT ok, finished_at IS NOT NULL FROM job_runs").fetchone()[:] == (0, 1)


def test_sentinel_cannot_write_prices_or_positions():
    raw = sqlite3.connect(":memory:")
    raw.executescript("CREATE TABLE daily_ohlcv (c REAL); INSERT INTO daily_ohlcv VALUES (1);"
                      "CREATE TABLE user_positions (s REAL); CREATE TABLE agent_findings (x);")
    conn = ag.guarded(raw, sn.WRITABLE, "sentinel")
    for sql in ("DELETE FROM daily_ohlcv", "WITH t AS (SELECT 1) DELETE FROM daily_ohlcv",
                "INSERT INTO user_positions VALUES (5)"):
        with pytest.raises(ag.WriteBoundaryViolation):
            ag.run(conn, sql, agent="sentinel")
    ag.run(conn, "INSERT INTO agent_findings VALUES (1)", agent="sentinel")


def test_finding_due_dates_follow_severity(temp_db):
    with get_connection() as conn:
        ag.ensure_tables(conn)
        ag.record_findings(conn, "t", [{"severity": "critical", "kind": "k", "subject": "a", "summary": "s"},
                                       {"severity": "medium", "kind": "k", "subject": "b", "summary": "s"}])
    df = ag.open_findings("t").set_index("subject")
    today = pd.Timestamp.now().normalize()
    assert pd.Timestamp(df.loc["a", "due_date"]) == today + pd.Timedelta(days=1)
    assert pd.Timestamp(df.loc["b", "due_date"]) == today + pd.Timedelta(days=14)
