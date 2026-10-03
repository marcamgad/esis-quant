"""
Job-health tracking. The central regression is the 2026-07-23 outage: the daily
job died and stayed dead for 3 days with nothing reporting it.
"""
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import health


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "h.db"))
    database.init_db()
    return config.DB_PATH


def _run_at(days_ago: int, ok: bool, msg=""):
    ts = (pd.Timestamp.now() - pd.Timedelta(days=days_ago)).isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("INSERT INTO job_runs (job, started_at, finished_at, ok, as_of_date, message) "
                     "VALUES ('daily_run', ?, ?, ?, '2026-07-22', ?)", (ts, ts, 1 if ok else 0, msg))


def test_no_runs_is_unhealthy(temp_db):
    h = health.health_status()
    assert not h["ok"]
    assert any("no run has ever been recorded" in p for p in h["problems"])


def test_recent_success_is_healthy(temp_db):
    _run_at(0, ok=True, msg="regime=RECOVERY")
    h = health.health_status()
    assert h["ok"] and not h["problems"]
    assert "HEALTHY" in health.format_health(h)


def test_failed_last_run_is_flagged(temp_db):
    _run_at(2, ok=True)
    _run_at(0, ok=False, msg="tradable universe is 0 names")
    h = health.health_status()
    assert not h["ok"]
    assert any("FAILED" in p for p in h["problems"])
    assert "tradable universe is 0 names" in health.format_health(h)


def test_silent_outage_is_caught_by_staleness(temp_db):
    """The real 2026-07-23 case: the job stopped running entirely, so there is
    no failure row at all -- only an old success. Absence of news is NOT good
    news, and staleness alone must raise the alarm."""
    _run_at(5, ok=True, msg="regime=RECOVERY")     # last success 5 days ago
    h = health.health_status(max_days=3)
    assert not h["ok"]
    assert h["days_since_success"] == 5
    assert any("no successful run in 5 days" in p for p in h["problems"])
    assert "BEAR" in health.format_health(h)       # explains the consequence


def test_record_run_round_trip(temp_db):
    health.record_run("daily_run", ok=True, as_of_date="2026-07-26", message="ok")
    assert health.last_run()["ok"] == 1
    assert health.last_success()["as_of_date"] == "2026-07-26"


def test_alert_never_raises(temp_db, monkeypatch):
    """Alerting must never take down the job it reports on."""
    def boom(*a, **k):
        raise OSError("no GUI")
    monkeypatch.setattr(health.subprocess, "run", boom)
    assert health.alert("t", "m") is False          # returns, does not raise
