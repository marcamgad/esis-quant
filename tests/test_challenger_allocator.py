"""Agents 5 (challenger) and 6 (allocator): limits, sizing, write boundary. Temp DB only."""

import sqlite3

import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from ops import agent_guard as ag
from ops import allocator as al
from ops import challenger as ch


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "ch.db"))
    database.init_db()
    days = pd.bdate_range(end="2026-10-01", periods=200, freq="C", weekmask="Sun Mon Tue Wed Thu")
    rows = []
    for i, d in enumerate(days):
        for t, p in (("BIG", 10 + 0.01 * i), ("SMALL", 5.0 + 0.005 * i), ("OTHER", 2.0)):
            rows.append((t, str(d.date()), p, p, 1e6, 1e8))
    for k in range(12):                                     # enough names for the calendar
        rows += [(f"F{k}", str(d.date()), 1.0, 1.0, 1e3, 1e3) for d in days]
    with get_connection() as conn:
        conn.executemany("INSERT INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, "
                         "value_traded_egp) VALUES (?,?,?,?,?,?)", rows)
        from data.security_master import _ensure_columns
        _ensure_columns(conn)
        for t, sec in (("BIG", "Construction"), ("SMALL", "Energy"), ("OTHER", "Construction")):
            conn.execute("INSERT OR REPLACE INTO stocks (ticker, sector, security_type) VALUES (?,?,'ordinary')",
                         (t, sec))
        conn.execute("INSERT INTO user_positions (ticker, shares, avg_cost, updated_at) VALUES ('BIG', 500, 10, 'x')")
        conn.execute("INSERT INTO user_positions (ticker, shares, avg_cost, updated_at) VALUES ('SMALL', 200, 5, 'x')")
        conn.execute("INSERT OR REPLACE INTO user_account (id, cash, updated_at) VALUES (1, 2000, 'x')")
    return config.DB_PATH


def _levels(res):
    return {c["check"]: c["level"] for c in res["checks"]}


def test_position_and_sector_limits_block(book):
    # BIG ~ 500 x 11.99 = 5,995 of ~9,370 account: already over 20%
    res = ch.review("BIG", 500)
    assert res["verdict"] == "BLOCKED" and _levels(res)["position size"] == "BLOCK"
    res = ch.review("OTHER", 1500)                 # same sector as BIG -> sector breach
    assert _levels(res)["sector"] == "BLOCK"


def test_small_diversifying_order_is_not_blocked(book):
    res = ch.review("SMALL", 200)
    assert res["n_block"] == 0


def test_override_is_logged(book):
    rid = ch.record(ch.review("BIG", 500), override_reason="testing the log")
    with get_connection() as conn:
        v, why = conn.execute("SELECT verdict, override_reason FROM trade_reviews WHERE id=?", (rid,)).fetchone()
    assert v == "OVERRIDDEN" and why == "testing the log"


def test_challenger_cannot_write_positions_or_ledger():
    raw = sqlite3.connect(":memory:")
    raw.executescript("CREATE TABLE user_positions (x); CREATE TABLE user_trade_ledger (x);"
                      "CREATE TABLE trade_reviews (x);")
    conn = ag.guarded(raw, ch.WRITABLE, ch.AGENT)
    for sql in ("INSERT INTO user_positions VALUES (1)", "INSERT INTO user_trade_ledger VALUES (1)"):
        with pytest.raises(ag.WriteBoundaryViolation):
            ag.run(conn, sql, agent=ch.AGENT)


@pytest.mark.parametrize("excess,drag,sigma,expected", [
    (0.086, 0.02, 0.10, 1.0),         # diversified book: capped at 100%
    (0.086, 0.02, 0.41, 0.196),       # concentrated book: ~20%
    (0.086, 0.16, 0.41, 0.0),         # costs exceed the edge: nothing in stocks
])
def test_half_kelly_allocation(excess, drag, sigma, expected):
    assert al.allocate(excess, drag, sigma) == pytest.approx(expected, abs=0.005)
