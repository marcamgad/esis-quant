"""Agent 2 (Ledger clerk): staging, approval, residue rule, realised P&L. Temp DB only."""

import sqlite3

import pytest

import config
from data import database
from data.database import get_connection
from ops import agent_guard as ag
from ops import ledger_clerk as lc


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "lc.db"))
    database.init_db()
    with get_connection() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(user_trade_ledger)")}
        if "fee" not in cols:
            conn.execute("ALTER TABLE user_trade_ledger ADD COLUMN fee REAL DEFAULT 0.0")
        if "price_source" not in cols:
            conn.execute("ALTER TABLE user_trade_ledger ADD COLUMN price_source TEXT DEFAULT 'broker'")
        conn.executemany("INSERT INTO user_trade_ledger (traded_at, action, ticker, shares, price, fee, "
                         "price_source) VALUES (?,?,?,?,?,?,?)", [
            ("2026-09-01", "buy", "BIG", 2, 290.0, 5.0, "broker"),     # 2 shares but ~580 EGP
            ("2026-09-01", "buy", "DUST", 3, 9.0, 0.0, "accepted"),    # residue worth ~27 EGP
            ("2026-09-01", "buy", "KEEP", 100, 2.0, 5.0, "broker"),
        ])
        conn.executemany("INSERT INTO daily_ohlcv (ticker, date, close, adjusted_close) VALUES (?,?,?,?)",
                         [("BIG", "2026-09-30", 300.0, 300.0), ("DUST", "2026-09-30", 9.0, 9.0),
                          ("KEEP", "2026-09-30", 2.2, 2.2)])
        conn.execute("INSERT INTO user_positions (ticker, shares, avg_cost, updated_at) "
                     "VALUES ('KEEP', 150, 2.1, 'x')")
    return config.DB_PATH


def test_stage_classifies_by_value_not_share_count(book):
    q = lc.stage().set_index("ticker")
    assert q.loc["BIG", "action"] == "sell"            # 2 shares, but a real ~600 EGP exit
    assert q.loc["DUST", "action"] == "correction"     # worth < 100 EGP
    assert q.loc["KEEP", "action"] == "buy"            # 50 shares missing from the ledger
    # cost of the missing 50 = broker total 315 - ledger total 205 = 110 -> 2.20 a share
    assert q.loc["KEEP", "price"] == pytest.approx(2.20)
    assert len(lc.stage()) == 3                        # staging twice does not duplicate


def test_approval_reconciles_and_pnl_skips_corrections(book):
    q = lc.stage().set_index("ticker")
    with pytest.raises(ValueError):
        lc.approve(int(q.loc["BIG", "id"]))            # a sell needs its price
    lc.approve(int(q.loc["BIG", "id"]), price=310.0, fee=6.0, trade_date="2026-09-24")
    lc.approve(int(q.loc["DUST", "id"]))
    lc.approve(int(q.loc["KEEP", "id"]))
    assert lc.differences().empty
    p = lc.realised_pnl()
    assert list(p["ticker"]) == ["BIG"]                # the correction is not a trade
    # avg cost (580 + 5) / 2 = 292.5 ; (310 - 292.5) * 2 - 6 = 29
    assert p["pnl"].iloc[0] == pytest.approx(29.0)


def test_clerk_cannot_write_the_ledger_itself():
    raw = sqlite3.connect(":memory:")
    raw.executescript("CREATE TABLE user_trade_ledger (x); CREATE TABLE user_positions (x);"
                      "CREATE TABLE ledger_staging (x);")
    conn = ag.guarded(raw, lc.WRITABLE, lc.AGENT)
    for sql in ("INSERT INTO user_trade_ledger VALUES (1)", "UPDATE user_positions SET x = 0"):
        with pytest.raises(ag.WriteBoundaryViolation):
            ag.run(conn, sql, agent=lc.AGENT)
    ag.run(conn, "INSERT INTO ledger_staging VALUES (1)", agent=lc.AGENT)
