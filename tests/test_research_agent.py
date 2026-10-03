"""
Tests for the research agent -- mostly ATTACKS ON ITS WRITE BOUNDARY.

The boundary is the only safety property this module has. Everything else it
does is advisory: a wrong finding wastes attention, but a write that escapes
the guard could corrupt prices, marks or the trade ledger, and those are the
records the whole system reasons from.

The first implementation had exactly such a hole -- __getattr__ forwarded
`cursor` to the unguarded connection, so pandas.read_sql (and anything else
using a cursor) bypassed the check entirely. It was found by running the agent,
not by reading it. These tests exist so the next hole is found by CI.
"""

import pytest

from data.database import get_connection
from ops import research_agent as ra


# --------------------------------------------------------------------------
# The parser. Wrong table name -> wrong verdict, so it is tested directly.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sql,expected", [
    ("INSERT INTO stocks (ticker) VALUES ('X')", "stocks"),
    ("insert into daily_ohlcv values (1)", "daily_ohlcv"),
    ("UPDATE user_positions SET shares = 0", "user_positions"),
    ("DELETE FROM user_trade_ledger WHERE id = 1", "user_trade_ledger"),
    ("CREATE TABLE IF NOT EXISTS research_findings (id INTEGER)", "research_findings"),
    ("CREATE TABLE research_findings (id INTEGER)", "research_findings"),
    ("DROP TABLE IF EXISTS daily_ohlcv", "daily_ohlcv"),
    ("ALTER TABLE stocks ADD COLUMN x REAL", "stocks"),
    ("INSERT OR REPLACE INTO stocks (ticker) VALUES ('X')", "stocks"),
    # CREATE INDEX resolves to the table it indexes, not the index name
    ("CREATE INDEX IF NOT EXISTS idx_x ON research_findings (severity)", "research_findings"),
    ("CREATE INDEX idx_y ON daily_ohlcv (ticker)", "daily_ohlcv"),
    # quoted / bracketed identifiers
    ('INSERT INTO "daily_ohlcv" VALUES (1)', "daily_ohlcv"),
    ("INSERT INTO [daily_ohlcv] VALUES (1)", "daily_ohlcv"),
    ("INSERT INTO stocks(ticker) VALUES ('X')", "stocks"),
])
def test_target_table_parsing(sql, expected):
    assert ra._GuardedConnection._target_table(sql) == expected


def test_unparseable_write_fails_closed():
    """A statement the parser cannot read must be REFUSED, not allowed.

    '?' is not in WRITABLE, so an unrecognised write is blocked. This is the
    property that makes parser gaps safe: a gap costs a false refusal, never a
    silent write."""
    assert ra._GuardedConnection._target_table("INSERT weird syntax") == "?"
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        with pytest.raises(ra.WriteBoundaryViolation):
            conn.execute("INSERT weird syntax")


# --------------------------------------------------------------------------
# Attacks. Each of these is a path an unguarded wrapper would have permitted.
# --------------------------------------------------------------------------

FORBIDDEN = [
    "INSERT INTO daily_ohlcv (ticker, date, close) VALUES ('ZZZZ','2026-01-01',1.0)",
    "UPDATE daily_ohlcv SET close = 999 WHERE ticker = 'OIH'",
    "DELETE FROM daily_ohlcv WHERE ticker = 'OIH'",
    "UPDATE user_positions SET shares = 0 WHERE ticker = 'OIH'",
    "INSERT INTO user_trade_ledger (traded_at, action, ticker, shares, price) "
    "VALUES ('2026-01-01','buy','ZZZZ',1,1.0)",
    "DELETE FROM user_trade_ledger",
    "DROP TABLE daily_ohlcv",
    "UPDATE tradable_universe SET is_tradable = 1",
]


@pytest.mark.parametrize("sql", FORBIDDEN)
def test_connection_blocks_forbidden_writes(sql):
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        with pytest.raises(ra.WriteBoundaryViolation):
            conn.execute(sql)


@pytest.mark.parametrize("sql", FORBIDDEN)
def test_cursor_blocks_forbidden_writes(sql):
    """THE REGRESSION TEST FOR THE ACTUAL BUG.

    conn.cursor() used to return the RAW cursor, so every statement above
    executed happily against the live database."""
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        cur = conn.cursor()
        with pytest.raises(ra.WriteBoundaryViolation):
            cur.execute(sql)


def _scratch_db():
    """In-memory DB: these attacks must never be aimed at the real one -- if the
    guard regressed, the audit's reproduction showed the table is emptied."""
    import sqlite3
    raw = sqlite3.connect(":memory:")
    raw.executescript(
        "CREATE TABLE daily_ohlcv (ticker TEXT, close REAL);"
        "INSERT INTO daily_ohlcv VALUES ('OIH', 2.0), ('AMOC', 13.0);"
        "CREATE TABLE stocks (ticker TEXT PRIMARY KEY, sector TEXT);"
        "CREATE TABLE research_findings (id INTEGER PRIMARY KEY AUTOINCREMENT, note TEXT);")
    return raw


ENGINE_ATTACKS = [
    # the audit's bypass: the statement's first word is WITH, not a write verb
    "WITH t AS (SELECT 1) DELETE FROM daily_ohlcv",
    "WITH t AS (SELECT 1) UPDATE daily_ohlcv SET close = 0",
    "WITH t AS (SELECT 1) INSERT INTO daily_ohlcv VALUES ('X', 1)",
    # comment and whitespace prefixes defeat a first-token parser too
    "/* x */ DELETE FROM daily_ohlcv",
    "-- x\nDELETE FROM daily_ohlcv",
    "ATTACH DATABASE ':memory:' AS other",
    "PRAGMA writable_schema = 1",
    "CREATE TRIGGER tr AFTER INSERT ON stocks BEGIN DELETE FROM daily_ohlcv; END",
    "CREATE VIEW v AS SELECT * FROM daily_ohlcv",
]


@pytest.mark.parametrize("sql", ENGINE_ATTACKS)
def test_engine_authorizer_blocks_parser_bypasses(sql):
    raw = _scratch_db()
    conn = ra._GuardedConnection(raw)
    with pytest.raises(ra.WriteBoundaryViolation):
        conn.execute(sql)
    with pytest.raises(ra.WriteBoundaryViolation):
        conn.cursor().execute(sql)
    raw.set_authorizer(None)
    assert raw.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0] == 2


def test_engine_authorizer_allows_the_agents_own_writes():
    raw = _scratch_db()
    conn = ra._GuardedConnection(raw)
    conn.execute("INSERT INTO research_findings (note) VALUES ('ok')")       # AUTOINCREMENT
    conn.execute("INSERT INTO stocks (ticker, sector) VALUES ('X', 'Y') "
                 "ON CONFLICT(ticker) DO UPDATE SET sector = excluded.sector")
    conn.execute("ALTER TABLE stocks ADD COLUMN listing_date TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS i ON research_findings (note)")
    assert conn.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone()[0] == 2
    assert [r[1] for r in conn.execute("PRAGMA table_info(stocks)")][-1] == "listing_date"


def test_executemany_is_guarded():
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        with pytest.raises(ra.WriteBoundaryViolation):
            conn.executemany("INSERT INTO daily_ohlcv (ticker) VALUES (?)", [("A",)])
        with pytest.raises(ra.WriteBoundaryViolation):
            conn.cursor().executemany("INSERT INTO daily_ohlcv (ticker) VALUES (?)",
                                      [("A",)])


def test_executescript_checks_every_statement():
    """A script must not smuggle a forbidden write in behind a permitted one."""
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        with pytest.raises(ra.WriteBoundaryViolation):
            conn.executescript(
                "INSERT INTO research_findings (fingerprint) VALUES ('x');"
                "DELETE FROM daily_ohlcv;")


def test_reads_are_allowed():
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        assert conn.execute("SELECT COUNT(*) FROM daily_ohlcv").fetchone() is not None
        assert conn.cursor().execute("SELECT 1").fetchone()[0] == 1


def test_permitted_writes_succeed():
    """The allow-list must actually allow, or the agent cannot log anything."""
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        ra._ensure_findings_table(conn)          # CREATE TABLE + CREATE INDEX
        conn.execute(
            "INSERT INTO research_findings (first_seen, last_seen, fingerprint, "
            "severity, kind, summary) VALUES ('t','t','__test__','info','test','x') "
            "ON CONFLICT(fingerprint) DO UPDATE SET last_seen='t2'")
        conn.execute("DELETE FROM research_findings WHERE fingerprint='__test__'")


def test_pandas_read_sql_goes_through_the_guard():
    """pandas takes the cursor path; confirm it is the guarded one."""
    import pandas as pd
    with get_connection() as raw:
        conn = ra._GuardedConnection(raw)
        df = pd.read_sql("SELECT 1 AS one", conn)
        assert df["one"].iloc[0] == 1


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------

def test_fingerprint_is_stable_and_distinct():
    a = ra._fingerprint("dead_vendor_feed", "CRST")
    assert a == ra._fingerprint("dead_vendor_feed", "CRST")
    assert a != ra._fingerprint("dead_vendor_feed", "KORA")
    assert a != ra._fingerprint("unknown_security", "CRST")


def test_scan_survives_a_failing_check():
    """A reporter that dies on the first thing it cannot evaluate is worse than
    no reporter, so a raising check must degrade to a finding."""
    def boom(conn):
        raise ValueError("synthetic failure")
    df = ra.scan(persist=False, checks=(boom,))
    assert len(df) == 1
    assert df.iloc[0]["kind"] == "check_failed"
    assert "synthetic failure" in df.iloc[0]["detail"]


def test_scan_returns_expected_shape():
    df = ra.scan(persist=False)
    for col in ("severity", "kind", "subject", "summary", "detail", "needs_human"):
        assert col in df.columns
    assert set(df["severity"]) <= set(ra.SEVERITY_ORDER)


def test_format_scan_handles_empty():
    import pandas as pd
    assert "no integrity findings" in ra.format_scan(pd.DataFrame())


def test_scan_is_sorted_by_severity():
    df = ra.scan(persist=False)
    if len(df) > 1:
        ranks = [ra.SEVERITY_ORDER[s] for s in df["severity"]]
        assert ranks == sorted(ranks)
