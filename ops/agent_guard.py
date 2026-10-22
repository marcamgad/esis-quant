"""
Shared machinery for every agent (Stage 1, 2026-10-03).

Three things every agent needs and none may implement on its own:

  1. AN ENGINE-LEVEL WRITE BOUNDARY. guarded(conn, writable) installs a SQLite
     authorizer, so the database itself refuses any row write, DDL or ATTACH
     outside the agent's allow-list -- whatever the SQL text looks like. Parsing
     SQL text is not a boundary: `WITH t AS (SELECT 1) DELETE FROM daily_ohlcv`
     passed the old text check and emptied the table in the audit's reproduction.
     Prices, positions, the ledger and the trade list are in no agent's list.

  2. ONE FINDINGS LOG WITH CLOSURE. research_findings piled up with no owner and
     no due date -- a critical finding sat open from 11 Sep. agent_findings
     carries owner, due date and status, and a finding the agent no longer sees
     is auto-resolved, so "open" means "still true".

  3. A SYSTEM STATE the trade list must respect. system_state = HALT stops the
     ticket. Only an agent can set it and only an agent (or `sentinel clear`)
     clears it; it is re-evaluated every run, so a HALT lasts exactly as long as
     the condition behind it.
"""

import hashlib
import sqlite3

import pandas as pd

from data.database import get_connection

_INTERNAL = {"sqlite_master", "sqlite_temp_master", "sqlite_schema", "sqlite_sequence"}
_READ_PRAGMAS = {"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo",
                 "foreign_key_list", "table_list"}


class WriteBoundaryViolation(RuntimeError):
    """An agent attempted to write outside its allow-list."""


def make_authorizer(writable):
    """sqlite3 authorizer allowing reads everywhere and writes only to `writable`.

    Unknown action codes are denied, so a future SQLite feature fails closed.
    ATTACH, views, triggers, virtual tables and PRAGMA writes are always denied.
    """
    writable = set(writable)
    s = sqlite3

    def _authorize(action, arg1, arg2, dbname, source):
        if action in (s.SQLITE_READ, s.SQLITE_SELECT, s.SQLITE_FUNCTION, s.SQLITE_TRANSACTION,
                      s.SQLITE_SAVEPOINT, getattr(s, "SQLITE_RECURSIVE", 33)):
            return s.SQLITE_OK
        if action == s.SQLITE_REINDEX:
            return s.SQLITE_OK                    # CREATE INDEX builds it; no row changes
        if action == s.SQLITE_PRAGMA:
            # reads only: a bare pragma, or schema introspection on a named table
            return s.SQLITE_OK if (arg2 is None or str(arg1).lower() in _READ_PRAGMAS) else s.SQLITE_DENY
        if action in (s.SQLITE_INSERT, s.SQLITE_UPDATE, s.SQLITE_DELETE):
            return s.SQLITE_OK if (arg1 in writable or arg1 in _INTERNAL) else s.SQLITE_DENY
        if action in (s.SQLITE_CREATE_TABLE, s.SQLITE_DROP_TABLE):
            return s.SQLITE_OK if arg1 in writable else s.SQLITE_DENY
        if action in (s.SQLITE_CREATE_INDEX, s.SQLITE_DROP_INDEX, s.SQLITE_ALTER_TABLE):
            return s.SQLITE_OK if arg2 in writable else s.SQLITE_DENY
        return s.SQLITE_DENY

    return _authorize


def guarded(conn, writable, agent: str = "agent"):
    """Install the authorizer on `conn` and return it. Writes outside `writable`
    raise sqlite3.DatabaseError('not authorized'); use run() to convert."""
    conn.set_authorizer(make_authorizer(writable))
    return conn


def run(conn, sql, params=(), agent: str = "agent"):
    """Execute on a guarded connection, turning an engine refusal into a clear error."""
    try:
        return conn.execute(sql, params)
    except sqlite3.DatabaseError as exc:
        if "not authorized" in str(exc):
            raise WriteBoundaryViolation(f"{agent}: write refused by the database engine: "
                                         f"{sql[:120]!r}") from exc
        raise


# --------------------------------------------------------------------------
# findings with closure
# --------------------------------------------------------------------------
FINDINGS_TABLES = {"agent_findings"}
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "info": 3}
DUE_DAYS = {"critical": 1, "high": 3, "medium": 14, "info": 30}


def ensure_tables(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS agent_findings (
        fingerprint TEXT PRIMARY KEY, agent TEXT NOT NULL, severity TEXT NOT NULL,
        kind TEXT NOT NULL, subject TEXT, summary TEXT, fix_hint TEXT,
        status TEXT NOT NULL DEFAULT 'open', owner TEXT, due_date TEXT,
        first_seen TEXT, last_seen TEXT, resolved_at TEXT, times_seen INTEGER DEFAULT 1)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS system_state (
        id INTEGER PRIMARY KEY CHECK (id = 1), state TEXT NOT NULL, reason TEXT,
        set_at TEXT, set_by TEXT)""")


def fingerprint(agent: str, kind: str, subject: str) -> str:
    return hashlib.sha1(f"{agent}|{kind}|{subject}".encode()).hexdigest()[:16]


def record_findings(conn, agent: str, findings: list, owner: str = "you") -> dict:
    """Upsert this run's findings and auto-resolve the agent's findings that
    were NOT seen this run. Returns {new: [...], resolved: n, open: n}.

    Call ensure_tables() on an unguarded connection first: schema setup is not
    an agent action, and an agent's guard rightly refuses DDL on tables it does
    not own (system_state belongs to the Sentinel alone)."""
    now = pd.Timestamp.now()
    seen, new = set(), []
    for f in findings:
        fp = fingerprint(agent, f["kind"], f.get("subject") or "")
        seen.add(fp)
        due = (now + pd.Timedelta(days=DUE_DAYS.get(f["severity"], 14))).date().isoformat()
        row = conn.execute("SELECT status FROM agent_findings WHERE fingerprint = ?", (fp,)).fetchone()
        if row is None or row[0] == "resolved":
            new.append(f)
        conn.execute(
            "INSERT INTO agent_findings (fingerprint, agent, severity, kind, subject, summary, fix_hint, "
            "status, owner, due_date, first_seen, last_seen) VALUES (?,?,?,?,?,?,?, 'open', ?,?,?,?) "
            "ON CONFLICT(fingerprint) DO UPDATE SET severity=excluded.severity, summary=excluded.summary, "
            "fix_hint=excluded.fix_hint, last_seen=excluded.last_seen, times_seen=times_seen+1, "
            "status='open', resolved_at=NULL, "
            "due_date=CASE WHEN agent_findings.status='resolved' THEN excluded.due_date "
            "ELSE agent_findings.due_date END, "
            "first_seen=CASE WHEN agent_findings.status='resolved' THEN excluded.first_seen "
            "ELSE agent_findings.first_seen END",
            (fp, agent, f["severity"], f["kind"], f.get("subject"), f["summary"], f.get("fix_hint"),
             owner, due, now.isoformat(timespec="seconds"), now.isoformat(timespec="seconds")))
    open_rows = [r[0] for r in conn.execute(
        "SELECT fingerprint FROM agent_findings WHERE agent = ? AND status = 'open'", (agent,))]
    stale = [fp for fp in open_rows if fp not in seen]
    for fp in stale:
        conn.execute("UPDATE agent_findings SET status='resolved', resolved_at=? WHERE fingerprint=?",
                     (now.isoformat(timespec="seconds"), fp))
    return {"new": new, "resolved": len(stale), "open": len(seen)}


def open_findings(agent: str = None) -> pd.DataFrame:
    with get_connection() as conn:
        ensure_tables(conn)
        q = "SELECT * FROM agent_findings WHERE status = 'open'"
        params = ()
        if agent:
            q += " AND agent = ?"
            params = (agent,)
        df = pd.read_sql(q, conn, params=params)
    if df.empty:
        return df
    df["_o"] = df["severity"].map(SEVERITY_ORDER)
    return df.sort_values(["_o", "due_date"]).drop(columns="_o")


# --------------------------------------------------------------------------
# system state
# --------------------------------------------------------------------------
def set_state(conn, state: str, reason: str, by: str):
    ensure_tables(conn)
    conn.execute("INSERT INTO system_state (id, state, reason, set_at, set_by) VALUES (1,?,?,?,?) "
                 "ON CONFLICT(id) DO UPDATE SET state=excluded.state, reason=excluded.reason, "
                 "set_at=excluded.set_at, set_by=excluded.set_by",
                 (state, reason, pd.Timestamp.now().isoformat(timespec="seconds"), by))


def get_state() -> dict:
    with get_connection() as conn:
        ensure_tables(conn)
        r = conn.execute("SELECT state, reason, set_at, set_by FROM system_state WHERE id = 1").fetchone()
    return dict(r) if r else {"state": "OK", "reason": None, "set_at": None, "set_by": None}


class SystemHalted(RuntimeError):
    """The sentinel has halted ticket production."""


def assert_not_halted():
    s = get_state()
    if s["state"] == "HALT":
        raise SystemHalted(f"system HALTED by {s['set_by']} at {s['set_at']}: {s['reason']}. "
                           "Fix the cause (py -3.11 -m ops.sentinel) or clear it "
                           "(py -3.11 -m ops.sentinel clear).")
