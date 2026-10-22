"""
Agent 1 -- SENTINEL. Liveness and freshness of everything the ticket depends on.

WHY IT EXISTS. Every critical failure found in the 2026-10-02 audit was
operational, and each was visible in the database for weeks while nothing read
it: the regime ensemble frozen at 2026-07-30, the daily job killed on 23, 24, 28
Sep and 1 Oct, a null close collapsing the universe to one name. The Sentinel is
the thing that reads it.

DETERMINISTIC ON PURPOSE. No model, no prose generation: an invariant table of
(artifact, how far behind it may be), a job-liveness check, and a small library
mapping known errors to their cause. The reliable 75% of an agent is rules.

WHAT IT MAY DO -- enforced by the database engine (ops.agent_guard), not by
convention:
  writes   agent_findings, system_state, job_runs (only to close killed runs)
  never    prices, positions, the ledger, the trade list, weights
It can STOP the ticket (system_state = HALT). It can never add a name or a weight.

HALT RULE. Any CRITICAL finding sets HALT; the trade list and the ticket refuse
to run while it stands. The state is recomputed on every run, so a HALT lasts
exactly as long as the condition behind it. `sentinel clear` overrides once.

    py -3.11 -m ops.sentinel            # check now, print the brief
    py -3.11 -m ops.sentinel --alert    # also pop an alert on new problems (scheduled use)
    py -3.11 -m ops.sentinel clear      # clear a HALT by hand
"""

import logging
import os
import subprocess
import sys

import pandas as pd

from data.database import get_connection
from ops import agent_guard as ag

logger = logging.getLogger("esis_quant.sentinel")

AGENT = "sentinel"
WRITABLE = {"agent_findings", "system_state", "job_runs"}
CAIRO = "Africa/Cairo"
DAILY_JOB_DONE_HOUR = 18          # the 17:00 job should have ingested today's session by 18:00

# (artifact, table, column, compare_to, warn_after, critical_after, why)
#   compare_to "expected": EGX weekdays behind the session that SHOULD be in the DB by now
#   compare_to "prices":   sessions behind the last valid price session actually in the DB
#   compare_to "days":     calendar days behind now
INVARIANTS = [
    ("prices", "daily_ohlcv", "date", "expected", 2, 3,
     "the price fetch or the daily job is not running (or a multi-day market holiday)"),
    ("macro (USD/EGP)", "macro_daily", "date", "days", 4, 8,
     "macro feed stale; the regime signal is built from it"),
    ("regime_history", "regime_history", "date", "prices", 2, 3,
     "the daily regime step did not finish"),
    ("regime_ensemble", "regime_ensemble", "date", "prices", 2, 3,
     "the exposure signal is stale; the ticket's exposure would be old"),
    ("trade list", "trade_list_log", "as_of_date", "prices", 2, 99,
     "no fresh target book; the ticket would be built from an old one"),
    ("paper record", "paper_trading_nav", "date", "prices", 3, 99,
     "the forward paper record (the go-live evidence) is not being kept"),
    ("news", "news_items", "first_seen_at", "days", 2, 99,
     "the news collector is not running"),
    ("fundamentals", "fundamentals", "fetched_at", "days", 10, 99,
     "the weekly fundamentals refresh is not running"),
    ("backup", "backup_log", "finished_at", "days", 2, 99,
     "the nightly backup (ops/backup.py, 02:00) is not running: the ledger and history are unprotected"),
]

# Known failures -> what they mean. Matched against the end of the daily log and
# Windows Task Scheduler result codes.
LOG_PATTERNS = [
    (r"\^C\s*$", "the run was interrupted: the PC was shut down or slept, or the console window was closed"),
    (r"database is locked", "another process held the database; usually two jobs overlapping"),
    (r"macro \(usd_egp\) is \d+ days stale", "the macro feed is stale; rerun the fetch"),
    (r"tradable universe is \d+ names", "the tradable universe collapsed; usually a bad price row"),
    (r"price fetch FAILED", "Yahoo failed for some tickers; usually transient, rerun later"),
    (r"timeout after \d+ min", "the job hung and the watchdog stopped it"),
]
TASK_RESULTS = {
    0x0: "succeeded",
    0xC000013A: "killed: the PC was shut down, went to sleep, or the console was closed during the run",
    0x8004131F: "an earlier instance was still running, so this one was skipped",
    0x41301: "still running",
    0x1: "the program exited with an error (see logs/daily_run.log)",
}
# Tasks whose result is checked, and when each is meant to run. The Daily Popup
# is excluded: it opens a window you close, which always ends as 0xC000013A.
TASKS = {"ESIS-QUANT Daily": "17:00 Sun-Thu", "ESIS-QUANT Fetch": "09:00 daily",
         "ESIS-QUANT News": "08:30 and 16:00 daily", "ESIS-QUANT Fundamentals Weekly": "Saturday 11:00",
         "ESIS-QUANT Sentinel": "hourly Sun-Thu", "ESIS-QUANT Backup": "02:00 daily"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _holidays() -> set:
    try:
        from data.egx_holidays import holidays
        return holidays()
    except Exception:  # noqa: BLE001
        return set()


def _egx_weekdays_between(a, b) -> int:
    """EGX sessions in (a, b]: Sun-Thu minus known holidays (data/egx_holidays.py)."""
    a, b = pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()
    if b <= a:
        return 0
    hol = _holidays()
    days = pd.date_range(a + pd.Timedelta(days=1), b, freq="D")
    return int(sum(d.dayofweek not in (4, 5) and d not in hol for d in days))


def expected_session(now=None) -> pd.Timestamp:
    """The newest session the database SHOULD hold by now: today after the daily
    job has had time to run on a trading day, otherwise the previous session
    (Sun-Thu, skipping known holidays)."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=CAIRO).tz_localize(None)
    hol = _holidays()
    d = now.normalize()
    if d.dayofweek in (4, 5) or d in hol or now.hour < DAILY_JOB_DONE_HOUR:
        d -= pd.Timedelta(days=1)
    while d.dayofweek in (4, 5) or d in hol:
        d -= pd.Timedelta(days=1)
    return d


def _last_valid_session():
    from data.calendar import valid_trading_days
    s = valid_trading_days()
    return s[-1] if len(s) else None


def _max(conn, table, col):
    try:
        r = conn.execute(f"SELECT MAX({col}) FROM {table}").fetchone()
        return pd.Timestamp(r[0]) if r and r[0] else None
    except Exception:  # noqa: BLE001 -- table missing
        return None


def task_results() -> dict:
    """{task: (last_run, result_code)} from Windows Task Scheduler; {} elsewhere."""
    if not sys.platform.startswith("win"):
        return {}
    cmd = ("Get-ScheduledTask | Where-Object TaskName -like 'ESIS-QUANT*' | ForEach-Object { "
           "$i = Get-ScheduledTaskInfo -TaskName $_.TaskName; "
           "'{0}|{1:yyyy-MM-ddTHH:mm:ss}|{2}' -f $_.TaskName, $i.LastRunTime, $i.LastTaskResult }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return {}
    res = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and parts[2].lstrip("-").isdigit():
            res[parts[0]] = (parts[1], int(parts[2]) & 0xFFFFFFFF)
    return res


def diagnose_log(path=None, tail_bytes: int = 20000) -> str:
    """Cause of the last daily-run failure, from the end of the daily log."""
    import re
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "logs", "daily_run.log")
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - tail_bytes))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    last_run = tail[tail.rfind("validate_startup"):] if "validate_startup" in tail else tail
    if "DAILY JOB OK" in last_run:
        return ""
    for pat, meaning in LOG_PATTERNS:
        if re.search(pat, last_run, re.M):
            return meaning
    if "Traceback" in last_run:
        line = [l for l in last_run.splitlines() if l.strip()][-1]
        return f"crashed: {line.strip()[:160]}"
    return ""


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check(now=None, with_tasks: bool = True) -> list:
    """All findings, as dicts {severity, kind, subject, summary, fix_hint}."""
    out = []
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=CAIRO).tz_localize(None)
    expected = expected_session(now)
    last_px = _last_valid_session()
    from data.calendar import valid_trading_days
    sessions = valid_trading_days()

    with get_connection() as conn:
        for name, table, col, cmp, warn, crit, why in INVARIANTS:
            last = _max(conn, table, col)
            if table == "daily_ohlcv":
                last = last_px                       # the newest GENUINE session, not a stray print
            if last is None:
                out.append({"severity": "critical" if crit < 99 else "high", "kind": "missing",
                            "subject": name, "summary": f"{name}: no data at all", "fix_hint": why})
                continue
            if cmp == "expected":
                lag, unit = _egx_weekdays_between(last, expected), "trading days"
            elif cmp == "prices":
                lag = int((sessions > last).sum()) if len(sessions) else 0
                unit = "sessions"
            else:
                lag, unit = (now.normalize() - last.normalize()).days, "days"
            sev = "critical" if lag >= crit else ("high" if lag >= warn else None)
            if sev == "high" and name in ("news", "fundamentals"):
                sev = "medium"                       # context feeds: worth fixing, never a halt
            if sev:
                out.append({"severity": sev, "kind": "stale", "subject": name,
                            "summary": f"{name} ends {last.date()}, {lag} {unit} behind",
                            "fix_hint": why})

        # universe on the newest genuine session
        try:
            from data.quality import MIN_UNIVERSE
            r = conn.execute("SELECT SUM(is_tradable) FROM tradable_universe WHERE date = "
                             "(SELECT MAX(date) FROM tradable_universe WHERE date <= ?)",
                             (str(last_px.date()) if last_px is not None else "9999",)).fetchone()
            n = int(r[0] or 0)
            if n < MIN_UNIVERSE:
                out.append({"severity": "critical", "kind": "universe", "subject": "tradable universe",
                            "summary": f"tradable universe is {n} names (< {MIN_UNIVERSE})",
                            "fix_hint": "usually one bad price row on the newest date; rebuild the universe"})
        except Exception:  # noqa: BLE001
            pass

        # null closes in the last 10 days
        nulls = conn.execute("SELECT COUNT(*) FROM daily_ohlcv WHERE (close IS NULL OR adjusted_close "
                             "IS NULL) AND date >= ?", (str((now - pd.Timedelta(days=10)).date()),)).fetchone()[0]
        if nulls:
            out.append({"severity": "high", "kind": "null_close", "subject": "daily_ohlcv",
                        "summary": f"{nulls} recent price row(s) have no close",
                        "fix_hint": "ingest should reject these; delete them and rebuild the universe"})

        # jobs: rows still 'running' long after any normal duration
        jr = pd.read_sql("SELECT id, job, started_at, finished_at FROM job_runs", conn)
    if not jr.empty:
        done = jr.dropna(subset=["finished_at"]).copy()
        done["mins"] = (pd.to_datetime(done["finished_at"]) - pd.to_datetime(done["started_at"])).dt.total_seconds() / 60
        med = done.groupby("job")["mins"].median().to_dict()
        hung = jr[jr["finished_at"].isna()].copy()
        for r in hung.itertuples():
            limit = max(20.0, 3 * med.get(r.job, 20.0))
            age = (now - pd.Timestamp(r.started_at)).total_seconds() / 60
            if age > limit:
                hint = ((diagnose_log() if r.job == "daily_run" else "") or
                        "most often the PC was shut down or slept while it ran; rerun it by hand")
                out.append({"severity": "high", "kind": "killed_run", "subject": f"{r.job}#{r.id}",
                            "summary": f"{r.job} started {r.started_at} never finished "
                                       f"(>{limit:.0f} min); marked killed",
                            "fix_hint": hint, "_close_run": int(r.id)})

    if not with_tasks:                 # called from inside the daily job: its own log is mid-run
        return out
    # The job record outranks the log and the scheduler: a later successful run
    # (by hand or catch-up) means the earlier failure is history, not a problem.
    last_ok = None
    if not jr.empty:
        with get_connection() as conn:
            r = conn.execute("SELECT MAX(started_at) FROM job_runs WHERE job = 'daily_run' AND ok = 1").fetchone()
            last_ok = pd.Timestamp(r[0]) if r and r[0] else None
            r = conn.execute("SELECT ok FROM job_runs WHERE job = 'daily_run' AND finished_at IS NOT NULL "
                             "ORDER BY started_at DESC LIMIT 1").fetchone()
            last_daily_ok = bool(r and r[0] == 1)
    else:
        last_daily_ok = False
    why = "" if last_daily_ok else diagnose_log()
    if why:
        out.append({"severity": "high", "kind": "daily_job_failed", "subject": "daily_run",
                    "summary": f"the last daily run did not complete: {why}",
                    "fix_hint": "run it by hand: py -3.11 -m ops.daily_run"})
    for task, (when, code) in task_results().items():
        if task not in TASKS or code in (0, 0x41301):
            continue
        if task == "ESIS-QUANT Daily" and last_ok is not None and last_ok >= pd.Timestamp(when):
            continue                    # a successful run since then has caught up
        meaning = TASK_RESULTS.get(code, f"failed with code 0x{code:X}")
        out.append({"severity": "high" if task == "ESIS-QUANT Daily" else "medium",
                    "kind": "task_failed", "subject": task,
                    "summary": f"{task} last run {when}: {meaning}",
                    "fix_hint": f"it runs {TASKS[task]}; keep the laptop on and plugged in then, "
                                "or run it by hand"})
    return out


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def run(alert: bool = False, now=None, with_tasks: bool = True) -> dict:
    findings = check(now=now, with_tasks=with_tasks)
    crit = [f for f in findings if f["severity"] == "critical"]
    prev = ag.get_state()
    with get_connection() as raw:
        ag.ensure_tables(raw)                            # schema setup is not an agent action
    with get_connection() as raw:
        conn = ag.guarded(raw, WRITABLE, AGENT)
        for f in findings:
            rid = f.pop("_close_run", None)
            if rid is not None:
                ag.run(conn, "UPDATE job_runs SET finished_at = ?, ok = 0, message = ? WHERE id = ? "
                             "AND finished_at IS NULL",
                       (pd.Timestamp.now().isoformat(timespec="seconds"),
                        f"killed (closed by sentinel): {f['fix_hint']}"[:2000], rid), AGENT)
        rec = ag.record_findings(conn, AGENT, findings)
        if crit:
            reason = "; ".join(f["summary"] for f in crit)
            ag.set_state(conn, "HALT", reason, AGENT)
        elif prev.get("state") == "HALT" and prev.get("set_by") == AGENT:
            ag.set_state(conn, "OK", "all sentinel invariants green", AGENT)
    state = ag.get_state()
    if alert:
        newly = [f for f in rec["new"] if f["severity"] in ("critical", "high")]
        if newly or (state["state"] == "HALT" and prev.get("state") != "HALT"):
            try:
                from ops.health import alert as _alert
                _alert("ESIS-QUANT Sentinel" + (" — HALT" if state["state"] == "HALT" else ""),
                       "\n".join(f"[{f['severity']}] {f['summary']}" for f in (newly or crit))
                       + "\n\nDetails: py -3.11 -m ops.sentinel")
            except Exception:  # noqa: BLE001
                pass
    return {"findings": findings, "state": state, "new": rec["new"], "resolved": rec["resolved"]}


def format_brief(res: dict) -> str:
    s = res["state"]
    L = ["=" * 66, f"SENTINEL — system {s['state']}" + (f"  ({s['reason']})" if s["state"] == "HALT" else ""),
         "=" * 66]
    if not res["findings"]:
        L.append("  all invariants green: prices, macro, regime, ensemble, ticket, jobs.")
    for sev in ("critical", "high", "medium", "info"):
        for f in [f for f in res["findings"] if f["severity"] == sev]:
            L.append(f"  [{sev.upper():8}] {f['summary']}")
            if f.get("fix_hint"):
                L.append(f"             -> {f['fix_hint']}")
    if res.get("resolved"):
        L.append(f"  {res['resolved']} earlier finding(s) no longer true and closed.")
    return "\n".join(L)


def clear(reason: str = "cleared by hand") -> None:
    with get_connection() as raw:
        ag.ensure_tables(raw)
        ag.set_state(ag.guarded(raw, WRITABLE, AGENT), "OK", reason, "manual")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if argv[:1] == ["clear"]:
        clear()
        print("system state cleared to OK (the next sentinel run re-checks)")
        return 0
    res = run(alert="--alert" in argv)
    print(format_brief(res))
    return 2 if res["state"]["state"] == "HALT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
