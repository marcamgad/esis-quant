"""
Job health tracking + failure alerting.

THE FAILURE THIS EXISTS TO PREVENT (it already happened). On 2026-07-23 the
daily job died: yfinance printed 2 of 27 tickers on Revolution Day (EGX shut),
the calendar filter had a right-edge bug, the tradable universe evaluated to 0
names, and the data-quality gate correctly REFUSED to emit a trade list. All
of that was right -- except nobody was told. The job stayed dead for three days
and the user only found out by asking. For a system whose entire edge is a
de-risk signal, a silent outage means the one message that matters (BEAR -> go
to cash) can never arrive.

So: every run records its outcome, and a failure is made LOUD.
  * record_run() writes success/failure to the job_runs table (audit trail).
  * health_status() reports whether the last run failed, or whether too long
    has passed since the last SUCCESSFUL run (a job that never starts leaves no
    failure row -- absence of news is not good news, so staleness is its own
    alarm).
  * alert() raises a Windows message box, so a failure is visible without the
    user going looking. No credentials, no network, nothing to configure.

The popup and the daily report both call health_status() and show a banner, so
the state is impossible to miss from either entry point.
"""

import logging
import subprocess
import sys

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.health")

# No successful run in this many days on a trading-day schedule = something is
# wrong (EGX trades Sun-Thu, so a healthy system runs at least every ~3 days).
MAX_DAYS_SINCE_SUCCESS = 3


def record_run(job: str, ok: bool, started_at=None, as_of_date=None, message: str = "") -> int:
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    started = pd.Timestamp(started_at).isoformat(timespec="seconds") if started_at is not None else now
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO job_runs (job, started_at, finished_at, ok, as_of_date, message) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job, started, now, 1 if ok else 0,
             str(as_of_date) if as_of_date is not None else None, str(message)[:2000]))
        return int(cur.lastrowid)


def start_run(job: str, message: str = "") -> int:
    """Claim a row BEFORE the work, so a killed run is visible instead of absent.

    THE GAP THIS CLOSES. job_runs was only ever written on completion, so a run
    that died mid-flight left no trace at all -- and "no row" is exactly what a
    run that never fired also looks like. Between 2026-09-01 and 09-03 the
    scheduler reported 0xC000013A (machine shut down mid-run) on three
    consecutive days while job_runs stayed empty for that window, so from the
    database alone the outage was indistinguishable from a laptop that had
    simply been switched off. Health checks read this table, so they scored
    three kills as three no-shows.

    Pair with finish_run(). A row still marked 'running' long after the fact IS
    the evidence of a kill.
    """
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO job_runs (job, started_at, finished_at, ok, as_of_date, message) "
            "VALUES (?, ?, NULL, NULL, NULL, ?)", (job, now, ("running: " + str(message))[:2000]))
        return int(cur.lastrowid)


def finish_run(run_id: int, ok: bool, as_of_date=None, message: str = "") -> None:
    """Close a row opened by start_run()."""
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute(
            "UPDATE job_runs SET finished_at = ?, ok = ?, as_of_date = ?, message = ? "
            "WHERE id = ?",
            (now, 1 if ok else 0,
             str(as_of_date) if as_of_date is not None else None, str(message)[:2000], run_id))


def killed_runs(job: str = None, older_than_hours: float = 3.0) -> pd.DataFrame:
    """Rows still open long enough that the process cannot plausibly be alive.

    These are the runs that were killed rather than the runs that failed. A
    failure writes ok=0 and a reason; a kill writes nothing, which is why it
    needed its own query rather than a flag.
    """
    cutoff = (pd.Timestamp.now() - pd.Timedelta(hours=older_than_hours)).isoformat(timespec="seconds")
    q = ("SELECT id, job, started_at, message FROM job_runs "
         "WHERE finished_at IS NULL AND started_at < ?")
    params = [cutoff]
    if job:
        q += " AND job = ?"
        params.append(job)
    with get_connection() as conn:
        return pd.read_sql(q + " ORDER BY started_at DESC", conn, params=tuple(params))


def last_run(job: str = "daily_run") -> dict:
    with get_connection() as conn:
        row = conn.execute(
            # finished_at IS NOT NULL keeps an in-flight row (start_run, not yet
            # closed) from being read as "the last run", which would report a
            # currently-running job as a failure because its ok is still NULL.
            "SELECT job, started_at, finished_at, ok, as_of_date, message FROM job_runs "
            "WHERE job = ? AND finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT 1", (job,)).fetchone()
    return dict(row) if row else {}


def last_success(job: str = "daily_run") -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT job, finished_at, as_of_date, message FROM job_runs "
            "WHERE job = ? AND ok = 1 ORDER BY finished_at DESC LIMIT 1", (job,)).fetchone()
    return dict(row) if row else {}


def health_status(job: str = "daily_run", max_days: int = MAX_DAYS_SINCE_SUCCESS) -> dict:
    """Is the pipeline healthy? Two independent alarms: the last run FAILED, or
    no run has SUCCEEDED recently (covers the case where it never ran at all)."""
    lr, ls = last_run(job), last_success(job)
    problems = []

    # A killed run is its own alarm, distinct from a failure. Without this the
    # three consecutive 0xC000013A kills of 2026-09-01..03 read as "no run",
    # which is also what an switched-off laptop looks like -- so the health
    # check could not tell "your job is dying" from "you weren't here".
    killed = killed_runs(job)
    if len(killed):
        problems.append(f"{len(killed)} run(s) started but never finished -- killed mid-run "
                        f"(most recent {str(killed.iloc[0]['started_at'])[:16]})")

    if not lr:
        problems.append("no run has ever been recorded")
        days = None
    else:
        if not lr.get("ok"):
            problems.append(f"last run FAILED: {str(lr.get('message',''))[:160]}")
        days = None
        if ls:
            days = (pd.Timestamp.now().normalize()
                    - pd.Timestamp(ls["finished_at"]).normalize()).days
            if days > max_days:
                problems.append(f"no successful run in {days} days "
                                f"(last {str(ls['finished_at'])[:16]})")
        else:
            problems.append("no successful run on record")

    return {"ok": not problems, "problems": problems, "last_run": lr,
            "last_success": ls, "days_since_success": days, "job": job}


def format_health(h: dict) -> str:
    if h["ok"]:
        ls = h.get("last_success") or {}
        return (f"  HEALTHY — last successful run {str(ls.get('finished_at',''))[:16]}"
                f" (book as-of {ls.get('as_of_date','?')})")
    lines = ["  *** PIPELINE PROBLEM — the daily job is not producing suggestions ***"]
    for p in h["problems"]:
        lines.append(f"    - {p}")
    lines.append("    fix: run  py -3.11 -m ops.daily_run   and read the error it prints.")
    lines.append("    until then, any regime change (BEAR -> cash) will NOT reach you.")
    return "\n".join(lines)


def alert(title: str, message: str) -> bool:
    """Pop a Windows message box. Best-effort and never raises -- an alerting
    failure must not take down the job it is reporting on."""
    if not sys.platform.startswith("win"):
        logger.warning("ALERT (%s): %s", title, message)
        return False
    try:
        safe = str(message).replace("'", "''")[:900]
        safe_title = str(title).replace("'", "''")[:120]
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
             "Add-Type -AssemblyName System.Windows.Forms | Out-Null; "
             f"[System.Windows.Forms.MessageBox]::Show('{safe}','{safe_title}',"
             "'OK','Error') | Out-Null"],
            timeout=30, capture_output=True)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("health.alert could not display a message box: %s", exc)
        return False


def alert_if_unhealthy(job: str = "daily_run") -> dict:
    h = health_status(job)
    if not h["ok"]:
        alert("ESIS-QUANT — daily job problem",
              "The daily job is not producing suggestions.\n\n"
              + "\n".join(h["problems"])
              + "\n\nRun:  py -3.11 -m ops.daily_run\n"
                "Until fixed, a BEAR de-risk signal cannot reach you.")
    return h


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    h = health_status()
    print(format_health(h))
    if not h["ok"] and "--alert" in sys.argv:
        alert_if_unhealthy()


# --- drift alerting beyond job failure  (audit P6.20) -----------------------
# The audit found alerting covered only "the job died" and regime-consensus
# drift. Two things that silently invalidate every number were unwatched: fills
# coming in worse than modelled, and the fee model diverging from what the
# broker actually charges. Both degrade gradually, which is exactly the failure
# mode a human never notices unaided.

EXEC_SHORTFALL_ALERT = 0.010     # measured one-way shortfall above 1.0%
COST_MODEL_ERROR_ALERT = 0.20    # modelled fees off by more than 20%


def execution_drift() -> dict:
    """Is realised slippage drifting away from the modelled assumption?"""
    try:
        from ops.execution_quality import measured_slippage
        m = measured_slippage()
    except Exception as exc:  # noqa: BLE001
        return {"check": "execution_drift", "ok": True, "conclusive": False,
                "reason": f"execution quality unreadable ({exc})"}
    if not m or not m.get("usable"):
        # Not enough fills is NOT a pass -- it is an absence of evidence, and
        # saying so is the difference between "fine" and "unknown".
        return {"check": "execution_drift", "ok": True, "conclusive": False,
                "n_fills": m.get("n_fills", 0),
                "reason": f"only {m.get('n_fills', 0)} fills — too few to conclude"}
    breach = m["measured"] > EXEC_SHORTFALL_ALERT
    return {"check": "execution_drift", "ok": not breach, "conclusive": True,
            "n_fills": m["n_fills"], "measured": m["measured"],
            "prior": m["prior"], "estimate": m["estimate"],
            "reason": ("" if not breach else
                       f"measured shortfall {m['measured']*100:.2f}% exceeds the "
                       f"{EXEC_SHORTFALL_ALERT*100:.1f}% alert level (modelled "
                       f"{m['prior']*100:.2f}%) — backtest costs are optimistic")}


def cost_model_drift() -> dict:
    """Do the fees we were actually charged match backtest/fees.py?

    This is the check that would have caught the EGP 2 -> EGP 4.95 error before
    it distorted a year of backtests, so it is worth running every day even
    though it is cheap and usually silent.
    """
    import pandas as pd
    from data.database import get_connection
    try:
        with get_connection() as conn:
            led = pd.read_sql("SELECT ticker, shares, price, fee FROM user_trade_ledger "
                              "WHERE fee > 0", conn)
    except Exception as exc:  # noqa: BLE001
        return {"check": "cost_model_drift", "ok": True, "conclusive": False,
                "reason": f"ledger unreadable ({exc})"}
    if led.empty:
        return {"check": "cost_model_drift", "ok": True, "conclusive": False,
                "reason": "no fees recorded yet"}
    from backtest.fees import one_side_cost
    notional = (led["shares"] * led["price"]).astype(float)
    actual = float(led["fee"].sum())
    predicted = float(sum(one_side_cost(float(v), spread_per_side=0.0)["total_egp"]
                          for v in notional))
    if predicted <= 0:
        return {"check": "cost_model_drift", "ok": True, "conclusive": False,
                "reason": "model predicted zero fees — cannot compare"}
    err = (actual - predicted) / predicted
    breach = abs(err) > COST_MODEL_ERROR_ALERT
    return {"check": "cost_model_drift", "ok": not breach, "conclusive": True,
            "n_orders": int(len(led)), "actual_egp": actual, "predicted_egp": predicted,
            "error_pct": err,
            "reason": ("" if not breach else
                       f"fees charged {actual:,.2f} EGP vs {predicted:,.2f} modelled "
                       f"({err*100:+.0f}%) — backtest/fees.py needs recalibrating")}


def drift_report(send: bool = False) -> dict:
    """Run every drift check; optionally alert on breaches only."""
    checks = [execution_drift(), cost_model_drift()]
    breaches = [c for c in checks if not c["ok"]]
    if send and breaches:
        alert("ESIS-QUANT — COST/EXECUTION DRIFT",
              "\n\n".join(c["reason"] for c in breaches) +
              "\n\nEvery backtest number assumes these stay in line.")
    return {"ok": not breaches, "checks": checks, "breaches": breaches}


def format_drift(d: dict) -> str:
    lines = ["DRIFT CHECKS"]
    for c in d["checks"]:
        mark = "ok " if c["ok"] else "!! "
        detail = c["reason"] or ("within limits" if c.get("conclusive") else "inconclusive")
        lines.append(f"  {mark}{c['check']}: {detail}")
    return "\n".join(lines)
