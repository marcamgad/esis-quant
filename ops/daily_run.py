"""
Scheduled-job entry point (audit Part 2.2). This is what the OS scheduler
invokes -- a thin, hardened wrapper around main.run_daily_job that:

  * configures file+console logging (so a scheduled run leaves a trace),
  * runs the daily job inside a top-level guard, and
  * exits NON-ZERO on any failure, so the scheduler records the run as failed
    rather than silently "succeeding" while having traded nothing / bad data.

Run manually:
    py -3.11 -m ops.daily_run
    py -3.11 -m ops.daily_run --alert     # also send the Telegram brief

Schedule (EGX trades Sun-Thu): see register_daily_task.ps1 / the docstring in
main.run_daily_job for the cron/schtasks line.
"""

import sys
import logging


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    send_alert = "--alert" in argv

    # the report's news section prints Arabic headlines; a cp1252 console (the
    # scheduled-task default) chokes on them, so force utf-8 up front.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    import main as esis_main
    logger = esis_main.setup_logging()
    log = logging.getLogger("esis_quant.daily_run")

    import pandas as pd
    from ops.health import record_run, alert, start_run, finish_run, killed_runs

    # HEARTBEAT (Stage 0). The runs of 23, 24 and 28 Sep were killed inside the
    # regime refit and left NO row, because job_runs was written only at the end
    # -- from the database the job looked like it had never been scheduled.
    # Close any earlier row that is still 'running' as killed, then claim a row
    # BEFORE the work starts.
    for r in killed_runs("daily_run", older_than_hours=1.0).itertuples():
        finish_run(int(r.id), ok=False, message=f"killed: no finish recorded (started {r.started_at})")
    run_id = start_run("daily_run")

    # HARD TIMEOUT. With frozen regime vintages the job takes ~1-2 minutes; a run
    # still alive after DAILY_TIMEOUT_MIN is hung. It is failed loudly and the
    # process exits, rather than holding the slot until Windows kills it silently.
    import os
    import threading
    DAILY_TIMEOUT_MIN = 30

    def _timeout():
        try:
            finish_run(run_id, ok=False, message=f"timeout after {DAILY_TIMEOUT_MIN} min")
            alert("ESIS-QUANT — DAILY JOB TIMED OUT",
                  f"No trade list after {DAILY_TIMEOUT_MIN} minutes; the process was stopped.")
        finally:
            os._exit(3)
    watchdog = threading.Timer(DAILY_TIMEOUT_MIN * 60, _timeout)
    watchdog.daemon = True
    watchdog.start()

    try:
        result = esis_main.run_daily_job(send_alert=send_alert)
    except Exception as exc:  # noqa: BLE001 -- top-level guard: log loudly, fail the run
        watchdog.cancel()
        log.exception("DAILY JOB FAILED (no trade list produced): %s", exc)
        # Record it AND shout. The 2026-07-23 outage proved that failing quietly
        # is the dangerous part: a dead job means a BEAR signal never arrives.
        finish_run(run_id, ok=False, message=str(exc))
        alert("ESIS-QUANT — DAILY JOB FAILED",
              f"No trade list was produced.\n\n{exc}\n\n"
              "Run:  py -3.11 -m ops.daily_run\n"
              "Until this is fixed, a BEAR de-risk signal cannot reach you.")
        return 1

    # PART 5 -- post-close RATIONALE record. The ticket says WHAT; this says WHY.
    # Without it a losing stretch cannot be diagnosed as process failure versus
    # ordinary variance, because the reasoning behind each day is gone.
    try:
        from ops.health import record_run as _rr
        t_ = result["trade_list"]
        flagged = [k for k, v in (result.get("flags") or {}).items()] if isinstance(
            result.get("flags"), dict) else []
        # PER-NAME RATIONALE (audit P6.18). An honest caveat first: a
        # min-variance weight CANNOT be explained in factor terms, because the
        # live model contains no factors -- mu is exactly zero and every weight
        # comes from the covariance matrix alone. The factor regression in
        # analysis/factor_exposure.py is a RESEARCH artifact measuring what the
        # book turned out to load on; it is not an input to any weight. So the
        # only truthful per-name explanation is the covariance one: how volatile
        # the name is, how it correlates with the rest, and whether a constraint
        # bound. Anything phrased as "we bought X because momentum/value" would
        # be a fabricated narrative.
        w = t_.get("weights")
        per_name = ""
        try:
            if w is not None and len(w):
                from portfolio.system import _cost_scalar  # noqa: F401  (import check)
                top = w.sort_values(ascending=False).head(5)
                per_name = " | ".join(
                    f"{k}={v*100:.1f}%" + ("(at 15% cap)" if v >= 0.1499 else "")
                    for k, v in top.items())
        except Exception:  # noqa: BLE001
            per_name = "unavailable"
        _rr("daily_rationale", ok=True, as_of_date=t_.get("as_of"),
            message=(f"regime={t_['regime']} exposure={t_['exposure']*100:.0f}% "
                     f"source={t_.get('regime_source','?')} p_bear={t_.get('p_bear')} "
                     f"positions={t_['n_positions']} flagged={flagged or 'none'} "
                     f"| weights are min-variance (mu=0, no factors in the live model); "
                     f"top: {per_name}"))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not record daily rationale: %s", exc)

    t = result["trade_list"]
    log.info("DAILY JOB OK: run=%s regime=%s exposure=%.0f%% positions=%d",
             t["run_id"], t["regime"], t["exposure"] * 100, t["n_positions"])
    watchdog.cancel()
    finish_run(run_id, ok=True, as_of_date=t.get("as_of"),
               message=f"regime={t['regime']} exposure={t['exposure']*100:.0f}% "
                       f"source={t.get('regime_source', '?')} positions={t['n_positions']}")

    # Refresh the profit-estimate cache at most weekly (it runs the backtest, so
    # not every day). Guarded so a failure never fails the run.
    try:
        import os, time
        from analysis.profit_estimate import CACHE_PATH, compute_and_cache
        stale = (not os.path.exists(CACHE_PATH) or
                 time.time() - os.path.getmtime(CACHE_PATH) > 7 * 86400)
        if stale:
            compute_and_cache()
            log.info("profit-estimate cache refreshed")
    except Exception as exc:  # noqa: BLE001
        log.warning("profit-estimate cache refresh skipped: %s", exc)

    # Write the order ticket here too, not only in the interactive popup. The
    # popup only runs when you are logged in, so if the PC was off at 10:00 the
    # CSV never appeared. Generating it in the headless job means a fresh ticket
    # is on disk whenever the job runs -- including a catch-up run at power-on.
    try:
        from ops.order_ticket import write_ticket
        tk = write_ticket()
        log.info("order ticket written: %s (%d lines)", tk["path"], tk["rows"])
        print(f"\norder ticket: {tk['path']}  ({tk['rows']} lines, as-of {tk['as_of']})")
    except Exception as exc:  # noqa: BLE001
        log.warning("order ticket could not be written: %s", exc)

    # Produce (and snapshot) the daily report so there's a fresh one waiting each
    # morning. Never let a report failure fail the trade-list run -- the trade
    # list is the product; the report is a read-only view of it.
    try:
        from ops.daily_report import daily_report
        print("\n" + daily_report(write=True))
    except Exception as exc:  # noqa: BLE001
        log.warning("daily report could not be generated: %s", exc)

    # Agent 8: one brief across every agent, saved to reports/brief_YYYY-MM-DD.md
    try:
        from ops.chief_of_staff import run as _brief
        print("\n" + _brief())
    except Exception as exc:  # noqa: BLE001
        log.warning("morning brief could not be generated: %s", exc)
    return 0


if __name__ == "__main__":
    from ops.keep_awake import keep_awake
    with keep_awake():                 # a sleeping laptop killed runs mid-way before
        raise SystemExit(main())
