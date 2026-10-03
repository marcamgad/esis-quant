"""
Data refresh, decoupled from signal generation.

THE FAILURE THIS FIXES. Fetching lived inside run_daily_job, ahead of an
~8-minute walk-forward regime refit. That coupling turned every interruption
into a data outage:

    2026-08-25 17:00  daily_run OK              <- last successful automated run
    2026-08-26..30    no runs at all
    2026-08-31 17:00  FAILED: "database is 8 days stale (limit 5)"
    2026-09-01..03    killed, 0xC000013A (machine shut down mid-run)

The 2026-08-31 gate failure was correct but self-inflicted. The gate fired
because nothing had fetched for eight days; nothing had fetched because the only
thing that fetches is a ten-minute job that must survive to completion. One
successful automated run in ten days, and each failure made the next failure
more likely -- a laptop that sleeps at 17:05 can never refresh data at all.

Fetching takes ~2 minutes and has no dependencies. Splitting it out means an
interrupted refit costs a signal (recoverable: run it again) instead of costing
data (not recoverable: the gate locks the system out until a full run survives).

Run it before the signal job, from its own scheduled task:

    py -3.11 -m ops.fetch_only
"""

import logging
import sys

logger = logging.getLogger("esis_quant.fetch_only")


def fetch_only(refresh_days: int = 30) -> dict:
    """Refresh prices, macro and reference data. No modelling, no gate."""
    import pandas as pd

    from data.collectors import DataCollector
    from data.universe import build_tradable_universe, live_universe_tickers

    start = (pd.Timestamp.today() - pd.Timedelta(days=refresh_days)).strftime("%Y-%m-%d")
    dc = DataCollector()

    out = {}
    # Same live-universe argument as run_daily_job. Defaulting here would refresh
    # config.EGX_SEED_TICKERS (47 names) while the universe is 203 -- the exact
    # bug that presented as an eleven-day scheduler failure.
    fetch = dc.fetch_all_historical(tickers=live_universe_tickers(), start_date=start)
    out["prices"] = {"fetched": fetch.get("fetched"), "failed": len(fetch.get("failed", []))}

    # Macro is a separate staleness gate (usd_egp, limit 7 days) and has its own
    # way of going stale, so it is refreshed here rather than only in the signal
    # job -- devaluation is the one risk this system has measured and cannot
    # hedge, and a stale FX series truncates the regime features silently.
    try:
        macro = dc.fetch_macro_daily(start_date=start)
        out["macro"] = macro
    except Exception as exc:  # noqa: BLE001
        logger.warning("macro refresh failed (non-fatal): %s", exc)
        out["macro"] = {"error": str(exc)}

    try:
        from data.stockanalysis import refresh as ref_refresh
        out["reference"] = ref_refresh()
    except Exception as exc:  # noqa: BLE001
        logger.warning("reference refresh failed (non-fatal): %s", exc)
        out["reference"] = {"error": str(exc)}

    build_tradable_universe()

    # Report the gate's verdict WITHOUT enforcing it. This job's contract is to
    # leave the data as fresh as the vendors allow; refusing to trade is the
    # signal job's decision, and raising here would defeat the whole point of
    # separating them.
    try:
        from data.quality import data_quality_report
        q = data_quality_report()
        out["quality"] = {"as_of": str(q.get("as_of")), "db_age_days": q.get("db_age_days"),
                          "ok": q.get("ok"), "fatal": q.get("fatal")}
    except Exception as exc:  # noqa: BLE001
        out["quality"] = {"error": str(exc)}

    # Integrity scan. Runs HERE rather than in the signal job for two reasons:
    # this job actually completes (the signal job was killed three days running
    # in September), and the scan checks the data this job just wrote, so it is
    # the first thing that can see a bad refresh. Never fatal -- a reporter that
    # can take down the pipeline it reports on is a liability.
    try:
        from ops.research_agent import scan
        f = scan()
        out["integrity"] = {
            "findings": len(f),
            "critical": int((f["severity"] == "critical").sum()) if len(f) else 0,
            "high": int((f["severity"] == "high").sum()) if len(f) else 0,
            "needs_human": int(f["needs_human"].sum()) if len(f) else 0,
            "top": list(f.head(3)["summary"]) if len(f) else [],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("integrity scan skipped (non-fatal): %s", exc)
        out["integrity"] = {"error": str(exc)}
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = fetch_only()
    p, q = res.get("prices", {}), res.get("quality", {})
    print(f"prices: fetched {p.get('fetched')} failed {p.get('failed')}")
    ref = res.get("reference", {})
    if "error" not in ref:
        gf = ref.get("gap_fill", {})
        print(f"reference: {ref.get('rows')} rows, {ref.get('flagged')}/{ref.get('compared')} "
              f"disagree >5%, gap-filled {gf.get('filled', 0)}")
    print(f"quality: as_of {q.get('as_of')} age {q.get('db_age_days')}d ok={q.get('ok')}")
    if q.get("fatal"):
        print("  gate would FAIL: " + "; ".join(q["fatal"]))
    ig = res.get("integrity", {})
    if "error" not in ig:
        print(f"integrity: {ig.get('findings',0)} finding(s) "
              f"({ig.get('critical',0)} critical, {ig.get('high',0)} high, "
              f"{ig.get('needs_human',0)} need a human)")
        for line in ig.get("top", []):
            print(f"  ! {line}")
    # Exit 0 even when the gate would fail: this job's job is to fetch, and a
    # non-zero exit would make the scheduler report a data refresh that worked
    # as a failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())
