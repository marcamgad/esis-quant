"""
Weekly fundamentals refresh -- its own job, on its own schedule.

WHY WEEKLY AND WHY SEPARATE. Financial statements change quarterly, so a daily
refresh would make ~690 requests a day to a free public site for data that is
almost always unchanged -- rude to the source and the quickest way to lose it.
Since 2026-09-14 it fetches annual AND quarterly pages (6 per ticker, ~1,270
requests), so the full universe takes ~45 minutes at a polite rate, far longer
than the 2-minute fetch job, and the September outages showed that a long job
on a laptop is the job that gets killed. Keeping it separate means a killed
fundamentals run costs a week-old statement, never a day's prices.

It also rebuilds the security master (ISIN / FIGI / sector) first, because a new
listing needs a sector before its fundamentals mean anything.

    py -3.11 -m ops.fundamentals_refresh
"""

import logging
import sys
import time


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from ops.health import start_run, finish_run

    run_id = start_run("fundamentals_refresh")
    t0 = time.time()
    try:
        from data.security_master import build
        m = build()
        print(f"security master: {m.get('stored', 0)} tickers, {m['with_figi']} FIGIs")

        from data.database import get_connection
        from data.fundamentals import refresh
        with get_connection() as conn:
            tickers = [r[0] for r in conn.execute(
                "SELECT DISTINCT ticker FROM stocks "
                "WHERE security_type = 'ordinary' OR security_type IS NULL ORDER BY ticker")]
        r = refresh(tickers=tickers)
        msg = (f"fundamentals: {r['tickers']} tickers, {r['rows_stored']} rows, "
               f"{len(r['failed'])} failed, {(time.time() - t0) / 60:.1f} min")
        print(msg)
        finish_run(run_id, ok=True, message=msg)
        return 0
    except Exception as exc:  # noqa: BLE001
        finish_run(run_id, ok=False, message=f"fundamentals_refresh failed: {exc}")
        print(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
