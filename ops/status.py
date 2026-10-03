"""
One-shot health check (`py -3.11 -m ops.status`). Prints, in plain text:
  * data freshness + quality gate status,
  * the most recent logged trade list (what the system decided, and when),
  * the FORWARD paper record so far (never the backtest -- see paper_trading),
so you can confirm at a glance whether the scheduled job did the right thing.

This does NOT re-run anything or fetch data -- it only reads what the last
scheduled run already produced.
"""

import logging

import pandas as pd

from data.quality import data_quality_report, format_report
from ops.trade_list import latest_trade_list
from ops.paper_trading import paper_track_record


def print_status():
    print("=" * 60)
    print("ESIS-QUANT — SYSTEM STATUS  (read-only; nothing is re-run)")
    print("=" * 60)

    # 1) data quality / freshness -------------------------------------------
    try:
        rep = data_quality_report()
        print("\n[1] DATA")
        print(format_report(rep))
        if not rep["ok"]:
            print("  >> GATE WOULD BLOCK a trade list on this data.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[1] DATA — could not evaluate: {exc}")

    # 2) latest trade list --------------------------------------------------
    print("\n[2] LATEST TRADE LIST (audit trail)")
    tl = latest_trade_list()
    if tl.empty:
        print("  none logged yet — the daily job has not produced a trade list.")
    else:
        gen = tl["generated_at"].iloc[0]
        print(f"  run {tl['run_id'].iloc[0]}  generated {gen}  as-of {tl['as_of_date'].iloc[0].date()}")
        print(f"  regime={tl['regime'].iloc[0]}  exposure={tl['exposure'].iloc[0]*100:.0f}%")
        holdings = tl[tl["ticker"] != "__CASH__"].sort_values("weight", ascending=False)
        if holdings.empty:
            print("  book: FULLY IN CASH (regime de-risk)")
        else:
            print(f"  {len(holdings)} positions: " +
                  ", ".join(f"{r.ticker} {r.weight*100:.1f}%" for r in holdings.head(6).itertuples()) +
                  (" ..." if len(holdings) > 6 else ""))

    # 3) forward paper record ----------------------------------------------
    print("\n[3] FORWARD PAPER RECORD  (out-of-sample; NOT the backtest)")
    rec = paper_track_record()
    if rec["n_days"] == 0:
        print("  no forward record yet.")
    else:
        print(f"  {rec['n_days']} sessions  {rec['start']} -> {rec['end']}  cum {rec['cum_return']*100:+.2f}%")
        print(f"  BEAR days observed: {rec['bear_days_observed']}")
        if "sharpe" in rec:
            print(f"  (so far) Sharpe {rec['sharpe']:.2f}  MaxDD {rec['max_drawdown']*100:.1f}%")
        print(f"  evidence quality: {'MINIMUM VIABLE' if rec['sufficient'] else 'NOT YET EVIDENCE'} — {rec['note']}")

    print("\n" + "=" * 60)
    print("Reminder: forward record is the ONLY out-of-sample evidence.")
    print("The backtest is in-sample, survivorship-biased, and unproven live.")
    print("=" * 60)


def week_review(days: int = 7):
    """Practice-week check: how the SUGGESTED book (paper) did over the last
    `days` sessions, whether the regime changed, and whether the system ever
    fired its de-risk (BEAR -> cash) signal. Read-only."""
    import pandas as pd
    from data.database import get_connection
    with get_connection() as conn:
        nav = pd.read_sql("SELECT date, nav, daily_return, exposure, regime FROM paper_trading_nav "
                          "ORDER BY date", conn, parse_dates=["date"])
        tl = pd.read_sql("SELECT DISTINCT as_of_date, regime, exposure FROM trade_list_log "
                         "ORDER BY as_of_date", conn, parse_dates=["as_of_date"])
    print("=" * 60)
    print(f"PRACTICE-WEEK REVIEW (last {days} sessions of the SUGGESTED book)")
    print("=" * 60)
    if nav.empty:
        print("no forward record yet — run the daily job for a few days first.")
        return
    wk = nav.tail(days)
    cum = wk["nav"].iloc[-1] / wk["nav"].iloc[0] - 1 if len(wk) > 1 else 0.0
    print(f"\nwindow: {wk['date'].min().date()} -> {wk['date'].max().date()}  ({len(wk)} sessions)")
    print(f"suggested book return this window: {cum*100:+.2f}%")
    print(f"best day {wk['daily_return'].max()*100:+.2f}%   worst day {wk['daily_return'].min()*100:+.2f}%")
    print(f"regimes seen: {wk['regime'].value_counts().to_dict()}")
    derisk = wk[wk["exposure"] < 0.5]
    if not derisk.empty:
        print(f">> DE-RISK FIRED on {len(derisk)} day(s): the system went (partly) to cash.")
    else:
        print("   de-risk signal did NOT fire (stayed invested all week).")
    # did the recommendation change?
    if len(tl) >= 2:
        prev, cur = tl.iloc[-2], tl.iloc[-1]
        changed = (prev["regime"] != cur["regime"]) or (abs(prev["exposure"] - cur["exposure"]) > 1e-6)
        print(f"\nlatest call: regime={cur['regime']} exposure={cur['exposure']*100:.0f}%  "
              f"({'CHANGED from ' + prev['regime'] if changed else 'unchanged'})")
    print("\nReminder: one calm week proves nothing. The real test is a live BEAR regime.")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.ERROR)
    if len(sys.argv) > 1 and sys.argv[1] == "week":
        week_review()
    else:
        print_status()
