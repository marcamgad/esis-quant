"""
Agent 8 -- CHIEF OF STAFF. Runs the agents in order and writes ONE morning brief.

WHY. Before this, findings lived in four places (research_findings, the
sentinel, the clerk, the librarian) and nothing chased them: a critical finding
-- 38% of the book unmeasurable -- sat open from 11 Sep. Eight agents producing
eight reports is noise. One brief, red items first, with owners and due dates,
is the thing a person can act on in two minutes.

DETERMINISTIC. A dependency-ordered runner (each step timed, isolated, never
able to take down the next) and a templated brief. No generated prose.

ORDER (a step's failure is reported, the run continues):
  1 sentinel        liveness and freshness; may HALT the ticket
  2 ledger_clerk    ledger vs holdings
  3 disclosure      new reported figures from today's news
  4 allocator       stock share vs cash, cost drag
  5 brief           red first, changes since the last brief, overdue findings

    py -3.11 -m ops.chief_of_staff          # run all, print + save reports/brief_YYYY-MM-DD.md
"""

import concurrent.futures as cf
import logging
import os
import sys
import time
import traceback

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.chief_of_staff")

STEP_TIMEOUT_S = 600
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")


def _step(name, fn):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            out = fut.result(timeout=STEP_TIMEOUT_S)
            return {"step": name, "ok": True, "secs": round(time.time() - t0, 1), "out": out}
        except cf.TimeoutError:
            return {"step": name, "ok": False, "secs": STEP_TIMEOUT_S, "error": "timed out"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("step %s failed: %s", name, traceback.format_exc(limit=3))
            return {"step": name, "ok": False, "secs": round(time.time() - t0, 1), "error": str(exc)[:200]}


def run_agents() -> list:
    from ops import sentinel, ledger_clerk, allocator, index_paper_trade
    from data import disclosure_reader
    return [
        _step("sentinel", lambda: sentinel.run(alert=False)),
        _step("ledger_clerk", ledger_clerk.stage),
        _step("disclosure", disclosure_reader.run),
        _step("allocator", allocator.run),
        _step("index_paper", index_paper_trade.update),
        _step("watchlist", _watchlist_refresh),
    ]


def _watchlist_refresh():
    from ops.watchlist import refresh
    from ops import system_picks
    r = refresh()
    from ops import high_risk_picks
    from ops.forecast import forecast
    sig = system_picks.signals()
    r["recorded_signals"] = system_picks.record(sig)
    hr = high_risk_picks.signals(careful=sig)
    r["recorded_high_risk"] = high_risk_picks.record(hr)
    # save a 4-week forecast for everything held or signalled, so calibration is scored live
    from ops.portfolio_tracker import holdings
    pos, _ = holdings()
    names = set(pos["ticker"]) if not pos.empty else set()
    names |= {b["ticker"] for b in sig["buy"] + hr["buy"]}
    for t in sorted(names):
        forecast(t, record_it=True)
    return r


def _last_brief_time():
    if not os.path.isdir(REPORT_DIR):
        return None
    files = sorted(f for f in os.listdir(REPORT_DIR) if f.startswith("brief_") and f.endswith(".md"))
    if not files:
        return None
    return pd.Timestamp(os.path.getmtime(os.path.join(REPORT_DIR, files[-1])), unit="s")


def _findings():
    from ops.agent_guard import ensure_tables
    with get_connection() as conn:
        ensure_tables(conn)
        f = pd.read_sql("SELECT agent, severity, kind, subject, summary, fix_hint, status, due_date, "
                        "first_seen, last_seen, resolved_at FROM agent_findings", conn)
        try:
            rf = pd.read_sql("SELECT severity, kind, subject, summary, first_seen, resolved_at FROM "
                             "research_findings WHERE resolved_at IS NULL", conn)
            rf["agent"], rf["status"] = "research_agent", "open"
            rf["due_date"] = (pd.to_datetime(rf["first_seen"]) + pd.Timedelta(days=14)).dt.date.astype(str)
            f = pd.concat([f, rf], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    return f


def brief(steps: list) -> str:
    from ops.agent_guard import get_state, SEVERITY_ORDER
    from ops.portfolio_tracker import holdings, invested_capital
    today = pd.Timestamp.now().normalize()
    since = _last_brief_time()
    state = get_state()
    f = _findings()
    open_ = f[f["status"] == "open"].copy()
    open_["_o"] = open_["severity"].map(SEVERITY_ORDER).fillna(9)
    open_ = open_.sort_values(["_o", "due_date"])
    overdue = open_[pd.to_datetime(open_["due_date"], errors="coerce") < today]
    new = open_[pd.to_datetime(open_["first_seen"], errors="coerce") > since] if since is not None else open_
    resolved = f[(f["status"] == "resolved") & (pd.to_datetime(f["resolved_at"], errors="coerce") > since)] \
        if since is not None else f.iloc[0:0]

    L = [f"# ESIS-QUANT morning brief — {today.date()}", ""]
    L.append(f"**System: {state['state']}**" + (f" — {state['reason']}" if state["state"] == "HALT" else ""))
    L.append("")

    # account
    pos, cash = holdings()
    value = float(pos["value"].sum() + cash) if not pos.empty else cash
    cap = invested_capital()
    L.append("## Account")
    line = f"- value **{value:,.0f} EGP** (stocks {value - cash:,.0f} + cash {cash:,.0f}) at the system's last marks"
    if cap:
        line += f"; vs money put in {cap:,.0f}: **{value - cap:+,.0f} EGP ({value / cap - 1:+.1%})**"
    L.append(line)
    if not pos.empty:
        w = (pos.set_index("ticker")["value"] / value).sort_values(ascending=False)
        L.append("- holdings: " + ", ".join(f"{t} {x:.0%}" for t, x in w.items()))
    try:
        from models.regime_ensemble import exposure_for
        e = exposure_for(today)
        L.append(f"- regime exposure **{e['exposure']:.0%}** (P(bear) {e['p_bear']}, signal {e['as_of_signal']})")
    except Exception:  # noqa: BLE001
        pass
    try:
        from ops.yardstick import race
        y = race()
        if y:
            L.append(f"- vs the same money in a cash fund ({y['cash_yield']:.1%}) since {y['since']}: "
                     f"**{y['trading_added']:+,.0f} EGP** from trading")
    except Exception:  # noqa: BLE001
        pass
    alloc = next((s["out"] for s in steps if s["step"] == "allocator" and s["ok"]), None)
    if alloc:
        sc = {r["scenario"]: r["stock_share"] for r in alloc["scenarios"]}
        c = alloc["costs"]
        L.append(f"- allocator: cash pays {alloc['cash_yield']:.1%} ({alloc['real_cash']:+.1%} real); "
                 f"stock share by the rule — system book {sc.get('system book, low turnover', float('nan')):.0%}, "
                 f"your book {sc.get('your book, low turnover', float('nan')):.0%}, "
                 f"your book at your trading pace {sc.get('your book, your current trading pace', float('nan')):.0%}")
        L.append(f"- last 30 days: {c['orders']} orders, fees ~{c['fees']:,.0f} EGP, cost drag ~{c['drag']:.0%}/yr")
    L.append("")

    # red first
    red = open_[open_["severity"].isin(["critical", "high"])]
    L.append(f"## Needs attention ({len(red)})")
    if red.empty:
        L.append("- nothing critical or high.")
    for r in red.head(15).itertuples():
        due = f" — due {r.due_date}" if isinstance(r.due_date, str) else ""
        L.append(f"- **[{r.severity}]** {r.summary} _( {r.agent}{due} )_")
        if isinstance(r.fix_hint, str) and r.fix_hint:
            L.append(f"  - → {r.fix_hint}")
    L.append("")
    L.append(f"## Changes since the last brief")
    L.append(f"- new: {len(new)} | closed: {len(resolved)} | overdue: {len(overdue)} | open in total: {len(open_)}")
    for r in overdue.head(8).itertuples():
        L.append(f"- overdue since {r.due_date}: [{r.severity}] {r.summary} _( {r.agent} )_")
    L.append("")

    # reported numbers for what you hold
    held = pos["ticker"].tolist() if not pos.empty else []
    if held:
        with get_connection() as conn:
            try:
                nf = pd.read_sql("SELECT f.ticker, f.metric, f.value_m, f.pct_change, f.period, f.fiscal_year, "
                                 "f.first_seen_at, i.headline FROM news_figures f JOIN news_items i ON i.id=f.news_id "
                                 "WHERE f.kind='result' AND f.confidence >= 0.6 AND f.ticker IN (%s) "
                                 "ORDER BY f.first_seen_at DESC LIMIT 6" % ",".join("?" * len(held)),
                                 conn, params=held)
            except Exception:  # noqa: BLE001
                nf = pd.DataFrame()
        L.append("## Latest reported figures for your holdings (from the press)")
        if nf.empty:
            L.append("- none extracted yet.")
        for r in nf.itertuples():
            per = " ".join(x for x in (r.period if pd.notna(r.period) else "",
                                       str(int(r.fiscal_year)) if pd.notna(r.fiscal_year) else "") if x)
            L.append(f"- {r.ticker} {per}: "
                     f"{r.headline[:110]} _(seen {str(r.first_seen_at)[:10]})_")
        L.append("")

    # forward paper test of the index-addition rule (ops/index_paper_trade.py)
    try:
        from ops import index_paper_trade as ipt
        rep = ipt.report()
        rep.pop("table", None)
        cand = ipt.detect()
        L.append("## EGX30 index paper test")
        L.append(f"- rule {rep['rule']}: {rep['closed']} closed paper trades, {rep['verdict']}"
                 + (f"; mean net vs market {rep['mean_net_abnormal']:+.2%}" if "mean_net_abnormal" in rep else ""))
        if not cand.empty:
            L.append("- **possible review announcement in the news — enter it with "
                     "`py -3.11 -m ops.index_paper_trade announce ...`:**")
            for r in cand.head(4).itertuples():
                L.append(f"  - {r.headline[:120]} _({str(r.first_seen_at)[:10]})_")
        L.append("")
    except Exception:  # noqa: BLE001
        pass

    # the system's decisions (ops/system_picks.py, fixed rule)
    try:
        from ops import system_picks
        sig = system_picks.signals()
        L.append(f"## System signals — rule {sig['rule']}")
        for b in sig["buy"]:
            m = b["size"]
            L.append(f"- **BUY {b['ticker']}: {m['shares']:,} shares** (~{m['value_egp']:,.0f} EGP at "
                     f"{m['price']:,.2f}) — {b['triggers'][0][:90] if b['triggers'] else ''}")
        if not sig["buy"]:
            L.append("- BUY: none today (no watchlist name passes all seven gates)")
        for x in sig["sell"]:
            L.append(f"- **SELL {x['ticker']}** ({x['shares']:,} shares): {'; '.join(x['why'])}")
        for x in sig.get("trim", []):
            L.append(f"- **TRIM {x['ticker']}**: sell {x['shares']:,} shares — {'; '.join(x['why'])}")
        for b in sig.get("wait", []):
            L.append(f"- WAIT {b['ticker']}: passes every gate, needs {b['needs']:,.0f} EGP more cash")
        if sig["hold"]:
            L.append("- HOLD: " + ", ".join(x["ticker"] for x in sig["hold"]))
        L.append("")
        from ops import high_risk_picks as hrp
        hr = hrp.signals(careful=sig)
        L.append(f"## High-risk signals — rule {hr['rule']} (stop {hrp.STOP:.0%}, target {hrp.TARGET:+.0%})")
        for b in hr["buy"]:
            m, q = b["size"], b["forecast"]["quantiles"]
            L.append(f"- **BUY (HIGH RISK) {b['ticker']}: {m['shares']:,} shares** (~{m['value_egp']:,.0f} EGP); "
                     f"4-week 80% range {q['p10']:,.2f}–{q['p90']:,.2f}")
        for x in hr["sell"]:
            L.append(f"- **SELL {x['ticker']}**: {'; '.join(x['why'])}")
        if not hr["buy"] and not hr["sell"]:
            L.append("- nothing today")
        L.append("")
    except Exception:  # noqa: BLE001
        pass

    # 4-week ranges for what is held (ops/forecast.py: calibrated ranges, no direction call)
    try:
        from ops.forecast import forecast
        from ops.portfolio_tracker import holdings as _h
        hp, _ = _h()
        if not hp.empty:
            L.append("## 4-week price ranges for your holdings (80% band; the middle is today's price)")
            for t in hp["ticker"]:
                f = forecast(t)
                if f and "error" not in f:
                    q = f["quantiles"]
                    L.append(f"- {t} {f['price']:,.2f}: {q['p10']:,.2f} to {q['p90']:,.2f} "
                             f"({q['p10'] / f['price'] - 1:+.0%} / {q['p90'] / f['price'] - 1:+.0%})")
            L.append("")
    except Exception:  # noqa: BLE001
        pass

    # watchlist (ops/watchlist.py): fresh triggers only, entries expire after 7 days
    try:
        from ops.watchlist import active
        wl = active()
        L.append(f"## Watchlist ({len(wl)}) — evidence, not recommendations")
        if wl.empty:
            L.append("- empty: nothing had a fresh trigger in the last 7 days.")
        for r in wl.head(8).itertuples():
            first = r.reasons[0][1] if r.reasons else ""
            more = f" (+{len(r.reasons) - 1} more)" if len(r.reasons) > 1 else ""
            caut = f" — cautions: {', '.join(k for k, _ in r.cautions)}" if r.cautions else ""
            L.append(f"- **{r.ticker}** {r.score:.1f}: {first[:110]}{more}{caut} _(until {r.expires})_")
        L.append("")
    except Exception:  # noqa: BLE001
        pass

    L.append("## Agent runs")
    for s in steps:
        L.append(f"- {s['step']}: {'ok' if s['ok'] else 'FAILED — ' + s.get('error', '')} ({s['secs']}s)")
    L += ["", "_Facts and the system's own arithmetic. Decisions are yours._"]
    return "\n".join(L)


def run(write: bool = True) -> str:
    steps = run_agents()
    text = brief(steps)
    if write:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, f"brief_{pd.Timestamp.now():%Y-%m-%d}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
