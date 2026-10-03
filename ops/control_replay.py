"""
Do the safety controls actually fire?  (Tier 3, item 9)

THE GAP. `risk.kill_switch.live_state` and `risk.kill_switch.book_state` are
covered by unit tests, but every one of those tests feeds them SYNTHETIC numbers
chosen to trip them. The forward record contains no adverse day at all -- worst
paper session -0.58%, worst book day -0.02%, zero days below -1%. So the
controls have never been exercised on anything real, and "the tests pass" is not
evidence they would fire when it matters.

WHAT THIS DOES INSTEAD. Replays REAL historical EGX return sequences -- the
worst days and worst rolling weeks the backtest actually contains, including the
2016 and 2024 devaluation crashes -- through the same threshold logic the live
controls use, and reports whether each would have tripped and on which date.

WHAT IT IS NOT. This is a REPLAY, not a live firing. It proves the logic reacts
correctly to real historical magnitudes; it does not prove the plumbing fires in
production, because production has never had a bad enough day. That distinction
is the whole point of the exercise and is preserved in the output: the verdict
is REPLAY-VERIFIED, never PASS.

Thresholds are READ from config, never redefined here. A replay that hardcoded
its own limits would pass while the live control was misconfigured.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("esis_quant.control_replay")


def worst_historical_windows(returns: pd.Series, n: int = 5) -> pd.DataFrame:
    """The n worst single days and n worst rolling 5-session stretches."""
    r = pd.Series(returns).dropna()
    weekly = (1 + r).rolling(5).apply(np.prod, raw=True) - 1
    days = r.nsmallest(n).rename("ret").to_frame().assign(kind="daily")
    weeks = weekly.nsmallest(n).rename("ret").to_frame().assign(kind="weekly-5d")
    return pd.concat([days, weeks]).sort_values("ret")


def replay_kill_switch(returns: pd.Series = None) -> dict:
    """Would the model kill switch have tripped on real history?

    Uses `risk.kill_switch.check_kill_switch` -- the same function the live path
    calls -- so a change to the live rule changes this test too.
    """
    from risk.kill_switch import check_kill_switch, kill_switch_events
    if returns is None:
        from portfolio.system import run_full_system
        returns = run_full_system()["returns"]["full_system"]
    r = pd.Series(returns).dropna()
    events = kill_switch_events(r)
    worst = worst_historical_windows(r)
    return {"n_sessions": int(len(r)),
            "daily_limit": config.KILL_SWITCH_MAX_DAILY_LOSS,
            "weekly_limit": config.KILL_SWITCH_MAX_WEEKLY_LOSS,
            "n_trips": int(len(events)),
            "events": events,
            "worst": worst,
            "would_have_fired": bool(len(events) > 0)}


def replay_book_guard(returns: pd.Series = None, start_nav: float = 10_000.0) -> dict:
    """Would the ACCOUNT drawdown guard have escalated on real history?

    Rebuilds a NAV path from real returns and walks the same escalation ladder
    `risk.kill_switch.book_state` applies (warn -> block -> flatten).
    """
    from risk.kill_switch import (BOOK_WARN_DRAWDOWN, BOOK_BLOCK_DRAWDOWN,
                                  BOOK_FLATTEN_DRAWDOWN)
    if returns is None:
        from portfolio.system import run_full_system
        returns = run_full_system()["returns"]["full_system"]
    r = pd.Series(returns).dropna()
    nav = start_nav * (1 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    hits = {}
    for name, level in (("warn", BOOK_WARN_DRAWDOWN),
                        ("block", BOOK_BLOCK_DRAWDOWN),
                        ("flatten", BOOK_FLATTEN_DRAWDOWN)):
        breached = dd[dd <= level]
        hits[name] = {"level": level, "n_days": int(len(breached)),
                      "first": (str(breached.index[0].date()) if len(breached) else None),
                      "worst": float(dd.min())}
    return {"n_sessions": int(len(r)), "max_drawdown": float(dd.min()),
            "levels": hits,
            "would_have_fired": bool(hits["flatten"]["n_days"] > 0)}


def replay_drift_alerts() -> dict:
    """Do the drift checks distinguish INCONCLUSIVE from PASS on real data?

    The failure mode being guarded against is a check that returns "ok" because
    it had nothing to look at. That reads as a pass and is not one.
    """
    from ops.health import execution_drift, cost_model_drift
    out = {}
    for fn in (execution_drift, cost_model_drift):
        d = fn()
        out[d["check"]] = {"ok": d["ok"], "conclusive": d.get("conclusive"),
                           "reason": d.get("reason", "")}
    return out


def format_replay(ks: dict, bg: dict, drift: dict) -> str:
    lines = ["CONTROL REPLAY — do the safety controls fire on REAL historical moves?",
             "  (REPLAY, not a live firing: the forward record contains no adverse day)",
             ""]
    lines.append(f"  MODEL KILL SWITCH  (daily {ks['daily_limit']*100:.0f}%, "
                 f"weekly {ks['weekly_limit']*100:.0f}%) over {ks['n_sessions']} sessions")
    lines.append(f"    trips on real history: {ks['n_trips']}  -> "
                 f"{'REPLAY-VERIFIED (fires)' if ks['would_have_fired'] else 'NEVER FIRES — investigate'}")
    w = ks["worst"]
    if not w.empty:
        lines.append("    worst real moves in the sample:")
        for row in w.head(6).itertuples():
            lines.append(f"      {row.Index.date()}  {row.kind:>10}  {row.ret*100:+7.2f}%")
    lines.append("")
    lines.append(f"  ACCOUNT DRAWDOWN GUARD over {bg['n_sessions']} sessions "
                 f"(worst real drawdown {bg['max_drawdown']*100:.1f}%)")
    for name, h in bg["levels"].items():
        status = f"first breached {h['first']}" if h["n_days"] else "never breached"
        lines.append(f"    {name:>8} at {h['level']*100:>5.0f}%: {h['n_days']:>5} days, {status}")
    lines.append(f"    -> {'REPLAY-VERIFIED (escalates)' if bg['would_have_fired'] else 'NEVER ESCALATES — investigate'}")
    lines.append("")
    lines.append("  DRIFT CHECKS on live data")
    for name, d in drift.items():
        verdict = ("INCONCLUSIVE" if not d["conclusive"] else ("ok" if d["ok"] else "BREACH"))
        lines.append(f"    {name:>18}: {verdict}  {d['reason']}")
    lines.append("")
    lines.append("  REPLAY-VERIFIED means the logic reacts correctly to real historical")
    lines.append("  magnitudes. It does NOT mean the control has fired in production.")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from portfolio.system import run_full_system
    rets = run_full_system()["returns"]["full_system"]
    print(format_replay(replay_kill_switch(rets), replay_book_guard(rets),
                        replay_drift_alerts()))
