"""
Pre-committed go-live gate (audit Tier-2 #9).

WHY. The single most dangerous moment is deciding to risk real money AFTER a
good-looking week, when you're already emotionally invested in the outcome. So
the criteria are written down NOW, in config, before that moment -- and this
module just checks reality against them. It cannot be argued with in the heat
of the moment; it either passes or it does not.

It reads ONLY the FORWARD paper record (ops.paper_trading) and the real-terms
conversion -- never the backtest, which is in-sample and survivorship-biased.
Every criterion in config.GO_LIVE_* must pass. The hardest one on purpose:
the system must have LIVED THROUGH a real BEAR stretch, because the BEAR->cash
overlay -- the entire edge -- has never once fired live.
"""

import config
from ops.paper_trading import paper_track_record


def _crit(name, ok, detail, conclusive: bool = True):
    """A criterion is PASS only if it is BOTH satisfied and CONCLUSIVE.

    WHY conclusive EXISTS. On 2026-08-08 the gate printed
    "[PASS] forward Sharpe 7.67" and "[PASS] forward max drawdown -0.58%" from a
    21-session record that the same gate's own `sufficient` flag called too
    short to be evidence. Both were artifacts of annualising three weeks of a
    rising market -- a Sharpe of 7.67 is not a result, and "no drawdown yet" is
    not drawdown control.

    Printing PASS there is greenwashing: it invites a reader to count green
    boxes on a sample the system has already declared insufficient. A statistic
    computed from too little data is INCONCLUSIVE, which is neither pass nor
    fail, and it blocks go-live exactly as a fail does.
    """
    return {"criterion": name, "pass": bool(ok and conclusive),
            "conclusive": bool(conclusive), "detail": detail}


def evaluate_go_live() -> dict:
    """Check the forward paper record against every pre-committed criterion."""
    rec = paper_track_record()
    checks = []

    n = rec.get("n_days", 0)
    # Performance statistics are only meaningful once the record is long enough.
    # This is the SAME threshold the duration criterion uses -- it is not a new
    # or looser bar, it just stops derived statistics being scored on a sample
    # already judged too short.
    enough = n >= config.GO_LIVE_MIN_PAPER_DAYS
    checks.append(_crit("forward duration", n >= config.GO_LIVE_MIN_PAPER_DAYS,
                        f"{n} sessions (need >= {config.GO_LIVE_MIN_PAPER_DAYS})"))

    bear = rec.get("bear_days_observed", 0)
    checks.append(_crit("survived a BEAR regime", bear >= config.GO_LIVE_MIN_BEAR_DAYS,
                        f"{bear} BEAR sessions with the overlay acting (need >= "
                        f"{config.GO_LIVE_MIN_BEAR_DAYS}); "
                        f"{rec.get('bear_days_overlay_inactive', 0)} more BEAR sessions "
                        "ran fully invested and do not count"))

    sharpe = rec.get("sharpe")
    checks.append(_crit("forward Sharpe", sharpe is not None and sharpe >= config.GO_LIVE_MIN_SHARPE,
                        (f"{sharpe:.2f} (need >= {config.GO_LIVE_MIN_SHARPE})" if sharpe is not None
                         else f"n/a (need >= {config.GO_LIVE_MIN_SHARPE})")
                        + ("" if enough else f" — from only {n} sessions, not meaningful"),
                        conclusive=enough))

    mdd = rec.get("max_drawdown")
    checks.append(_crit("forward max drawdown", mdd is not None and mdd >= config.GO_LIVE_MAX_DRAWDOWN,
                        (f"{mdd*100:.2f}% (must be >= {config.GO_LIVE_MAX_DRAWDOWN*100:.0f}%)"
                         if mdd is not None else f"n/a (must be >= {config.GO_LIVE_MAX_DRAWDOWN})")
                        + ("" if enough else f" — only {n} sessions, no adverse day yet"),
                        conclusive=enough))

    # real-terms CAGR: only computable once there's a forward return series; if
    # the record isn't long enough yet this stays a fail (unknown != pass).
    real_cagr = rec.get("real_usd_cagr")
    checks.append(_crit("positive REAL (USD) CAGR",
                        real_cagr is not None and real_cagr >= config.GO_LIVE_MIN_REAL_CAGR,
                        f"{real_cagr if real_cagr is not None else 'n/a'} "
                        f"(need >= {config.GO_LIVE_MIN_REAL_CAGR})",
                        conclusive=enough))

    passed = all(c["pass"] for c in checks)
    return {"go_live": passed, "checks": checks, "n_days": n, "note": rec.get("note", "")}


def format_go_live(d: dict) -> str:
    lines = ["=" * 60, "GO-LIVE GATE (forward paper record vs pre-committed criteria)", "=" * 60]
    for c in d["checks"]:
        mark = ("PASS" if c["pass"] else
                ("INCONC" if not c.get("conclusive", True) else "FAIL"))
        lines.append(f"  [{mark}] {c['criterion']:26s} {c['detail']}")
    lines.append("-" * 60)
    if d["go_live"]:
        lines.append("  RESULT: all criteria met. Real capital is JUSTIFIED (not guaranteed).")
    else:
        n_fail = sum(1 for c in d["checks"] if not c["pass"] and c.get("conclusive", True))
        n_inc = sum(1 for c in d["checks"] if not c.get("conclusive", True))
        lines.append(f"  RESULT: NOT READY -- {n_fail} failed, {n_inc} inconclusive "
                     "(insufficient record). Keep paper trading.")
        lines.append("  INCONC = not enough forward data to judge. It is not a pass.")
    lines.append("  This gate reads the forward record only. It never reads the backtest.")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_go_live(evaluate_go_live()))
