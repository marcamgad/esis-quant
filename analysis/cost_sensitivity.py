"""
Does the edge survive costs being worse than modelled?  (audit 2026-08-05, P1.4)

WHY THIS EXISTS. The audit found the real fee schedule (backtest/fees.py) was
never wired into any headline number: every published CAGR/Sharpe was charged a
flat 0.55% round trip. That is now fixed. But fixing the LEVEL of a cost
assumption is not the same as showing the result is robust to it, and the
correct response to "we had the cost wrong once" is not "we have it right now"
-- it is "here is what happens if we are still wrong."

TWO DIMENSIONS, BECAUSE COST HAS TWO DIMENSIONS HERE:

  1. MULTIPLIER (1x / 2x / 3x). Scales the whole schedule. Answers: if Thndr's
     fee, the stamp duty, and slippage are all worse than measured, does the
     strategy still clear? The measured schedule is a fit to 8 contract notes at
     one book size on one day -- 2x and 3x are not paranoia, they are the honest
     uncertainty band around a sample of 8.

  2. BOOK SIZE. The flat EGP-per-order leg does not scale, so the SAME strategy
     has a completely different cost profile at 1,000 vs 1,000,000 EGP. A single
     "cost of the strategy" number cannot exist once a flat fee is present, and
     quoting one was the original error.

WHAT WOULD FALSIFY THE STRATEGY: if the edge only survives at 1x and dies at 2x,
the result is a bet on the cost estimate rather than on the market. If it only
survives at large book sizes, capacity (analysis/capacity.py, ~12.5M EGP) and
cost squeeze it from both ends and there is no size at which it works.

Nothing here is tuned. The multipliers are 1/2/3 and the sizes are decades;
neither was chosen by looking at the answer.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("esis_quant.cost_sensitivity")

MULTIPLIERS = (1.0, 2.0, 3.0)
BOOK_SIZES = (1_000.0, 10_000.0, 100_000.0, 1_000_000.0)


def _scaled_schedule(mult: float):
    """A copy of the real schedule with every leg multiplied. Returned as a
    context-manager-free pair so the caller can patch and restore explicitly --
    monkeypatching module state is ugly but keeps ONE schedule definition
    rather than a parallel 'stress' copy that could drift from the real one."""
    from backtest import fees
    broker = [(d, flat * mult, pct * mult) for d, flat, pct in fees.THNDR_SCHEDULE]
    stamp = [(d, s * mult, i * mult) for d, s, i in fees.STAMP_DUTY_SCHEDULE]
    return broker, stamp


def run_one(aum_egp: float, mult: float, rebalance_days: int = 21) -> dict:
    """Full system at one (book size, cost multiplier). Restores the schedule
    afterwards even on failure -- a leaked stress schedule would silently
    poison every later number in the process."""
    from backtest import fees
    from portfolio.system import run_full_system

    keep_b, keep_s = fees.THNDR_SCHEDULE, fees.STAMP_DUTY_SCHEDULE
    keep_slip = config.SLIPPAGE_PCT
    try:
        fees.THNDR_SCHEDULE, fees.STAMP_DUTY_SCHEDULE = _scaled_schedule(mult)
        config.SLIPPAGE_PCT = keep_slip * mult
        r = run_full_system(rebalance_days=rebalance_days, aum_egp=aum_egp)
        m = r["metrics"]["full_system"]
        c = r["metrics"]["core"]
        return {"aum_egp": aum_egp, "mult": mult,
                "system_cagr": m["ann_return"], "system_sharpe": m["sharpe"],
                "system_maxdd": m["max_drawdown"],
                "core_cagr": c["ann_return"], "core_sharpe": c["sharpe"]}
    finally:
        fees.THNDR_SCHEDULE, fees.STAMP_DUTY_SCHEDULE = keep_b, keep_s
        config.SLIPPAGE_PCT = keep_slip


def sweep(book_sizes=BOOK_SIZES, multipliers=MULTIPLIERS) -> pd.DataFrame:
    rows = []
    for aum in book_sizes:
        for m in multipliers:
            try:
                rows.append(run_one(aum, m))
            except Exception as exc:  # noqa: BLE001
                logger.warning("cost sweep failed at aum=%s mult=%s: %s", aum, m, exc)
    return pd.DataFrame(rows)


def format_sweep(df: pd.DataFrame) -> str:
    if df.empty:
        return "cost sensitivity: no results"
    lines = ["COST SENSITIVITY — does the edge survive costs being worse than measured?",
             "  schedule = backtest/fees.py (EGP 4.95 + 0.112% per order + stamp), scaled by mult",
             "",
             f"  {'book EGP':>12}{'mult':>6}{'sys CAGR':>10}{'sys Sharpe':>12}"
             f"{'sys MaxDD':>11}{'core CAGR':>11}"]
    for aum, g in df.groupby("aum_egp"):
        for r in g.itertuples():
            lines.append(f"  {r.aum_egp:>12,.0f}{r.mult:>6.0f}x{r.system_cagr*100:>9.2f}%"
                         f"{r.system_sharpe:>12.3f}{r.system_maxdd*100:>10.1f}%"
                         f"{r.core_cagr*100:>10.2f}%")
        lines.append("")
    # the falsification test, stated before the numbers were seen
    base = df[(df["mult"] == 1.0)]
    worst = df[(df["mult"] == 3.0)]
    if not base.empty and not worst.empty:
        lines.append("  VERDICT")
        for aum in sorted(df["aum_egp"].unique()):
            b = base[base["aum_egp"] == aum]
            w = worst[worst["aum_egp"] == aum]
            if b.empty or w.empty:
                continue
            b, w = b.iloc[0], w.iloc[0]
            drop = (b["system_cagr"] - w["system_cagr"]) * 100
            verdict = ("survives 3x" if w["system_cagr"] > 0 else "DIES at 3x")
            lines.append(f"    {aum:>12,.0f} EGP: {b['system_cagr']*100:.1f}% -> "
                         f"{w['system_cagr']*100:.1f}% at 3x cost ({drop:+.1f}pp)  {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    df = sweep()
    print(format_sweep(df))
