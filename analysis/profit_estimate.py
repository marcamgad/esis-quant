"""
Profit ESTIMATE for the daily report (honest ranges, never a promise).

WHY A RANGE, NOT A NUMBER. "Estimate my profit" has only one honest answer: a
distribution. The strategy's real-USD Sharpe is ~0.5 with a CI that includes
zero, so any single "you will make X" figure would be a lie. Instead we
bootstrap the strategy's own history to show, for each horizon, the MEDIAN
outcome, a 10-90% range, and the probability of ending profitable -- then scale
those percentages to your actual book size in EGP.

HONESTY LABELS baked in:
  * These are BACKTEST-derived (in-sample, survivorship-biased) -> optimistic.
    Live odds are almost certainly worse (closer to a coin flip short-term).
  * P(profit) barely rises with time (~64% at 1 month, ~68% at 1 year): short-
    horizon profit/loss is mostly noise, which is the whole point to convey.

The heavy bootstrap runs once and caches to a JSON file; the daily report reads
the cache and just multiplies by your book value, so it stays fast. Re-cache
weekly (or when the data changes) via compute_and_cache().
"""

import json
import os

import numpy as np
import pandas as pd

import config

CACHE_PATH = os.path.join(config.BASE_DIR, "analysis", "_profit_estimate_cache.json")
HORIZONS = {"1 month": 21, "3 months": 63, "6 months": 126, "1 year": 252}


def compute_and_cache(n_boot: int = 5000, seed: int = 0) -> dict:
    """Run the strategy, convert to real USD, bootstrap each horizon's
    cumulative-return distribution, and cache the percentiles. Heavy -- run
    weekly, not per report."""
    import logging
    logging.disable(logging.CRITICAL)
    from portfolio.system import run_full_system
    from analysis import real_returns as rr

    res = run_full_system(aum_egp=1_000)          # small-size (impact ~0) baseline
    usd = rr.to_hard_currency(res["returns"]["full_system"]).dropna().values
    rng = np.random.default_rng(seed)
    L = 5
    out = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
           "basis": "real-USD, small-size (1K EGP) backtest -- in-sample, optimistic",
           "horizons": {}}
    for label, days in HORIZONS.items():
        dist = np.empty(n_boot)
        for b in range(n_boot):
            nb = int(np.ceil(days / L))
            starts = rng.integers(0, len(usd) - L, nb)
            idx = (starts[:, None] + np.arange(L)).reshape(-1)[:days]
            dist[b] = np.prod(1 + usd[idx]) - 1
        p10, p50, p90 = np.percentile(dist, [10, 50, 90])
        out["horizons"][label] = {"p10": float(p10), "p50": float(p50), "p90": float(p90),
                                  "p_profit": float((dist > 0).mean()), "days": days}
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def estimate(book_egp: float) -> dict:
    """Scale the cached percentage distribution to a book size in EGP. Returns
    {} if the cache is missing (report then prompts to compute it)."""
    cache = _load_cache()
    if not cache:
        return {}
    book = book_egp if book_egp and book_egp > 0 else 1_000.0
    rows = []
    for label, h in cache["horizons"].items():
        rows.append({"horizon": label, "p_profit": h["p_profit"],
                     "median_pct": h["p50"], "lo_pct": h["p10"], "hi_pct": h["p90"],
                     "median_egp": h["p50"] * book, "lo_egp": h["p10"] * book, "hi_egp": h["p90"] * book})
    return {"book_egp": book, "basis": cache.get("basis", ""),
            "generated_at": cache.get("generated_at", ""), "rows": rows}


def format_estimate(d: dict) -> str:
    if not d:
        return ("    profit estimate not cached yet -- run:\n"
                "       py -3.11 -m analysis.profit_estimate")
    lines = [f"    estimated outcome for your {d['book_egp']:,.0f} EGP book "
             f"(real terms; BACKTEST-derived, so optimistic):"]
    for r in d["rows"]:
        lines.append(f"       {r['horizon']:>9}: median {r['median_pct']*100:+5.1f}% "
                     f"({r['median_egp']:+,.0f} EGP)   "
                     f"range [{r['lo_egp']:+,.0f}, {r['hi_egp']:+,.0f}] EGP   "
                     f"P(profit) {r['p_profit']*100:.0f}%")
    lines.append("    NOTE: P(profit) barely rises with time -> short-run result is mostly luck.")
    return "\n".join(lines)


if __name__ == "__main__":
    print("computing profit-estimate cache (runs the backtest, ~30s)...")
    out = compute_and_cache()
    print(f"cached to {CACHE_PATH}")
    print(format_estimate(estimate(1_000)))
