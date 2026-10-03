"""
Capacity check (audit Tier-2 #8).

WHY. Every backtested return assumes your fills don't move the market. That
holds only up to some capital; past it, your own buying pushes prices against
you and the paper edge evaporates -- worst in exactly the small-cap EGX names
the min-var book likes. Before sizing real money you must know: how many EGP
can this book actually absorb?

MODEL. For each name we cap our daily trading at PARTICIPATION_CAP of its ADV
(average daily value traded), optionally spread over `spread_days` sessions.
The most capital the book can hold before name i breaches its cap is

    capital_max_i = participation_cap * ADV_i * spread_days / weight_i

and the BOOK capacity is the minimum over held names -- the first name to choke
sets the ceiling. We report that binding name and the per-name headroom, so you
see which position is the bottleneck (usually the smallest-ADV name at a
non-trivial weight).

This is a liquidity ceiling, not a promise: real capacity is lower once you
account for other participants and volatile volume. Conservative by design.
"""

import numpy as np
import pandas as pd

from backtest.costs import average_daily_value


def book_capacity(weights: pd.Series, participation_cap: float = 0.10,
                  spread_days: int = 3, adv: pd.Series = None) -> dict:
    """Max EGP the target book can hold at `participation_cap` of ADV per name,
    trading built up over `spread_days` sessions."""
    w = weights[weights > 0]
    if w.empty:
        return {"capacity_egp": 0.0, "binding_ticker": None, "per_name": pd.DataFrame()}
    if adv is None:
        adv = average_daily_value(list(w.index))
    rows = []
    for t, wi in w.items():
        a = float(adv.get(t, np.nan))
        cap_i = participation_cap * a * spread_days / float(wi) if a > 0 and np.isfinite(a) else float("nan")
        rows.append({"ticker": t, "weight": float(wi), "adv_egp": a, "capacity_egp": cap_i})
    per = pd.DataFrame(rows).sort_values("capacity_egp")
    valid = per.dropna(subset=["capacity_egp"])
    if valid.empty:
        return {"capacity_egp": float("nan"), "binding_ticker": None, "per_name": per,
                "participation_cap": participation_cap, "spread_days": spread_days}
    binding = valid.iloc[0]
    return {"capacity_egp": float(binding["capacity_egp"]), "binding_ticker": str(binding["ticker"]),
            "per_name": per, "participation_cap": participation_cap, "spread_days": spread_days,
            "n_names_no_adv": int(per["capacity_egp"].isna().sum())}


def format_capacity(d: dict) -> str:
    if not d or d.get("binding_ticker") is None:
        return "capacity: could not estimate (no ADV data)."
    lines = [f"CAPACITY CHECK ({d['participation_cap']*100:.0f}% of ADV per name, "
             f"built over {d['spread_days']} sessions)",
             f"  book capacity ~ {d['capacity_egp']:,.0f} EGP  (binding name: {d['binding_ticker']})",
             f"  above this, your own trades in {d['binding_ticker']} start moving the price."]
    per = d["per_name"].dropna(subset=["capacity_egp"]).head(5)
    if not per.empty:
        lines.append(f"  tightest names:  " +
                     ", ".join(f"{r.ticker} {r.capacity_egp/1e6:.1f}M" for r in per.itertuples()))
    if d.get("n_names_no_adv"):
        lines.append(f"  ({d['n_names_no_adv']} name(s) had no ADV data and were skipped)")
    return "\n".join(lines)
