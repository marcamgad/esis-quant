"""
Sector limits for the target book.

THE FAILURE. On 2026-09-04 a three-name addition came back as MILS + AFMC + LUTS
with "average pairwise correlation 0.227". MILS (North Cairo Flour Mills) and
AFMC (Alexandria Flour Mills) are the same business -- same wheat input, same
subsidy regime -- correlated 0.644 and stable at 0.62-0.70 across every window
tested. The average hid one strong pair behind two near-zero ones, and nothing
in the optimiser could object, because it had never been told what sector
anything was in (`stocks.sector` was NULL until 2026-09-11).

Minimum-variance optimisation trusts the correlation matrix, and correlations
estimated on quiet periods understate how much same-sector names move together
in a sector-wide shock -- a subsidy change, an input-price spike, a regulatory
decision. A sector limit is the cheap insurance against that estimation error.

TWO CONTROLS, because they fail in different places:
  * select_diverse -- when a small ticket keeps only the top N names, pick them
    in weight order but cap how many may come from one sector. Top-N by weight is
    exactly the selection that can return two flour mills.
  * apply_sector_cap -- after weights exist, no sector may exceed a share of the
    book; excess is redistributed to other sectors within the per-name cap.

UNKNOWN SECTORS ARE NOT POOLED. A name with no classification is treated as its
own sector. Pooling every unknown into one bucket would cap them together and
evict names for sharing an absence of data, which is a data problem posing as a
risk decision.

This module never adds a return view. It only removes concentration the
optimiser could not see.
"""

import logging

import pandas as pd

logger = logging.getLogger("esis_quant.portfolio.sector_cap")

try:
    import config as _config
except Exception:  # noqa: BLE001
    _config = None

# Largest share of the book any single sector may carry.
SECTOR_WEIGHT_CAP = float(getattr(_config, "SECTOR_WEIGHT_CAP", 0.30))
# Most names one sector may contribute when a ticket is cut to a few names.
MAX_NAMES_PER_SECTOR = int(getattr(_config, "MAX_NAMES_PER_SECTOR", 2))


def load_sectors() -> dict:
    """ticker -> exchange sector, from the `stocks` table. Empty on any failure,
    so a missing classification degrades to 'no sector limit', never a crash."""
    try:
        from data.database import get_connection
        with get_connection() as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(stocks)")}
            if "sector" not in cols:
                return {}
            rows = conn.execute(
                "SELECT ticker, sector FROM stocks WHERE sector IS NOT NULL").fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_sectors failed (sector limits disabled): %s", exc)
        return {}


def _sector_of(ticker: str, sectors: dict) -> str:
    s = sectors.get(ticker)
    # unknowns are their own sector -- see module docstring
    return s if s else f"__unknown__:{ticker}"


def select_diverse(target: pd.Series, sectors: dict, max_names: int,
                   held=None, max_per_sector: int = None) -> set:
    """Choose up to `max_names` names in descending weight, at most
    `max_per_sector` from any one sector.

    Held names are always kept and count toward their sector's quota. If the
    quota leaves the ticket short of `max_names`, the remaining slots are filled
    in weight order ignoring the quota -- a ticket with fewer names than it can
    afford concentrates cost, which is its own risk.
    """
    max_per_sector = MAX_NAMES_PER_SECTOR if max_per_sector is None else int(max_per_sector)
    held = set(held or ()) & set(target.index)
    chosen = set(held)
    counts = {}
    for t in held:
        k = _sector_of(t, sectors)
        counts[k] = counts.get(k, 0) + 1

    ordered = list(target.sort_values(ascending=False).index)
    for t in ordered:
        if len(chosen) >= max_names:
            break
        if t in chosen:
            continue
        k = _sector_of(t, sectors)
        if counts.get(k, 0) >= max_per_sector:
            continue
        chosen.add(t)
        counts[k] = counts.get(k, 0) + 1

    if len(chosen) < max_names:
        for t in ordered:
            if len(chosen) >= max_names:
                break
            chosen.add(t)
    return chosen


def apply_sector_cap(weights: pd.Series, sectors: dict, cap: float = None,
                     max_weight: float = 1.0, max_iter: int = 100) -> tuple:
    """Limit every sector to `cap` of the book. Returns (weights, info).

    Excess from an over-cap sector is shared across names in under-cap sectors
    in proportion to their existing weights, respecting `max_weight`. If the
    constraint cannot be met (too few sectors, or no room under the per-name
    cap), it is RELAXED and reported rather than forced -- an infeasible cap that
    silently drops money is worse than a looser cap stated openly.
    """
    cap = SECTOR_WEIGHT_CAP if cap is None else float(cap)
    w = weights.astype(float).copy()
    w = w[w > 0]
    if w.empty or not sectors:
        return weights, {"applied": False, "reason": "no weights or no sector data"}
    w = w / w.sum()

    sec = pd.Series({t: _sector_of(t, sectors) for t in w.index})
    n_sectors = sec.nunique()
    if n_sectors * cap < 1.0 - 1e-9:
        return w, {"applied": False,
                   "reason": f"infeasible: {n_sectors} sectors x cap {cap:.0%} < 100%"}

    before = w.groupby(sec).sum()
    for _ in range(max_iter):
        tot = w.groupby(sec).sum()
        over = tot[tot > cap + 1e-9]
        if over.empty:
            break
        excess = 0.0
        for k, v in over.items():
            names = sec.index[sec == k]
            scale = cap / v
            excess += float(w[names].sum() * (1.0 - scale))
            w[names] = w[names] * scale
        room_names = [t for t in w.index
                      if tot.get(sec[t], 0.0) < cap - 1e-9 and w[t] < max_weight - 1e-9]
        if not room_names or excess <= 1e-12:
            break
        base = w[room_names]
        add = excess * (base / base.sum()) if base.sum() > 0 else pd.Series(
            excess / len(room_names), index=room_names)
        w[room_names] = (w[room_names] + add).clip(upper=max_weight)

    w = w / w.sum()
    after = w.groupby(sec).sum()
    breach = after[after > cap + 1e-6]
    info = {"applied": True, "cap": cap,
            "max_sector_before": float(before.max()),
            "max_sector_after": float(after.max()),
            "capped_sectors": sorted(k for k in before.index if before[k] > cap + 1e-9
                                     and not str(k).startswith("__unknown__")),
            "residual_breach": {k: float(v) for k, v in breach.items()}}
    if info["capped_sectors"]:
        logger.info("sector cap %.0f%%: reduced %s (max sector %.1f%% -> %.1f%%)",
                    cap * 100, info["capped_sectors"],
                    info["max_sector_before"] * 100, info["max_sector_after"] * 100)
    return w, info
