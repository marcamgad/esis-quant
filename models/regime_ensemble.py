"""
Ensemble regime signal -- the LIVE regime source.

WHY THE LIVE PATH USES THIS INSTEAD OF A SINGLE HMM FIT. Measured, not assumed
(analysis/regime_stability.py): refitting the same model on the SAME data with
only a different EM random seed moved the BEAR-day count between 478 and 790,
and swung the overlay's contribution by ~10 percentage points of CAGR. Refitting
on random 70% ticker subsets was worse -- the overlay was negative in all six
trials. A signal that unstable cannot be traded: whichever fit you happened to
save is the "edge" you think you have.

Averaging P(BEAR) across independent refits removes that. Two DISJOINT five-seed
ensembles agreed (single fits ranged -14.4% to +3.4% delta-CAGR), and the
ensembled overlay reproduced the project's long-standing honest finding:

    core 28.1% CAGR / Sharpe 1.58 / MaxDD -42.2%
    ensemble continuous  23.8% CAGR / Sharpe 1.63 / MaxDD -31.0%

i.e. it pays about 4.3pp of return for about 11pp less drawdown, with Sharpe
essentially unchanged. That is DRAWDOWN CONTROL, not alpha, and it is the only
claim this signal supports.

CONTINUOUS, NOT BINARY. Exposure is 1 - p_bear. When the fits disagree the book
is partially de-risked rather than flipped on or off, which is the entire point
of ensembling -- a hard vote throws away the disagreement that makes the
estimate honest.

COST AND FRESHNESS. Each refit is a full walk-forward EM estimation. It used to
be rebuilt monthly, which meant the stored series ENDED at the build date and
exposure_for() kept serving the last row for up to a month -- and because
nothing ever called refresh_if_stale(), it stopped at 2026-07-30 and the live
book stayed 100% invested through the Aug 2026 BEAR call. Since 2026-10-03 the
vintages are frozen (models.regime_pipeline._cached_fit), so a rebuild only
filters forward and runs daily; is_stale() now also means "does not reach the
latest observation date".

CAVEATS THAT DO NOT GO AWAY: the result is in-sample, has not been through a
deflated-Sharpe test, and the overlay was 100% invested through both EGP
devaluations -- it does not defend the risk that dominates real-terms returns.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.regime_ensemble")

DEFAULT_SEEDS = (2001, 2002, 2003, 2004, 2005)
MAX_AGE_DAYS = 30          # rebuild roughly monthly; the signal moves slowly
MIN_FITS = 3               # below this the average is still mostly noise


def build_ensemble(seeds=DEFAULT_SEEDS, tickers=None, n_init: int = 3) -> pd.Series:
    """Fraction of independent walk-forward refits labelling each day BEAR."""
    from analysis.regime_stability import ensemble_bear_probability
    p = ensemble_bear_probability(list(seeds), tickers=tickers, n_init=n_init,
                                  use_cache=tickers is None)
    if p.empty:
        logger.error("build_ensemble: no fits succeeded -- ensemble NOT updated")
    return p


def store_ensemble(p_bear: pd.Series, n_fits: int) -> int:
    if p_bear is None or p_bear.empty:
        return 0
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = [(pd.Timestamp(d).strftime("%Y-%m-%d"), float(v), int(n_fits), now)
            for d, v in p_bear.items() if v == v]
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO regime_ensemble (date, p_bear, n_fits, computed_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(date) DO UPDATE SET "
            "p_bear=excluded.p_bear, n_fits=excluded.n_fits, computed_at=excluded.computed_at",
            rows)
    logger.info("store_ensemble: wrote %d rows from %d fits", len(rows), n_fits)
    return len(rows)


def load_ensemble() -> pd.Series:
    with get_connection() as conn:
        df = pd.read_sql("SELECT date, p_bear FROM regime_ensemble ORDER BY date",
                         conn, parse_dates=["date"])
    return df.set_index("date")["p_bear"] if not df.empty else pd.Series(dtype=float)


def ensemble_status() -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*), MAX(date), MAX(computed_at), MAX(n_fits) "
                           "FROM regime_ensemble").fetchone()
    n, last_date, computed, n_fits = (row[0], row[1], row[2], row[3]) if row else (0, None, None, None)
    age = None
    if computed:
        age = (pd.Timestamp.now().normalize() - pd.Timestamp(computed).normalize()).days
    return {"n_rows": int(n or 0), "last_date": last_date, "computed_at": computed,
            "n_fits": int(n_fits or 0), "age_days": age}


def _latest_regime_date():
    with get_connection() as conn:
        row = conn.execute("SELECT MAX(date) FROM regime_history").fetchone()
    return row[0] if row else None


def is_stale(max_age_days: int = MAX_AGE_DAYS, target_date=None) -> bool:
    """Should the daily job rebuild it? Missing, too few fits, too old, or not
    reaching the latest regime_history date (the single-fit series the daily
    job has just extended)."""
    s = ensemble_status()
    if s["n_rows"] == 0 or s["n_fits"] < MIN_FITS:
        return True
    target = target_date or _latest_regime_date()
    if target and (s["last_date"] is None or str(s["last_date"])[:10] < str(target)[:10]):
        return True
    return s["age_days"] is None or s["age_days"] > max_age_days


def exposure_for(as_of, fallback_regime: str = None) -> dict:
    """Live exposure multiplier for a date: 1 - p_bear, from the stored ensemble.

    Falls back to the single-fit hard label if the ensemble is unavailable, so a
    missing ensemble degrades to the OLD behaviour rather than failing to trade
    -- but the caller is told which path was used, because they are not
    equally trustworthy.
    """
    p = load_ensemble()
    as_of = pd.Timestamp(as_of)
    if not p.empty:
        avail = p.loc[:as_of]
        if len(avail):
            pb = float(avail.iloc[-1])
            return {"exposure": float(np.clip(1.0 - pb, 0.0, 1.0)), "p_bear": pb,
                    "source": "ensemble", "as_of_signal": str(avail.index[-1].date())}
    exp = 0.0 if fallback_regime == "BEAR" else 1.0
    logger.warning("exposure_for: ensemble unavailable, falling back to single-fit label %s",
                   fallback_regime)
    return {"exposure": exp, "p_bear": None, "source": "single_fit_fallback",
            "as_of_signal": None}


def refresh_if_stale(seeds=DEFAULT_SEEDS, max_age_days: int = MAX_AGE_DAYS) -> dict:
    """Rebuild and store the ensemble when stale. Safe to call from the daily
    job -- returns without doing the expensive work when it is fresh."""
    if not is_stale(max_age_days):
        return {"rebuilt": False, **ensemble_status()}
    # store_ensemble upserts by date; ensemble_status reads MAX(computed_at),
    # so a forward-filter rebuild correctly reads as fresh afterwards.
    p = build_ensemble(seeds=seeds)
    n = store_ensemble(p, n_fits=len(seeds))
    return {"rebuilt": bool(n), "rows": n, **ensemble_status()}
