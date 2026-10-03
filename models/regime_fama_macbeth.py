"""
Phase 2.1 -- regime-conditional Fama-MacBeth premia with empirical-Bayes
shrinkage, replacing the HAND-SET regime multiplier table.

THE BUG THIS ATTACKS. models/alpha_score.py multiplies each factor's premium by
a hand-picked regime multiplier (BULL: momentum x1.5, BEAR: liquidity x1.5,
HIGH_VOL: EGP x1.5, RECOVERY: all 1.0). Its own docstring admits these were
adapted from a 9-factor spec when only 5 factors exist, that two of the
instructions were dropped because the factors don't exist, and that RECOVERY is
neutral only because there was no principled substitute. Those are guesses, and
guessed weights are a plausible contributor to the measured -18.4% selection
contribution.

THE FIX. Estimate the premia per regime instead of guessing multipliers:
  1. Run the cross-sectional regression each period (standard Fama-MacBeth).
  2. Average the per-period coefficients WITHIN each regime -> a regime-specific
     premium, plus its standard error across periods in that regime.
  3. Shrink each regime premium toward the global (all-period) premium by
     empirical Bayes, because a regime cell has few periods and its raw mean is
     mostly noise:

         w_r = tau^2 / (tau^2 + se_r^2)
         shrunk_r = w_r * mu_r + (1 - w_r) * mu_global

     tau^2 is the BETWEEN-regime variance of the premia net of sampling error
     (a moment estimator, floored at 0). When a regime has few periods, se_r^2
     is large, w_r -> 0, and the estimate falls back to the global premium --
     which is the honest behaviour: with no information, do not tilt.

PRE-REGISTERED DECISION RULE (written BEFORE the result was seen, per the
build-plan requirement -- do not edit these thresholds after reading output):

  NULL  -- "the 5-factor set carries no usable regime-conditional selection
           alpha at this universe size" if EITHER
             (a) mean shrinkage weight across regimes/factors < 0.30, i.e. the
                 data cannot distinguish regime premia from the global premium
                 and the estimator collapses to the prior; OR
             (b) no factor-regime cell has |t| >= 2.0 for its deviation from
                 the global premium; OR
             (c) the A/B backtest shows selection contribution NOT improving
                 (new minus old <= 0).

  REAL  -- only if the shrinkage weights are materially above 0.30 AND at least
           one cell is statistically distinguishable (|t| >= 2) AND the A/B
           selection contribution improves.

A NULL result here is a publishable, useful outcome and must be reported as
plainly as a positive one. Do not iterate on this if the first honest test
comes back negative.

DATA NOTE: factor_exposures covers the ORIGINAL 27 tickers (46,008 rows) with 5
usable betas; beta_size/value/quality are NULL because no market-cap, book-value
or profitability data exists anywhere in Layer 1. That absence, not the
estimator, is the binding constraint on selection alpha here.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection
from models.factor_model import BETA_COLUMNS

logger = logging.getLogger("esis_quant.regime_fm")

# The dependent variable: next-month return. Betas in the panel are already
# lagged to the PRIOR month-end (_attach_prior_month_betas), so regressing
# this month's return on last month's betas is point-in-time by construction.
RETURN_COL = "monthly_return"

MIN_CROSS_SECTION = 8        # names needed for a meaningful cross-sectional fit
MIN_PERIODS_PER_REGIME = 6   # below this a regime cell is reported but not trusted
SHRINK_WEIGHT_NULL_THRESHOLD = 0.30
T_STAT_THRESHOLD = 2.0


def _load_panel(tickers=None) -> pd.DataFrame:
    """Monthly forward returns joined to PRIOR-month betas and the regime."""
    from models.fama_macbeth import compute_monthly_returns, load_factor_exposures, \
        _attach_prior_month_betas
    exposures = load_factor_exposures(tickers)
    monthly = compute_monthly_returns(tickers)
    merged = _attach_prior_month_betas(monthly, exposures)
    with get_connection() as conn:
        reg = pd.read_sql("SELECT date, regime FROM regime_history", conn, parse_dates=["date"])
    reg["period"] = reg["date"].dt.to_period("M")
    # one regime per month: the modal label, so a single odd day cannot flip it
    modal = reg.groupby("period")["regime"].agg(lambda s: s.value_counts().idxmax())
    merged["period"] = pd.PeriodIndex(pd.to_datetime(merged["date"]), freq="M")
    merged["regime"] = merged["period"].map(modal)
    return merged.dropna(subset=["regime"])


def _cross_section_coefs(panel: pd.DataFrame, factors) -> pd.DataFrame:
    """One OLS cross-sectional regression per period -> coefficient per factor.

    From scratch via lstsq on [1, betas]; periods with too few names are
    skipped rather than fitted on noise.
    """
    rows = []
    for period, g in panel.groupby("period"):
        g = g.dropna(subset=list(factors) + [RETURN_COL])
        if len(g) < MIN_CROSS_SECTION:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[f].to_numpy(float) for f in factors])
        y = g[RETURN_COL].to_numpy(float)
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        rows.append({"period": period, "regime": g["regime"].iloc[0], "n": len(g),
                     **{f: coef[i + 1] for i, f in enumerate(factors)}})
    return pd.DataFrame(rows)


def _empirical_bayes(mu_r: float, se_r: float, mu_g: float, tau2: float) -> tuple:
    """Shrink a noisy group mean toward the global mean. Returns (estimate, w)."""
    if not np.isfinite(se_r) or se_r <= 0:
        return mu_g, 0.0
    w = tau2 / (tau2 + se_r ** 2) if (tau2 + se_r ** 2) > 0 else 0.0
    w = float(np.clip(w, 0.0, 1.0))
    return w * mu_r + (1.0 - w) * mu_g, w


def estimate_regime_premia(tickers=None, factors=None) -> pd.DataFrame:
    """Regime-conditional premia, raw and empirical-Bayes shrunk, with the
    t-stat of each cell's deviation from the global premium."""
    factors = list(factors or [c for c in BETA_COLUMNS])
    panel = _load_panel(tickers)
    coefs = _cross_section_coefs(panel, factors)
    if coefs.empty:
        logger.error("estimate_regime_premia: no usable cross-sections")
        return pd.DataFrame()

    out = []
    for f in factors:
        x = coefs[f].dropna()
        if x.empty:
            continue
        mu_g = float(x.mean())
        # between-regime variance net of sampling noise (moment estimator)
        grp = coefs.groupby("regime")[f]
        means, ses, ns = {}, {}, {}
        for r, s in grp:
            s = s.dropna()
            ns[r] = len(s)
            means[r] = float(s.mean()) if len(s) else np.nan
            ses[r] = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
        valid = [r for r in means if np.isfinite(means[r]) and np.isfinite(ses[r])]
        if not valid:
            continue
        between = float(np.var([means[r] for r in valid], ddof=1)) if len(valid) > 1 else 0.0
        mean_se2 = float(np.mean([ses[r] ** 2 for r in valid]))
        tau2 = max(between - mean_se2, 0.0)
        for r in valid:
            est, w = _empirical_bayes(means[r], ses[r], mu_g, tau2)
            t = (means[r] - mu_g) / ses[r] if ses[r] > 0 else np.nan
            out.append({"factor": f, "regime": r, "n_periods": ns[r],
                        "premium_raw": means[r], "premium_global": mu_g,
                        "se": ses[r], "tau2": tau2, "shrink_w": w,
                        "premium_shrunk": est, "t_vs_global": t,
                        "trustworthy": ns[r] >= MIN_PERIODS_PER_REGIME})
    return pd.DataFrame(out)


def verdict(df: pd.DataFrame) -> dict:
    """Apply the PRE-REGISTERED rule. Returns the decision and its reasons."""
    if df.empty:
        return {"verdict": "NULL", "reasons": ["no estimable premia"]}
    mean_w = float(df["shrink_w"].mean())
    max_t = float(df["t_vs_global"].abs().max())
    reasons = []
    if mean_w < SHRINK_WEIGHT_NULL_THRESHOLD:
        reasons.append(f"mean shrinkage weight {mean_w:.3f} < {SHRINK_WEIGHT_NULL_THRESHOLD} "
                       "-- estimator collapses to the global premium")
    if max_t < T_STAT_THRESHOLD:
        reasons.append(f"no cell distinguishable from global (max |t| = {max_t:.2f} "
                       f"< {T_STAT_THRESHOLD})")
    return {"verdict": "NULL" if reasons else "PENDING_AB",
            "mean_shrink_w": mean_w, "max_abs_t": max_t, "reasons": reasons,
            "n_cells": int(len(df)),
            "n_trustworthy": int(df["trustworthy"].sum())}


def format_premia(df: pd.DataFrame, v: dict) -> str:
    if df.empty:
        return "regime-conditional Fama-MacBeth: no estimable premia."
    lines = ["REGIME-CONDITIONAL FAMA-MACBETH (empirical-Bayes shrunk)",
             f"  {'factor':24}{'regime':10}{'n':>4}{'raw':>10}{'global':>10}"
             f"{'shrunk':>10}{'w':>7}{'t':>7}"]
    for r in df.sort_values(["factor", "regime"]).itertuples():
        flag = "" if r.trustworthy else "  (thin)"
        lines.append(f"  {r.factor:24}{r.regime:10}{r.n_periods:>4}{r.premium_raw:>10.4f}"
                     f"{r.premium_global:>10.4f}{r.premium_shrunk:>10.4f}"
                     f"{r.shrink_w:>7.2f}{r.t_vs_global:>7.2f}{flag}")
    lines.append("  " + "-" * 74)
    lines.append(f"  mean shrinkage weight {v['mean_shrink_w']:.3f}   max |t| vs global {v['max_abs_t']:.2f}")
    lines.append(f"  PRE-REGISTERED VERDICT: {v['verdict']}")
    for reason in v["reasons"]:
        lines.append(f"     - {reason}")
    return "\n".join(lines)


def shrunk_weight_table(df: pd.DataFrame) -> dict:
    """{regime: {factor: premium}} -- the ESTIMATED replacement for the hand-set
    REGIME_ADJUSTMENTS multipliers."""
    if df.empty:
        return {}
    return {r: dict(zip(g["factor"], g["premium_shrunk"]))
            for r, g in df.groupby("regime")}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    d = estimate_regime_premia()
    print(format_premia(d, verdict(d)))
