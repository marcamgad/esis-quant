"""
Regime-model stability test  --  is the overlay an edge, or an artifact?

WHY THIS EXISTS. Expanding the universe from 27 to 47 tickers changed the HMM's
regime labels (BEAR days 560 -> 384) and flipped the overlay's contribution from
+0.4pp CAGR to -10pp. The core stock book was IDENTICAL across both runs
(29.3% / Sharpe 1.59), so the entire difference came from the regime labeling.

That is alarming, because adding twenty liquid EGX names should not change
whether the market was in a bear regime in 2018. If a benign input change moves
the labels that much, the overlay's measured +46% cumulative edge may be a
property of one particular fit rather than of the market.

WHAT THIS MEASURES. The core book is held FIXED, and only the regime model is
re-estimated, under two independent perturbations:

  * UNIVERSE  -- refit the observation matrix on random subsets of the tickers.
    A regime is supposed to be a market-wide state; a model that reads it
    differently from a random 70% of the same market is not measuring the market.
  * SEED      -- refit on the identical data with different random restarts.
    The HMM's EM is non-convex, so different initialisations can land in
    different optima. Any variation here is pure estimation noise with no
    economic content at all.

For each refit we record the BEAR-day count and, crucially, the overlay's effect
on the same core returns (delta CAGR and delta Sharpe versus no overlay). The
spread of those deltas is the answer: if it straddles zero, the overlay is not a
reliable edge, whatever any single run reported.

HOW TO READ THE RESULT. This is not a test the overlay can "pass" by being
positive on average. The question is whether its contribution is STABLE. A mean
of +5pp with a range of -15pp to +20pp means the historical +46% was a draw from
a wide distribution, not a measurement of something real.

Nothing here writes to regime_history -- refits are held in memory so the live
labels are untouched.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.regime_stability")

from backtest.fees import book_pct_per_side
COST = book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)


def _perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}
    sd = r.std(ddof=1)
    nav = (1 + r).cumprod()
    return {"cagr": float((1 + r).prod() ** (252 / len(r)) - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
            "max_dd": float((nav / nav.cummax() - 1).min())}


def core_returns(tickers=None, lookback: int = 180, rebalance_days: int = 21) -> pd.Series:
    """The min-variance book. Held FIXED across every refit so that any change
    in measured performance is attributable to the regime labels alone."""
    from data.universe import load_tradable
    from backtest.engine import BacktestEngine, load_price_returns
    from portfolio.system import _min_variance_fn
    rets = load_price_returns(tickers=tickers)
    eng = BacktestEngine(rebalance_days=rebalance_days, cost_per_turnover=COST)
    return eng.run_portfolio(_min_variance_fn, rets, lookback=lookback,
                             tradable=load_tradable())["net_returns"]


def fit_regimes(tickers=None, n_states: int = 4, n_init: int = 3,
                random_state: int = 0, obs: pd.DataFrame = None,
                use_cache: bool = False) -> pd.Series:
    """Refit the regime model IN MEMORY on a given ticker set and seed.

    Uses the same observation matrix and labelling the live pipeline uses, so
    this measures the real model's stability, not a simplified stand-in.

    use_cache defaults to False here: a stability study on a ticker subset must
    really refit. The live ensemble passes True (frozen vintages).
    """
    from models.regime_features import build_observation_matrix
    from models.regime_pipeline import compute_regime_series

    if obs is None:
        obs = build_observation_matrix(tickers=tickers)
    if obs is None or obs.empty:
        return pd.Series(dtype=object)
    # walkforward = the mode the live system uses (annual refit + filtered
    # decode). Using anything else here would measure the stability of a model
    # that is not the one being traded.
    df = compute_regime_series(obs, mode="walkforward", n_states=n_states,
                               n_init=n_init, random_state=random_state, use_cache=use_cache)
    if df is None or df.empty:
        return pd.Series(dtype=object)
    return pd.Series(df["regime"].to_numpy(), index=pd.DatetimeIndex(df["date"]))


def overlay_delta(core: pd.Series, regime: pd.Series, bear_scale: float = 0.0) -> dict:
    """Effect of applying the overlay to a fixed core book."""
    from risk.overlay import apply_overlay
    reg = regime.reindex(core.index).ffill()
    managed = apply_overlay(core, reg, bear_scale=bear_scale,
                            cost_per_turnover=COST)["managed_returns"]
    base, mgd = _perf(core), _perf(managed)
    return {"d_cagr": mgd["cagr"] - base["cagr"], "d_sharpe": mgd["sharpe"] - base["sharpe"],
            "bear_days": int((reg == "BEAR").sum()), "managed_cagr": mgd["cagr"],
            "managed_sharpe": mgd["sharpe"]}


def stability_study(n_universe: int = 8, n_seed: int = 6, subset_frac: float = 0.7,
                    seed: int = 0, tickers=None) -> pd.DataFrame:
    """Refit under universe subsets and under reseeding; return one row each."""
    all_t = [t["ticker"] for t in config.EGX_SEED_TICKERS] if tickers is None else list(tickers)
    core = core_returns()
    base = _perf(core)
    logger.info("stability_study: core CAGR %.3f sharpe %.2f", base["cagr"], base["sharpe"])
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_seed):
        lab = fit_regimes(tickers=None, random_state=100 + i)
        if lab.empty:
            continue
        d = overlay_delta(core, lab)
        rows.append({"perturbation": "seed", "trial": i, "n_tickers": len(all_t), **d})

    k = max(5, int(len(all_t) * subset_frac))
    for i in range(n_universe):
        sub = list(rng.choice(all_t, size=k, replace=False))
        lab = fit_regimes(tickers=sub, random_state=0)
        if lab.empty:
            continue
        d = overlay_delta(core, lab)
        rows.append({"perturbation": "universe", "trial": i, "n_tickers": k, **d})

    df = pd.DataFrame(rows)
    df.attrs["core_cagr"] = base["cagr"]
    df.attrs["core_sharpe"] = base["sharpe"]
    return df


def format_study(df: pd.DataFrame) -> str:
    if df.empty:
        return "regime stability: no successful refits."
    lines = ["REGIME-MODEL STABILITY  (core book held fixed: "
             f"CAGR {df.attrs.get('core_cagr', float('nan'))*100:.1f}%, "
             f"Sharpe {df.attrs.get('core_sharpe', float('nan')):.2f})",
             "  overlay contribution across refits -- if this straddles zero,",
             "  the overlay is not a reliable edge.",
             f"  {'perturbation':14}{'n':>4}{'BEAR days':>11}{'dCAGR':>9}{'dSharpe':>9}"]
    for g, sub in df.groupby("perturbation"):
        for r in sub.itertuples():
            lines.append(f"  {g:14}{r.trial:>4}{r.bear_days:>11}"
                         f"{r.d_cagr*100:>8.1f}%{r.d_sharpe:>9.2f}")
    lines.append("  " + "-" * 46)
    for g, sub in df.groupby("perturbation"):
        lines.append(f"  {g:12} dCAGR mean {sub['d_cagr'].mean()*100:+6.1f}%  "
                     f"range [{sub['d_cagr'].min()*100:+.1f}%, {sub['d_cagr'].max()*100:+.1f}%]"
                     f"   dSharpe mean {sub['d_sharpe'].mean():+.2f} "
                     f"range [{sub['d_sharpe'].min():+.2f}, {sub['d_sharpe'].max():+.2f}]")
    pos = (df["d_cagr"] > 0).mean()
    lines.append(f"  overlay HELPED in {pos*100:.0f}% of refits, hurt in {(1-pos)*100:.0f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    df = stability_study()
    print(format_study(df))
    df.to_csv("regime_stability.csv", index=False)
    print("\nfull results -> regime_stability.csv")


# --------------------------------------------------------------------------
# Ensemble fix: average out the estimation noise instead of trusting one fit
# --------------------------------------------------------------------------
def ensemble_bear_probability(seeds, tickers=None, n_init: int = 3,
                              use_cache: bool = False) -> pd.Series:
    """Fraction of independent HMM refits that label each day BEAR.

    THE IDEA. A single fit's BEAR label is contaminated by EM initialisation
    noise -- the stability study showed BEAR-day counts swinging 478-790 on
    IDENTICAL data purely from the random start. Averaging many fits should
    cancel that noise and leave whatever genuine signal exists, and it yields a
    CONTINUOUS confidence (0..1) rather than a brittle hard label, so days where
    the fits disagree get partial de-risking instead of an arbitrary on/off.

    This is the last honest defence of the regime thesis: if the ensembled
    signal is BOTH stable and positive, the overlay is real and the earlier
    instability was an estimation artifact. If it is not, the thesis is dead.
    """
    from models.regime_features import build_observation_matrix
    obs = build_observation_matrix(tickers=tickers)       # once, not once per seed
    labels = []
    for s in seeds:
        lab = fit_regimes(tickers=tickers, random_state=int(s), n_init=n_init,
                          obs=obs, use_cache=use_cache)
        if not lab.empty:
            labels.append(lab)
    if not labels:
        return pd.Series(dtype=float)
    idx = labels[0].index
    for l in labels[1:]:
        idx = idx.union(l.index)
    votes = pd.DataFrame({i: l.reindex(idx) for i, l in enumerate(labels)})
    return (votes == "BEAR").mean(axis=1).rename("p_bear")


def ensemble_overlay(core: pd.Series, p_bear: pd.Series, mode: str = "continuous",
                     threshold: float = 0.5) -> dict:
    """Apply the ensemble consensus as exposure.

    "continuous" -- exposure = 1 - P(BEAR): partial de-risking in proportion to
                    how many fits agree. This is the point of ensembling.
    "vote"       -- exposure = 0 when a majority of fits say BEAR, else 1.
                    Included only as the hard-label comparison.
    """
    from risk.overlay import apply_overlay
    p = p_bear.reindex(core.index).ffill().fillna(0.0).clip(0.0, 1.0)
    reg = pd.Series(np.where(p >= threshold, "BEAR", "OTHER"), index=core.index)
    if mode == "vote":
        managed = apply_overlay(core, reg, bear_scale=0.0, cost_per_turnover=COST)["managed_returns"]
    else:
        # continuous exposure decided at t-1 (no look-ahead), cost on |change|
        e = (1.0 - p).shift(1).fillna(1.0).to_numpy()
        turn = np.abs(np.diff(np.concatenate([[0.0], e]))) * COST
        managed = pd.Series(e * core.to_numpy() - turn, index=core.index)
    base, mgd = _perf(core), _perf(managed)
    return {"d_cagr": mgd["cagr"] - base["cagr"], "d_sharpe": mgd["sharpe"] - base["sharpe"],
            "managed_cagr": mgd["cagr"], "managed_sharpe": mgd["sharpe"],
            "core_max_dd": base["max_dd"], "managed_max_dd": mgd["max_dd"],
            "d_max_dd": mgd["max_dd"] - base["max_dd"],
            "mean_exposure": float((1.0 - p).mean()), "returns": managed}


def ensemble_study(n_per_ensemble: int = 5, n_ensembles: int = 2, n_init: int = 3) -> pd.DataFrame:
    """Build several INDEPENDENT ensembles from disjoint seed sets.

    The trap this avoids: an ensemble that happens to look good once is the same
    error as a single fit that happens to look good once. Two disjoint ensembles
    must AGREE for the result to mean anything.
    """
    core = core_returns()
    rows = []
    for e in range(n_ensembles):
        seeds = [1000 * (e + 1) + i for i in range(n_per_ensemble)]
        p = ensemble_bear_probability(seeds, n_init=n_init)
        if p.empty:
            continue
        for mode in ("continuous", "vote"):
            r = ensemble_overlay(core, p, mode=mode)
            r.pop("returns", None)
            rows.append({"ensemble": e, "mode": mode, "n_fits": n_per_ensemble,
                         "mean_p_bear": float(p.mean()), **r})
    df = pd.DataFrame(rows)
    b = _perf(core)
    df.attrs["core_cagr"] = b["cagr"]
    df.attrs["core_sharpe"] = b["sharpe"]
    return df


def format_ensemble(df: pd.DataFrame) -> str:
    if df.empty:
        return "ensemble study: no successful fits."
    lines = [f"ENSEMBLE REGIME OVERLAY  (core fixed: CAGR "
             f"{df.attrs.get('core_cagr', float('nan'))*100:.1f}%, "
             f"Sharpe {df.attrs.get('core_sharpe', float('nan')):.2f})",
             f"  {'ens':>4}{'mode':>12}{'meanP(BEAR)':>13}{'exposure':>10}{'dCAGR':>9}{'dSharpe':>9}"]
    for r in df.itertuples():
        lines.append(f"  {r.ensemble:>4}{r.mode:>12}{r.mean_p_bear:>13.3f}"
                     f"{r.mean_exposure:>10.2f}{r.d_cagr*100:>8.1f}%{r.d_sharpe:>9.2f}")
    lines.append("  " + "-" * 58)
    for mode, sub in df.groupby("mode"):
        agree = "AGREE" if (sub["d_cagr"] > 0).all() or (sub["d_cagr"] <= 0).all() else "DISAGREE"
        lines.append(f"  {mode:12} across ensembles: dCAGR "
                     f"[{sub['d_cagr'].min()*100:+.1f}%, {sub['d_cagr'].max()*100:+.1f}%]  "
                     f"dSharpe [{sub['d_sharpe'].min():+.2f}, {sub['d_sharpe'].max():+.2f}]  -> {agree}")
    return "\n".join(lines)
