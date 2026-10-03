"""
Monte Carlo stress + parameter robustness  (deferred items P6.16, P6.17).

TWO QUESTIONS THE EXISTING TOOLS DO NOT ANSWER.

  * analysis/significance.py block-bootstraps the REALISED return series. That
    asks "could this Sharpe have come from luck given these returns?" It cannot
    ask "what range of outcomes should I expect going forward?", because it only
    ever resamples the one path that happened.

  * analysis/overlay_tuning.py sweeps bear_scale. Nothing sweeps the two
    parameters the whole book actually rests on: the covariance LOOKBACK (180
    days) and the HMM REGIME COUNT. A result that only works at 180 days is a
    fitted parameter wearing a backtest.

MONTE CARLO METHOD. Stationary-block resampling of daily returns to build many
synthetic paths of the same length, preserving autocorrelation and volatility
clustering that an i.i.d. bootstrap would destroy. Reports the distribution of
terminal outcomes and, specifically, P(loss) at 1 and 3 years -- the number that
actually matters to someone deciding whether to fund the account.

THIS IS NOT A FORECAST. It resamples history, so it assumes the future is drawn
from the same distribution as 2015-2026 -- a period containing two currency
devaluations. It cannot produce a shock bigger than the worst block it has seen,
so the left tail is understated by construction. Stated here so the P(loss)
figure is never read as a bound.

PARAMETER SWEEP METHOD. Re-run the whole system at each setting. A CLIFF at the
current setting is a red flag, not something to route around: if 180 days works
and 150/210 do not, the 180 was fitted. Flat-ish is what robustness looks like.

Nothing here is tuned. The grids are round numbers chosen before the results
were seen, and the current setting is not centred in either grid on purpose.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.robustness_suite")

TRADING_DAYS = 252
LOOKBACK_GRID = (90, 120, 180, 250, 360)
REGIME_GRID = (2, 3, 4, 5)
DEFAULT_PATHS = 2000


def _stationary_blocks(r: np.ndarray, n_out: int, mean_block: int, rng) -> np.ndarray:
    """One synthetic path by stationary bootstrap (Politis-Romano): geometric
    block lengths, wrapped indexing. Preserves short-range dependence without
    imposing a single fixed block length."""
    n = len(r)
    out = np.empty(n_out)
    i = 0
    while i < n_out:
        start = rng.integers(0, n)
        length = max(1, int(rng.geometric(1.0 / mean_block)))
        length = min(length, n_out - i)
        idx = (start + np.arange(length)) % n
        out[i:i + length] = r[idx]
        i += length
    return out


def monte_carlo(returns, horizons=(1, 3), n_paths: int = DEFAULT_PATHS,
                mean_block: int = 21, seed: int = 0) -> dict:
    """Distribution of outcomes over each horizon (in years)."""
    r = pd.Series(returns).dropna().values
    if len(r) < 100:
        return {"ok": False, "reason": f"only {len(r)} observations"}
    rng = np.random.default_rng(seed)
    out = {"ok": True, "n_paths": n_paths, "mean_block": mean_block,
           "n_obs": len(r), "horizons": {}}
    for years in horizons:
        n_out = int(TRADING_DAYS * years)
        totals = np.empty(n_paths)
        worst_dd = np.empty(n_paths)
        for p in range(n_paths):
            path = _stationary_blocks(r, n_out, mean_block, rng)
            nav = np.cumprod(1.0 + path)
            totals[p] = nav[-1] - 1.0
            worst_dd[p] = float((nav / np.maximum.accumulate(nav) - 1.0).min())
        out["horizons"][years] = {
            "median": float(np.median(totals)),
            "p05": float(np.percentile(totals, 5)),
            "p25": float(np.percentile(totals, 25)),
            "p75": float(np.percentile(totals, 75)),
            "p95": float(np.percentile(totals, 95)),
            "p_loss": float((totals < 0).mean()),
            "median_maxdd": float(np.median(worst_dd)),
            "p05_maxdd": float(np.percentile(worst_dd, 5)),
        }
    return out


def format_monte_carlo(d: dict, label: str = "") -> str:
    if not d.get("ok"):
        return f"monte carlo: {d.get('reason')}"
    lines = [f"MONTE CARLO — {label} ({d['n_paths']:,} stationary-block paths, "
             f"mean block {d['mean_block']}d, from {d['n_obs']} observations)",
             f"  {'horizon':>8}{'p05':>9}{'p25':>9}{'median':>9}{'p75':>9}{'p95':>9}"
             f"{'P(loss)':>10}{'med MaxDD':>11}"]
    for years, h in d["horizons"].items():
        lines.append(f"  {years:>6}yr{h['p05']*100:>8.1f}%{h['p25']*100:>8.1f}%"
                     f"{h['median']*100:>8.1f}%{h['p75']*100:>8.1f}%{h['p95']*100:>8.1f}%"
                     f"{h['p_loss']*100:>9.1f}%{h['median_maxdd']*100:>10.1f}%")
    lines.append("  Resamples history, so it cannot produce a shock larger than the worst")
    lines.append("  block already seen — the left tail is understated by construction.")
    return "\n".join(lines)


def lookback_sweep(grid=LOOKBACK_GRID, rebalance_days: int = 21) -> pd.DataFrame:
    """Whole system re-run at each covariance lookback."""
    from portfolio.system import run_full_system
    rows = []
    for lb in grid:
        try:
            r = run_full_system(rebalance_days=rebalance_days, lookback=lb)
            m, c = r["metrics"]["full_system"], r["metrics"]["core"]
            rows.append({"lookback": lb, "system_cagr": m["ann_return"],
                         "system_sharpe": m["sharpe"], "system_maxdd": m["max_drawdown"],
                         "core_cagr": c["ann_return"], "core_sharpe": c["sharpe"]})
        except Exception as exc:  # noqa: BLE001
            logger.warning("lookback %s failed: %s", lb, exc)
    return pd.DataFrame(rows)


def regime_count_sweep(grid=REGIME_GRID) -> pd.DataFrame:
    """Refit the HMM at each state count and measure the resulting BEAR overlay.

    *** THESE NUMBERS ARE IN-SAMPLE AND ARE NOT COMPARABLE TO THE LIVE SYSTEM. ***

    An earlier version of this docstring claimed it used "the same walk-forward
    machinery the live path uses". That was FALSE and is corrected here. This
    function fits the HMM once on the WHOLE history and decodes the whole
    history with those parameters -- the smoothed, look-ahead-contaminated mode
    that models/regime_pipeline exists to avoid.

    WHY THAT MATTERS FOR READING THE RESULT. Lower state counts label far more
    days BEAR (2 states: 1,557 days vs 330 at 4 states). An in-sample fit knows
    which days those were. So the much better Sharpe and drawdown at 2-3 states
    is exactly what look-ahead produces, and is NOT evidence that the live model
    should use fewer states. Concluding "switch to 3 states" from this table
    would be adopting a look-ahead artifact as a strategy change.

    It is still worth running: it shows the result is SENSITIVE to state count,
    which is a robustness finding regardless of the direction. A walk-forward
    version of this sweep is the honest follow-up and has not been built.
    """
    from models.hmm import GaussianHMM
    from models.regime_features import build_observation_matrix
    from backtest.engine import load_price_returns
    from portfolio.system import _min_variance_fn
    from backtest.engine import BacktestEngine
    from data.universe import load_tradable

    rets = load_price_returns()
    eng = BacktestEngine(rebalance_days=21)
    core = eng.run_portfolio(_min_variance_fn, rets, lookback=180,
                             tradable=load_tradable())["net_returns"]
    try:
        feats = build_observation_matrix()
    except Exception as exc:  # noqa: BLE001
        logger.warning("regime features unavailable: %s", exc)
        return pd.DataFrame()

    X = feats.dropna()
    rows = []
    for k in grid:
        try:
            hmm = GaussianHMM(n_states=k, random_state=0)
            hmm.fit(X.values)
            states = pd.Series(hmm.predict_proba(X.values).argmax(axis=0)
                               if hmm.predict_proba(X.values).shape[0] == k
                               else hmm.predict_proba(X.values).argmax(axis=1),
                               index=X.index)
            # BEAR = the state with the lowest mean market return, defined from
            # the fitted parameters rather than assumed by index
            means = {s: float(X.iloc[:, 0][states == s].mean()) for s in states.unique()}
            bear = min(means, key=means.get)
            expo = (states != bear).astype(float).reindex(core.index).ffill().fillna(1.0).shift(1).fillna(1.0)
            managed = expo * core
            nav = (1 + managed).cumprod()
            sd = managed.std(ddof=1)
            rows.append({"n_states": k,
                         "bear_days": int((states == bear).sum()),
                         "cagr": float((1 + managed).prod() ** (TRADING_DAYS / len(managed)) - 1),
                         "sharpe": float(managed.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
                         "maxdd": float((nav / nav.cummax() - 1).min())})
        except Exception as exc:  # noqa: BLE001
            logger.warning("regime count %s failed: %s", k, exc)
    return pd.DataFrame(rows)


def format_sweeps(lb: pd.DataFrame, rg: pd.DataFrame, current_lookback: int = 180) -> str:
    lines = ["PARAMETER ROBUSTNESS — is the result a plateau or a spike?", ""]
    if not lb.empty:
        lines.append(f"  COVARIANCE LOOKBACK (live setting: {current_lookback})")
        lines.append(f"    {'days':>6}{'sys CAGR':>11}{'sys Sharpe':>12}{'core CAGR':>11}{'core Sharpe':>13}")
        for r in lb.itertuples():
            mark = "  <-- live" if r.lookback == current_lookback else ""
            lines.append(f"    {r.lookback:>6}{r.system_cagr*100:>10.2f}%{r.system_sharpe:>12.3f}"
                         f"{r.core_cagr*100:>10.2f}%{r.core_sharpe:>13.3f}{mark}")
        spread = lb["core_sharpe"].max() - lb["core_sharpe"].min()
        best = lb.loc[lb["core_sharpe"].idxmax(), "lookback"]
        lines.append(f"    core Sharpe range {spread:.3f}; best at {best} days -> "
                     f"{'PLATEAU (robust)' if spread < 0.30 else 'SENSITIVE — the setting matters'}")
        lines.append("")
    if not rg.empty:
        lines.append("  HMM REGIME COUNT (live setting: 4)")
        lines.append("    *** IN-SAMPLE FIT — NOT comparable to the live walk-forward system.")
        lines.append("    Fewer states label more days BEAR and an in-sample fit knows which")
        lines.append("    days those were, so the 2-3 state advantage is look-ahead, not edge.")
        lines.append(f"    {'states':>7}{'bear days':>11}{'CAGR':>10}{'Sharpe':>9}{'MaxDD':>9}")
        for r in rg.itertuples():
            mark = "  <-- live" if r.n_states == 4 else ""
            lines.append(f"    {r.n_states:>7}{r.bear_days:>11}{r.cagr*100:>9.2f}%"
                         f"{r.sharpe:>9.3f}{r.maxdd*100:>8.1f}%{mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from portfolio.system import run_full_system
    r = run_full_system()
    print(format_monte_carlo(monte_carlo(r["returns"]["full_system"]), "full system"))
    print()
    print(format_sweeps(lookback_sweep(), regime_count_sweep()))
