"""
Is the de-risk overlay worth what it costs -- and is it even well-timed?

TWO SEPARATE QUESTIONS, DELIBERATELY NOT CONFLATED.

  (1) EFFICIENCY. The overlay currently goes fully to cash in BEAR
      (bear_scale = 0). That is the most aggressive possible setting: it gives
      up 100% of the upside during any period the model calls bear. A partial
      scale (say 0.3-0.5) gives up less return per unit of drawdown protection.
      The right measure is not "which has the highest CAGR" but the EXCHANGE
      RATE: how many points of CAGR are surrendered per point of drawdown
      avoided. A setting that saves 10pp of drawdown for 2pp of return is
      better than one that saves 12pp for 8pp.

  (2) TIMING ACCURACY. Efficiency tuning is meaningless if the bear calls are
      simply wrong. So this also measures, unconditionally: what did the market
      ACTUALLY do after each regime label? If forward returns following BEAR are
      no worse than following other regimes, the overlay is not mistimed -- it
      is uninformative, and no amount of scaling fixes that. That check must
      come first, because tuning a broken signal produces a fitted parameter,
      not an improvement.

WHAT WOULD JUSTIFY KEEPING THE OVERLAY: forward returns after BEAR are
materially worse than after other regimes (so the signal carries information),
AND some scale delivers a better drawdown-per-return exchange rate than simply
holding. If neither holds, the honest conclusion is to retire the overlay and
run a plain min-variance core -- a simpler, more honest product.

Everything runs on the SAME core book so any difference is attributable to the
overlay alone, and on the ENSEMBLE regime signal (not a single HMM fit), since
single fits were shown to be unstable.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("esis_quant.overlay_tuning")

TRADING_DAYS = 252
from backtest.fees import book_pct_per_side
COST = book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)
BEAR_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
FORWARD_HORIZONS = (5, 21, 63)


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    nav = (1 + r).cumprod()
    sd = r.std(ddof=1)
    return {"cagr": float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "max_dd": float((nav / nav.cummax() - 1).min())}


def core_and_regime():
    """The fixed core book plus the ensemble p_bear, aligned."""
    from data.universe import load_tradable
    from backtest.engine import BacktestEngine, load_price_returns
    from portfolio.system import _min_variance_fn
    from models.regime_ensemble import load_ensemble

    rets = load_price_returns()
    eng = BacktestEngine(rebalance_days=21, cost_per_turnover=COST)
    core = eng.run_portfolio(_min_variance_fn, rets, lookback=180,
                             tradable=load_tradable())["net_returns"]
    p = load_ensemble().reindex(core.index).ffill().fillna(0.0).clip(0, 1)
    return core, p


def timing_accuracy(core: pd.Series, p_bear: pd.Series,
                    horizons=FORWARD_HORIZONS, threshold: float = 0.5) -> pd.DataFrame:
    """What did the book ACTUALLY do after a bear call, versus after a non-bear
    call? Uses only realized forward returns -- no model, no assumption."""
    rows = []
    is_bear = p_bear >= threshold
    for h in horizons:
        fwd = (1 + core).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        b, n = fwd[is_bear].dropna(), fwd[~is_bear].dropna()
        if len(b) < 20 or len(n) < 20:
            continue
        # Welch t-stat: are the two forward-return distributions different?
        se = np.sqrt(b.var(ddof=1) / len(b) + n.var(ddof=1) / len(n))
        t_raw = float((b.mean() - n.mean()) / se) if se > 0 else np.nan
        # OVERLAP CORRECTION -- essential, not cosmetic. Rolling h-day forward
        # returns share h-1 days with their neighbours, so consecutive
        # observations are almost the same number. Treating them as independent
        # inflates the t-stat by ~sqrt(h): a 63-day window over 2,670 days gives
        # roughly 2670/63 ~ 42 independent observations, not 2,670. Reporting
        # the raw figure would manufacture significance out of double-counting.
        t_adj = t_raw / np.sqrt(h) if t_raw == t_raw else np.nan
        rows.append({"horizon": h, "n_bear": len(b), "n_other": len(n),
                     "mean_after_bear": float(b.mean()), "mean_after_other": float(n.mean()),
                     "diff": float(b.mean() - n.mean()),
                     "t_raw": t_raw, "t_stat": t_adj,
                     "n_independent": int(min(len(b), len(n)) / h),
                     "significant": bool(abs(t_adj) >= 2.0) if t_adj == t_adj else False,
                     "bear_worse": bool(b.mean() < n.mean())})
    return pd.DataFrame(rows)


def scale_sweep(core: pd.Series, p_bear: pd.Series, scales=BEAR_SCALES,
                continuous: bool = False) -> pd.DataFrame:
    """Performance at each bear_scale. `continuous` blends by p_bear instead of
    switching on a hard threshold."""
    base = _stats(core)
    rows = []
    for s in scales:
        if continuous:
            e = (1.0 - (1.0 - s) * p_bear).shift(1).fillna(1.0)
        else:
            e = pd.Series(np.where(p_bear.shift(1).fillna(0.0) >= 0.5, s, 1.0), index=core.index)
        turn = e.diff().abs().fillna(0.0) * COST
        managed = e * core - turn
        st = _stats(managed)
        # the exchange rate: CAGR surrendered per point of drawdown avoided
        dd_saved = st["max_dd"] - base["max_dd"]          # positive = less drawdown
        cagr_given = base["cagr"] - st["cagr"]            # positive = gave up return
        rows.append({"bear_scale": s, "mode": "continuous" if continuous else "hard",
                     **st, "dd_saved_pp": dd_saved, "cagr_given_pp": cagr_given,
                     "cost_per_dd_point": (cagr_given / dd_saved) if dd_saved > 1e-9 else np.nan,
                     "mean_exposure": float(e.mean())})
    return pd.DataFrame(rows)


def format_report(acc: pd.DataFrame, sweep: pd.DataFrame, base: dict) -> str:
    lines = ["OVERLAY TUNING — is de-risking well-timed, and is it worth the cost?",
             f"  core book (no overlay): CAGR {base['cagr']*100:.1f}%  "
             f"Sharpe {base['sharpe']:.2f}  MaxDD {base['max_dd']*100:.1f}%", ""]

    lines.append("  (1) TIMING ACCURACY — what actually happened after a BEAR call?")
    if acc.empty:
        lines.append("      insufficient data")
    else:
        lines.append(f"      {'horizon':>8}{'after BEAR':>12}{'after other':>13}"
                     f"{'diff':>9}{'t(raw)':>8}{'t(adj)':>8}{'indep n':>9}  verdict")
        for r in acc.itertuples():
            v = ("worse & significant" if (r.bear_worse and r.significant)
                 else "worse, NOT significant" if r.bear_worse else "NOT worse")
            lines.append(f"      {r.horizon:>7}d{r.mean_after_bear*100:>11.2f}%"
                         f"{r.mean_after_other*100:>12.2f}%{r.diff*100:>8.2f}%"
                         f"{r.t_raw:>8.2f}{r.t_stat:>8.2f}{r.n_independent:>9}  {v}")
        lines.append("      t(adj) divides by sqrt(horizon) for overlapping windows --")
        lines.append("      the raw figure double-counts and is not usable.")
    lines.append("")
    lines.append("  (2) EXCHANGE RATE — CAGR given up per point of drawdown avoided")
    lines.append(f"      {'scale':>7}{'mode':>12}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"
                 f"{'ddSaved':>9}{'cagrGiven':>11}{'cost/pt':>9}")
    for r in sweep.itertuples():
        cp = f"{r.cost_per_dd_point:.2f}" if r.cost_per_dd_point == r.cost_per_dd_point else "  n/a"
        lines.append(f"      {r.bear_scale:>7.2f}{r.mode:>12}{r.cagr*100:>8.1f}%"
                     f"{r.sharpe:>8.2f}{r.max_dd*100:>8.1f}%{r.dd_saved_pp*100:>8.1f}"
                     f"{r.cagr_given_pp*100:>10.1f}{cp:>9}")
    lines.append("      cost/pt < 1 means each point of drawdown avoided cost less than a")
    lines.append("      point of CAGR. Lower is better; negative means it improved both.")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    core, p = core_and_regime()
    base = _stats(core)
    acc = timing_accuracy(core, p)
    sweep = pd.concat([scale_sweep(core, p, continuous=False),
                       scale_sweep(core, p, continuous=True)], ignore_index=True)
    print(format_report(acc, sweep, base))
