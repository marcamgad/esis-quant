"""
Multi-timeframe factor screen -- the standard institutional stock-selection stack.

WHAT THIS IS. The screen a discretionary or systematic equity desk actually
runs: cross-sectional momentum over several horizons, volatility-adjusted so a
wild name does not out-rank a steady one on raw return alone, trend structure
against moving averages, volume confirmation, and relative strength against the
market. Every component here is standard and none of it is exotic.

WHY IT IS BUILT WITH ITS OWN BACKTEST ATTACHED. A ranking is not a finding. This
module deliberately makes it impossible to get the pretty table (`rank_universe`)
without the ugly one (`walk_forward_test`) being available on the same import,
because the entire history of this project is that the pretty table looks
convincing every single time:

  * measured selection alpha  -2.46%  (t = -0.61)
  * regime-conditional Fama-MacBeth: max |t| 1.93, shrink weight 0.12
  * rank persistence:                t = 1.01
  * BEAR overlay: worked EGX and US, FAILED and hurt in India

Each of those started as a ranking that looked exactly as sensible as whatever
this file prints today. The scoring below is honest work; the question of
whether the score predicts anything is a separate empirical question that
`walk_forward_test` answers, and the answer is allowed to be no.

HORIZONS. 1w / 1m / 3m / 6m / 1y, per the request to see the market on several
clocks at once. Note that these are not independent views -- a 1-year window
contains the 6-month one -- so agreement across horizons is much weaker evidence
than it looks. `momentum_agreement` reports it because desks look at it, and the
docstring says what it is worth.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.multi_timeframe")

TRADING_DAYS = 252

# Sessions per horizon. EGX trades Sun-Thu, ~21 sessions/month.
HORIZONS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}

# Standard academic momentum skips the most recent month: 1-month reversal is one
# of the few robust cross-sectional effects, and letting it into a 12-month
# momentum score mixes two signals with opposite signs.
MOM_SKIP_RECENT = 21


def _safe_ret(s: pd.Series, n: int) -> float:
    s = s.dropna().tail(n)
    if len(s) < max(3, n // 3):
        return np.nan
    return float((1.0 + s).prod() - 1.0)


def horizon_returns(rets: pd.DataFrame, tickers=None) -> pd.DataFrame:
    """Trailing return on every horizon, per name."""
    cols = tickers if tickers is not None else list(rets.columns)
    out = {}
    for t in cols:
        if t not in rets.columns:
            continue
        s = rets[t]
        out[t] = {k: _safe_ret(s, n) for k, n in HORIZONS.items()}
    return pd.DataFrame(out).T


def factor_table(rets: pd.DataFrame, prices: pd.Series, volumes: pd.DataFrame = None,
                 tickers=None, market: pd.Series = None) -> pd.DataFrame:
    """The screen itself: one row per name, one column per factor.

    Factors, and why each is here:
      mom_12_1   12-month return skipping the last month -- the classic
                 cross-sectional momentum factor.
      mom_6_1    the same idea on a 6-month clock, which reacts sooner.
      sharpe_6m  6-month return divided by realised volatility. Raw momentum
                 systematically favours the most volatile names; dividing it out
                 is what stops the screen from simply ranking by risk.
      rev_1m     last month's return. Enters the composite NEGATIVELY (short-term
                 reversal), which is why it is kept separate from momentum.
      trend      price relative to its 200-session mean -- above or below trend.
      ma_stack   fraction of (50 > 100 > 200) moving-average orderings that hold;
                 1.0 is a fully aligned uptrend.
      vol_ann    annualised volatility. Reported, not scored: it is the risk you
                 are taking, and the request was explicitly for MORE of it.
      rs_6m      return minus the equal-weight market's return over 6 months.
      vol_conf   recent volume vs its own 3-month norm -- is the move being
                 confirmed by participation, or is it a drift on no trade?
    """
    cols = tickers if tickers is not None else list(rets.columns)
    cols = [t for t in cols if t in rets.columns]
    rows = {}
    mkt = market if market is not None else rets[cols].mean(axis=1)

    for t in cols:
        s = rets[t].dropna()
        if len(s) < 150:
            continue
        skip = s.iloc[:-MOM_SKIP_RECENT] if len(s) > MOM_SKIP_RECENT else s
        vol6 = s.tail(126).std() * np.sqrt(TRADING_DAYS)
        r6 = _safe_ret(s, 126)

        px = (1.0 + s).cumprod()
        ma50, ma100, ma200 = px.tail(50).mean(), px.tail(100).mean(), px.tail(200).mean()
        last = float(px.iloc[-1])
        stack = np.mean([ma50 > ma100, ma100 > ma200, last > ma50])

        vc = np.nan
        if volumes is not None and t in volumes.columns:
            v = volumes[t].dropna()
            if len(v) > 63 and v.tail(63).median() > 0:
                vc = float(v.tail(10).median() / v.tail(63).median())

        rows[t] = {
            "mom_12_1": _safe_ret(skip, 231),
            "mom_6_1": _safe_ret(skip, 105),
            "sharpe_6m": (r6 / vol6) if (vol6 and np.isfinite(vol6) and vol6 > 0) else np.nan,
            "rev_1m": _safe_ret(s, 21),
            "trend": (last / ma200 - 1.0) if ma200 > 0 else np.nan,
            "ma_stack": float(stack),
            "vol_ann": float(s.tail(126).std() * np.sqrt(TRADING_DAYS)),
            "rs_6m": (r6 - _safe_ret(mkt, 126)) if np.isfinite(r6) else np.nan,
            "vol_conf": vc,
        }
    return pd.DataFrame(rows).T


def _z(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorised. Clipping matters more than it looks:
    EGX has names up +471% in a year, and an unclipped z-score lets one such
    outlier dominate a composite built from five factors."""
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-clip, clip)


# Composite weights. Signs are the point: reversal enters negative.
COMPOSITE = {"mom_12_1": 0.30, "mom_6_1": 0.20, "sharpe_6m": 0.25,
             "trend": 0.10, "ma_stack": 0.05, "rs_6m": 0.20, "rev_1m": -0.10}


def score(factors: pd.DataFrame) -> pd.DataFrame:
    """Composite score. Every input z-scored cross-sectionally first, so factors
    measured in different units (a return, a ratio, a fraction) are comparable."""
    out = factors.copy()
    zs = pd.DataFrame({k: _z(out[k]) for k in COMPOSITE if k in out.columns}, index=out.index)
    out["score"] = sum(zs[k] * w for k, w in COMPOSITE.items() if k in zs.columns)
    # How many of the five horizons are positive. Reported because desks look at
    # it -- but the horizons NEST (1y contains 6m contains 3m), so agreement is
    # far weaker evidence than five independent confirmations would be.
    return out.sort_values("score", ascending=False)


def momentum_agreement(hr: pd.DataFrame) -> pd.Series:
    return (hr > 0).sum(axis=1)


def walk_forward_test(rets: pd.DataFrame, prices_hist: pd.DataFrame = None,
                      top_n: int = 5, rebalance: int = 21, start_after: int = 400,
                      n_perm: int = 200, seed: int = 0) -> dict:
    """Does the composite score actually predict the next period's return?

    The test: at each rebalance date, rank on data available UP TO that date,
    buy the top N equally, hold one rebalance period, record the forward return.
    Compare against the equal-weight universe over the identical dates.

    Then permute. Shuffling the scores across names, holding the dates and the
    return panel fixed, destroys any real link between rank and outcome while
    preserving the market's own path. If the true spread does not stand out
    against the permuted distribution, the strategy's apparent edge is what a
    coin produces on this data -- which is the outcome every previous selection
    test in this project has produced.
    """
    rng = np.random.default_rng(seed)
    dates = rets.index
    if len(dates) < start_after + rebalance * 3:
        return {"ok": False, "reason": "not enough history"}

    picks_r, bench_r = [], []
    rebal_dates = list(range(start_after, len(dates) - rebalance, rebalance))
    stored = []
    for i in rebal_dates:
        hist = rets.iloc[:i]
        fwd = rets.iloc[i:i + rebalance]
        live = [c for c in rets.columns
                if hist[c].tail(252).notna().sum() > 150 and fwd[c].notna().sum() > rebalance // 2]
        if len(live) < 20:
            continue
        f = factor_table(hist, prices=None, tickers=live)
        if f.empty:
            continue
        sc = score(f)["score"].dropna()
        if len(sc) < 20:
            continue
        top = list(sc.head(top_n).index)
        fr = (1.0 + fwd[live]).prod() - 1.0
        picks_r.append(float(fr[top].mean()))
        bench_r.append(float(fr.mean()))
        stored.append((sc, fr))

    if len(picks_r) < 8:
        return {"ok": False, "reason": f"only {len(picks_r)} rebalances"}

    p, b = np.array(picks_r), np.array(bench_r)
    spread = p - b
    n = len(spread)
    t_stat = float(spread.mean() / (spread.std(ddof=1) / np.sqrt(n))) if spread.std(ddof=1) > 0 else 0.0

    # permutation null: same dates, same returns, ranks shuffled
    null = []
    for _ in range(n_perm):
        d = []
        for sc, fr in stored:
            names = list(sc.index)
            pick = rng.choice(names, size=min(top_n, len(names)), replace=False)
            d.append(float(fr[list(pick)].mean()) - float(fr[list(sc.index)].mean()))
        null.append(np.mean(d))
    null = np.array(null)
    p_val = float((np.abs(null) >= abs(spread.mean())).mean())

    per_year = TRADING_DAYS / rebalance
    return {"ok": True, "n_rebalances": n, "top_n": top_n,
            "picks_mean": float(p.mean()), "bench_mean": float(b.mean()),
            "spread_mean": float(spread.mean()), "spread_ann": float(spread.mean() * per_year),
            "t_stat": t_stat, "p_permutation": p_val,
            "hit_rate": float((spread > 0).mean()),
            "null_mean": float(null.mean()), "null_sd": float(null.std())}
