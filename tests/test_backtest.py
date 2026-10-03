"""
Tests for the backtest engine + analytics (from scratch). Correctness anchors:
  * analytics match hand-computed values on known series.
  * a PERFECT-FORESIGHT signal makes money and beats equal-weight; an
    ANTI-signal loses -- proving the engine actually acts on the signal.
  * transaction costs make net < gross.
  * NO LOOK-AHEAD: a signal known only at T cannot use T's own return.
"""

import numpy as np
import pandas as pd

from backtest import analytics as an
from backtest.engine import BacktestEngine, equal_weight_benchmark


def test_analytics_known_values():
    # constant +1% daily for 252 days
    r = pd.Series([0.01] * 252)
    assert abs(an.annualized_return(r) - (1.01 ** 252 - 1)) < 1e-6
    assert an.max_drawdown(r) == 0.0            # never declines
    assert np.isnan(an.sharpe_ratio(r))        # zero volatility -> Sharpe undefined
    assert an.hit_rate(r) == 1.0
    # a positive-drift noisy series has a positive, finite Sharpe
    rng = np.random.default_rng(0)
    r_noisy = pd.Series(rng.normal(0.001, 0.01, 252))
    assert an.sharpe_ratio(r_noisy) > 0
    # a series with one big loss has negative drawdown
    r2 = pd.Series([0.01, 0.01, -0.5, 0.01])
    assert an.max_drawdown(r2) < -0.4


def _synthetic_market(n_days=300, n_tickers=12, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = pd.DataFrame(rng.normal(0.0003, 0.02, size=(n_days, n_tickers)),
                        index=dates, columns=[f"T{i}" for i in range(n_tickers)])
    return rets


def _forward_signal(returns_wide, rebalance_days, sign=1.0):
    """Perfect-foresight (sign=+1) or anti (sign=-1) signal: score at T is the
    realized return over the NEXT rebalance window. Long form (ticker,date,score).
    """
    rows = []
    dates = returns_wide.index
    rebal = list(dates[::rebalance_days])
    for i, T in enumerate(rebal):
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1]
        window = dates[(dates > T) & (dates <= end)]
        if len(window) == 0:
            continue
        fwd = (1 + returns_wide.loc[window]).prod() - 1  # per-ticker forward return
        for tk, v in fwd.items():
            rows.append({"ticker": tk, "date": T, "score": sign * v})
    return pd.DataFrame(rows)


def test_perfect_foresight_beats_equal_weight_and_anti_loses():
    rets = _synthetic_market()
    # use_fee_schedule=False is REQUIRED for a genuinely cost-free run: since
    # the 2026-08-05 audit the real fee schedule is charged by default, so
    # cost_per_turnover=0.0 alone no longer means zero cost. This test is
    # about signal mechanics, so it opts out explicitly.
    eng = BacktestEngine(rebalance_days=10, top_fraction=0.3, min_names=3,
                         cost_per_turnover=0.0, tradable_only=False,
                         use_fee_schedule=False)

    good = eng.run(_forward_signal(rets, 10, sign=+1.0), rets)
    bad = eng.run(_forward_signal(rets, 10, sign=-1.0), rets)
    bench = equal_weight_benchmark(rets, start=good["net_returns"].index[0])

    good_total = float((1 + good["net_returns"]).prod())
    bad_total = float((1 + bad["net_returns"]).prod())
    bench_total = float((1 + bench.reindex(good["net_returns"].index).fillna(0)).prod())

    assert good_total > bench_total            # foresight beats owning everything
    assert good_total > bad_total              # foresight beats its opposite
    assert an.sharpe_ratio(good["net_returns"]) > an.sharpe_ratio(bad["net_returns"])


def test_costs_reduce_net_below_gross():
    rets = _synthetic_market(seed=1)
    sig = _forward_signal(rets, 10, sign=1.0)
    eng = BacktestEngine(rebalance_days=10, top_fraction=0.3, min_names=3, cost_per_turnover=0.00275, tradable_only=False)
    res = eng.run(sig, rets)
    gross = float((1 + res["gross_returns"]).prod())
    net = float((1 + res["net_returns"]).prod())
    assert net < gross                          # turnover costs bite
    assert all(t >= 0 for t in res["turnover"].values())


def test_no_lookahead_signal_at_T_uses_only_future_returns():
    # Signal is nonzero on exactly ONE ticker at ONE rebalance date; the
    # engine must earn that ticker's return only AFTER T, never T's own day.
    rets = _synthetic_market(n_days=60, n_tickers=6, seed=2)
    dates = rets.index
    T = dates[9]
    sig = pd.DataFrame([{"ticker": "T0", "date": T, "score": 1.0}])
    # min_names=1 so the single-name book is allowed
    eng = BacktestEngine(rebalance_days=10, top_fraction=1.0, min_names=1, cost_per_turnover=0.0, tradable_only=False)
    res = eng.run(sig, rets)
    # the first earned day is strictly after T
    earned = res["net_returns"][res["net_returns"] != 0]
    assert earned.index.min() > T
