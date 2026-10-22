"""Agent 7 (research lab): no lookahead, trial accounting, pass rule."""

import numpy as np
import pandas as pd
import pytest

from research import lab


def _panel(n=400, k=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n, freq="C", weekmask="Sun Mon Tue Wed Thu")
    r = pd.DataFrame(rng.normal(0, 0.02, (n, k)), index=idx, columns=[f"T{i}" for i in range(k)])
    vt = pd.DataFrame(1e6, index=idx, columns=r.columns)
    uni = pd.DataFrame(True, index=idx, columns=r.columns)
    return r, vt, uni


@pytest.mark.parametrize("template", lab.TEMPLATES)
def test_factor_uses_no_future_data(template):
    """Changing returns AFTER date d must not change any score at or before d."""
    r, vt, _ = _panel()
    d = r.index[300]
    a = lab.factor(template, 60, r, vt).loc[:d]
    r2 = r.copy()
    r2.loc[r2.index > d] = 0.5
    b = lab.factor(template, 60, r2, vt).loc[:d]
    pd.testing.assert_frame_equal(a, b)


def test_planted_signal_is_found_and_noise_is_not():
    r, vt, uni = _panel()
    noise = lab.backtest(lab.factor("mom", 20, r, vt), r, uni, 20, book=1e12)
    # plant: tomorrow's return is visible to a cheating score -> must look excellent
    cheat = r.shift(-1)
    planted = lab.backtest(cheat, r, uni, 1, book=1e12)
    assert planted.mean() > 0.005 and abs(noise.mean()) < 0.002


def test_deflation_bar_rises_with_the_number_of_trials():
    rng = np.random.default_rng(1)
    ex = pd.Series(rng.normal(0.0004, 0.01, 2000),
                   index=pd.bdate_range("2016-01-01", periods=2000))
    few = lab.evaluate(ex, n_trials=2, trial_sharpes=[0.5, -0.3])
    many = lab.evaluate(ex, n_trials=500, trial_sharpes=list(rng.normal(0, 0.6, 500)))
    assert many["dsr"] < few["dsr"]


def test_pass_rule_needs_every_criterion():
    idx = pd.bdate_range("2016-01-01", periods=2000)
    good_then_bad = pd.Series(np.r_[np.full(1000, 0.002), np.full(1000, -0.0005)], index=idx)
    ev = lab.evaluate(good_then_bad + np.random.default_rng(2).normal(0, 1e-4, 2000), 1, None)
    assert ev["half2"] < 0 and ev["verdict"] == "NULL"
