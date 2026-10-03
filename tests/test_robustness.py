"""
Offline tests for the cross-market robustness harness (no network). Covers the
only NEW logic in ops/robustness.py -- the market-agnostic regime labeller and
feature builder; the rest reuses already-tested production modules.
"""

import numpy as np
import pandas as pd

from ops.robustness import _label_states, market_features, walkforward_regimes


def test_label_states_economic_assignment():
    # rows = states, cols = [ret20, vol20, breadth]
    mu = np.array([
        [0.05, 0.15, 0.7],   # high return, low vol  -> BULL
        [-0.06, 0.20, 0.3],  # lowest return         -> BEAR
        [0.00, 0.45, 0.5],   # highest vol           -> HIGH_VOL
        [0.02, 0.18, 0.6],   # remainder             -> RECOVERY
    ])
    lab = _label_states(mu)
    assert lab[0] == "BULL"
    assert lab[1] == "BEAR"
    assert lab[2] == "HIGH_VOL"
    assert lab[3] == "RECOVERY"
    assert set(lab.values()) == {"BULL", "BEAR", "HIGH_VOL", "RECOVERY"}


def test_market_features_are_backward_looking():
    # feature at date t must not change when future rows are appended
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=300)
    rets = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 6)), index=dates, columns=list("ABCDEF"))
    full = market_features(rets)
    trunc = market_features(rets.iloc[:200])
    common = full.index.intersection(trunc.index)
    assert len(common) > 50
    assert np.allclose(full.loc[common].values, trunc.loc[common].values, atol=1e-12)


def test_walkforward_regimes_are_point_in_time():
    # regime label at date t must be invariant to future data (walk-forward +
    # filtered decode) -- the same causality guarantee the production path has.
    rng = np.random.default_rng(1)
    n = 1600
    dates = pd.bdate_range("2016-01-01", periods=n)
    # two long alternating regimes so the HMM has something real to find
    block = (np.arange(n) // 400) % 2
    drift = np.where(block == 0, 0.001, -0.001)
    rets = pd.DataFrame(rng.normal(0.0, 0.01, (n, 6)) + drift[:, None],
                        index=dates, columns=list("ABCDEF"))
    feat = market_features(rets)
    full = walkforward_regimes(feat, min_train=400)
    trunc = walkforward_regimes(feat.loc[:dates[1000]], min_train=400)
    common = full.index.intersection(trunc.index)
    common = common[common <= dates[900]]     # well inside both
    assert len(common) > 100
    assert (full.loc[common] == trunc.loc[common]).mean() == 1.0
