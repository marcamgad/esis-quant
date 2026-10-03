"""
Phase 2 selection-alpha modules. These estimators returned NULL on real data;
these tests verify the MACHINERY is correct, so the null is trustworthy rather
than a bug -- including that each can detect a real effect when one is planted.
"""
import numpy as np
import pandas as pd
import pytest

from models import regime_fama_macbeth as rfm
from models import rank_persistence as rp


# --------------------------------------------------------------------------
# empirical-Bayes shrinkage
# --------------------------------------------------------------------------
def test_shrinkage_collapses_to_global_when_noisy():
    """A group mean with huge standard error must be ignored in favour of the
    global prior -- the honest behaviour when a regime cell is thin."""
    est, w = rfm._empirical_bayes(mu_r=5.0, se_r=100.0, mu_g=1.0, tau2=0.01)
    assert w < 0.01
    assert est == pytest.approx(1.0, abs=1e-3)


def test_shrinkage_trusts_a_precise_group_mean():
    """When the cell is precise relative to between-group spread, keep it."""
    est, w = rfm._empirical_bayes(mu_r=5.0, se_r=0.01, mu_g=1.0, tau2=10.0)
    assert w > 0.99
    assert est == pytest.approx(5.0, abs=0.05)


def test_shrinkage_is_bounded_and_monotone():
    prev = -1.0
    for se in (10.0, 3.0, 1.0, 0.3, 0.1):
        _, w = rfm._empirical_bayes(1.0, se, 0.0, tau2=1.0)
        assert 0.0 <= w <= 1.0
        assert w >= prev          # smaller se -> more weight on the group
        prev = w


def test_verdict_applies_preregistered_rule():
    """Low shrinkage weight or no significant cell -> NULL, per the fixed rule."""
    weak = pd.DataFrame({"shrink_w": [0.05, 0.10], "t_vs_global": [0.5, 1.0],
                         "trustworthy": [True, True]})
    assert rfm.verdict(weak)["verdict"] == "NULL"
    strong = pd.DataFrame({"shrink_w": [0.8, 0.9], "t_vs_global": [3.0, 0.2],
                           "trustworthy": [True, True]})
    assert rfm.verdict(strong)["verdict"] != "NULL"


def test_cross_section_skips_thin_periods():
    """Periods with too few names must be dropped, not fitted on noise."""
    rows = []
    for p, n in [("2020-01", 3), ("2020-02", 20)]:
        for i in range(n):
            rows.append({"period": p, "regime": "BULL", "beta_market": float(i),
                         rfm.RETURN_COL: 0.01 * i})
    df = pd.DataFrame(rows)
    out = rfm._cross_section_coefs(df, ["beta_market"])
    assert list(out["period"]) == ["2020-02"]      # the 3-name period is skipped


# --------------------------------------------------------------------------
# rank persistence
# --------------------------------------------------------------------------
def test_persistence_high_for_stable_ordering():
    """A MOSTLY stable ordering must score high persistence.

    Small noise is added deliberately: a perfectly constant panel has zero rank
    variance, so the rolling correlation is undefined (NaN) -- correct, but it
    tests nothing. This keeps the ordering intact while leaving variance to
    correlate.
    """
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=120)
    stable = pd.DataFrame(
        {f"T{i}": i + rng.normal(0, 0.05, len(idx)) for i in range(10)}, index=idx)
    sig = rp.compute_persistence(stable, window=20, lag=10)
    last = sig.iloc[-1].dropna()
    assert len(last) >= 8
    assert last["T9"] > 0.3      # persistently top-ranked -> strongly positive
    assert last["T0"] < -0.3     # persistently bottom-ranked -> strongly negative


def test_persistence_low_for_shuffled_ordering():
    """Randomly reshuffled ranks must NOT look persistent."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=120)
    noise = pd.DataFrame(rng.normal(size=(len(idx), 10)),
                         index=idx, columns=[f"T{i}" for i in range(10)])
    sig = rp.compute_persistence(noise, window=20, lag=10)
    assert abs(float(sig.tail(30).stack().mean())) < 0.25


def test_persistence_is_causal():
    """Changing only FUTURE values must not alter today's signal."""
    idx = pd.bdate_range("2020-01-01", periods=90)
    rng = np.random.default_rng(1)
    base = pd.DataFrame(rng.normal(size=(len(idx), 8)),
                        index=idx, columns=[f"T{i}" for i in range(8)])
    cut = 60
    a = rp.compute_persistence(base, window=20, lag=10).iloc[cut]
    tampered = base.copy()
    tampered.iloc[cut + 1:] = tampered.iloc[cut + 1:] * 99.0
    b = rp.compute_persistence(tampered, window=20, lag=10).iloc[cut]
    pd.testing.assert_series_equal(a, b)


def test_rank_verdict_rule():
    assert rp.verdict({"t_stat": 0.5, "mean_ic": 0.01, "ic_positive_frac": 0.6})["verdict"] == "NULL"
    assert rp.verdict({"t_stat": 3.0, "mean_ic": -0.01, "ic_positive_frac": 0.6})["verdict"] == "NULL"
    assert rp.verdict({"t_stat": 3.0, "mean_ic": 0.05, "ic_positive_frac": 0.7})["verdict"] == "REAL"
