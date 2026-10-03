"""
Tests for the from-scratch 2SLS IV estimator (Layer 5).

DGP with a KNOWN causal effect theta, an endogenous treatment D (correlated
with the structural error through a confounder u), and valid instruments Z
(drive D, excluded from Y). Naive OLS of Y on D is biased by the endogeneity;
2SLS with strong instruments recovers theta.
"""

import numpy as np

from data.linalg import ols_fit
from models.causal.iv_estimator import (
    two_stage_least_squares, first_stage_f, sargan_test,
)


def _make_iv_data(n=5000, theta=2.0, seed=0, strong=True):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)                      # unobserved confounder
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    strength = 1.0 if strong else 0.02
    D = strength * (z1 + z2) + u + rng.normal(scale=0.5, size=n)  # endogenous via u
    Y = theta * D + 2.0 * u + rng.normal(scale=0.5, size=n)       # u also in Y -> endogeneity
    Z = np.column_stack([z1, z2])
    return Y, D, Z, u


def test_2sls_recovers_effect_under_endogeneity():
    theta = 2.0
    Y, D, Z, u = _make_iv_data(theta=theta, seed=1)
    iv = two_stage_least_squares(Y, D, Z)
    assert abs(iv["theta"] - theta) < 0.1


def test_naive_ols_biased_but_iv_not():
    theta = 2.0
    Y, D, Z, u = _make_iv_data(theta=theta, seed=2)
    # naive OLS of Y on [1, D]
    Xi = np.column_stack([np.ones(len(D)), D])
    ols_theta = ols_fit(Xi, Y)[1]
    iv = two_stage_least_squares(Y, D, Z)
    assert abs(ols_theta - theta) > 0.3          # OLS is biased upward by endogeneity
    assert abs(iv["theta"] - theta) < abs(ols_theta - theta)


def test_first_stage_f_strong_vs_weak():
    Y_s, D_s, Z_s, _ = _make_iv_data(seed=3, strong=True)
    Y_w, D_w, Z_w, _ = _make_iv_data(seed=3, strong=False)
    ones = np.ones((len(D_s), 1))
    f_strong = first_stage_f(D_s, Z_s, ones)
    f_weak = first_stage_f(D_w, Z_w, ones)
    assert f_strong["f_stat"] > 10 and not f_strong["weak"]
    assert f_weak["f_stat"] < 10 and f_weak["weak"]


def test_sargan_not_rejected_for_valid_instruments():
    theta = 2.0
    Y, D, Z, u = _make_iv_data(theta=theta, seed=4)
    iv = two_stage_least_squares(Y, D, Z)
    sargan = iv["sargan"]
    assert sargan["overidentified"] is True
    assert sargan["df"] == 1                      # 2 instruments - 1 endogenous
    # valid instruments: statistic well below chi2(1) 1% critical value (~6.63)
    assert sargan["statistic"] < 6.63


def test_sargan_undefined_when_just_identified():
    theta = 2.0
    Y, D, Z, u = _make_iv_data(theta=theta, seed=5)
    iv = two_stage_least_squares(Y, D, Z[:, [0]])  # one instrument, one endogenous
    assert iv["sargan"]["overidentified"] is False
    assert np.isnan(iv["sargan"]["statistic"])
