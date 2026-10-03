"""
Tests for the from-scratch Double ML estimator (Layer 5).

Correctness anchors use a DGP with a KNOWN causal effect theta and
confounding by X that biases naive OLS:

    D = m(X) + v         (X drives treatment)
    Y = theta*D + g(X) + eps   (X also drives outcome)

Because X drives both D and Y, naive OLS of Y on D is biased away from
theta; DML partialling-out with cross-fitting should recover theta and its
naive baseline should be visibly more biased.
"""

import numpy as np

from models.causal.double_ml import double_ml, naive_ols_effect


def _make_confounded_data(n=4000, theta=1.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    # nonlinear confounding through X
    gX = 2.0 * X[:, 0] + 1.5 * np.sin(X[:, 1]) - X[:, 2] ** 2
    mX = 1.0 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
    D = mX + rng.normal(scale=1.0, size=n)
    Y = theta * D + gX + rng.normal(scale=1.0, size=n)
    return X, D, Y


def test_dml_recovers_known_effect():
    theta = 1.5
    X, D, Y = _make_confounded_data(theta=theta, seed=1)
    res = double_ml(Y, D, X)
    # DML estimate close to the true causal effect
    assert abs(res["theta"] - theta) < 0.15
    # and within a few standard errors
    assert abs(res["theta"] - theta) < 4 * res["se"]


def test_dml_less_biased_than_naive():
    theta = 1.5
    X, D, Y = _make_confounded_data(theta=theta, seed=2)
    res = double_ml(Y, D, X)
    naive = res["naive"]
    dml_bias = abs(res["theta"] - theta)
    naive_bias = abs(naive["theta"] - theta)
    # confounding makes the naive estimate materially more biased
    assert naive_bias > dml_bias
    assert naive_bias > 0.3


def test_dml_zero_effect_detected():
    X, D, Y = _make_confounded_data(theta=0.0, seed=3)
    res = double_ml(Y, D, X)
    assert abs(res["theta"]) < 0.12
    # not statistically distinguishable from zero
    assert abs(res["t_stat"]) < 3.0


def test_naive_ols_effect_matches_manual():
    rng = np.random.default_rng(4)
    D = rng.normal(size=500)
    Y = 2.0 * D + 1.0 + rng.normal(scale=0.1, size=500)
    res = naive_ols_effect(Y, D)
    assert abs(res["theta"] - 2.0) < 0.05


def test_dml_residuals_aligned_and_finite():
    X, D, Y = _make_confounded_data(seed=5)
    res = double_ml(Y, D, X)
    assert res["Y_tilde"].shape == Y.shape
    assert res["D_tilde"].shape == D.shape
    assert np.all(np.isfinite(res["Y_tilde"]))
    assert np.all(np.isfinite(res["D_tilde"]))
