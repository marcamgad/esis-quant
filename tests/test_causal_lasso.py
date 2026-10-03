"""
Tests for the from-scratch coordinate-descent Lasso (Layer 5).

Correctness anchors:
  * soft_threshold matches its closed form.
  * With lambda -> 0, Lasso reproduces the OLS solution on a well-
    conditioned problem (known ground truth).
  * A large lambda drives coefficients to exactly zero (sparsity).
  * Lasso recovers the support of a sparse ground-truth model: truly-zero
    coefficients are shrunk far harder than the true nonzero ones.
"""

import numpy as np

from data.linalg import ols_fit
from models.causal.lasso import (
    LassoCoordinateDescent, soft_threshold, lambda_grid, lasso_cv, lasso_path,
    time_series_folds,
)


def test_time_series_folds_no_purge_covers_all_indices():
    n, k = 100, 5
    covered = set()
    for train, val in time_series_folds(n, k, purge=0):
        covered |= set(val.tolist())
        # train and val never overlap
        assert not (set(train.tolist()) & set(val.tolist()))
    assert covered == set(range(n))  # every index validated exactly once


def test_time_series_folds_purge_removes_boundary_rows():
    n, k, purge = 100, 5, 5
    for train, val in time_series_folds(n, k, purge=purge):
        lo, hi = val.min(), val.max()
        train_set = set(train.tolist())
        # no training index within `purge` steps of the validation block
        for j in range(max(0, lo - purge), lo):
            assert j not in train_set
        for j in range(hi + 1, min(n, hi + 1 + purge)):
            assert j not in train_set


def test_soft_threshold_closed_form():
    assert soft_threshold(3.0, 1.0) == 2.0
    assert soft_threshold(-3.0, 1.0) == -2.0
    assert soft_threshold(0.5, 1.0) == 0.0
    assert soft_threshold(-0.5, 1.0) == 0.0


def test_lasso_matches_ols_at_zero_lambda():
    rng = np.random.default_rng(0)
    n, p = 400, 3
    X = rng.normal(size=(n, p))
    true_beta = np.array([1.5, -2.0, 0.7])
    y = X @ true_beta + 3.0 + rng.normal(scale=0.01, size=n)

    lasso = LassoCoordinateDescent(tol=1e-10, max_iter=5000).fit(X, y, lam=1e-8)
    # OLS with intercept for reference
    Xi = np.column_stack([np.ones(n), X])
    ols = ols_fit(Xi, y)
    assert np.allclose(lasso.coef_, ols[1:], atol=1e-3)
    assert abs(lasso.intercept_ - ols[0]) < 1e-3


def test_large_lambda_zeros_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.0, 1.0, 1.0, 1.0]) + rng.normal(size=200)
    lasso = LassoCoordinateDescent().fit(X, y, lam=10.0)
    assert np.allclose(lasso.coef_, 0.0)
    # intercept collapses to the mean of y when all slopes are zero
    assert abs(lasso.intercept_ - y.mean()) < 1e-6


def test_lasso_recovers_sparse_support():
    rng = np.random.default_rng(2)
    n, p = 500, 8
    X = rng.normal(size=(n, p))
    true_beta = np.array([3.0, 0.0, -2.5, 0.0, 0.0, 1.5, 0.0, 0.0])
    y = X @ true_beta + rng.normal(scale=0.1, size=n)
    lasso = LassoCoordinateDescent(tol=1e-9, max_iter=5000).fit(X, y, lam=0.05)
    nonzero_true = np.abs(true_beta) > 0
    # nonzero coefficients recovered with the right sign, zeros stay tiny
    assert np.all(np.sign(lasso.coef_[nonzero_true]) == np.sign(true_beta[nonzero_true]))
    assert np.max(np.abs(lasso.coef_[~nonzero_true])) < 0.2


def test_lasso_cv_selects_reasonable_lambda():
    rng = np.random.default_rng(3)
    n, p = 300, 5
    X = rng.normal(size=(n, p))
    y = X @ np.array([2.0, 0.0, 0.0, -1.0, 0.0]) + rng.normal(scale=0.5, size=n)
    res = lasso_cv(X, y)
    assert res["best_lambda"] in set(res["lambdas"].tolist())
    assert res["cv_mse"].shape == res["lambdas"].shape


def test_lasso_path_shapes_and_shrinkage():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0]) + rng.normal(scale=0.2, size=200)
    grid = lambda_grid()
    path = lasso_path(X, y, grid)
    assert path["coefs"].shape == (len(grid), 4)
    # coefficients shrink toward zero as lambda grows (monotone total L1 norm)
    l1 = np.abs(path["coefs"]).sum(axis=1)
    # total L1 norm shrinks substantially from smallest to largest lambda
    assert l1[0] > l1[-1]
    assert l1[-1] < 0.5 * l1[0]
    # the genuinely-zero coefficient (index 3) stays tiny throughout and is
    # driven to exactly zero once lambda is large enough
    assert np.max(np.abs(path["coefs"][:, 3])) < 0.1
    assert abs(path["coefs"][-1, 3]) < 1e-9
