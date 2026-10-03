import numpy as np
import pandas as pd

from data.linalg import ols_fit, rolling_ols_beta


def test_ols_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 500
    true_intercept, true_slope = 2.0, -3.5
    x = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.01, n)
    y = true_intercept + true_slope * x + noise
    X = np.column_stack([np.ones(n), x])

    beta = ols_fit(X, y)
    assert np.isclose(beta[0], true_intercept, atol=0.01)
    assert np.isclose(beta[1], true_slope, atol=0.01)


def test_rolling_ols_beta_recovers_known_slope():
    rng = np.random.default_rng(1)
    n = 300
    true_slope = 0.75
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x = pd.Series(rng.normal(0, 1, n), index=idx)
    y = pd.Series(true_slope * x.to_numpy() + rng.normal(0, 0.05, n), index=idx)

    beta = rolling_ols_beta(y, x, window=100)
    estimated = beta.dropna().iloc[-1]
    assert np.isclose(estimated, true_slope, atol=0.05)


def test_rolling_ols_beta_nan_when_regressor_constant():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    x = pd.Series(np.full(50, 5.0), index=idx)  # constant -> singular X'X, must not fabricate a beta
    y = pd.Series(np.arange(50, dtype=float), index=idx)

    beta = rolling_ols_beta(y, x, window=20)
    assert beta.dropna().empty
