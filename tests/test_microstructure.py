import numpy as np
import pandas as pd

from data.microstructure import MicrostructureAnalyzer


def test_amihud_illiquidity_basic():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.01, 0.02, -0.02, 0.01], index=idx)
    value_traded = pd.Series([1000.0] * 10, index=idx)

    illiq = MicrostructureAnalyzer.amihud_illiquidity(returns, value_traded, window=5)

    # By day 5 (min_periods=2), value should be the rolling mean of |r|/vol.
    expected_day5 = returns.abs().iloc[:5].div(1000.0).mean()
    assert np.isclose(illiq.iloc[4], expected_day5, atol=1e-9)


def test_amihud_illiquidity_handles_zero_volume():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, 0.02, 0.01, 0.02, 0.01], index=idx)
    value_traded = pd.Series([1000.0, 0.0, 1000.0, 1000.0, 1000.0], index=idx)

    illiq = MicrostructureAnalyzer.amihud_illiquidity(returns, value_traded, window=5)
    assert np.isfinite(illiq.iloc[-1])  # zero-volume day excluded, not inf/nan-poisoned


def test_roll_spread_recovers_known_bounce():
    # Simulate bid-ask bounce: price alternates true_value +/- half_spread,
    # which produces Cov(dP_t, dP_{t-1}) = -half_spread^2 exactly.
    rng = np.random.default_rng(0)
    half_spread = 0.05
    n = 200
    bounce = rng.choice([-1, 1], size=n) * half_spread
    true_value = 100 + np.cumsum(rng.normal(0, 0.001, n))  # near-flat fundamental
    prices = pd.Series(true_value + bounce, index=pd.date_range("2024-01-01", periods=n, freq="D"))

    # normalize=False: validate the raw Roll formula magnitude (in price
    # units), which recovers 2*half_spread = 0.10 asymptotically.
    spread = MicrostructureAnalyzer.roll_spread_estimator(prices, window=60, normalize=False)
    estimated = spread.dropna().median()
    # allow generous tolerance since this is a finite noisy sample.
    assert 0.05 < estimated < 0.20

    # normalize=True (default) divides by the ~100 price level -> ~0.001.
    norm_spread = MicrostructureAnalyzer.roll_spread_estimator(prices, window=60)
    assert 0.0005 < norm_spread.dropna().median() < 0.002


def test_roll_spread_nan_when_no_negative_autocovariance():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    # Monotonically increasing prices -> positive-ish autocovariance -> NaN.
    prices = pd.Series(np.arange(30, dtype=float), index=idx)
    spread = MicrostructureAnalyzer.roll_spread_estimator(prices, window=10)
    assert spread.dropna().empty


def test_kyle_lambda_recovers_true_slope():
    rng = np.random.default_rng(1)
    n = 150
    true_lambda = 0.002
    signed_volume = pd.Series(rng.normal(0, 1000, n), index=pd.date_range("2024-01-01", periods=n, freq="D"))
    noise = rng.normal(0, 0.01, n)
    price_changes = pd.Series(true_lambda * signed_volume.to_numpy() + noise, index=signed_volume.index)

    lam = MicrostructureAnalyzer.kyle_lambda(price_changes, signed_volume, window=100)
    estimated = lam.dropna().iloc[-1]
    assert np.isclose(estimated, true_lambda, atol=0.0005)


def test_realized_variance_components_recovers_beta_and_decomposition():
    rng = np.random.default_rng(2)
    n = 150
    true_beta = 1.3
    market_returns = pd.Series(rng.normal(0, 0.01, n), index=pd.date_range("2024-01-01", periods=n, freq="D"))
    idio = rng.normal(0, 0.005, n)
    stock_returns = pd.Series(true_beta * market_returns.to_numpy() + idio, index=market_returns.index)

    result = MicrostructureAnalyzer.realized_variance_components(stock_returns, market_returns, window=100)
    last = result.dropna().iloc[-1]

    assert np.isclose(last["beta"], true_beta, atol=0.15)
    # Variance decomposition should roughly add back up to total variance.
    assert np.isclose(last["systematic_var"] + last["idiosyncratic_var"], last["total_var"], rtol=0.2)
