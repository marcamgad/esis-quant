import numpy as np

from visualization.sentiment_charts import (
    _ols_with_ci, fit_temperature, reliability_curve, information_coefficient,
)
import pandas as pd


def test_ols_with_ci_recovers_known_slope():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    true_slope, true_intercept = 0.5, 1.0
    y = true_intercept + true_slope * x + rng.normal(0, 0.05, n)

    fit = _ols_with_ci(x, y)
    assert np.isclose(fit["slope"], true_slope, atol=0.05)
    assert np.isclose(fit["intercept"], true_intercept, atol=0.05)
    assert fit["p_value"] < 0.001  # strong, low-noise real relationship should be highly significant
    assert fit["se_slope"] > 0


def test_ols_with_ci_high_pvalue_for_pure_noise():
    rng = np.random.default_rng(1)
    n = 30
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)  # unrelated to x
    fit = _ols_with_ci(x, y)
    assert fit["p_value"] > 0.05  # should NOT look significant


def test_ols_with_ci_band_widens_relative_to_point_estimate():
    rng = np.random.default_rng(2)
    n = 20
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 1, n)
    fit = _ols_with_ci(x, y)
    assert (fit["ci_hi"] >= fit["y_pred"]).all()
    assert (fit["ci_lo"] <= fit["y_pred"]).all()


def test_fit_temperature_above_one_when_overconfident():
    # Construct badly overconfident logits (huge magnitude) with only
    # 60% accuracy -- temperature scaling should push T > 1 to soften them.
    rng = np.random.default_rng(3)
    n = 100
    labels = rng.integers(0, 3, n)
    logits = np.zeros((n, 3))
    for i in range(n):
        correct = rng.random() < 0.6
        predicted_class = labels[i] if correct else (labels[i] + 1) % 3
        logits[i, predicted_class] = 10.0  # extreme, overconfident logit
    T = fit_temperature(logits, labels, n_steps=100)
    assert T > 1.0


def test_reliability_curve_matches_manual_binning():
    confidences = np.array([0.05, 0.15, 0.55, 0.65, 0.95])
    correctness = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    bin_conf, bin_acc = reliability_curve(confidences, correctness, n_bins=10)
    # Each value falls in its own decile bin here -> 5 non-empty bins.
    assert len(bin_conf) == 5
    assert len(bin_acc) == 5
    np.testing.assert_allclose(sorted(bin_conf), sorted(confidences))


def test_information_coefficient_matches_manual_correlation():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    rng = np.random.default_rng(4)
    sentiment = pd.Series(rng.normal(0, 1, 50), index=idx)
    returns = 0.3 * sentiment + pd.Series(rng.normal(0, 1, 50), index=idx)
    ic = information_coefficient(sentiment, returns)
    expected = sentiment.corr(returns)
    assert np.isclose(ic, expected)


def test_information_coefficient_nan_when_too_few_samples():
    sentiment = pd.Series([0.1, 0.2], index=pd.date_range("2024-01-01", periods=2))
    returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
    assert np.isnan(information_coefficient(sentiment, returns))
