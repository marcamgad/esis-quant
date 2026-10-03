"""
Deflated Sharpe: known-answer checks on the normal CDF, the monotone behaviour
of the trial hurdle, and that a genuinely strong series survives while a noise
series does not.
"""
import numpy as np
import pytest

from analysis import deflated_sharpe as ds


def test_norm_cdf_known_values():
    assert ds.norm_cdf(0.0) == pytest.approx(0.5, abs=1e-6)
    assert ds.norm_cdf(1.96) == pytest.approx(0.975, abs=1e-4)
    assert ds.norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-4)
    assert ds.norm_cdf(-8.0) < 1e-6 and ds.norm_cdf(8.0) > 1 - 1e-6


def test_hurdle_rises_with_trial_count():
    """The more things you try, the higher the bar the winner must clear."""
    prev = -1.0
    for n in (1, 5, 25, 100, 1000):
        sr0 = ds.expected_max_sharpe(n, sigma_sr=0.05)
        assert sr0 >= prev
        prev = sr0
    assert ds.expected_max_sharpe(1, 0.05) == 0.0      # one trial -> no deflation


def test_psr_benchmark_monotonicity():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0008, 0.01, 1500)
    p_low = ds.probabilistic_sharpe(r, sr_benchmark=0.0)["psr"]
    p_high = ds.probabilistic_sharpe(r, sr_benchmark=0.05)["psr"]
    assert p_low > p_high                              # harder benchmark -> lower PSR


def test_strong_series_survives_noise_series_does_not():
    rng = np.random.default_rng(1)
    strong = rng.normal(0.0015, 0.008, 2000)           # very high Sharpe
    noise = rng.normal(0.0, 0.01, 2000)                # no edge
    d_strong = ds.deflated_sharpe(strong, n_trials=50, sigma_sr=0.03)
    d_noise = ds.deflated_sharpe(noise, n_trials=50, sigma_sr=0.03)
    assert d_strong["survives"]
    assert not d_noise["survives"]
    assert d_strong["dsr"] > d_noise["dsr"]


def test_more_trials_lowers_dsr():
    rng = np.random.default_rng(2)
    r = rng.normal(0.0006, 0.01, 1500)
    a = ds.deflated_sharpe(r, n_trials=1, sigma_sr=0.03)["dsr"]
    b = ds.deflated_sharpe(r, n_trials=200, sigma_sr=0.03)["dsr"]
    assert a > b                                        # same track record, more searching
