"""
Overlay tuning. The critical test is the OVERLAP CORRECTION -- without it,
rolling forward-return windows manufacture significance by double-counting.
"""
import numpy as np
import pandas as pd
import pytest

from analysis import overlay_tuning as ot


def test_overlap_correction_deflates_t_by_sqrt_horizon():
    """A 63-day overlapping window shares 62 of 63 days with its neighbour, so
    the naive t-stat must be divided by ~sqrt(63) before it means anything."""
    idx = pd.bdate_range("2020-01-01", periods=1500)
    rng = np.random.default_rng(0)
    core = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    p = pd.Series(np.where(np.arange(len(idx)) % 200 < 60, 0.9, 0.1), index=idx)
    acc = ot.timing_accuracy(core, p, horizons=(63,))
    r = acc.iloc[0]
    assert r["t_stat"] == pytest.approx(r["t_raw"] / np.sqrt(63), rel=1e-9)
    assert abs(r["t_stat"]) < abs(r["t_raw"])
    assert r["n_independent"] < len(idx) / 60      # not the raw observation count


def test_noise_signal_is_not_called_significant():
    """A regime label unrelated to returns must not clear the adjusted bar."""
    idx = pd.bdate_range("2020-01-01", periods=1200)
    rng = np.random.default_rng(1)
    core = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    p = pd.Series(rng.random(len(idx)), index=idx)          # random labels
    acc = ot.timing_accuracy(core, p, horizons=(5, 21))
    assert not acc["significant"].any()


def test_planted_effect_is_detected_directionally():
    """When bear periods really are worse, the diff must be negative."""
    idx = pd.bdate_range("2020-01-01", periods=1500)
    rng = np.random.default_rng(2)
    bear = (np.arange(len(idx)) % 250) < 80
    core = pd.Series(np.where(bear, rng.normal(-0.003, 0.01, len(idx)),
                              rng.normal(0.001, 0.01, len(idx))), index=idx)
    p = pd.Series(np.where(bear, 0.9, 0.1), index=idx)
    acc = ot.timing_accuracy(core, p, horizons=(21,))
    assert acc.iloc[0]["bear_worse"]
    assert acc.iloc[0]["diff"] < 0


def test_scale_one_is_a_no_op():
    """bear_scale = 1 means never de-risk, so it must reproduce the core."""
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    core = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    p = pd.Series(0.8, index=idx)
    sw = ot.scale_sweep(core, p, scales=(1.0,), continuous=True)
    assert sw.iloc[0]["dd_saved_pp"] == pytest.approx(0.0, abs=1e-9)
    assert sw.iloc[0]["cagr_given_pp"] == pytest.approx(0.0, abs=1e-9)


def test_derisking_saves_more_drawdown_the_harder_it_is_applied():
    idx = pd.bdate_range("2020-01-01", periods=1200)
    rng = np.random.default_rng(4)
    bear = (np.arange(len(idx)) % 250) < 80
    core = pd.Series(np.where(bear, rng.normal(-0.004, 0.015, len(idx)),
                              rng.normal(0.0012, 0.01, len(idx))), index=idx)
    p = pd.Series(np.where(bear, 0.9, 0.05), index=idx)
    sw = ot.scale_sweep(core, p, scales=(0.0, 0.5, 1.0), continuous=True).set_index("bear_scale")
    assert sw.loc[0.0, "dd_saved_pp"] > sw.loc[0.5, "dd_saved_pp"] >= 0


def test_return_effect_depends_on_whether_bear_periods_were_actually_bad():
    """When bear periods genuinely lose money, de-risking ADDS return -- so
    'cagr_given' goes NEGATIVE. When bear periods are merely more volatile but
    still positive, de-risking gives return up. The sign must follow the data,
    not an assumption that de-risking always costs something."""
    idx = pd.bdate_range("2020-01-01", periods=1200)
    rng = np.random.default_rng(5)
    bear = (np.arange(len(idx)) % 250) < 80
    p = pd.Series(np.where(bear, 0.9, 0.05), index=idx)

    truly_bad = pd.Series(np.where(bear, rng.normal(-0.004, 0.015, len(idx)),
                                   rng.normal(0.0012, 0.01, len(idx))), index=idx)
    sw_bad = ot.scale_sweep(truly_bad, p, scales=(0.0,), continuous=True)
    assert sw_bad.iloc[0]["cagr_given_pp"] < 0        # de-risking GAINED return

    just_volatile = pd.Series(np.where(bear, rng.normal(0.0015, 0.03, len(idx)),
                                       rng.normal(0.0010, 0.008, len(idx))), index=idx)
    sw_vol = ot.scale_sweep(just_volatile, p, scales=(0.0,), continuous=True)
    assert sw_vol.iloc[0]["cagr_given_pp"] > 0        # de-risking COST return
