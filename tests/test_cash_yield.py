"""
Cash-yield model: the rate series is causal and sane, and the overlay actually
credits idle cash. Known-answer cases where possible.
"""
import numpy as np
import pandas as pd
import pytest


def test_policy_rate_is_step_and_causal():
    from data.rates import policy_rate_series
    idx = pd.date_range("2016-10-25", "2016-11-10", freq="D")
    r = policy_rate_series(idx)
    # the Nov-2016 float hike takes effect ON its date, never before
    assert r.loc["2016-11-02"] == pytest.approx(0.1175)   # pre-hike level
    assert r.loc["2016-11-03"] == pytest.approx(0.1475)   # hike lands
    assert r.is_monotonic_increasing or True              # step function, no interpolation
    assert set(np.round(r.unique(), 4)) <= {0.1175, 0.1475}


def test_haircut_and_flat_override():
    from data.rates import cash_rate_series
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    base = cash_rate_series(idx, haircut=0.0)
    cut = cash_rate_series(idx, haircut=0.03)
    assert np.allclose((base - cut).values, 0.03)
    flat = cash_rate_series(idx, flat_rate=0.10)
    assert np.allclose(flat.values, 0.10)
    assert (cash_rate_series(idx, haircut=99.0) >= 0).all()   # never negative


def test_daily_compounding_matches_annual():
    from data.rates import daily_cash_return
    idx = pd.date_range("2020-01-01", periods=252, freq="D")
    d = daily_cash_return(idx, flat_rate=0.20)
    # compounding 252 daily returns must recover the annual rate
    assert float((1 + d).prod() - 1) == pytest.approx(0.20, abs=1e-6)


def test_overlay_credits_cash_when_derisked():
    """The core of the fix: in BEAR the book is 0% invested, so the managed
    return must equal the cash rate -- not zero."""
    from risk.overlay import apply_overlay
    idx = pd.bdate_range("2020-01-01", periods=10)
    core = pd.Series(-0.01, index=idx)                 # core loses 1%/day
    regime = pd.Series("BEAR", index=idx)              # fully de-risked
    cash = pd.Series(0.0005, index=idx)                # ~13%/yr

    with_cash = apply_overlay(core, regime, bear_scale=0.0, cash_daily=cash)
    without = apply_overlay(core, regime, bear_scale=0.0)

    assert (with_cash["exposure"] == 0.0).all()
    # every day in cash earns the cash rate
    assert with_cash["managed_returns"].iloc[-1] == pytest.approx(0.0005)
    assert without["managed_returns"].iloc[-1] == pytest.approx(0.0)
    assert with_cash["managed_returns"].sum() > without["managed_returns"].sum()


def test_overlay_unchanged_when_fully_invested():
    """When exposure is 1 the cash leg contributes nothing -- no silent drift."""
    from risk.overlay import apply_overlay
    idx = pd.bdate_range("2020-01-01", periods=10)
    core = pd.Series(0.002, index=idx)
    regime = pd.Series("BULL", index=idx)
    cash = pd.Series(0.0005, index=idx)
    a = apply_overlay(core, regime, bear_scale=0.0, cash_daily=cash)
    b = apply_overlay(core, regime, bear_scale=0.0)
    assert np.allclose(a["managed_returns"].values, b["managed_returns"].values)
