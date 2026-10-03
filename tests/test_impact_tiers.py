"""
Phase 3 liquidity tiers. The fitted curve was NOT adopted (see the module's
verdict) -- these tests verify the tiering and fitting machinery is correct so
that rejection is a judgement about the evidence, not a bug.
"""
import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from backtest import impact_tiers as it


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "i.db"))
    database.init_db()
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    rows = []
    # 8 tickers spanning 3 orders of magnitude in traded value
    for i, adv in enumerate([50e6, 40e6, 8e6, 6e6, 1e6, 8e5, 5e4, 4e4]):
        px = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
        for d, p in zip(idx, px):
            v = adv * float(np.exp(rng.normal(0, 0.3)))
            rows.append((f"T{i}", str(d.date()), float(p), float(p), 0, v))
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO daily_ohlcv (ticker,date,close,adjusted_close,"
                         "volume,value_traded_egp) VALUES (?,?,?,?,?,?)", rows)
    return config.DB_PATH


def test_tiers_order_by_liquidity(temp_db):
    tiers = it.assign_tiers(n_tiers=4)
    assert len(tiers) == 8
    assert tiers["T0"] == 0          # most liquid -> tier 0
    assert tiers["T7"] == 3          # least liquid -> last tier
    assert tiers["T0"] < tiers["T7"]


def test_every_ticker_gets_exactly_one_tier(temp_db):
    tiers = it.assign_tiers(n_tiers=4)
    assert tiers.notna().all()
    assert set(tiers.unique()) <= {0, 1, 2, 3}


def test_fit_recovers_a_planted_power_law():
    """Known-answer: build |r| = k * p^alpha exactly and check the log-log OLS
    recovers k and alpha -- proves the estimator itself is right."""
    rng = np.random.default_rng(1)
    p = np.exp(rng.normal(0, 0.5, 4000))
    k_true, a_true = 0.02, 0.45
    y = np.log(k_true * p ** a_true)
    X = np.column_stack([np.ones(len(p)), np.log(p)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert float(np.exp(coef[0])) == pytest.approx(k_true, rel=1e-6)
    assert float(coef[1]) == pytest.approx(a_true, rel=1e-6)


def test_thin_tier_falls_back_to_sqrt_law(temp_db):
    fits = pd.DataFrame([{"tier": 0, "n_obs": 5, "k": np.nan, "alpha": it.DEFAULT_ALPHA,
                          "fitted": False}])
    got = it.tiered_impact(0.04, tier=0, fits=fits)
    assert got == pytest.approx(config.COST_IMPACT_COEF * np.sqrt(0.04))


def test_impact_is_monotone_in_participation(temp_db):
    fits = pd.DataFrame([{"tier": 0, "n_obs": 9999, "k": 0.02, "alpha": 0.4, "fitted": True}])
    a = it.tiered_impact(0.01, 0, fits)
    b = it.tiered_impact(0.10, 0, fits)
    c = it.tiered_impact(0.50, 0, fits)
    assert a < b < c                       # bigger order -> more impact
    assert (b / a) < (0.10 / 0.01)         # concave: sub-linear growth


def test_compare_to_flat_flags_direction(temp_db):
    fits = pd.DataFrame([{"tier": 0, "n_obs": 9999, "k": 0.002, "alpha": 0.3, "fitted": True}])
    cmp = it.compare_to_flat(fits, participations=(0.05,))
    assert (cmp["ratio"] < 1).all()        # this fit costs LESS than flat -> ratio < 1
