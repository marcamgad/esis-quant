"""
Business case: the T-bill comparison must be like-for-like (both legs deflated)
and must not flatter the strategy.
"""
import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from analysis import business_case as bc


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "bc.db"))
    database.init_db()
    idx = pd.bdate_range("2020-01-01", periods=504)
    with get_connection() as conn:   # flat FX so real == nominal, isolating the logic
        conn.executemany("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, 20.0)",
                         [(str(d.date()),) for d in idx])
    return idx


def test_tbill_daily_compounds_to_the_annual_yield(temp_db):
    d = bc.tbill_daily(temp_db, flat_yield=0.20)
    one_year = (1 + d.iloc[:252]).prod() - 1
    assert one_year == pytest.approx(0.20, abs=1e-6)


def test_tbill_leg_is_deflated_too(temp_db, monkeypatch):
    """The trap this guards: comparing a REAL strategy return against a NOMINAL
    bill return would overstate the strategy by the whole devaluation."""
    idx = temp_db
    with get_connection() as conn:   # FX doubles -> EGP halves in USD terms
        fx = np.linspace(20.0, 40.0, len(idx))
        conn.executemany("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, ?)",
                         [(str(d.date()), float(v)) for d, v in zip(idx, fx)])
    strat = pd.Series(0.0, index=idx)
    c = bc.compare(strat, flat_yield=0.20)
    assert c["tbill_nominal"]["cagr"] > 0.15          # nominally the bill earns ~20%
    assert c["tbill_real"]["cagr"] < c["tbill_nominal"]["cagr"]   # real is much worse
    assert c["tbill_real"]["cagr"] < 0.05


def test_flat_fx_makes_real_equal_nominal(temp_db):
    rng = np.random.default_rng(0)
    strat = pd.Series(rng.normal(0.0005, 0.01, len(temp_db)), index=temp_db)
    c = bc.compare(strat, flat_yield=0.10)
    assert c["strategy_real"]["cagr"] == pytest.approx(c["strategy_nominal"]["cagr"], abs=0.02)


def test_beats_flag_tracks_real_not_nominal(temp_db):
    """A strategy losing in real terms must NOT be marked as beating bills just
    because its nominal number is larger."""
    idx = temp_db
    with get_connection() as conn:
        fx = np.linspace(20.0, 60.0, len(idx))
        conn.executemany("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, ?)",
                         [(str(d.date()), float(v)) for d, v in zip(idx, fx)])
    weak = pd.Series(0.00005, index=idx)              # tiny nominal gain
    c = bc.compare(weak, flat_yield=0.25)
    assert c["beats_tbill_real"] == (c["strategy_real"]["cagr"] > c["tbill_real"]["cagr"])


def test_yield_sweep_reports_sensitivity(temp_db):
    rng = np.random.default_rng(1)
    strat = pd.Series(rng.normal(0.0004, 0.01, len(temp_db)), index=temp_db)
    sw = bc.yield_sweep(strat, yields=(0.05, 0.30))
    assert len(sw) == 2
    assert sw["tbill_real_cagr"].iloc[1] > sw["tbill_real_cagr"].iloc[0]   # higher yield, better bill
