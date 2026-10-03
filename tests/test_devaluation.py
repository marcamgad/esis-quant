"""
Devaluation study: the machinery must be correct so the NULL result is
trustworthy. Known-answer cases on synthetic data.
"""
import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from analysis import devaluation as dv


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "dv.db"))
    database.init_db()
    idx = pd.bdate_range("2020-01-01", periods=400)
    fx = pd.Series(20.0, index=idx)
    fx.iloc[200:] = 30.0          # one +50% jump on day 200
    fx.iloc[350:] = 45.0          # a second jump on day 350
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, ?)",
                         [(str(d.date()), float(v)) for d, v in fx.items()])
    return idx


def test_episodes_collapse_cascades(temp_db):
    eps = dv.devaluation_episodes()
    assert len(eps) == 2                       # two jumps, not two hundred
    assert eps[0].date() == temp_db[200].date()


def test_forward_window_marks_days_before_only(temp_db):
    eps = dv.devaluation_episodes()
    fwd = dv.forward_window(temp_db, eps, horizon=30)
    i = temp_db.get_loc(eps[0])                     # index positions: the panel is
    assert fwd.iloc[i - 10]                         # business days, so calendar
    assert not fwd.iloc[i]                          # arithmetic can miss a date
    assert not fwd.iloc[i + 10]                     # after the event -> not flagged


def test_perfect_signal_scores_high_lift(temp_db):
    """A signal that fires exactly in the pre-event window must show a big lift
    -- proves the scorer can detect a real effect when one exists."""
    eps = dv.devaluation_episodes()
    perfect = dv.forward_window(temp_db, eps, horizon=dv.HORIZON_DAYS)
    e = dv.evaluate_signal(perfect, episodes=eps, index=temp_db)
    assert e["precision"] == pytest.approx(1.0)
    assert e["lift"] > 3.0
    assert e["recall"] == 2


def test_useless_signal_scores_about_one(temp_db):
    """An always-on signal has zero information: precision == base rate."""
    eps = dv.devaluation_episodes()
    always = pd.Series(True, index=temp_db)
    e = dv.evaluate_signal(always, episodes=eps, index=temp_db)
    assert e["lift"] == pytest.approx(1.0, abs=1e-9)


def test_permutation_rejects_a_random_signal(temp_db):
    """The honest guard: a coin-flip signal must NOT come out significant."""
    rng = np.random.default_rng(3)
    noise = pd.Series(rng.random(len(temp_db)) > 0.8, index=temp_db)
    out = dv.permutation_pvalue(noise, n_perm=400, n_trials=4)
    assert not out["significant"]
