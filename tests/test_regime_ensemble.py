"""
Ensemble regime signal on the LIVE path. The properties that matter: exposure is
continuous, it falls back safely when the ensemble is missing, and staleness is
detected so the daily job rebuilds it.
"""
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from models import regime_ensemble as re_


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "e.db"))
    database.init_db()
    return config.DB_PATH


def _seed(p_by_date, n_fits=5, computed_at=None):
    now = (computed_at or pd.Timestamp.now()).isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO regime_ensemble (date,p_bear,n_fits,computed_at) "
                         "VALUES (?,?,?,?)",
                         [(d, float(v), n_fits, now) for d, v in p_by_date.items()])


def test_exposure_is_continuous_not_binary(temp_db):
    """The whole point of ensembling: partial de-risking when fits disagree."""
    _seed({"2026-07-01": 0.0, "2026-07-02": 0.4, "2026-07-03": 1.0})
    assert re_.exposure_for("2026-07-01")["exposure"] == pytest.approx(1.0)
    assert re_.exposure_for("2026-07-02")["exposure"] == pytest.approx(0.6)   # not 0 or 1
    assert re_.exposure_for("2026-07-03")["exposure"] == pytest.approx(0.0)
    assert re_.exposure_for("2026-07-02")["source"] == "ensemble"


def test_uses_latest_signal_at_or_before_the_date(temp_db):
    """No look-ahead: a date after the last signal carries the last KNOWN value."""
    _seed({"2026-07-01": 0.3, "2026-07-10": 0.8})
    r = re_.exposure_for("2026-07-15")
    assert r["p_bear"] == pytest.approx(0.8)
    assert r["as_of_signal"] == "2026-07-10"
    early = re_.exposure_for("2026-07-05")
    assert early["p_bear"] == pytest.approx(0.3)      # never the future 0.8


def test_falls_back_to_single_fit_label_when_missing(temp_db):
    """A missing ensemble must degrade to the OLD behaviour, not fail to trade."""
    bear = re_.exposure_for("2026-07-01", fallback_regime="BEAR")
    bull = re_.exposure_for("2026-07-01", fallback_regime="BULL")
    assert bear["exposure"] == 0.0 and bull["exposure"] == 1.0
    assert bear["source"] == "single_fit_fallback"    # caller is told which path ran
    assert bear["p_bear"] is None


def test_staleness_triggers_rebuild(temp_db):
    assert re_.is_stale()                              # nothing stored yet
    _seed({"2026-07-01": 0.2}, n_fits=5)
    assert not re_.is_stale(max_age_days=30)           # fresh
    _seed({"2026-07-01": 0.2}, n_fits=5,
          computed_at=pd.Timestamp.now() - pd.Timedelta(days=90))
    assert re_.is_stale(max_age_days=30)               # too old
    _seed({"2026-07-01": 0.2}, n_fits=1)               # too few fits = still noise
    assert re_.is_stale(max_age_days=30)


def test_store_and_load_round_trip(temp_db):
    s = pd.Series({pd.Timestamp("2026-07-01"): 0.25, pd.Timestamp("2026-07-02"): 0.75})
    assert re_.store_ensemble(s, n_fits=5) == 2
    loaded = re_.load_ensemble()
    assert len(loaded) == 2
    assert loaded.iloc[1] == pytest.approx(0.75)
    assert re_.ensemble_status()["n_fits"] == 5
