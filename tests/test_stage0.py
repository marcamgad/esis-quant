"""
Stage 0 (2026-10-03) regression tests. Each one pins a failure the audit found
in the live system:

  * the regime refit took 3-8 minutes daily and the job was killed inside it
    -> frozen vintages must be reused, and must refit when training data changes
  * the exposure signal froze for two months and nothing stopped the ticket
    -> the quality gate must refuse when regime artifacts lag the last session
  * 14 BEAR days at exposure 1.0 counted toward the go-live gate
  * a deposit read as profit in the book guard
  * T+2 settled a Thursday sale on Saturday
  * one null-close bar collapsed the tradable universe

All run against a temporary database, never esis_quant.db.
"""

import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "s0.db"))
    database.init_db()
    return config.DB_PATH


def test_tests_never_point_at_the_real_database():
    import os
    from tests.conftest import REAL_DB
    assert os.path.abspath(config.DB_PATH) != os.path.abspath(REAL_DB)


# --- frozen regime vintages -------------------------------------------------

def _two_regime_data(n=400, seed=0):
    """Four blocks with distinct means/scales, as wide as the live feature set
    (label_states needs 4 states and the real column order)."""
    from models.regime_features import OBSERVATION_COLUMNS
    rng = np.random.default_rng(seed)
    D = len(OBSERVATION_COLUMNS)
    blocks = [rng.normal(m, s, (n // 4, D)) for m, s in ((0.8, 0.3), (-0.9, 1.2), (0.0, 2.0), (0.3, 0.5))]
    return np.vstack(blocks)


def test_vintage_is_fit_once_then_reused(temp_db, monkeypatch):
    from models import regime_pipeline as rp
    X = _two_regime_data()
    calls = []
    real = rp._fit_and_label
    monkeypatch.setattr(rp, "_fit_and_label",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    h1, lab1 = rp._cached_fit("2025-01-01", X, 4, 1, 7, 50, "f")
    h2, lab2 = rp._cached_fit("2025-01-01", X, 4, 1, 7, 50, "f")
    assert len(calls) == 1                                   # second call read the cache
    np.testing.assert_allclose(h1.mu_, h2.mu_)
    np.testing.assert_allclose(h1.filtered_proba(X), h2.filtered_proba(X))
    assert lab1 == lab2


def test_vintage_refits_when_its_training_data_changes(temp_db, monkeypatch):
    from models import regime_pipeline as rp
    X = _two_regime_data()
    calls = []
    real = rp._fit_and_label
    monkeypatch.setattr(rp, "_fit_and_label",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    rp._cached_fit("2025-01-01", X, 4, 1, 7, 50, "f")
    X2 = X.copy(); X2[10, 0] += 1.0                          # a revised past observation
    rp._cached_fit("2025-01-01", X2, 4, 1, 7, 50, "f")
    rp._cached_fit("2025-01-01", X, 4, 1, 8, 50, "f")        # a different seed
    assert len(calls) == 3


# --- signal freshness gate ---------------------------------------------------

def _seed_prices(days):
    rows = [(f"T{i}", str(d.date()), 10.0, 10.0, 1000.0, 1e6) for d in days for i in range(12)]
    with get_connection() as conn:
        conn.executemany("INSERT INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, "
                         "value_traded_egp) VALUES (?,?,?,?,?,?)", rows)


def test_stale_regime_ensemble_is_fatal(temp_db):
    from data import quality
    days = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=40)
    _seed_prices(days)
    with get_connection() as conn:
        conn.execute("INSERT INTO regime_history (date, regime) VALUES (?, 'BULL')",
                     (str(days[-1].date()),))
        conn.execute("INSERT INTO regime_ensemble (date, p_bear, n_fits, computed_at) "
                     "VALUES (?, 0.0, 5, ?)", (str(days[-20].date()), str(days[-20].date())))
    rep = quality.data_quality_report()
    assert rep["signal_lag"]["regime_ensemble"]["sessions_behind"] == 19
    assert any("regime_ensemble" in f for f in rep["fatal"])
    assert not any("regime_history" in f for f in rep["fatal"])


# --- go-live gate --------------------------------------------------------------

def test_bear_days_count_only_when_the_overlay_acted(temp_db, monkeypatch):
    from ops import paper_trading as pt
    days = pd.bdate_range("2026-08-02", periods=30)
    rows = []
    for i, d in enumerate(days):
        regime = "BEAR" if i < 20 else "BULL"
        exposure = 1.0 if i < 14 else (0.2 if i < 20 else 1.0)   # 14 inactive, 6 acted
        rows.append((str(d.date()), 1.0, 0.0, exposure, regime, "r", 5))
    with get_connection() as conn:
        conn.executemany("INSERT INTO paper_trading_nav (date, nav, daily_return, exposure, "
                         "regime, run_id, n_positions) VALUES (?,?,?,?,?,?,?)", rows)
    rec = pt.paper_track_record()
    assert rec["bear_days_observed"] == 6
    assert rec["bear_days_overlay_inactive"] == 14


# --- cash flows ----------------------------------------------------------------

def test_deposit_is_not_profit_in_the_book_guard():
    from risk.kill_switch import book_twr
    idx = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"])
    nav = pd.Series([10_000.0, 10_000.0, 13_000.0, 11_700.0], index=idx)
    flows = pd.Series([3_000.0], index=pd.to_datetime(["2026-09-03"]))
    twr = book_twr(nav, flows)
    assert twr.iloc[2] == pytest.approx(1.0)                 # +3,000 was money in, not a gain
    assert twr.iloc[3] == pytest.approx(0.9)                 # the later -10% is real
    no_flows = book_twr(nav, pd.Series(dtype=float))
    assert no_flows.iloc[2] == pytest.approx(1.3)            # what the old guard saw


# --- settlement ------------------------------------------------------------------

def test_settlement_fallback_skips_friday_and_saturday(temp_db):
    from ops.portfolio_tracker import _settlement_date
    assert _settlement_date("2030-01-03", days=2) == "2030-01-07"   # Thu -> Mon
    assert _settlement_date("2030-01-06", days=2) == "2030-01-08"   # Sun -> Tue


# --- stockanalysis dated history backfill -----------------------------------------

HIST_PAGE = ('data:[{a:3.02,c:3.02,h:3.02,l:2.54,o:2.74,t:"2026-10-01",v:147612435,ch:10.22},'
             '{a:2.74,c:2.74,h:2.76,l:1.99,o:2.48,t:"2026-09-30",v:195447348,ch:10.48},'
             '{a:.743,c:.743,h:.753,l:.72,o:.741,t:"2026-09-29",v:26596979,ch:.27}]')


def test_parse_history_reads_dated_bars():
    from data.stockanalysis import parse_history
    h = parse_history(HIST_PAGE)
    assert list(h.index.strftime("%Y-%m-%d")) == ["2026-09-29", "2026-09-30", "2026-10-01"]
    assert h.loc["2026-09-30", "low"] == 1.99 and h.loc["2026-09-29", "close"] == 0.743


def test_backfill_never_creates_dates_or_overwrites(temp_db, monkeypatch):
    from data import stockanalysis as sa
    from data import fundamentals
    days = pd.bdate_range("2026-08-03", "2026-09-30")
    _seed_prices(days)                                      # sessions exist through 09-30
    from data.collectors import DataCollector
    with get_connection() as conn:
        DataCollector._ensure_source_column(conn)
        conn.execute("INSERT INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, source) "
                     "VALUES ('CRST', '2026-09-30', 9.99, 9.99, 1, 'vendor')")
    monkeypatch.setattr(fundamentals, "_fetch", lambda url, timeout=25: HIST_PAGE)
    monkeypatch.setattr("time.sleep", lambda s: None)
    sa.backfill_history(["CRST"])
    with get_connection() as conn:
        rows = dict(conn.execute("SELECT date, close FROM daily_ohlcv WHERE ticker='CRST'").fetchall())
    assert "2026-10-01" not in rows                          # not an accepted session: not minted
    assert rows["2026-09-30"] == 9.99                        # vendor row untouched
    assert rows["2026-09-29"] == 0.743                       # gap filled


# --- ingest ------------------------------------------------------------------------

def test_null_close_bars_are_not_stored(temp_db):
    from data.collectors import DataCollector
    df = pd.DataFrame({"open": [1.0, None], "high": [1.0, None], "low": [1.0, None],
                       "close": [1.0, None], "volume": [10.0, 0.0],
                       "value_traded_egp": [10.0, None], "adjusted_close": [1.0, None]},
                      index=pd.to_datetime(["2026-09-30", "2026-10-01"]))
    DataCollector()._store_ohlcv("MFSC", df)
    with get_connection() as conn:
        dates = [r[0] for r in conn.execute("SELECT date FROM daily_ohlcv WHERE ticker='MFSC'")]
    assert dates == ["2026-09-30"]
