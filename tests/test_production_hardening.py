"""
Tests for the production-hardening work (audit Part 2). These cover exactly
the gaps the audit found UNTESTED: data gaps, stray non-session dates,
degenerate universes, yfinance API failures, staleness gates, and the forward
paper-trading record.
"""

import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "prod.db"))
    database.init_db()
    database.seed_stocks()
    return config.DB_PATH


def _insert(ticker, dates, close, value=5_000_000.0):
    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, value_traded_egp) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            [(ticker, d.strftime("%Y-%m-%d"), float(c), float(c), value) for d, c in zip(dates, close)])


# ---------------- CAL-01: trading calendar ----------------

def test_calendar_rejects_stray_non_session_dates(temp_db):
    from data.calendar import valid_trading_days, spurious_dates
    dates = pd.bdate_range("2024-01-01", periods=50)
    rng = np.random.default_rng(0)
    for t in [f"T{i}" for i in range(10)]:
        _insert(t, dates, 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))))
    # a stray "session" INSIDE everyone's listing window: a Saturday on which
    # exactly one ticker prints (this is the real yfinance failure mode).
    holiday = pd.Timestamp("2024-02-03")          # Saturday, mid-sample
    _insert("T0", [holiday], [123.0])

    good = valid_trading_days()
    assert holiday not in good                    # the stray date is rejected
    assert len(good) == 50                        # the real sessions survive
    assert holiday in set(pd.to_datetime(spurious_dates()["date"]))


def test_returns_never_span_a_rejected_date(temp_db):
    from backtest.engine import load_price_returns
    dates = pd.bdate_range("2024-01-01", periods=30)
    for t in ["A", "B", "C"]:
        _insert(t, dates, np.full(len(dates), 100.0))
    # stray mid-sample print with an absurd price: if the date survived, it
    # would inject a +400% return and a -80% return on either side.
    _insert("A", [pd.Timestamp("2024-01-13")], [500.0])   # Saturday
    rets = load_price_returns(use_calendar=True)
    assert np.nanmax(np.abs(rets.values)) < 0.01


# ---------------- UNIV-01: coverage-based eligibility ----------------

def test_single_missing_print_does_not_disqualify_a_ticker():
    # THE BUG: the old rule was window.notna().all() -- one NaN in 180 days
    # removed the name entirely. Coverage-based eligibility must keep it.
    from backtest.engine import BacktestEngine
    dates = pd.bdate_range("2024-01-01", periods=100)
    w = pd.DataFrame(0.01, index=dates, columns=["A", "B", "C"])
    w.loc[dates[50], "A"] = np.nan          # one missing print for A
    eng = BacktestEngine(min_window_coverage=0.90)
    cols = eng._eligible_names(w, dates[-1], None)
    assert "A" in cols                       # kept (99% coverage)
    assert set(cols) == {"A", "B", "C"}


def test_badly_broken_series_is_still_excluded():
    from backtest.engine import BacktestEngine
    dates = pd.bdate_range("2024-01-01", periods=100)
    w = pd.DataFrame(0.01, index=dates, columns=["GOOD", "BROKEN"])
    w.loc[dates[:60], "BROKEN"] = np.nan     # 40% coverage
    eng = BacktestEngine(min_window_coverage=0.90)
    cols = eng._eligible_names(w, dates[-1], None)
    assert cols == ["GOOD"]


def test_degenerate_universe_skips_rebalance_instead_of_trading_two_names():
    from backtest.engine import BacktestEngine
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0.0005, 0.01, (300, 3)), index=dates, columns=["A", "B", "C"])
    eng = BacktestEngine(rebalance_days=21, min_names=5)   # only 3 names exist
    ulog = []
    res = eng.run_portfolio(lambda w: pd.Series(1 / w.shape[1], index=w.columns),
                            rets, lookback=100, universe_log=ulog)
    assert res["n_skipped_rebalances"] > 0                  # refused to trade
    assert all(u["n_eligible"] == 3 for u in ulog)          # and logged why


# ---------------- FETCH-01: API failures ----------------

def test_fetch_failure_raises_instead_of_looking_like_no_data(monkeypatch):
    from data import collectors
    from data.collectors import DataCollector, FetchError

    class _Boom:
        def __init__(self, *a, **k): pass
        def history(self, **k): raise ConnectionError("network down")

    monkeypatch.setattr(collectors.yf, "Ticker", _Boom)
    monkeypatch.setattr(config, "FETCH_RETRIES", 2)
    monkeypatch.setattr(config, "FETCH_BACKOFF_SECONDS", 0.0)
    with pytest.raises(FetchError):
        DataCollector().fetch_egx_prices("COMI", "2024-01-01")


def test_fetch_retries_then_succeeds(monkeypatch):
    from data import collectors
    from data.collectors import DataCollector
    calls = {"n": 0}

    class _Flaky:
        def __init__(self, *a, **k): pass
        def history(self, **k):
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("transient")
            idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=3), name="Date")
            return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                                 "Adj Close": 1.0, "Volume": 10}, index=idx)

    monkeypatch.setattr(collectors.yf, "Ticker", _Flaky)
    monkeypatch.setattr(config, "FETCH_RETRIES", 3)
    monkeypatch.setattr(config, "FETCH_BACKOFF_SECONDS", 0.0)
    df = DataCollector().fetch_egx_prices("COMI", "2024-01-01")
    assert not df.empty and calls["n"] == 2      # retried once, then worked


# ---------------- quality gate + validation ----------------

def test_quality_gate_blocks_on_stale_db(temp_db):
    from data.quality import assert_data_fit_to_trade, DataQualityError
    old = pd.bdate_range("2020-01-01", periods=80)
    for t in ["A", "B", "C"]:
        _insert(t, old, np.full(len(old), 100.0))
    with pytest.raises(DataQualityError):        # years stale
        assert_data_fit_to_trade()


def test_validate_startup_fails_on_empty_prices(temp_db):
    from ops.validate import validate_startup, ConfigError
    with pytest.raises(ConfigError):             # daily_ohlcv empty
        validate_startup(require_trading=True)


# ---------------- forward paper record ----------------

def test_paper_track_record_refuses_to_call_short_history_evidence(temp_db):
    from ops.paper_trading import paper_track_record
    with get_connection() as conn:
        conn.executemany("INSERT INTO paper_trading_nav (date, nav, daily_return, exposure, regime, run_id, n_positions) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)",
                         [(d.strftime("%Y-%m-%d"), 1.0 + i * 0.001, 0.001, 1.0, "BULL", "r1", 5)
                          for i, d in enumerate(pd.bdate_range("2026-01-01", periods=30))])
    rec = paper_track_record()
    assert rec["n_days"] == 30
    assert rec["sufficient"] is False            # 30 days is not a track record
    assert "too short" in rec["note"]


def test_paper_record_flags_missing_bear_regime(temp_db):
    from ops.paper_trading import paper_track_record
    with get_connection() as conn:
        conn.executemany("INSERT INTO paper_trading_nav (date, nav, daily_return, exposure, regime, run_id, n_positions) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?)",
                         [(d.strftime("%Y-%m-%d"), 1.0, 0.0, 1.0, "BULL", "r1", 5)
                          for d in pd.bdate_range("2024-01-01", periods=300)])
    rec = paper_track_record()
    # 300 days but zero BEAR days -> the overlay (the entire edge) is untested
    assert rec["sufficient"] is False
    assert "BEAR" in rec["note"]


def test_calendar_rejects_phantom_date_at_right_edge():
    """Regression (2026-07-23, Revolution Day): yfinance printed 2 of 27 tickers
    on a day EGX was CLOSED. Because every other ticker's last print was the day
    before, the old listing rule marked them delisted and scored the phantom day
    at 100% coverage -- so it survived, became the live as-of date, and collapsed
    the tradable universe to 0. The right-edge grace fix must reject it."""
    import numpy as np
    import pandas as pd
    from data.calendar import coverage_by_date, valid_trading_days

    idx = pd.bdate_range("2026-06-01", periods=40)
    cols = [f"T{i:02d}" for i in range(27)]
    panel = pd.DataFrame(100.0, index=idx, columns=cols)
    # final date: only 2 tickers print, the other 25 are NaN (market was shut)
    panel.iloc[-1, :] = np.nan
    panel.iloc[-1, :2] = 100.0

    cov = coverage_by_date(panel)
    assert cov.iloc[-1] < 0.10          # measured against the FULL universe, not 2/2
    good = valid_trading_days(panel)
    assert idx[-1] not in good          # phantom date rejected
    assert idx[-2] in good              # the real prior session survives


def test_calendar_still_honours_genuine_delisting():
    """The grace fix must not resurrect a stock that genuinely stopped trading
    mid-sample -- it should still be excluded from coverage after it delists."""
    import numpy as np
    import pandas as pd
    from data.calendar import coverage_by_date

    idx = pd.bdate_range("2026-01-01", periods=60)
    panel = pd.DataFrame(100.0, index=idx, columns=["A", "B", "C"])
    panel.loc[idx[20]:, "C"] = np.nan   # C delists early, far from the right edge
    cov = coverage_by_date(panel)
    # A and B still print every day -> full coverage despite C being gone
    assert cov.iloc[-1] == pytest.approx(1.0)
    assert cov.iloc[30] == pytest.approx(1.0)


def test_quality_gate_catches_stale_macro_while_prices_are_fresh():
    """Regression: prices ran to 2026-07-30 while usd_egp stopped at 2026-07-07,
    so the regime signal was truncated 23 days and merely forward-filled -- yet
    every price-based check passed. Macro must be checked on its own."""
    import pandas as pd
    import config
    from data import database
    from data.database import get_connection
    from data.quality import data_quality_report

    old = config.DB_PATH
    import tempfile, os
    tmp = os.path.join(tempfile.mkdtemp(), "q.db")
    config.DB_PATH = tmp
    try:
        database.init_db()
        # tradable_universe is created by build_tradable_universe (pandas to_sql),
        # not by init_db, so the test must create it explicitly.
        with get_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS tradable_universe "
                         "(ticker TEXT, date DATE, is_tradable INTEGER)")
        today = pd.Timestamp.now().normalize()
        idx = pd.bdate_range(today - pd.Timedelta(days=400), today)
        with get_connection() as conn:
            for t_ in [f"T{i:02d}" for i in range(15)]:
                conn.executemany(
                    "INSERT OR REPLACE INTO daily_ohlcv (ticker,date,close,adjusted_close,volume,value_traded_egp) "
                    "VALUES (?,?,100,100,1000,5000000)", [(t_, str(d.date())) for d in idx])
                conn.executemany(
                    "INSERT OR REPLACE INTO tradable_universe (ticker,date,is_tradable) VALUES (?,?,1)",
                    [(t_, str(d.date())) for d in idx])
            # macro deliberately stops 30 days ago while prices are current
            conn.executemany("INSERT OR REPLACE INTO macro_daily (date, usd_egp) VALUES (?, 48.0)",
                             [(str(d.date()),) for d in idx if d <= today - pd.Timedelta(days=30)])
        rep = data_quality_report()
        assert not rep["ok"]
        assert any("macro" in f for f in rep["fatal"])
        assert rep["macro_age_days"] >= 30
    finally:
        config.DB_PATH = old


# --- go-live gate must never PASS on an insufficient record -----------------

def test_gate_reports_inconclusive_not_pass_on_short_record(monkeypatch):
    """On 2026-08-08 the gate printed "[PASS] forward Sharpe 7.67" from 21
    sessions it had itself declared insufficient. A statistic from too little
    data is INCONCLUSIVE -- neither pass nor fail -- and must block go-live."""
    import config
    from ops import go_live_gate as g
    monkeypatch.setattr(g, "paper_track_record", lambda: {
        "n_days": 21, "bear_days_observed": 0, "sharpe": 7.67,
        "max_drawdown": -0.0058, "real_usd_cagr": None, "sufficient": False})
    d = g.evaluate_go_live()
    by = {c["criterion"]: c for c in d["checks"]}
    assert by["forward Sharpe"]["pass"] is False
    assert by["forward Sharpe"]["conclusive"] is False
    assert by["forward max drawdown"]["pass"] is False
    assert d["go_live"] is False
    assert "INCONC" in g.format_go_live(d)


def test_gate_passes_those_criteria_once_the_record_is_long_enough(monkeypatch):
    """The inconclusive rule must not become a permanent block -- it lifts when
    the sample reaches the SAME threshold the duration criterion uses."""
    import config
    from ops import go_live_gate as g
    monkeypatch.setattr(g, "paper_track_record", lambda: {
        "n_days": config.GO_LIVE_MIN_PAPER_DAYS, "bear_days_observed": 25,
        "sharpe": 1.2, "max_drawdown": -0.10, "real_usd_cagr": 0.03,
        "sufficient": True})
    d = g.evaluate_go_live()
    assert all(c["pass"] for c in d["checks"])
    assert d["go_live"] is True


def test_inconclusive_blocks_go_live_exactly_like_a_fail(monkeypatch):
    import config
    from ops import go_live_gate as g
    monkeypatch.setattr(g, "paper_track_record", lambda: {
        "n_days": config.GO_LIVE_MIN_PAPER_DAYS, "bear_days_observed": 25,
        "sharpe": None, "max_drawdown": -0.10, "real_usd_cagr": 0.03,
        "sufficient": True})
    assert g.evaluate_go_live()["go_live"] is False
