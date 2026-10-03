import numpy as np
import pandas as pd
import pytest

import config
from data import database
from models import factor_model as fm
from models import regime_features as rf


def _fake_panel(idx, rng, names):
    panel = {}
    for name in names:
        price = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
        vol = rng.uniform(1000, 2000, len(idx))
        df = pd.DataFrame({"close": price, "volume": vol}, index=idx)
        df["value_traded_egp"] = df["close"] * df["volume"]
        panel[name] = df
    return panel


@pytest.fixture
def patched_data(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    rng = np.random.default_rng(7)
    panel = _fake_panel(idx, rng, ["A", "B", "C", "D", "E", "F"])

    def fake_panel(tickers=None):
        return panel

    def fake_macro(columns=("usd_egp",)):
        data = {}
        if "usd_egp" in columns:
            data["usd_egp"] = 30 + np.cumsum(rng.normal(0, 0.02, len(idx)))
        if "brent_oil_usd" in columns:
            data["brent_oil_usd"] = 75 + np.cumsum(rng.normal(0, 0.3, len(idx)))
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr(rf, "_load_price_panel", fake_panel)
    monkeypatch.setattr(rf, "load_macro", fake_macro)
    monkeypatch.setattr(rf, "MIN_TICKER_COVERAGE", 3)
    monkeypatch.setattr(fm, "_load_price_panel", fake_panel)
    return idx, panel


def test_compute_factor_exposures_returns_expected_columns(patched_data):
    exposures = fm.compute_factor_exposures(window=60)
    for col in fm.BETA_COLUMNS + ["idiosyncratic_vol", "ticker", "date"]:
        assert col in exposures.columns
    assert exposures["ticker"].nunique() == 6
    assert len(exposures) > 0


def test_compute_factor_exposures_no_lookahead_within_window(patched_data):
    # A beta estimate at date t must only depend on data up to and
    # including t -- verified indirectly: the first non-NaN beta for any
    # ticker/factor should not appear before `window` trading days have
    # elapsed since that ticker's data begins.
    exposures = fm.compute_factor_exposures(window=60)
    idx, panel = patched_data
    first_date_by_ticker = {t: df.index.min() for t, df in panel.items()}

    for ticker, group in exposures.groupby("ticker"):
        earliest_beta_date = group["date"].min()
        days_elapsed = (earliest_beta_date - first_date_by_ticker[ticker]).days
        assert days_elapsed >= 55  # ~60 trading days minus small slack for weekends/holidays elsewhere


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_esis.db"))
    database.init_db()
    return config.DB_PATH


def test_store_factor_exposures_roundtrip(temp_db):
    df = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "beta_market": [1.1, 1.2],
        "beta_momentum": [0.3, np.nan],
        "beta_liquidity": [-0.2, -0.1],
        "beta_egp_sensitivity": [0.05, 0.06],
        "beta_oil_sensitivity": [-0.01, 0.02],
        "idiosyncratic_vol": [0.25, 0.26],
    })
    fm.store_factor_exposures(df)

    with database.get_connection() as conn:
        rows = conn.execute(
            "SELECT ticker, date, beta_market, beta_momentum, beta_size FROM factor_exposures ORDER BY date"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["beta_market"] == 1.1
    assert rows[1]["beta_momentum"] is None  # NaN correctly stored as NULL, not 0 or "nan"
    assert rows[0]["beta_size"] is None  # unavailable factor, never fabricated


def test_store_factor_exposures_upsert_updates_existing_row(temp_db):
    df1 = pd.DataFrame({
        "ticker": ["AAA"], "date": pd.to_datetime(["2024-01-01"]),
        "beta_market": [1.0], "beta_momentum": [0.1], "beta_liquidity": [0.1],
        "beta_egp_sensitivity": [0.1], "beta_oil_sensitivity": [0.1], "idiosyncratic_vol": [0.2],
    })
    df2 = df1.copy()
    df2["beta_market"] = [2.0]

    fm.store_factor_exposures(df1)
    fm.store_factor_exposures(df2)

    with database.get_connection() as conn:
        rows = conn.execute("SELECT beta_market FROM factor_exposures").fetchall()
    assert len(rows) == 1
    assert rows[0]["beta_market"] == 2.0
