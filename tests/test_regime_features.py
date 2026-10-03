import numpy as np
import pandas as pd
import pytest

from models import regime_features as rf


@pytest.fixture(autouse=True)
def lower_coverage_threshold(monkeypatch):
    # Tests use tiny synthetic panels (2-3 tickers); the real
    # MIN_TICKER_COVERAGE=10 would drop every row as NaN.
    monkeypatch.setattr(rf, "MIN_TICKER_COVERAGE", 2)


def _make_ticker_df(closes, volumes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=idx)
    volume = pd.Series(volumes, index=idx)
    value_traded = close * volume
    return pd.DataFrame({"close": close, "volume": volume, "value_traded_egp": value_traded})


def test_market_proxy_liquidity_weighting_matches_manual_calc():
    # Ticker A: big, heavily-traded (should dominate the weighted average).
    # Ticker B: small, lightly-traded.
    a = _make_ticker_df([100, 110, 121], [1_000_000, 1_000_000, 1_000_000])  # +10%, +10%
    b = _make_ticker_df([100, 90, 81], [1_000, 1_000, 1_000])  # -10%, -10%
    panel = {"A": a, "B": b}

    proxy = rf.build_market_proxy(panel)

    # Day index 1 (second day): weights are based on day-0 (t-1) traded value.
    # A's day-0 value = 100*1e6 = 1e8; B's = 100*1e3 = 1e5. A completely dominates.
    day1_return = proxy["return"].iloc[1]
    # Expected: essentially all-A weight -> return close to A's +10%, nowhere near B's -10%.
    assert np.isclose(day1_return, 0.10, atol=0.001)


def test_market_proxy_falls_back_to_equal_weight_on_first_valid_day():
    a = _make_ticker_df([100, 110], [500, 500])
    b = _make_ticker_df([100, 90], [500, 500])
    panel = {"A": a, "B": b}

    proxy = rf.build_market_proxy(panel)

    # Day 0 has no t-1 weights at all (nothing before it) -- but pct_change()
    # already makes day 0's return NaN for every ticker, so the first row
    # with any real return is day 1, which DOES have valid t-1 (day 0) weights
    # here. To exercise the fallback, use a case with an actual missing weight:
    # zero volume on day 0 for both -> lagged weight is 0 for everyone on day 1.
    a2 = _make_ticker_df([100, 110], [0, 500])
    b2 = _make_ticker_df([100, 90], [0, 500])
    proxy2 = rf.build_market_proxy({"A": a2, "B": b2})
    day1_return = proxy2["return"].iloc[0] if len(proxy2) and proxy2.index[0] == a2.index[1] else None
    # With both t-1 weights being 0, weight_sum <= 0 -> fallback to equal-weight:
    # mean(+10%, -10%) = 0%.
    assert day1_return is not None
    assert np.isclose(day1_return, 0.0, atol=1e-9)


def test_breadth_fraction_above_moving_average():
    # 3 tickers with enough history for a 2-day MA; construct so exactly 2/3
    # are above their own MA on the last day.
    a = _make_ticker_df([100, 100, 110], [100, 100, 100])  # last close 110 > MA(100,100)=100 -> above
    b = _make_ticker_df([100, 100, 105], [100, 100, 100])  # 105 > 100 -> above
    c = _make_ticker_df([100, 100, 90], [100, 100, 100])  # 90 < 100 -> below
    panel = {"A": a, "B": b, "C": c}

    breadth = rf.build_breadth(panel, ma_window=2)
    last = breadth.iloc[-1]
    assert np.isclose(last, 2 / 3, atol=1e-9)
    assert breadth.dtype == float  # regression guard for the object-dtype bug found earlier


def test_build_observation_matrix_columns_present(monkeypatch):
    # Full pipeline smoke test using an in-memory-style monkeypatch of the
    # loader functions, so this doesn't depend on a populated real DB.
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    rng = np.random.default_rng(0)

    def fake_price_panel(tickers=None):
        panel = {}
        for name in ["A", "B", "C"]:
            price = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
            vol = rng.uniform(1000, 2000, len(idx))
            df = pd.DataFrame({"close": price, "volume": vol}, index=idx)
            df["value_traded_egp"] = df["close"] * df["volume"]
            panel[name] = df
        return panel

    def fake_macro(columns=("usd_egp",)):
        return pd.DataFrame({"usd_egp": 30 + np.cumsum(rng.normal(0, 0.05, len(idx)))}, index=idx)

    monkeypatch.setattr(rf, "_load_price_panel", fake_price_panel)
    monkeypatch.setattr(rf, "load_macro", fake_macro)

    obs = rf.build_observation_matrix()
    for col in rf.OBSERVATION_COLUMNS:
        assert col in obs.columns
    assert not obs[rf.OBSERVATION_COLUMNS].isna().any().any()
