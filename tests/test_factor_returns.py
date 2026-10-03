import numpy as np
import pandas as pd

from models import factor_returns as fr
from models import regime_features as rf


def test_monthly_rebalance_dates_picks_first_day_of_each_month():
    idx = pd.DatetimeIndex(["2024-01-05", "2024-01-06", "2024-02-01", "2024-02-15", "2024-03-04"])
    dates = fr._monthly_rebalance_dates(idx)
    assert [str(d.date()) for d in dates] == ["2024-01-05", "2024-02-01", "2024-03-04"]


def test_build_long_short_factor_matches_manual_calc():
    # 4 tickers, 2 months. Month 1: score ranks C > A > D > B (top/bottom 1
    # each since quantile*4=1.2->int->1, but min_names*2=6 > 4 so the
    # *first* rebalance with only 4 names won't have enough names per the
    # default MIN_NAMES_PER_SIDE=3 -- use a lower min_names for this test.
    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    scores = pd.DataFrame(
        {"A": 0.10, "B": -0.10, "C": 0.20, "D": -0.20},
        index=idx,
    )
    # constant returns per ticker so the expected long-short spread is exact and trivial to check
    returns = pd.DataFrame(
        {"A": 0.01, "B": 0.02, "C": 0.05, "D": -0.03},
        index=idx,
    )

    factor = fr._build_long_short_factor(scores, returns, quantile=0.30, min_names=1)
    # Winner (highest score) = C, loser (lowest score) = D -> spread = 0.05 - (-0.03) = 0.08
    assert np.isclose(factor.iloc[0], 0.08)
    assert np.isclose(factor.iloc[-1], 0.08)


def test_build_long_short_factor_carries_over_when_undersubscribed():
    idx = pd.date_range("2024-01-01", periods=65, freq="D")
    # Month 1 (Jan): enough names. Month 2 (Feb): scores all NaN -> should
    # carry over January's winners/losers rather than going empty.
    scores = pd.DataFrame(index=idx, columns=["A", "B", "C", "D"], dtype=float)
    jan_mask = scores.index.month == 1
    scores.loc[jan_mask, :] = [0.10, -0.10, 0.20, -0.20]
    # Feb rows left as NaN (undersubscribed).

    returns = pd.DataFrame(
        {"A": 0.01, "B": 0.02, "C": 0.05, "D": -0.03},
        index=idx,
    )
    factor = fr._build_long_short_factor(scores, returns, quantile=0.30, min_names=1)
    # Should still have a value in February (carried-over winners/losers), not NaN.
    feb_values = factor[factor.index.month == 2]
    assert not feb_values.empty
    assert np.isclose(feb_values.iloc[0], 0.08)


def test_build_momentum_scores_uses_skip_and_lookback_correctly():
    idx = pd.date_range("2024-01-01", periods=300, freq="D")
    # Deterministic price path so momentum score is exactly computable.
    close = pd.Series(100 * 1.001 ** np.arange(300), index=idx)
    panel = {"A": pd.DataFrame({"close": close, "value_traded_egp": 1000.0}, index=idx)}

    scores = fr.build_momentum_scores(panel)
    t = 280
    expected = close.iloc[t - fr.MOMENTUM_SKIP_DAYS] / close.iloc[t - fr.MOMENTUM_SKIP_DAYS - fr.MOMENTUM_LOOKBACK_DAYS] - 1
    assert np.isclose(scores["A"].iloc[t], expected)


def test_build_factor_returns_smoke(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    rng = np.random.default_rng(42)

    def fake_panel(tickers=None):
        panel = {}
        for name in ["A", "B", "C", "D", "E", "F"]:
            price = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx)))
            vol = rng.uniform(1000, 2000, len(idx))
            df = pd.DataFrame({"close": price, "volume": vol}, index=idx)
            df["value_traded_egp"] = df["close"] * df["volume"]
            panel[name] = df
        return panel

    def fake_macro(columns=("usd_egp",)):
        data = {}
        if "usd_egp" in columns:
            data["usd_egp"] = 30 + np.cumsum(rng.normal(0, 0.02, len(idx)))
        if "brent_oil_usd" in columns:
            data["brent_oil_usd"] = 75 + np.cumsum(rng.normal(0, 0.3, len(idx)))
        return pd.DataFrame(data, index=idx)

    monkeypatch.setattr(fr, "_load_price_panel", fake_panel)
    monkeypatch.setattr(fr, "load_macro", fake_macro)
    monkeypatch.setattr(rf, "MIN_TICKER_COVERAGE", 3)  # only 6 fake tickers; real default (10) would zero everything out

    factors = fr.build_factor_returns()
    for name in fr.FACTOR_NAMES:
        assert name in factors.columns
    assert not factors.isna().any().any()
    assert len(factors) > 0


def test_long_short_factor_excludes_rebalance_day_return():
    # Build a simple 3-ticker panel spanning 2 months
    idx = pd.date_range("2024-01-01", "2024-03-31", freq="B")
    rng = np.random.default_rng(42)

    returns = pd.DataFrame(
        {t: rng.normal(0, 0.01, len(idx)) for t in ["A", "B", "C"]},
        index=idx,
    )
    # Score: A always wins (highest score), C always loses (lowest score)
    scores = pd.DataFrame(
        {"A": 1.0, "B": 0.0, "C": -1.0},
        index=idx,
    )

    factor = fr._build_long_short_factor(scores, returns)

    # Get the rebalance dates
    rebalance_dates = fr._monthly_rebalance_dates(scores.index)

    # The factor series must NOT have any value on a rebalance date itself
    for rdate in rebalance_dates:
        if rdate in factor.index:
            # rdate in factor.index means rebalance day was included -- BUG
            assert False, (
                f"Rebalance day {rdate} found in factor return series -- "
                "lookahead bias present. Fix 6 not applied correctly."
            )
