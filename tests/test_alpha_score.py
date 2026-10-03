import numpy as np
import pandas as pd
import pytest

import config
from data import database
from models import alpha_score as az
from models.factor_model import BETA_COLUMNS


def test_cross_sectional_zscore_basic_matches_manual_calc():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 5), "x": values})
    result = az._cross_sectional_zscore(df, ["x"], clip=2.5)
    expected = (values - values.mean()) / values.std(ddof=1)
    np.testing.assert_allclose(result["x"].to_numpy(), expected)


def test_cross_sectional_zscore_clips_extreme_outlier():
    # Samuelson's inequality bounds any single point's z-score at
    # sqrt(n-1) for a sample of n -- with only 6 points that cap is
    # sqrt(5) =~ 2.236, which can never exceed a 2.5 clip no matter how
    # extreme the outlier is. Need n>=8 (sqrt(7)=~2.65) for a clip of 2.5
    # to be reachable at all.
    values = np.array([0.0] * 8 + [1e9])
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 9), "x": values})
    raw_z = (values - values.mean()) / values.std(ddof=1)
    assert raw_z.max() > 2.5  # confirms this example actually exercises the clip path

    result = az._cross_sectional_zscore(df, ["x"], clip=2.5)
    assert result["x"].max() == pytest.approx(2.5)
    assert result["x"].min() < 0


def test_cross_sectional_zscore_nan_when_too_few_tickers():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3), "x": [1.0, 2.0, 3.0]})
    result = az._cross_sectional_zscore(df, ["x"], clip=2.5)  # only 3 < MIN_TICKERS_FOR_ZSCORE=5
    assert result["x"].isna().all()


def test_cross_sectional_zscore_zero_std_gives_zero_not_nan():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 6), "x": [5.0] * 6})
    result = az._cross_sectional_zscore(df, ["x"], clip=2.5)
    assert (result["x"] == 0.0).all()


@pytest.fixture
def synthetic_exposures_and_regimes(monkeypatch):
    tickers = [f"T{i}" for i in range(8)]
    date = pd.Timestamp("2024-03-01")
    # T0 has a strongly positive momentum beta and near-zero everything
    # else; T1 has the mirror-image negative momentum beta. Other
    # tickers fill out the cross-section with modest, varied values so
    # z-scoring has enough names to work with.
    rng = np.random.default_rng(0)
    rows = []
    for i, t in enumerate(tickers):
        row = {"ticker": t, "date": date}
        for col in BETA_COLUMNS:
            row[col] = rng.normal(0, 0.3)
        rows.append(row)
    rows[0]["beta_momentum"] = 3.0
    rows[1]["beta_momentum"] = -3.0
    exposures = pd.DataFrame(rows)
    regimes = pd.DataFrame({"date": [date], "regime": ["BULL"]})

    def fake_loader(tickers=None):
        return exposures.copy(), regimes.copy()

    monkeypatch.setattr(az, "_load_exposures_and_regimes", fake_loader)
    return exposures, regimes


def test_compute_alpha_scores_rewards_high_momentum_more_in_bull(monkeypatch, synthetic_exposures_and_regimes):
    premia = pd.Series({"beta_market": 0.0, "beta_momentum": 0.01, "beta_liquidity": 0.0,
                         "beta_egp_sensitivity": 0.0, "beta_oil_sensitivity": 0.0})

    bull_scores = az.compute_alpha_scores(premia)
    t0_bull = bull_scores.loc[bull_scores["ticker"] == "T0", "alpha_score"].iloc[0]
    t1_bull = bull_scores.loc[bull_scores["ticker"] == "T1", "alpha_score"].iloc[0]
    assert t0_bull > t1_bull  # high momentum beta should score higher than low/negative

    # Now flip the regime to one where momentum ISN'T upweighted (BEAR
    # upweights liquidity instead) and confirm the T0/T1 gap shrinks,
    # proving the regime adjustment actually changes the outcome, not
    # just the raw premium.
    exposures, _ = synthetic_exposures_and_regimes
    bear_regimes = pd.DataFrame({"date": [exposures["date"].iloc[0]], "regime": ["BEAR"]})
    monkeypatch.setattr(az, "_load_exposures_and_regimes", lambda tickers=None: (exposures.copy(), bear_regimes))

    bear_scores = az.compute_alpha_scores(premia)
    t0_bear = bear_scores.loc[bear_scores["ticker"] == "T0", "alpha_score"].iloc[0]
    t1_bear = bear_scores.loc[bear_scores["ticker"] == "T1", "alpha_score"].iloc[0]

    assert (t0_bull - t1_bull) > (t0_bear - t1_bear)


def test_compute_alpha_scores_clips_at_configured_bound(synthetic_exposures_and_regimes):
    premia = pd.Series({"beta_market": 0.0, "beta_momentum": 1.0, "beta_liquidity": 0.0,
                         "beta_egp_sensitivity": 0.0, "beta_oil_sensitivity": 0.0})
    scores = az.compute_alpha_scores(premia)
    assert scores["alpha_score"].abs().max() <= az.ALPHA_CLIP + 1e-9


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_esis.db"))
    database.init_db()
    return config.DB_PATH


def test_store_alpha_scores_roundtrip(temp_db):
    df = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "regime": ["BULL", "BEAR"],
        "alpha_score": [1.23, -0.87],
    })
    az.store_alpha_scores(df)

    with database.get_connection() as conn:
        rows = conn.execute("SELECT ticker, regime, alpha_score FROM signals ORDER BY ticker").fetchall()
    assert [dict(r) for r in rows] == [
        {"ticker": "AAA", "regime": "BULL", "alpha_score": 1.23},
        {"ticker": "BBB", "regime": "BEAR", "alpha_score": -0.87},
    ]
