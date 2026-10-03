import logging

import pandas as pd

import numpy as np

from models.regime_pipeline import (
    _warn_if_labels_look_inconsistent, compute_regime_durations, standardize, standardize_expanding,
)


def test_standardize_expanding_has_no_lookahead():
    # Row t's z-score must depend only on data up to t: appending FUTURE rows
    # must not change any already-computed value. Global standardization
    # fails this; the expanding version must pass it (STAT-LEAK-01).
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    full = standardize_expanding(df, ["a", "b"], min_periods=30)
    truncated = standardize_expanding(df.iloc[:200], ["a", "b"], min_periods=30)
    # the first 200 rows computed on the full series equal those computed
    # on only the first 200 rows -> no future information used
    common = full.iloc[:200].dropna()
    assert np.allclose(common.values, truncated.loc[common.index].values, atol=1e-12)


def test_standardize_expanding_warmup_is_nan():
    df = pd.DataFrame({"a": np.arange(50, dtype=float)})
    out = standardize_expanding(df, ["a"], min_periods=30)
    assert out["a"].iloc[:29].isna().all()   # not enough history yet
    assert out["a"].iloc[29:].notna().all()


def _make_stats(rows):
    return pd.DataFrame(rows).set_index("regime")


def _make_regime_series(labels):
    return pd.Series(labels, index=pd.date_range("2024-01-01", periods=len(labels), freq="D"))


def test_no_warning_when_labels_match_their_definitions(caplog):
    stats = _make_stats([
        {"regime": "BULL", "mean_return": 0.002, "volatility": 0.15, "avg_duration_days": 60},
        {"regime": "BEAR", "mean_return": -0.002, "volatility": 0.35, "avg_duration_days": 25},
        {"regime": "HIGH_VOL", "mean_return": 0.0, "volatility": 0.40, "avg_duration_days": 40},
        {"regime": "RECOVERY", "mean_return": 0.003, "volatility": 0.25, "avg_duration_days": 15},
    ])
    series = _make_regime_series(["BULL"] * 60 + ["BEAR"] * 20 + ["HIGH_VOL"] * 15 + ["RECOVERY"] * 5)

    with caplog.at_level(logging.WARNING, logger="esis_quant.regime_pipeline"):
        _warn_if_labels_look_inconsistent(series, stats)
    assert not caplog.records


def test_warns_when_high_vol_is_actually_the_lowest_vol_state(caplog):
    # Reproduces the real mismatch found on EGX data: HIGH_VOL ends up
    # with the lowest volatility of the four once BEAR claims the
    # genuinely highest-vol state.
    stats = _make_stats([
        {"regime": "BULL", "mean_return": 0.002, "volatility": 0.22, "avg_duration_days": 90},
        {"regime": "BEAR", "mean_return": -0.001, "volatility": 0.32, "avg_duration_days": 30},
        {"regime": "HIGH_VOL", "mean_return": -0.0001, "volatility": 0.10, "avg_duration_days": 95},  # lowest vol!
        {"regime": "RECOVERY", "mean_return": 0.0036, "volatility": 0.26, "avg_duration_days": 24},
    ])
    series = _make_regime_series(["BULL"] * 40 + ["BEAR"] * 15 + ["HIGH_VOL"] * 35 + ["RECOVERY"] * 10)

    with caplog.at_level(logging.WARNING, logger="esis_quant.regime_pipeline"):
        _warn_if_labels_look_inconsistent(series, stats)

    assert len(caplog.records) == 1
    assert "HIGH_VOL does not have the highest volatility" in caplog.records[0].message
    assert "BEAR" in caplog.records[0].message  # should name which regime does have it


def test_warns_when_recovery_dominates():
    stats = _make_stats([
        {"regime": "BULL", "mean_return": 0.004, "volatility": 0.20, "avg_duration_days": 27},
        {"regime": "BEAR", "mean_return": -0.0001, "volatility": 0.34, "avg_duration_days": 20},
        {"regime": "HIGH_VOL", "mean_return": -0.0005, "volatility": 0.40, "avg_duration_days": 70},
        {"regime": "RECOVERY", "mean_return": 0.0009, "volatility": 0.15, "avg_duration_days": 103},
    ])
    series = _make_regime_series(["RECOVERY"] * 62 + ["HIGH_VOL"] * 18 + ["BULL"] * 11 + ["BEAR"] * 9)

    caplog_records = []
    import logging as _logging

    class _Handler(_logging.Handler):
        def emit(self, record):
            caplog_records.append(record.getMessage())

    logger = _logging.getLogger("esis_quant.regime_pipeline")
    handler = _Handler()
    logger.addHandler(handler)
    try:
        _warn_if_labels_look_inconsistent(series, stats)
    finally:
        logger.removeHandler(handler)

    assert any("persistent 'calm/normal market' state" in msg for msg in caplog_records)


def test_compute_regime_durations_averages_consecutive_runs():
    series = pd.Series(["A", "A", "A", "B", "B", "A", "A"])
    durations = compute_regime_durations(series)
    # Run lengths: A=3, B=2, A=2 -> A average = (3+2)/2 = 2.5, B average = 2
    assert durations["A"] == 2.5
    assert durations["B"] == 2


def test_standardize_preserves_column_ordering_direction():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 5.0, 0.0]})
    arr, means, stds = standardize(df, ["x", "y"])
    # Standardization is a monotonic (positive-slope) linear transform per
    # column, so argmax/argmin over a column must be unchanged.
    assert arr[:, 0].argmax() == df["x"].to_numpy().argmax()
    assert arr[:, 1].argmin() == df["y"].to_numpy().argmin()


def test_compute_regime_durations_first_element_same_as_last():
    # When first and last labels differ, the old np.roll approach would
    # compare labels[0] against labels[-1] and potentially misassign
    # the first run_id. This test catches that regression.
    # First label "A", last label "B" -- the old bug triggers here
    series = pd.Series(["A", "A", "B", "B", "B"])
    durations = compute_regime_durations(series)
    assert durations["A"] == 2.0  # one run of length 2
    assert durations["B"] == 3.0  # one run of length 3


def test_compute_regime_durations_single_element():
    series = pd.Series(["BULL"])
    result = compute_regime_durations(series)
    assert result == {"BULL": 1.0}


def test_compute_regime_series_modes_and_vintage():
    # Synthetic 2-regime observation frame (enough rows for walk-forward refits)
    import numpy as np
    from models.regime_pipeline import compute_regime_series
    from models.regime_features import OBSERVATION_COLUMNS
    rng = np.random.default_rng(3)
    n = 1400
    dates = pd.date_range("2016-01-01", periods=n, freq="B")
    # two alternating multi-year regimes with distinct means across the obs cols
    block = np.repeat([0, 1, 0, 1, 0], n // 5 + 1)[:n]
    base = np.where(block[:, None] == 0, -1.0, 1.0)
    data = base + rng.normal(0, 0.5, size=(n, len(OBSERVATION_COLUMNS)))
    obs = pd.DataFrame(data, index=dates, columns=OBSERVATION_COLUMNS)

    wf = compute_regime_series(obs, mode="walkforward", n_init=2, min_train_obs=250)
    # walk-forward records a parameter vintage per row, and each vintage date
    # is strictly BEFORE the rows it labels (no future params)
    assert "param_vintage_date" in wf.columns
    assert (pd.to_datetime(wf["param_vintage_date"]) <= pd.to_datetime(wf["date"])).all()
    # probabilities form a distribution
    probs = wf[["prob_bull", "prob_bear", "prob_high_vol", "prob_recovery"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    # walk-forward drops the initial burn-in (starts later than the raw obs)
    assert wf["date"].min() > obs.index.min()

    # filtered mode covers the full (post-standardization) span and is a valid frame
    filt = compute_regime_series(obs, mode="filtered", n_init=2)
    assert len(filt) >= len(wf)
    assert set(filt["regime"].unique()).issubset({"BULL", "BEAR", "HIGH_VOL", "RECOVERY"})
