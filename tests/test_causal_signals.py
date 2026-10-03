"""
Tests for Layer 5 signal construction (models/causal/signals.py).

Unit tests use a small synthetic panel with KNOWN thetas so the signal
algebra (causal_signal = 0.5*theta_s*sentiment + 0.5*theta_iv*momentum,
then cross-sectional z-score) is checked exactly. A DB-backed smoke test
confirms the end-to-end run() produces a non-empty, finite, z-scored
signal on the real database.
"""

import numpy as np
import pandas as pd

import config
from models.causal import signals as cs


def test_construct_causal_signal_algebra():
    # two dates, three tickers each; known thetas
    panel = pd.DataFrame({
        "ticker": ["A", "B", "C", "A", "B", "C"],
        "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
        "momentum": [0.1, -0.2, 0.3, 0.0, 0.5, -0.5],
        "sentiment_score": [np.nan] * 6,   # no sentiment -> sentiment term drops out
    })
    results = {"dml_sentiment": {"theta": 0.0}, "iv_momentum": {"theta": 2.0}}
    out = cs.construct_causal_signal(panel, results)

    # raw = 0.5*0*sent + 0.5*2*momentum = momentum
    expected_raw = panel["momentum"].to_numpy()
    assert np.allclose(out["causal_signal_raw"].to_numpy(), expected_raw)

    # z-scored within each date -> mean 0 per date
    for _, g in out.groupby("date"):
        assert abs(g["causal_signal"].mean()) < 1e-9


def test_construct_causal_signal_uses_both_effects():
    panel = pd.DataFrame({
        "ticker": ["A", "B"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "momentum": [0.2, 0.2],
        "sentiment_score": [1.0, -1.0],
    })
    results = {"dml_sentiment": {"theta": 0.5}, "iv_momentum": {"theta": 1.0}}
    out = cs.construct_causal_signal(panel, results)
    # raw_A = 0.5*0.5*1 + 0.5*1*0.2 = 0.35 ; raw_B = 0.5*0.5*(-1) + 0.5*1*0.2 = -0.15
    raw = dict(zip(out["ticker"], out["causal_signal_raw"]))
    assert abs(raw["A"] - 0.35) < 1e-12
    assert abs(raw["B"] - (-0.15)) < 1e-12


def test_walkforward_signal_has_no_future_leak():
    # Phase 1 leak check: pre-2023 causal_signal must be identical whether or
    # not post-2022 data exists at estimation time. With the pooled-theta bug
    # this held only because z-scoring cancels theta; the walk-forward version
    # must hold structurally (2018 uses only <=2017 thetas).
    panel = cs.build_causal_panel()
    full = cs.construct_causal_signal_walkforward(panel)
    trunc = cs.construct_causal_signal_walkforward(panel[panel["date"] <= "2022-12-31"].copy())
    m = full["date"] <= pd.Timestamp("2022-12-31")
    a = full[m].set_index(["ticker", "date"])["causal_signal"]
    b = trunc.set_index(["ticker", "date"])["causal_signal"]
    j = a.to_frame("full").join(b.rename("trunc"), how="inner")
    assert len(j) > 1000
    assert np.allclose(j["full"], j["trunc"], atol=1e-9)


def test_run_end_to_end_on_real_db():
    out = cs.run(store=False)
    panel, results, sig = out["panel"], out["results"], out["signals"]
    assert not panel.empty
    assert not sig.empty
    # signal is finite and z-scored (std ~ 1 across the pooled sample when it varies)
    assert np.all(np.isfinite(sig["causal_signal"]))
    # momentum IV ran on real data with strong instruments
    iv = results["iv_momentum"]
    assert iv["n"] >= config.CAUSAL_MIN_OBS
    assert iv["first_stage_f"]["f_stat"] > config.CAUSAL_WEAK_IV_F_THRESHOLD
    # sentiment DML correctly flags insufficiency (no daily_sentiment yet)
    assert results["dml_sentiment"].get("insufficient", False) is True
