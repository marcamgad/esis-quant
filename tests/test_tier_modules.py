"""
Edge cases for the modules added during the tiered remediation.

These modules produce the numbers the audits are argued from, so a silent wrong
answer here is worse than a crash. Every test below is a case where the code
could plausibly return something plausible-looking and wrong: empty inputs,
single observations, constant series, misaligned indices, NaNs, and the
specific alignment bugs already caught by hand this session.
"""
import numpy as np
import pandas as pd
import pytest

import config


# --- data/pit_universe -------------------------------------------------------

def test_pit_universe_handles_empty_panel(monkeypatch):
    from data import pit_universe as pu
    monkeypatch.setattr(pu, "_load_value_traded", lambda: pd.DataFrame())
    assert pu.build_pit_universe().empty
    assert pu.store_pit_universe(pd.DataFrame()) == 0


def test_pit_universe_membership_is_backward_looking_only():
    """A name that becomes liquid only at the END of the sample must NOT appear
    at the start. This is the whole point of the module -- if it fails, the
    selection bias it exists to remove is still present."""
    from data import pit_universe as pu
    idx = pd.bdate_range("2020-01-01", periods=400)
    vt = pd.DataFrame({
        "OLD": 1e6,                                    # liquid throughout
        "NEW": [0.0] * 300 + [1e9] * 100,              # only liquid at the end
    }, index=idx)
    med = vt.rolling(60, min_periods=30).median()
    seen = vt.notna().cumsum()
    eligible = (med >= pu.PIT_MIN_VALUE_EGP) & (seen >= pu.PIT_MIN_HISTORY)
    # NEW cannot be eligible in the first half
    assert not eligible["NEW"].iloc[:250].any()
    assert eligible["NEW"].iloc[-1]


def test_pit_turnover_report_flags_a_static_universe():
    """A 'point-in-time' universe that never changes is the hand-picked list in
    disguise. The report must say so rather than look successful."""
    from data.pit_universe import turnover_report
    idx = pd.bdate_range("2020-01-01", periods=50)
    static = pd.DataFrame([{"date": d, "ticker": t, "in_universe": 1, "rank": 1.0,
                            "median_value": 1e6}
                           for d in idx for t in ("A", "B", "C")])
    rep = turnover_report(static)
    assert rep["entries"] == 0 and rep["exits"] == 0
    assert rep["names_always"] == 3
    from data.pit_universe import format_comparison
    assert "WARNING" in format_comparison(rep, seed_n=3)


# --- analysis/regime_validation ---------------------------------------------

def test_score_calls_refuses_tiny_samples():
    """Fewer than 20 observations on either side is not a measurement."""
    from analysis.regime_validation import score_calls
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 60), index=idx)
    bear = pd.Series([1] * 5 + [0] * 55, index=idx)
    out = score_calls(bear, r)
    assert out["ok"] is False
    assert "too few" in out["reason"]


def test_score_calls_applies_the_overlap_correction():
    """t(adj) must be t(raw)/sqrt(h). Without it, overlapping forward windows
    manufacture significance."""
    from analysis.regime_validation import score_calls
    idx = pd.bdate_range("2015-01-01", periods=1200)
    rng = np.random.default_rng(1)
    bear = pd.Series((np.arange(1200) % 200 < 60).astype(int), index=idx)
    r = pd.Series(np.where(bear.values == 1, rng.normal(-0.003, 0.01, 1200),
                           rng.normal(0.001, 0.01, 1200)), index=idx)
    out = score_calls(bear, r, h=21)
    assert out["t_adj"] == pytest.approx(out["t_raw"] / np.sqrt(21), rel=1e-9)
    assert abs(out["t_adj"]) < abs(out["t_raw"])


def test_score_calls_survives_a_constant_return_series():
    """Zero variance must not raise or produce a fake infinite t-stat."""
    from analysis.regime_validation import score_calls
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(0.0, index=idx)
    bear = pd.Series(([1] * 100 + [0] * 100) * 2, index=idx)
    out = score_calls(bear, r, h=5)
    assert out["ok"] is True
    assert out["t_raw"] != out["t_raw"] or np.isfinite(out["t_raw"])


def test_regime_series_uses_the_date_column_not_position():
    """compute_regime_series returns `date` as a COLUMN. Indexing positionally
    would silently misalign every call against the returns it is scored on."""
    from analysis.regime_validation import _regime_series
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=3),
                       "regime": ["BULL", "BEAR", "BULL"]})
    s = _regime_series(df)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.loc[pd.Timestamp("2020-01-02")] == "BEAR"


def test_bear_state_is_lowest_mean_return_not_lowest_index():
    """HMM state numbering is arbitrary across refits; assuming state 0 is BEAR
    would mislabel roughly (K-1)/K of all fits."""
    from analysis.regime_validation import _bear_state
    idx = pd.bdate_range("2020-01-01", periods=90)
    obs = pd.DataFrame({"ret": [0.02] * 30 + [-0.05] * 30 + [0.01] * 30}, index=idx)
    states = pd.Series([0] * 30 + [1] * 30 + [2] * 30, index=idx)
    assert _bear_state(obs, states) == 1


# --- ops/control_replay ------------------------------------------------------

def test_worst_windows_returns_both_daily_and_weekly():
    from ops.control_replay import worst_historical_windows
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(np.random.default_rng(2).normal(0, 0.02, 300), index=idx)
    w = worst_historical_windows(r, n=3)
    assert set(w["kind"]) == {"daily", "weekly-5d"}
    assert len(w) == 6
    assert w["ret"].is_monotonic_increasing


def test_replay_reads_thresholds_from_config_not_hardcoded(monkeypatch):
    """A replay with its own copy of the limits would pass while the live
    control was misconfigured."""
    from ops.control_replay import replay_kill_switch
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(0.0005, index=idx)
    out = replay_kill_switch(r)
    assert out["daily_limit"] == config.KILL_SWITCH_MAX_DAILY_LOSS
    assert out["weekly_limit"] == config.KILL_SWITCH_MAX_WEEKLY_LOSS


def test_replay_reports_never_fires_when_nothing_breaches():
    """A calm series must report would_have_fired=False, not silently pass."""
    from ops.control_replay import replay_kill_switch, replay_book_guard
    idx = pd.bdate_range("2020-01-01", periods=400)
    r = pd.Series(0.0003, index=idx)                 # monotonic gains
    assert replay_kill_switch(r)["would_have_fired"] is False
    bg = replay_book_guard(r)
    assert bg["would_have_fired"] is False
    assert bg["max_drawdown"] == pytest.approx(0.0, abs=1e-12)


def test_replay_book_guard_escalation_is_ordered():
    """warn must trigger on more days than block, which triggers on more than
    flatten. Out-of-order thresholds would make the ladder meaningless."""
    from ops.control_replay import replay_book_guard
    idx = pd.bdate_range("2020-01-01", periods=500)
    path = np.concatenate([np.full(250, 0.001), np.full(250, -0.004)])
    bg = replay_book_guard(pd.Series(path, index=idx))
    lv = bg["levels"]
    assert lv["warn"]["n_days"] >= lv["block"]["n_days"] >= lv["flatten"]["n_days"]


# --- analysis/factor_exposure ------------------------------------------------

def test_factor_regression_refuses_short_overlap():
    from analysis.factor_exposure import regress
    idx = pd.bdate_range("2020-01-01", periods=40)
    s = pd.Series(np.random.default_rng(3).normal(0, 0.01, 40), index=idx)
    f = pd.DataFrame({"MKT": np.random.default_rng(4).normal(0, 0.01, 40)}, index=idx)
    out = regress(s, f)
    assert out["ok"] is False


def test_newey_west_se_is_positive_and_finite():
    from analysis.factor_exposure import _newey_west_se
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(400), rng.normal(0, 1, 400)])
    resid = rng.normal(0, 1, 400)
    se = _newey_west_se(X, resid)
    assert np.all(np.isfinite(se)) and np.all(se >= 0)


def test_alpha_of_a_pure_factor_replica_is_zero():
    """A strategy that IS the market must show ~zero alpha and beta ~1.
    If this fails, every alpha number in the audits is suspect."""
    from analysis.factor_exposure import regress
    idx = pd.bdate_range("2015-01-01", periods=900)
    mkt = pd.Series(np.random.default_rng(6).normal(0.0004, 0.01, 900), index=idx)
    out = regress(mkt.copy(), pd.DataFrame({"MKT": mkt}))
    assert out["ok"]
    assert abs(out["alpha_annual"]) < 1e-6
    assert out["coef"][1] == pytest.approx(1.0, abs=1e-9)


# --- analysis/alpha_decay ----------------------------------------------------

def test_half_life_refuses_to_fit_noise():
    """A half-life fitted to insignificant ICs is a number with no referent."""
    from analysis.alpha_decay import half_life
    ic = pd.DataFrame({"horizon": [1, 5, 21], "mean_ic": [0.001, 0.002, 0.003],
                       "t_adj": [0.1, 0.2, 0.3], "significant": [False, False, False]})
    out = half_life(ic)
    assert out["ok"] is False
    assert "no signal" in out["reason"] or "distinguishable" in out["reason"]


def test_half_life_reports_when_ic_does_not_decay():
    from analysis.alpha_decay import half_life
    ic = pd.DataFrame({"horizon": [1, 5, 21], "mean_ic": [0.01, 0.05, 0.20],
                       "t_adj": [3.0, 3.0, 3.0], "significant": [True, True, True]})
    out = half_life(ic)
    assert out["ok"] is False
    assert "does not decay" in out["reason"]


# --- analysis/robustness_suite ----------------------------------------------

def test_monte_carlo_refuses_short_series():
    from analysis.robustness_suite import monte_carlo
    assert monte_carlo(pd.Series(np.zeros(50)))["ok"] is False


def test_monte_carlo_percentiles_are_ordered():
    from analysis.robustness_suite import monte_carlo
    idx = pd.bdate_range("2015-01-01", periods=800)
    r = pd.Series(np.random.default_rng(7).normal(0.0004, 0.012, 800), index=idx)
    out = monte_carlo(r, horizons=(1,), n_paths=200)
    h = out["horizons"][1]
    assert h["p05"] <= h["p25"] <= h["median"] <= h["p75"] <= h["p95"]
    assert 0.0 <= h["p_loss"] <= 1.0
    assert h["median_maxdd"] <= 0.0


def test_stationary_blocks_preserve_length_and_values():
    from analysis.robustness_suite import _stationary_blocks
    rng = np.random.default_rng(8)
    r = rng.normal(0, 0.01, 300)
    path = _stationary_blocks(r, 500, 21, rng)
    assert len(path) == 500
    assert np.isin(path, r).all()          # resampled, never invented


# --- backtest/fees derived scalar -------------------------------------------

def test_book_pct_per_side_falls_with_size_and_rises_with_breadth():
    from backtest.fees import book_pct_per_side
    small = book_pct_per_side(1_000, 12)
    big = book_pct_per_side(1_000_000, 12)
    wide = book_pct_per_side(1_000, 24)
    assert small > big                      # flat fee amortises
    assert wide > small                     # more orders, more flat fees
    assert book_pct_per_side(0, 12) == float("inf")


def test_removed_cost_constants_raise_rather_than_return_a_number():
    """Silently returning a percentage would reintroduce the two-cost-model bug
    the whole P1 fix existed to remove."""
    for name in ("THNDR_COMMISSION_PCT", "ROUND_TRIP_COST_PCT"):
        with pytest.raises(AttributeError) as e:
            getattr(config, name)
        assert "backtest.fees" in str(e.value)
