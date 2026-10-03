"""
Tests for the Layer 8 risk engine + overlay. Known-answer anchors: norm_ppf
matches standard normal quantiles; Cornish-Fisher reduces to the normal VaR
when skew/kurtosis are ~0; the overlay de-risks in BEAR, trips + re-enters the
circuit breaker, and never uses same-day information.
"""

import numpy as np
import pandas as pd

from risk import metrics as m
from risk.overlay import apply_overlay, kelly_fraction


def test_norm_ppf_known_quantiles():
    assert abs(m.norm_ppf(0.975) - 1.959964) < 1e-4
    assert abs(m.norm_ppf(0.5)) < 1e-9
    assert abs(m.norm_ppf(0.05) + 1.644854) < 1e-4


def test_cornish_fisher_reduces_to_normal_when_gaussian():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 20000)   # near-Gaussian
    cf = m.cornish_fisher_var(r, alpha=0.95)
    normal_var = -(r.mean() + m.norm_ppf(0.05) * r.std())
    assert abs(cf - normal_var) < 0.001    # skew/kurt ~0 -> CF == normal VaR


def test_cornish_fisher_exceeds_normal_for_fat_left_tail():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 5000)
    r[::50] = -0.15                        # inject a fat, left-skewed tail
    cf = m.cornish_fisher_var(r, alpha=0.95)
    normal_var = -(r.mean() + m.norm_ppf(0.05) * r.std())
    assert cf > normal_var                 # CF widens the tail estimate


def test_drawdown_analytics_known_path():
    # up 10%, down to -20% from peak, recover past the peak
    r = pd.Series([0.10, -0.2727273, 0.5])  # 1.1 -> 0.8 -> 1.2
    da = m.drawdown_analytics(r)
    assert abs(da["max_drawdown"] - (-0.2727273)) < 1e-4
    assert da["recovered"] is True


# ---------------- overlay ----------------

def _series(vals, start="2020-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_overlay_goes_to_cash_in_bear():
    core = _series([0.01] * 10)
    regime = _series(["BULL"] * 5 + ["BEAR"] * 5)
    out = apply_overlay(core, regime, bear_scale=0.0)
    exp = out["exposure"]
    # day after the first BEAR label (index 6) exposure should be 0
    assert exp.iloc[6] == 0.0
    assert exp.iloc[1] == 1.0               # BULL -> fully invested


def test_overlay_circuit_breaker_trips_and_reenters():
    # steady losses drive drawdown below -20%, then a big recovery
    losses = [-0.03] * 10           # ~ -26% cumulative -> breaker trips
    gains = [0.05] * 20             # recovers
    core = _series(losses + gains)
    regime = _series(["BULL"] * len(core))
    out = apply_overlay(core, regime, breaker_dd=-0.20, reentry_dd=-0.10)
    exp = out["exposure"]
    assert (exp == 0.0).any()                # breaker forced cash at some point
    assert exp.iloc[-1] > 0.0                # and re-entered after recovery


def test_overlay_no_lookahead():
    # exposure at t is based on regime/drawdown through t-1; a BEAR label on
    # day t must NOT zero out day t's own return (only day t+1's).
    core = _series([0.02, 0.02, 0.02])
    regime = _series(["BULL", "BEAR", "BEAR"])
    out = apply_overlay(core, regime, bear_scale=0.0)
    # day 1 (first BEAR) still uses exposure from day-0 regime (BULL) -> invested
    assert out["exposure"].iloc[1] == 1.0
    assert out["exposure"].iloc[2] == 0.0    # day 2 sees BEAR-as-of-day-1


def test_kelly_fraction_scales_with_edge():
    strong = kelly_fraction(np.full(100, 0.001))   # positive drift, tiny vol -> capped
    weak = kelly_fraction(np.random.default_rng(0).normal(-0.001, 0.02, 100))
    assert strong >= weak
    assert 0.0 <= weak <= 1.0


# ---------------- Brinson attribution ----------------

def test_brinson_effects_reconcile_to_active_return():
    from risk.attribution import brinson_attribution
    # 4 names, 2 sectors, 2 periods; check alloc+sel+inter == Rp - Rb exactly
    dates = pd.bdate_range("2020-01-01", periods=40)
    rng = np.random.default_rng(4)
    rets = pd.DataFrame(rng.normal(0.001, 0.02, size=(40, 4)),
                        index=dates, columns=["A", "B", "C", "D"])
    sectors = pd.Series({"A": "S1", "B": "S1", "C": "S2", "D": "S2"})
    rebs = [dates[0], dates[20]]
    wbp = {dates[0]: pd.Series({"A": 0.4, "B": 0.1, "C": 0.4, "D": 0.1}),
           dates[20]: pd.Series({"A": 0.1, "B": 0.4, "C": 0.1, "D": 0.4})}
    out = brinson_attribution(wbp, rebs, rets, sectors)
    assert out["max_reconciliation_error"] < 1e-9      # BHB identity holds
    t = out["total"]
    assert abs((t["allocation"] + t["selection"] + t["interaction"]) - t["active"]) < 1e-9


def test_timing_effect_positive_when_overlay_dodges_losses():
    from risk.attribution import timing_effect
    core = pd.Series([0.01, -0.05, -0.05, 0.01], index=pd.bdate_range("2020-01-01", periods=4))
    managed = pd.Series([0.01, 0.0, 0.0, 0.01], index=core.index)  # sat out the losses
    te = timing_effect(managed, core)
    assert te["timing_contrib"] > 0
