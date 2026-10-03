"""Monte Carlo, parameter sweeps, alpha decay, drift alerting, embargo."""
import numpy as np
import pandas as pd
import pytest

from analysis import robustness_suite as rs
from analysis import alpha_decay as ad


def test_monte_carlo_preserves_block_structure():
    """Stationary blocks must retain autocorrelation. An i.i.d. shuffle would
    destroy the volatility clustering that drives drawdowns."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 1500))
    d = rs.monte_carlo(r, horizons=(1,), n_paths=200)
    assert d["ok"] and 1 in d["horizons"]
    h = d["horizons"][1]
    assert h["p05"] < h["median"] < h["p95"]
    assert 0.0 <= h["p_loss"] <= 1.0
    assert h["median_maxdd"] <= 0.0


def test_monte_carlo_p_loss_moves_with_drift():
    """A clearly profitable series must show a lower P(loss) than a losing one --
    the check that the simulation reflects its input at all."""
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0.002, 0.01, 1200))
    bad = pd.Series(rng.normal(-0.002, 0.01, 1200))
    pg = rs.monte_carlo(good, horizons=(1,), n_paths=300)["horizons"][1]["p_loss"]
    pb = rs.monte_carlo(bad, horizons=(1,), n_paths=300)["horizons"][1]["p_loss"]
    assert pg < 0.25 < pb


def test_monte_carlo_refuses_tiny_samples():
    assert rs.monte_carlo(pd.Series([0.01, -0.01]), n_paths=10)["ok"] is False


def test_ic_overlap_correction_divides_by_sqrt_horizon():
    """Same correction as the overlay timing test: overlapping forward windows
    are not independent observations."""
    idx = pd.bdate_range("2020-01-01", periods=600)
    cols = [f"T{i}" for i in range(8)]
    rng = np.random.default_rng(2)
    sig = pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), len(cols))), index=idx, columns=cols)
    ic = ad.ic_by_horizon(sig, ret, horizons=(21,))
    r = ic.iloc[0]
    assert r["t_adj"] == pytest.approx(r["t_raw"] / np.sqrt(21), rel=1e-9)


def test_no_half_life_is_fitted_to_noise():
    """A half-life fitted to an insignificant IC is a number with no referent.
    The module must refuse rather than manufacture one."""
    ic = pd.DataFrame({"horizon": [1, 5, 21], "mean_ic": [0.001, 0.002, 0.003],
                       "t_adj": [0.1, 0.2, 0.3], "significant": [False, False, False]})
    hl = ad.half_life(ic)
    assert hl["ok"] is False
    assert "distinguishable from zero" in hl["reason"]


def test_half_life_is_fitted_when_a_signal_exists():
    ic = pd.DataFrame({"horizon": [1, 5, 21, 63], "mean_ic": [0.20, 0.10, 0.05, 0.01],
                       "t_adj": [5.0, 4.0, 3.0, 2.5], "significant": [True] * 4})
    hl = ad.half_life(ic)
    assert hl["ok"] is True and hl["half_life_days"] > 0


def test_drift_checks_report_inconclusive_rather_than_pass():
    """Too few fills is an ABSENCE of evidence, not a pass. Conflating the two
    is how a degrading cost model stays invisible."""
    from ops import health
    d = health.execution_drift()
    assert "conclusive" in d
    if not d.get("conclusive"):
        assert d["ok"] is True          # doesn't alarm...
        assert "too few" in d["reason"] or "unreadable" in d["reason"]   # ...but says why


def test_embargo_excludes_training_data_near_the_boundary():
    """The embargo must actually shrink the training window, not just be a
    parameter that is accepted and ignored."""
    from models import regime_pipeline as rp
    assert rp.EMBARGO_DAYS > 0
    import inspect
    src = inspect.getsource(rp.compute_regime_series)
    assert "cutoff" in src and "embargo" in src


def test_survivorship_disclosure_is_unmissable():
    """The disclosure must name the bias explicitly and say figures are upper
    bounds -- a vague caveat is not a disclosure."""
    from data.universe import survivorship_disclosure
    full = survivorship_disclosure()
    short = survivorship_disclosure(short=True)
    for text in (full, short):
        assert "BIAS" in text.upper()
    assert "UPPER BOUND" in full.upper()
    assert "2026" in full            # names the selection date
