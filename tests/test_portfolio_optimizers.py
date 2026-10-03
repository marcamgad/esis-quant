"""
Known-answer + edge tests for the from-scratch portfolio optimizers
(covariance shrinkage, mean-variance, risk-parity). Each has a case where
the correct weights are derivable by symmetry.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio.covariance import ledoit_wolf_shrinkage, project_capped_simplex, sample_covariance
from portfolio.mv_optimizer import mean_variance_weights
from portfolio.risk_parity import risk_parity_weights


# ---------------- covariance / projection ----------------

def test_ledoit_wolf_is_pd_even_when_sample_is_singular():
    # T < N -> sample covariance is singular; shrinkage must fix it.
    rng = np.random.default_rng(0)
    R = rng.normal(size=(5, 12))  # 5 obs, 12 assets
    lw = ledoit_wolf_shrinkage(R)
    assert 0.0 <= lw["delta"] <= 1.0
    eig = np.linalg.eigvalsh(lw["sigma"])
    assert eig.min() > 0  # strictly positive-definite


def test_ledoit_wolf_shrinks_more_with_less_data():
    rng = np.random.default_rng(1)
    big = ledoit_wolf_shrinkage(rng.normal(size=(2000, 8)))
    small = ledoit_wolf_shrinkage(rng.normal(size=(15, 8)))
    assert small["delta"] > big["delta"]  # scarce data -> shrink harder


def test_project_capped_simplex_sums_to_one_and_respects_cap():
    v = np.array([0.9, 0.05, 0.02, 0.5, -0.3])
    w = project_capped_simplex(v, cap=0.3, total=1.0)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.max() <= 0.3 + 1e-9
    assert w.min() >= -1e-12


def test_project_capped_simplex_infeasible_raises():
    with pytest.raises(ValueError):
        project_capped_simplex(np.ones(4), cap=0.2, total=1.0)  # 4*0.2=0.8 < 1


# ---------------- mean-variance ----------------

def test_mv_equal_mu_identity_cov_gives_equal_weights():
    # Known answer: equal expected returns + scaled-identity covariance ->
    # the return term is constant on the simplex, so MV minimizes w'w ->
    # equal weights (interior, below the cap).
    n = 6
    mu = np.full(n, 0.01)
    sigma = 0.04 * np.eye(n)
    w = mean_variance_weights(mu, sigma, risk_aversion=5.0, max_weight=0.5)["weights"]
    assert np.allclose(w, 1.0 / n, atol=1e-4)


def test_mv_tilts_toward_higher_return_asset():
    n = 4
    mu = np.array([0.05, 0.01, 0.01, 0.01])
    sigma = 0.04 * np.eye(n)
    w = mean_variance_weights(mu, sigma, risk_aversion=1.0, max_weight=1.0)["weights"]
    assert w[0] == pytest.approx(w.max())      # most weight on the high-return asset
    assert w[0] > 1.0 / n


def test_mv_respects_max_weight_cap():
    n = 4
    mu = np.array([1.0, 0.0, 0.0, 0.0])         # wants everything in asset 0
    sigma = 0.01 * np.eye(n)
    w = mean_variance_weights(mu, sigma, risk_aversion=0.1, max_weight=0.4)["weights"]
    assert w[0] <= 0.4 + 1e-9
    assert abs(w.sum() - 1.0) < 1e-9


# ---------------- risk parity ----------------

def test_risk_parity_equal_correlation_gives_equal_weights():
    # Known answer: equal variances + equal pairwise correlation -> by
    # symmetry every asset has the same risk contribution at equal weights.
    n, rho, var = 5, 0.3, 0.04
    sigma = var * ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))
    w = risk_parity_weights(sigma)["weights"]
    assert np.allclose(w, 1.0 / n, atol=1e-6)


def test_risk_parity_equalizes_risk_contributions():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 8))
    sigma = A @ A.T / 8 + 0.01 * np.eye(8)     # random PD covariance
    res = risk_parity_weights(sigma)
    rc = res["risk_contrib"]
    assert np.allclose(rc, 1.0 / 8, atol=1e-4)  # all risk contributions equal
    assert abs(res["weights"].sum() - 1.0) < 1e-9
    assert res["weights"].min() > 0             # long-only


def test_risk_parity_underweights_the_volatile_asset():
    # One high-variance asset should get LESS weight than the low-variance ones.
    sigma = np.diag([0.01, 0.01, 0.25])
    w = risk_parity_weights(sigma)["weights"]
    assert w[2] < w[0] and w[2] < w[1]


# ---------------- CVaR ----------------

def test_cvar_single_scenario_maximizes_that_scenario_return():
    # Known answer: with T=1 the min-CVaR objective reduces to zeta = -r_1'w,
    # so minimizing it MAXIMIZES the single scenario's return -> load the
    # highest-return asset up to the weight cap.
    from portfolio.cvar_optimizer import min_cvar_weights
    scenarios = np.array([[0.05, 0.01, -0.02, 0.03]])  # asset 0 is best
    w = min_cvar_weights(scenarios, alpha=0.95, max_weight=0.4)["weights"]
    assert w[0] == pytest.approx(0.4, abs=1e-3)         # filled to the cap
    assert w[2] == pytest.approx(0.0, abs=1e-3)         # worst asset gets nothing


def test_cvar_symmetric_assets_equal_weight():
    # Two assets with mirror-image return scenarios -> symmetric problem ->
    # equal weights minimize tail risk.
    from portfolio.cvar_optimizer import min_cvar_weights
    rng = np.random.default_rng(5)
    g = rng.normal(0, 0.02, 100)
    h = rng.normal(0, 0.02, 100)
    # column-swap maps asset0<->asset1 AND the scenario set onto itself, so the
    # problem is exactly symmetric under swap -> equal weight is optimal.
    scenarios = np.column_stack([np.concatenate([g, h]), np.concatenate([h, g])])
    w = min_cvar_weights(scenarios, alpha=0.90, max_weight=1.0)["weights"]
    assert np.allclose(w, [0.5, 0.5], atol=0.02)


def test_cvar_avoids_the_tail_risky_asset():
    # asset 1 has occasional deep crashes; min-CVaR should underweight it.
    from portfolio.cvar_optimizer import min_cvar_weights
    rng = np.random.default_rng(6)
    safe = rng.normal(0.0005, 0.01, 300)
    risky = rng.normal(0.0005, 0.01, 300)
    risky[::20] = -0.25   # periodic crashes (fat left tail)
    scenarios = np.column_stack([safe, risky])
    w = min_cvar_weights(scenarios, alpha=0.95, max_weight=1.0)["weights"]
    assert w[0] > w[1]


# ---------------- ensemble + composite ----------------

def test_ensemble_iid_equal_variance_gives_near_equal_weights():
    # Known answer: uncorrelated, equal-variance assets -> every optimizer
    # (min-var, risk-parity, min-CVaR) returns ~equal weights, so the blend does too.
    from portfolio.ensemble import ensemble_weights
    rng = np.random.default_rng(11)
    R = rng.normal(0, 0.02, size=(400, 5))
    w = ensemble_weights(R, tickers=[f"A{i}" for i in range(5)], max_weight=0.5)["weights"]
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w.values, 0.2, atol=0.03)


def test_ensemble_respects_position_and_sector_caps():
    from portfolio.ensemble import ensemble_weights
    rng = np.random.default_rng(12)
    # 4 names, 2 sectors; give sector S1 much lower vol so optimizers pile in
    R = np.column_stack([rng.normal(0, 0.005, 500), rng.normal(0, 0.005, 500),
                         rng.normal(0, 0.05, 500), rng.normal(0, 0.05, 500)])
    tickers = ["A", "B", "C", "D"]
    sectors = pd.Series({"A": "S1", "B": "S1", "C": "S2", "D": "S2"})
    w = ensemble_weights(R, tickers, sectors=sectors, max_weight=0.4, max_sector=0.6)["weights"]
    assert w.max() <= 0.4 + 1e-6                         # position cap
    assert w[["A", "B"]].sum() <= 0.6 + 1e-6             # sector cap on S1
