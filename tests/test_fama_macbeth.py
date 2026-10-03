import numpy as np
import pandas as pd

from models.factor_model import BETA_COLUMNS
from models.fama_macbeth import newey_west_se, run_monthly_cross_sections, _attach_prior_month_betas


def test_newey_west_se_close_to_naive_for_iid_data():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    naive_se = x.std(ddof=1) / np.sqrt(len(x))
    nw_se = newey_west_se(x, lags=4)
    # For genuinely i.i.d. data, HAC and naive SE should be close (autocovariance
    # terms are just sampling noise around zero).
    assert 0.8 < nw_se / naive_se < 1.25


def test_newey_west_se_larger_for_positively_autocorrelated_data():
    rng = np.random.default_rng(1)
    n = 2000
    phi = 0.7
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1)

    naive_se = x.std(ddof=1) / np.sqrt(n)
    nw_se = newey_west_se(x, lags=4)
    # Positive serial correlation means the naive iid formula understates
    # the true variance of the sample mean -- NW should correct upward.
    assert nw_se > naive_se


def test_newey_west_se_nan_for_too_few_points():
    assert np.isnan(newey_west_se(np.array([1.0])))


def _build_synthetic_panel(true_premium, n_tickers=40, n_months=60, noise_std=0.005, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    true_betas = {c: rng.normal(0, 1, n_tickers) for c in BETA_COLUMNS}

    rows = []
    for m in range(n_months):
        month = pd.Timestamp("2015-01-01") + pd.DateOffset(months=m)
        for i, ticker in enumerate(tickers):
            signal = sum(true_premium[c] * true_betas[c][i] for c in BETA_COLUMNS)
            ret = signal + rng.normal(0, noise_std)
            row = {"ticker": ticker, "month": month, "monthly_return": ret}
            row.update({c: true_betas[c][i] for c in BETA_COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_fama_macbeth_recovers_known_premium_with_significant_t_stat():
    true_premium = {"beta_market": 0.01, "beta_momentum": 0.0, "beta_liquidity": 0.0,
                     "beta_egp_sensitivity": 0.0, "beta_oil_sensitivity": 0.0}
    merged = _build_synthetic_panel(true_premium)

    monthly_gammas = run_monthly_cross_sections(merged)
    assert len(monthly_gammas) == 60  # every month has 40 >= MIN_TICKERS_PER_MONTH tickers

    market_series = monthly_gammas["beta_market"].to_numpy()
    premium = market_series.mean()
    se = newey_west_se(market_series, lags=4)
    t_stat = premium / se

    assert abs(premium - 0.01) < 0.003
    assert t_stat > 3  # a real, engineered signal should be clearly significant

    # A factor with a true zero premium should have a much smaller point estimate.
    mom_premium = monthly_gammas["beta_momentum"].mean()
    assert abs(mom_premium) < abs(premium)


def test_fama_macbeth_skips_months_with_too_few_tickers():
    true_premium = {c: 0.0 for c in BETA_COLUMNS}
    merged = _build_synthetic_panel(true_premium, n_tickers=5, n_months=10)  # below MIN_TICKERS_PER_MONTH=10
    monthly_gammas = run_monthly_cross_sections(merged)
    assert monthly_gammas.empty


def test_attach_prior_month_betas_uses_only_information_known_before_the_month():
    # Two tickers, betas that change over time; a monthly return should be
    # paired with the beta estimate dated BEFORE that month started, never
    # a beta dated from later in (or after) the month itself.
    exposures = pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B"],
        "date": pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-20", "2024-01-10", "2024-02-05"]),
        "beta_market": [1.0, 1.5, 2.0, 0.5, 0.6],
        "beta_momentum": [0.0] * 5, "beta_liquidity": [0.0] * 5,
        "beta_egp_sensitivity": [0.0] * 5, "beta_oil_sensitivity": [0.0] * 5,
    })
    monthly_returns = pd.DataFrame({
        "ticker": ["A", "B"],
        "month": pd.to_datetime(["2024-02-01", "2024-02-01"]),
        "monthly_return": [0.03, -0.01],
    })

    merged = _attach_prior_month_betas(monthly_returns, exposures)

    row_a = merged[merged["ticker"] == "A"].iloc[0]
    # Ticker A's February return must use the 2024-01-31 beta (1.5), NOT
    # the 2024-02-20 beta (2.0) which wasn't known yet at the start of Feb.
    assert row_a["beta_market"] == 1.5

    row_b = merged[merged["ticker"] == "B"].iloc[0]
    # Ticker B has no exposure dated on/before 2024-02-01 as-of the month
    # start other than 2024-01-10 -- must use that, not leave it NaN just
    # because a later (2024-02-05) row also exists.
    assert row_b["beta_market"] == 0.5
