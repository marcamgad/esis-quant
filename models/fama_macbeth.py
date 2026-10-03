"""
Fama-MacBeth (1973) two-step cross-sectional regression, implemented from
scratch (no statsmodels). Step 1 (time-series betas) is
models/factor_model.py; this module does step 2 (monthly cross-sectional
regressions of returns on betas) and step 3 (averaging the monthly
premia + Newey-West HAC standard errors).
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection
from data.linalg import ols_fit
from models.factor_model import BETA_COLUMNS
from models.regime_features import _load_price_panel

logger = logging.getLogger("esis_quant.fama_macbeth")

NEWEY_WEST_LAGS = 4
MIN_TICKERS_PER_MONTH = 10


def newey_west_se(x, lags: int = NEWEY_WEST_LAGS) -> float:
    """Newey-West (1987) HAC standard error of the sample MEAN of a time
    series x (here, a series of monthly cross-sectional regression
    coefficients for one factor).

    Long-run variance: S = gamma_0 + 2 * sum_{l=1}^{lags} w_l * gamma_l,
    w_l = 1 - l/(lags+1) (Bartlett kernel), gamma_l = the lag-l sample
    autocovariance of the demeaned series. SE(mean) = sqrt(S / T).

    With lags=0 this reduces to the ordinary (non-HAC) standard error of
    the mean, sqrt(Var(x)/T) -- used as a sanity check in tests.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 2:
        return np.nan
    demeaned = x - x.mean()
    gamma_0 = np.sum(demeaned * demeaned) / T
    S = gamma_0
    for lag in range(1, min(lags, T - 1) + 1):
        gamma_l = np.sum(demeaned[lag:] * demeaned[:-lag]) / T
        weight = 1 - lag / (lags + 1)
        S += 2 * weight * gamma_l
    return float(np.sqrt(max(S, 0.0) / T))


def load_factor_exposures(tickers=None) -> pd.DataFrame:
    with get_connection() as conn:
        query = "SELECT ticker, date, " + ", ".join(BETA_COLUMNS) + " FROM factor_exposures"
        params = ()
        if tickers:
            placeholders = ",".join(["?"] * len(tickers))
            query += f" WHERE ticker IN ({placeholders})"
            params = tuple(tickers)
        df = pd.read_sql(query, conn, params=params, parse_dates=["date"])
    return df


def compute_monthly_returns(tickers=None) -> pd.DataFrame:
    """Realized return within each calendar month, per ticker -- the
    dependent variable for that month's cross-sectional regression.
    Long format: columns ticker, month, monthly_return.
    """
    panel = _load_price_panel(tickers)
    rows = []
    for ticker, df in panel.items():
        close = df["close"].dropna()
        if close.empty:
            continue
        # Month-over-month return = last close of month M vs last close of
        # month M-1 (pct_change on the month-end series). The previous
        # first-to-last-within-month version silently dropped the first
        # trading day's return every month and forced any single-trading-day
        # month to exactly 0 (MATH-BUG-01). Label each return at the month
        # START (period "MS") so it aligns with the cross-section indexing.
        month_end_close = close.resample("ME").last()
        monthly_return = month_end_close.pct_change()
        monthly_return.index = monthly_return.index.to_period("M").to_timestamp(how="start")
        for month, ret in monthly_return.items():
            if pd.notna(ret):
                rows.append({"ticker": ticker, "month": month, "monthly_return": ret})
    return pd.DataFrame(rows)


def _attach_prior_month_betas(monthly_returns: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    """For each (ticker, month) return, attach the most recent beta
    estimate dated on or before the start of that month -- i.e. only
    information already known when the month began, no lookahead into
    the return we're trying to explain.
    """
    # merge_asof requires the "on" column sorted globally (not just within
    # each "by" group), even though matching itself happens per-group.
    #
    # Match the return earned DURING month M (labeled at M's start) to the
    # most recent beta dated strictly BEFORE month M begins -- i.e. as of the
    # last trading day of month M-1. Matching on `month` itself (the 1st of
    # M) could pull a beta dated on that same day, whose estimate already
    # embeds month M's opening-day return, leaking contemporaneous
    # information into a "predetermined" exposure (STAT-LEAK-04). Subtracting
    # one day pushes the as-of match into the prior month.
    exposures_sorted = exposures.sort_values("date")
    returns_sorted = monthly_returns.sort_values("month").copy()
    returns_sorted["match_date"] = returns_sorted["month"] - pd.Timedelta(days=1)
    merged = pd.merge_asof(
        returns_sorted, exposures_sorted,
        left_on="match_date", right_on="date", by="ticker", direction="backward",
    )
    return merged.dropna(subset=BETA_COLUMNS, how="all")


def run_monthly_cross_sections(merged: pd.DataFrame) -> pd.DataFrame:
    """One OLS per month: monthly_return_i = gamma_0 + sum_k gamma_k * beta_ik.
    Returns a DataFrame indexed by month with columns
    ['intercept'] + BETA_COLUMNS + ['n_tickers'], one row per month that
    had enough tickers to regress.
    """
    rows = []
    for month, group in merged.groupby("month"):
        valid = group.dropna(subset=BETA_COLUMNS + ["monthly_return"])
        if len(valid) < MIN_TICKERS_PER_MONTH:
            continue
        y = valid["monthly_return"].to_numpy()
        X = np.column_stack([np.ones(len(valid))] + [valid[c].to_numpy() for c in BETA_COLUMNS])
        gamma = ols_fit(X, y)
        rows.append({"month": month, "intercept": gamma[0],
                     **{c: gamma[i + 1] for i, c in enumerate(BETA_COLUMNS)},
                     "n_tickers": len(valid)})
    return pd.DataFrame(rows).set_index("month") if rows else pd.DataFrame()


def run_fama_macbeth(tickers=None) -> dict:
    """End-to-end Fama-MacBeth: monthly cross-sectional regressions, then
    average premia with Newey-West (4-lag) t-stats.

    Returns {'premia': DataFrame[factor, premium, se, t_stat, n_months],
    'monthly': DataFrame of the monthly gamma_kt series}.
    """
    exposures = load_factor_exposures(tickers)
    if exposures.empty:
        raise ValueError("No factor_exposures rows found -- run models.factor_model.compute_factor_exposures first")

    monthly_returns = compute_monthly_returns(tickers)
    merged = _attach_prior_month_betas(monthly_returns, exposures)
    monthly_gammas = run_monthly_cross_sections(merged)

    if monthly_gammas.empty:
        raise ValueError(
            f"No month had >= {MIN_TICKERS_PER_MONTH} tickers with both valid betas and a return -- "
            "cannot run cross-sectional regressions"
        )

    factor_cols = ["intercept"] + BETA_COLUMNS
    premia_rows = []
    for col in factor_cols:
        series = monthly_gammas[col].dropna()
        premium = float(series.mean())
        se = newey_west_se(series.to_numpy(), lags=NEWEY_WEST_LAGS)
        t_stat = premium / se if se > 0 else np.nan
        premia_rows.append({"factor": col, "premium": premium, "se": se, "t_stat": t_stat, "n_months": len(series)})

    premia_df = pd.DataFrame(premia_rows).set_index("factor")
    logger.info("Fama-MacBeth: %d months, premia:\n%s", len(monthly_gammas), premia_df.to_string())

    return {"premia": premia_df, "monthly": monthly_gammas}
