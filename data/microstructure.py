"""
Market microstructure features, computed manually with NumPy/Pandas
(no statsmodels/sklearn). All functions take plain pandas Series so they
can be tested against synthetic data independent of the database.
"""

import logging

import numpy as np
import pandas as pd

from data.linalg import rolling_ols_beta

logger = logging.getLogger("esis_quant.microstructure")


class MicrostructureAnalyzer:
    @staticmethod
    def amihud_illiquidity(returns: pd.Series, value_traded_egp: pd.Series, window: int = 20) -> pd.Series:
        """ILLIQ_t = (1/window) * sum(|R_i| / Volume_i) over the trailing
        window, i.e. a rolling mean of |return| / EGP value traded per day.
        High ILLIQ = large price impact per unit traded = illiquid.

        returns, value_traded_egp: aligned pandas Series indexed by date.
        Days with zero/near-zero traded value are excluded from the mean
        (undefined price impact, not zero impact) rather than producing inf.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            daily_impact = (returns.abs() / value_traded_egp).replace([np.inf, -np.inf], np.nan)
        return daily_impact.rolling(window, min_periods=max(2, window // 2)).mean()

    @staticmethod
    def roll_spread_estimator(prices: pd.Series, window: int = 20, normalize: bool = True) -> pd.Series:
        """Roll (1984) implied spread: 2 * sqrt(max(-Cov(dP_t, dP_{t-1}), 0)).
        Negative serial covariance in price changes implies a bid-ask
        bounce; where covariance is non-negative the estimator is
        undefined and returned as NaN (not clamped to 0) so callers can
        distinguish "no signal" from "zero spread".

        The raw Roll spread is in PRICE (EGP) units, so an 80-EGP stock
        looks far less liquid than a 5-EGP stock purely from price level.
        With normalize=True (default) the spread is divided by the rolling
        mean price to yield a unitless fractional (%) spread that is
        comparable cross-sectionally (MATH-BUG-02). Pass normalize=False for
        the raw currency spread.
        """
        dP = prices.diff()
        dP_lag = dP.shift(1)

        cov = dP.rolling(window).cov(dP_lag)
        spread_sq = (-cov).clip(lower=0.0)
        spread = 2 * np.sqrt(spread_sq)
        spread[cov >= 0] = np.nan
        if normalize:
            mean_price = prices.rolling(window).mean()
            spread = spread / mean_price.replace(0.0, np.nan)
        return spread

    @staticmethod
    def kyle_lambda(price_changes: pd.Series, signed_volume: pd.Series, window: int = 60) -> pd.Series:
        """Kyle's (1985) lambda: price impact coefficient from
        dP_t = lambda * signed_volume_t + eps_t, estimated by rolling OLS
        (manual, via data.linalg.rolling_ols_beta) with an intercept.

        signed_volume should already be volume * sign(return) -- computing
        that from raw volume/returns is the caller's responsibility so this
        function stays a pure regression.
        Returns a Series of lambda estimates, NaN where a window has too
        few valid observations to regress.
        """
        return rolling_ols_beta(price_changes, signed_volume, window)

    @staticmethod
    def realized_variance_components(stock_returns: pd.Series, market_returns: pd.Series, window: int = 60):
        """Decompose rolling total variance into systematic (beta^2 *
        market variance) and idiosyncratic (total - systematic) components,
        via a rolling OLS beta (manual, via data.linalg.rolling_ols_beta)
        of stock on market returns.

        Returns a DataFrame with columns: beta, total_var, systematic_var,
        idiosyncratic_var (all rolling, annualization left to the caller).
        """
        total_var = stock_returns.rolling(window).var()
        market_var = market_returns.rolling(window).var()
        beta = rolling_ols_beta(stock_returns, market_returns, window)

        systematic_var = (beta ** 2) * market_var
        idiosyncratic_var = (total_var - systematic_var).clip(lower=0.0)

        return pd.DataFrame(
            {
                "beta": beta,
                "total_var": total_var,
                "systematic_var": systematic_var,
                "idiosyncratic_var": idiosyncratic_var,
            }
        )
