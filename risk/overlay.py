"""
Regime de-risking overlay + drawdown circuit breaker + Kelly sizing
(Layer 8.3), from scratch.

This is the productionized version of the edge the backtest actually found:
the core book (min-variance) is left to do the cross-sectional allocation, and
a TIME-SERIES exposure multiplier e_t in [0, 1] scales the whole book up and
down based on point-in-time regime and running drawdown. Realized return is

    r_managed_t = e_{t-1} * r_core_t

-- the exposure is decided using information available at the close of day
t-1 (regime as of t-1, drawdown through t-1), then applied to day t's return.
No look-ahead. Cash (the un-invested fraction) earns 0 here (a CBE/T-bill
rate can be wired in later).

Exposure is the product of three multipliers:
  * regime:   e_regime = bear_scale in BEAR, else 1  (the dominant lever).
  * breaker:  a hard circuit breaker -- if running drawdown breaches
              breaker_dd (e.g. -20%), force exposure to 0 until the drawdown
              RECOVERS above reentry_dd (e.g. -10%). The explicit re-entry
              rule is essential: a breaker with no re-entry is just "sell once
              and never come back", which would sit in cash through every
              subsequent recovery.
  * kelly:    optional half-Kelly / drawdown scaling (see kelly_fraction).
"""

import numpy as np
import pandas as pd


def kelly_fraction(trailing_returns: np.ndarray, cap: float = 1.0) -> float:
    """Half-Kelly exposure for a single return stream: f* = mean/variance,
    f = 0.5 f*, clipped to [0, cap]. EDGE ESTIMATOR (stated explicitly, per the
    house rule): mean and variance are the trailing SAMPLE moments of the
    strategy's own daily returns -- so this scales exposure up when the recent
    risk-adjusted return has been strong and down when it's been weak/negative.
    Long-only, so negative Kelly clamps to 0 (go to cash), never short.
    """
    r = np.asarray(trailing_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return cap
    var = r.var()
    if var <= 0:
        return cap
    f = 0.5 * (r.mean() / var) * (1.0 / 252)  # scale the daily mean/var to a per-day fraction
    return float(min(max(f, 0.0), cap))


def volatility_target_exposure(trailing_returns: np.ndarray, target_vol: float = 0.15,
                               lookback: int = 20, cap: float = 1.0,
                               floor: float = 0.0, ppy: int = 252) -> float:
    """Continuous exposure that targets a constant annualized volatility --
    the REPLACEMENT for the whipsawing hard drawdown breaker (Tier-2 #6).

    e_vol = clip(target_vol / realized_vol_trailing, floor, cap).

    WHY THIS INSTEAD OF THE BREAKER. The hard breaker is a step function: it
    slams exposure 1->0 the instant drawdown crosses a line, then 0->1 on
    re-entry, so on a choppy series it toggles constantly, each toggle a costly
    round trip into a falling market (the C5 gate: +824% -> -90%, ~1185
    toggles). Vol-targeting is CONTINUOUS: as realized vol rises, exposure eases
    down smoothly and comes back smoothly, so it de-risks in turbulence without
    the on/off whipsaw. Realized vol is the trailing SAMPLE std of the core's
    own returns (stated as the estimator, per the house rule), annualized.

    Causal: only returns through t-1 enter the trailing window; the result
    scales day t. Long-only, so exposure never goes negative.
    """
    r = np.asarray(trailing_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < max(5, lookback // 2):
        return cap
    realized = r[-lookback:].std(ddof=1) * np.sqrt(ppy)
    if realized <= 0:
        return cap
    return float(min(max(target_vol / realized, floor), cap))


def apply_overlay(core_returns: pd.Series, regime: pd.Series, bear_scale: float = 0.0,
                  breaker_dd: float = -1.0, reentry_dd: float = -0.10,
                  breaker_min_cash_days: int = 20,
                  use_kelly: bool = False, kelly_lookback: int = 126,
                  use_vol_target: bool = False, target_vol: float = 0.15,
                  vol_lookback: int = 20, vol_floor: float = 0.0,
                  cost_per_turnover: float = 0.0,
                  cash_daily: pd.Series = None) -> dict:
    """The drawdown circuit breaker is DISABLED by default (breaker_dd=-1.0
    never trips). The C5 gate showed a naive breaker whipsaws badly on this
    data -- tripping at -20% and re-entering at -10% toggled ~1185 times, each
    a costly round trip into a falling market, turning a +824% path into -90%.
    `breaker_min_cash_days` (a minimum cash lock-out once tripped) dampens that
    whipsaw for anyone who does enable it; the regime overlay alone already
    cuts drawdown more effectively (-43% -> -25%) with only ~38 trades, so on
    this data the breaker is redundant. Left in, guarded and off by default."""
    """Apply the regime + circuit-breaker (+ optional Kelly) overlay to a core
    daily-return series. `regime` is a per-date label series (BULL/BEAR/...).

    Returns {'managed_returns', 'exposure', 'breaker_active'} where exposure_t
    is the multiplier that was applied to core return at t (decided at t-1).
    """
    core = core_returns.dropna()
    reg = regime.reindex(core.index).ffill()
    # Return earned on the UN-INVESTED fraction. Passing None keeps the old
    # zero-yield assumption; passing a series makes "go to cash" earn the
    # prevailing rate, which is what actually happens in an Egyptian account.
    cash_vals = (np.zeros(len(core)) if cash_daily is None
                 else pd.Series(cash_daily).reindex(core.index).ffill().fillna(0.0).to_numpy())

    managed = np.zeros(len(core))
    exposure = np.zeros(len(core))
    breaker_flags = np.zeros(len(core), dtype=bool)

    managed_nav, managed_peak = 1.0, 1.0
    core_nav, core_peak = 1.0, 1.0   # the underlying, tracked even while in cash
    breaker_on = False
    days_in_cash = 0
    core_vals = core.to_numpy()

    for t in range(len(core)):
        # ---- decide exposure using info through t-1 ----
        managed_dd = managed_nav / managed_peak - 1.0
        core_dd = core_nav / core_peak - 1.0
        # Trip on the MANAGED drawdown (protect realized capital); re-enter when
        # the CORE/underlying has recovered above reentry_dd AND a minimum cash
        # lock-out has elapsed. Re-entry keys off the core, not the managed
        # series -- once in cash the managed NAV is flat, so its own drawdown
        # could never recover and the breaker would latch forever. The lock-out
        # stops the -20%/-10% band from toggling on every wiggle.
        if breaker_on:
            days_in_cash += 1
            if core_dd >= reentry_dd and days_in_cash >= breaker_min_cash_days:
                breaker_on = False
        elif managed_dd <= breaker_dd:
            breaker_on = True
            days_in_cash = 0

        e = 0.0 if breaker_on else 1.0
        reg_prev = reg.iloc[t - 1] if t > 0 else reg.iloc[0]  # regime as of t-1
        if e > 0 and reg_prev == "BEAR":
            e = bear_scale
        if e > 0 and use_kelly and t >= kelly_lookback:
            e *= kelly_fraction(core_vals[t - kelly_lookback:t])
        # vol-target multiplier (breaker replacement): scale by target/realized
        # vol using only returns through t-1. Smoothly de-risks in turbulence.
        if e > 0 and use_vol_target and t >= 1:
            e *= volatility_target_exposure(core_vals[:t], target_vol=target_vol,
                                            lookback=vol_lookback, floor=vol_floor)

        exposure[t] = e
        breaker_flags[t] = breaker_on
        # ---- apply to day t's core return, net of exposure-change trading ----
        # changing exposure (going to cash / re-investing) trades |Δe| of the
        # book, which costs; charge it so the overlay's whipsaw isn't free.
        prev_e = exposure[t - 1] if t > 0 else 0.0
        turn_cost = abs(e - prev_e) * cost_per_turnover
        # invested fraction earns the book; the UN-invested fraction earns the
        # cash rate (zero only if no cash_daily series was supplied)
        r_t = e * core_vals[t] + (1.0 - e) * cash_vals[t] - turn_cost
        managed[t] = r_t
        managed_nav *= (1 + r_t)
        managed_peak = max(managed_peak, managed_nav)
        core_nav *= (1 + core_vals[t])
        core_peak = max(core_peak, core_nav)

    idx = core.index
    return {"managed_returns": pd.Series(managed, index=idx),
            "exposure": pd.Series(exposure, index=idx),
            "breaker_active": pd.Series(breaker_flags, index=idx)}
