"""
Walk-forward long-portfolio backtest, from scratch.

Rules (deliberately conservative -- the point is an honest number, not a
flattering one):
  * NO LOOK-AHEAD. A signal observed at the close of rebalance date T sets
    the portfolio that earns returns from T+1 onward. Positions are held
    until the next rebalance. Prices are ADJUSTED close (split/dividend).
  * Long-only, top-fraction selection (EGX shorting is impractical). Each
    rebalance: rank the cross-section by the signal, take the top
    `top_fraction`, weight them (equal or signal-proportional), require at
    least `min_names` to hold (else stay in cash that period).
  * Transaction costs charged on TURNOVER at each rebalance: cost =
    sum_i |w_new,i - w_drifted,i| * cost_per_turnover. With
    the REAL fee schedule (backtest/fees.py): a flat EGP charge per ORDER
    plus a percentage plus date-aware stamp duty, charged against
    config.BACKTEST_DEFAULT_AUM_EGP unless aum_egp is given. This is ON BY
    DEFAULT as of the 2026-08-05 audit -- it previously defaulted OFF and was
    never enabled anywhere, so every headline number was charged a flat 0.55%
    against a measured 1.5% (10k book) to 8.2% (1k book).
  * TRADING CALENDAR: prices are restricted to genuine EGX sessions via
    data.calendar (audit fix CAL-01) -- stray non-session prints otherwise
    manufacture holiday-gap returns.
  * UNIVERSE COMPLETENESS: eligibility uses a COVERAGE floor, not
    zero-NaN (audit fix UNIV-01). See run_portfolio.

Returns a dict with the daily net-return Series, the gross series, NAV,
per-rebalance turnover, and the realized weights, ready for analytics.py.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.backtest")

# Legacy flat fallback, used ONLY when a caller explicitly opts out of the fee
# schedule (use_fee_schedule=False). Derived from the real schedule at the
# default book size rather than hardcoded, so there is still one source of truth.
def _default_cost_per_turnover() -> float:
    from backtest.fees import book_pct_per_side
    return book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)

# A name is eligible at rebalance T if it printed on at least this fraction of
# the lookback window. The previous rule demanded ZERO NaNs, so one missing
# print anywhere in 180 days disqualified a ticker -- which, combined with the
# stray-session dates CAL-01 now removes, collapsed the universe to 2 of 26
# names for ~5 months of 2024. A 90% floor still rejects genuinely broken
# series while tolerating the odd halt/missing print.
MIN_WINDOW_COVERAGE = 0.90


def load_signal_scores(score_col: str, tickers=None) -> pd.DataFrame:
    """Long (ticker, date, score) frame for one signals-table column."""
    with get_connection() as conn:
        q = f"SELECT ticker, date, {score_col} AS score FROM signals WHERE {score_col} IS NOT NULL"
        params = ()
        if tickers:
            q += " AND ticker IN (" + ",".join(["?"] * len(tickers)) + ")"
            params = tuple(tickers)
        df = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    return df


def load_price_returns(tickers=None, use_calendar: bool = True) -> pd.DataFrame:
    """Wide (date x ticker) daily returns from ADJUSTED close, restricted to
    genuine trading sessions.

    `use_calendar=True` drops stray non-session dates BEFORE differencing, so
    returns never span a holiday gap. Returns are computed after the filter,
    which is the correct order: filtering afterwards would leave the bogus
    return already baked in.
    """
    with get_connection() as conn:
        q = "SELECT ticker, date, adjusted_close FROM daily_ohlcv"
        params = ()
        if tickers:
            q += " WHERE ticker IN (" + ",".join(["?"] * len(tickers)) + ")"
            params = tuple(tickers)
        px = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    wide = px.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    if use_calendar:
        from data.calendar import valid_trading_days
        good = valid_trading_days(wide)
        wide = wide.loc[wide.index.isin(good)]
    # Splits/bonuses two sources agree on but the vendor never adjusted
    # (HDBK 2:1, EFID 1.5:1 ...). Self-disabling: see data/ca_factors.py.
    from data.ca_factors import adjust_panel
    wide = adjust_panel(wide)
    return wide.pct_change()


def load_fill_ratio(tickers=None) -> pd.DataFrame:
    """Typical price / close per (date, ticker): the in-session fill proxy.

    WHY NOT THE OPEN. EGX opens are not real: in 97% of rows (Yahoo and S&P
    alike) the stored open equals the previous close. A "T+1 open fill" would
    therefore be the old close-fill assumption under another name. The typical
    price (H+L+C)/3 of the fill day is the standard VWAP proxy; against 16 real
    broker fills (Aug 2026) it tracked actual shortfall with correlation 0.97,
    +1.77% vs +2.34% actual, every fill inside the day's range.

    Unit-free, so it is valid on the adjusted series. Days with no range
    (high == low, or no volume) get 1.0: no information, no adjustment.
    """
    with get_connection() as conn:
        q = "SELECT ticker, date, high, low, close, volume FROM daily_ohlcv WHERE close > 0"
        params = ()
        if tickers:
            q += " AND ticker IN (" + ",".join(["?"] * len(tickers)) + ")"
            params = tuple(tickers)
        px = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    ok = (px["high"] > px["low"]) & (px["volume"].fillna(0) > 0) & px["high"].notna() & px["low"].notna()
    px["ratio"] = np.where(ok, (px["high"] + px["low"] + px["close"]) / (3 * px["close"]), 1.0)
    return px.pivot_table(index="date", columns="ticker", values="ratio", aggfunc="last").sort_index()


def fill_adjustment(dw: pd.Series, hold: pd.DatetimeIndex, returns_wide: pd.DataFrame,
                    ratio: pd.DataFrame, lag: int = 1, slippage: float = 0.0) -> pd.Series:
    """Per-day return correction for executing a rebalance AFTER the signal.

    The default engine assumes the new book is held from the signal close.
    Here the trade fills at the typical price `lag` sessions later:
      * before the fill day the OLD book is still held: correction -sum(dw * r)
      * on the fill day, bought weight earns close/typical - 1 instead of the
        full-day return; sold weight earns typical/prev_close - 1 instead of 0
      * `slippage` is charged on traded weight on top (one side each)
    Only TRADED weight is affected; an unchanged holding is untouched.
    """
    out = pd.Series(0.0, index=hold)
    if len(hold) == 0 or dw.abs().sum() == 0:
        return out
    lag = max(1, int(lag))
    names = dw.index
    for k in range(min(lag - 1, len(hold))):
        r = returns_wide.loc[hold[k]].reindex(names).fillna(0.0)
        out.iloc[k] = -float((dw * r).sum())
    if lag - 1 >= len(hold):
        return out
    d = hold[lag - 1]
    r = returns_wide.loc[d].reindex(names).fillna(0.0)
    tpr = ratio.reindex(index=[d], columns=names).iloc[0].fillna(1.0) if d in ratio.index \
        else pd.Series(1.0, index=names)
    buy, sell = dw.clip(lower=0), (-dw).clip(lower=0)
    out.iloc[lag - 1] = float((buy * ((1.0 / tpr - 1.0) - r)).sum() + (sell * (tpr * (1.0 + r) - 1.0)).sum()
                              - slippage * dw.abs().sum())
    return out


def load_value_traded(tickers=None) -> pd.DataFrame:
    """Wide (date x ticker) daily value-traded in EGP -- the liquidity panel the
    tiered cost model needs to turn a weight change into a participation rate."""
    with get_connection() as conn:
        q = "SELECT ticker, date, value_traded_egp FROM daily_ohlcv"
        params = ()
        if tickers:
            q += " WHERE ticker IN (" + ",".join(["?"] * len(tickers)) + ")"
            params = tuple(tickers)
        vt = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    if vt.empty:
        return pd.DataFrame()
    return vt.pivot(index="date", columns="ticker", values="value_traded_egp").sort_index()


class BacktestEngine:
    def __init__(self, rebalance_days: int = 21, top_fraction: float = 0.30,
                 weighting: str = "equal", cost_per_turnover: float = None,
                 min_names: int = 5, max_weight: float = 0.15, tradable_only: bool = True,
                 min_window_coverage: float = MIN_WINDOW_COVERAGE,
                 aum_egp: float = None, adv_lookback: int = None,
                 use_fee_schedule: bool = True, fill: str = "close", fill_lag: int = 1,
                 fill_slippage: float = 0.0):
        # EXECUTION MODEL (2026-10-03). "close": the book changes at the signal
        # close (the historical assumption, kept as the default so every past
        # result stays reproducible). "typical": trades fill at the typical price
        # `fill_lag` sessions after the signal -- what manual execution actually
        # gets (see load_fill_ratio). The live account's fills came 2 sessions
        # after the signal and ~0.57% above the typical price.
        if fill not in ("close", "typical"):
            raise ValueError("fill must be 'close' or 'typical'")
        self.fill, self.fill_lag, self.fill_slippage = fill, int(fill_lag), float(fill_slippage)
        self._fill_ratio = None
        self.rebalance_days = rebalance_days
        self.top_fraction = top_fraction
        self.weighting = weighting
        self.cost_per_turnover = (_default_cost_per_turnover() if cost_per_turnover is None
                                  else cost_per_turnover)
        self.min_names = min_names
        self.max_weight = max_weight
        # filter the investable set through the single Phase-A3 tradability
        # definition so the backtest can't hold dead/illiquid names (the ORAS
        # problem) that no other layer would trade either.
        self.tradable_only = tradable_only
        self.min_window_coverage = min_window_coverage
        # LIQUIDITY-AWARE COSTS (Tier-2 #7 wired into the engine). When aum_egp
        # is set, each rebalance's cost is computed per-name from how big the
        # trade is relative to that name's trailing average value traded (ADV),
        # via the square-root market-impact model in backtest.costs -- instead
        # of the flat cost_per_turnover. Cost therefore rises with BOTH the
        # capital deployed and the illiquidity of the names traded, which is the
        # whole point: the flat 0.55% flatters exactly the trades that hurt most
        # at size. aum_egp=None keeps the old flat behaviour (backward compatible).
        # A flat EGP-per-order fee is meaningless without a capital base, so the
        # schedule path needs an AUM. Defaulting it (rather than silently falling
        # back to the percentage-only model when it is absent) is what makes the
        # real schedule the DEFAULT instead of dead code: the 2026-08-05 audit
        # found use_fee_schedule was never once set True in the whole repo, so
        # every headline number was charged ~0.55% against a measured 1.5-8.2%.
        # Charge the real broker/stamp schedule (flat EGP per order + pct +
        # date-aware stamp duty) instead of a pure percentage.
        self.use_fee_schedule = bool(use_fee_schedule)
        # Default the AUM only when the schedule is ON. Defaulting it
        # unconditionally silently broke the documented "aum_egp=None -> flat
        # cost_per_turnover" contract, so an explicit cost-free run
        # (cost_per_turnover=0.0, use_fee_schedule=False) started paying
        # impact costs it never asked for.
        self.aum_egp = (aum_egp if aum_egp is not None
                        else (config.BACKTEST_DEFAULT_AUM_EGP if self.use_fee_schedule else None))
        self.adv_lookback = adv_lookback or config.COST_ADV_LOOKBACK_DAYS
        self._value_traded = None   # lazily loaded liquidity panel

    def _target_weights(self, scores: pd.Series) -> pd.Series:
        """Top-fraction long weights from a cross-section of scores (index =
        tickers). Long-only: only positive-ranked names, capped at max_weight.
        """
        s = scores.dropna()
        if len(s) < self.min_names:
            return pd.Series(dtype=float)
        k = max(self.min_names, int(np.ceil(len(s) * self.top_fraction)))
        top = s.sort_values(ascending=False).head(k)
        if self.weighting == "score":
            w = top - top.min() + 1e-9  # shift so the lowest selected name still gets weight
            w = w / w.sum()
        else:
            w = pd.Series(1.0 / len(top), index=top.index)
        w = w.clip(upper=self.max_weight)
        return w / w.sum()

    def run(self, scores_df: pd.DataFrame, returns_wide: pd.DataFrame) -> dict:
        """Simulate the portfolio. scores_df: long (ticker,date,score).
        returns_wide: (date x ticker) daily returns. Returns net/gross daily
        return Series (indexed by date), NAV, turnover per rebalance, weights.
        """
        if self.tradable_only:
            from data.universe import tradable_mask
            scores_df = tradable_mask(scores_df)
        score_panel = scores_df.pivot(index="date", columns="ticker", values="score").sort_index()
        # only trade on dates we have both a signal and forward prices for
        common_dates = returns_wide.index
        score_dates = score_panel.index[score_panel.index.isin(common_dates)]
        if len(score_dates) == 0:
            raise ValueError("backtest: no overlapping signal/price dates")

        rebal_dates = list(score_dates[:: self.rebalance_days])
        all_dates = returns_wide.index
        net_ret = pd.Series(0.0, index=all_dates)
        gross_ret = pd.Series(0.0, index=all_dates)
        turnovers, weights_hist = {}, {}

        prev_w = pd.Series(dtype=float)
        for i, T in enumerate(rebal_dates):
            # weights decided at close of T using only info known at T
            target = self._target_weights(score_panel.loc[T])
            # cost charged at T on turnover vs the (drifted-to-cash-simplified) previous book
            all_names = prev_w.index.union(target.index)
            dw = target.reindex(all_names, fill_value=0.0) - prev_w.reindex(all_names, fill_value=0.0)
            turnover = float(dw.abs().sum())
            cost = self._turnover_cost(dw, T)
            turnovers[T] = turnover
            weights_hist[T] = target

            # holding window: T+1 .. next rebalance date (inclusive)
            end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
            window = all_dates[(all_dates > T) & (all_dates <= end)]
            if len(window) == 0:
                prev_w = target
                continue

            held = returns_wide.loc[window, target.index] if len(target) else pd.DataFrame(index=window)
            # daily portfolio gross return = sum_i w_i * r_i (missing -> 0, treated as cash)
            daily = held.mul(target, axis=1).sum(axis=1, min_count=1).fillna(0.0) if len(target) else pd.Series(0.0, index=window)
            if self.fill == "typical":
                daily = daily + self._execution(dw, window, returns_wide).values
            gross_ret.loc[window] = daily.values
            net = daily.copy()
            net.iloc[0] = net.iloc[0] - cost  # charge the rebalance cost on the first held day
            net_ret.loc[window] = net.values
            prev_w = target

        # trim to the traded span
        first = rebal_dates[0]
        mask = (all_dates > first)
        net_ret, gross_ret = net_ret[mask], gross_ret[mask]
        nav = (1 + net_ret).cumprod()
        logger.info("backtest: %d rebalances, avg turnover=%.2f, final NAV=%.3f",
                    len(rebal_dates), np.mean(list(turnovers.values())) if turnovers else 0.0,
                    float(nav.iloc[-1]) if len(nav) else float("nan"))
        return {"net_returns": net_ret, "gross_returns": gross_ret, "nav": nav,
                "turnover": turnovers, "weights": weights_hist, "rebalance_dates": rebal_dates}

    def _execution(self, dw: pd.Series, hold, returns_wide: pd.DataFrame) -> pd.Series:
        if self._fill_ratio is None:
            self._fill_ratio = load_fill_ratio(list(returns_wide.columns))
        return fill_adjustment(dw, hold, returns_wide, self._fill_ratio,
                               lag=self.fill_lag, slippage=self.fill_slippage)

    def _schedule_cost(self, dw: pd.Series, T) -> float:
        """Rebalance cost from the REAL fee schedule (backtest/fees.py): a flat
        EGP charge plus a percentage on every order, plus date-aware stamp duty.

        This is charged PER ORDER, which the old percentage-only model could not
        express. It matters most exactly where this book lives: at small AUM a
        rebalance touching 13 names pays the flat fee 13 times, so cost scales
        with the NUMBER of positions, not just turnover. Requires aum_egp to be
        set -- without a capital figure a flat EGP fee has no meaning.
        """
        from backtest.fees import one_side_cost
        aum = float(self.aum_egp)
        total = 0.0
        for name, d in dw.items():
            notional = abs(float(d)) * aum
            if notional <= 0:
                continue
            total += one_side_cost(notional, as_of=T, same_day=False,
                                   spread_per_side=0.0)["total_egp"]
        return total / aum if aum > 0 else 0.0

    def _turnover_cost(self, dw: pd.Series, T) -> float:
        """Cost of a rebalance as a fraction of NAV. Flat when aum_egp is None;
        otherwise per-name and size-aware: for each traded name the EGP notional
        is |dw_i| * aum, its participation is notional / ADV_i(T) (ADV = trailing
        mean value-traded through T -- known at T, no look-ahead), and the
        one-way cost is commission + slippage(participation). Total = sum_i
        |dw_i| * one_way_i, so a full round trip of illiquid names costs more
        than of liquid ones."""
        gross = dw.abs()
        if self.use_fee_schedule:
            # Market impact (size/liquidity) PLUS the explicit fee schedule.
            # _impact_only must charge NO commission here or commission is
            # counted twice -- once inside trade_cost_fraction and again in
            # _schedule_cost's broker leg. That double-count would make the
            # corrected numbers unfairly punitive, which is just as dishonest
            # as the flattering error it replaces.
            return self._impact_only(gross, T, commission=0.0) + self._schedule_cost(dw, T)
        if self.aum_egp is None:
            return float(gross.sum()) * self.cost_per_turnover
        return self._impact_only(gross, T)

    def _impact_only(self, gross: pd.Series, T=None, commission: float = None) -> float:
        """Size/liquidity market impact only -- no broker fees. Split out so the
        fee-schedule path can add impact and explicit fees separately."""
        from backtest.costs import trade_cost_fraction
        if self._value_traded is None:
            self._value_traded = load_value_traded()
        vt = self._value_traded
        cost = 0.0
        for name, dwi in gross.items():
            if dwi <= 0:
                continue
            adv = float("nan")
            if not vt.empty and name in vt.columns:
                col = vt[name]
                col = col[col.index <= T].tail(self.adv_lookback)
                col = col[col > 0]
                if len(col):
                    adv = float(col.mean())
            one_way = trade_cost_fraction(float(dwi) * self.aum_egp, adv,
                                          commission=commission)["one_way"]
            cost += float(dwi) * one_way
        return cost

    def _eligible_names(self, window: pd.DataFrame, T, trad) -> list:
        """Names eligible at rebalance T: printed on >= min_window_coverage of
        the lookback AND tradable on T (audit fix UNIV-01).

        The old rule was `window.columns[window.notna().all()]` -- zero
        tolerance for a single missing print across 180 days. Combined with the
        stray-session dates CAL-01 removes, it wiped the universe down to 2
        names for months. Coverage-based eligibility keeps a name whose series
        is essentially intact, and the remaining gaps are filled with 0 (no
        move) ONLY for the covariance estimate, never for realized P&L.
        """
        cov = window.notna().mean()
        cols = list(cov.index[cov >= self.min_window_coverage])
        if trad is not None:
            allowed = trad.get(T, set())
            cols = [c for c in cols if c in allowed]
        return cols

    def run_portfolio(self, weight_fn, returns_wide: pd.DataFrame, lookback: int = 252,
                      tradable=None, universe_log: list = None) -> dict:
        """Backtest a full-weight strategy: at each rebalance date T, call
        weight_fn(trailing_returns_df) -> pd.Series of target weights (indexed
        by ticker). The trailing frame is the last `lookback` sessions up to and
        INCLUDING T, for names passing `_eligible_names` (coverage + tradable).
        Weights earn from T+1; costs charged on turnover. Same no-lookahead/cost
        model as run().

        `universe_log`, if a list is passed, receives one dict per rebalance
        recording the universe size actually used -- so a silently-shrinking
        universe is visible in the audit trail instead of hiding inside a
        plausible-looking return series.
        """
        all_dates = returns_wide.index
        rebal_dates = list(all_dates[lookback::self.rebalance_days])
        net_ret = pd.Series(0.0, index=all_dates)
        gross_ret = pd.Series(0.0, index=all_dates)
        turnovers, weights_hist = {}, {}
        prev_w = pd.Series(dtype=float)

        trad = None
        if tradable is not None and not tradable.empty:
            trad = tradable[tradable["is_tradable"] == 1].groupby("date")["ticker"].apply(set)

        skipped = 0
        for i, T in enumerate(rebal_dates):
            window = returns_wide.loc[:T].tail(lookback)
            cols = self._eligible_names(window, T, trad)
            if universe_log is not None:
                universe_log.append({"date": T, "n_eligible": len(cols),
                                     "n_tradable": len(trad.get(T, set())) if trad is not None else None,
                                     "n_in_panel": int(window.shape[1])})
            if len(cols) < self.min_names:
                skipped += 1
                prev_w = pd.Series(dtype=float)
                continue
            # fill residual gaps with 0 for the ESTIMATOR only (a missing print
            # is "no observed move"); realized P&L below never uses filled data.
            target = weight_fn(window[cols].fillna(0.0))
            target = target[target > 1e-6]
            if target.empty:
                prev_w = pd.Series(dtype=float)
                continue
            target = target / target.sum()
            weights_hist[T] = target

            names = prev_w.index.union(target.index)
            dw = target.reindex(names, fill_value=0.0) - prev_w.reindex(names, fill_value=0.0)
            cost = self._turnover_cost(dw, T)
            turnovers[T] = float(dw.abs().sum())

            end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else all_dates[-1]
            hold = all_dates[(all_dates > T) & (all_dates <= end)]
            if len(hold) == 0:
                prev_w = target
                continue
            held = returns_wide.loc[hold, target.index]
            daily = held.mul(target, axis=1).sum(axis=1, min_count=1).fillna(0.0)
            if self.fill == "typical":
                daily = daily + self._execution(dw, hold, returns_wide).values
            gross_ret.loc[hold] = daily.values
            net = daily.copy()
            net.iloc[0] -= cost
            net_ret.loc[hold] = net.values
            prev_w = target

        if not rebal_dates:
            raise ValueError("run_portfolio: no rebalance dates (lookback too long for the data)")
        if skipped:
            logger.warning("run_portfolio: %d/%d rebalances SKIPPED (fewer than %d eligible names) -- "
                           "the book sat in cash on those dates", skipped, len(rebal_dates), self.min_names)
        mask = all_dates > rebal_dates[0]
        net_ret, gross_ret = net_ret[mask], gross_ret[mask]
        return {"net_returns": net_ret, "gross_returns": gross_ret, "nav": (1 + net_ret).cumprod(),
                "turnover": turnovers, "weights": weights_hist, "rebalance_dates": rebal_dates,
                "n_skipped_rebalances": skipped}


def equal_weight_benchmark(returns_wide: pd.DataFrame, start=None) -> pd.Series:
    """Equal-weight-all-names daily return.

    NOT the market (2026-10-04): since 2016 it returned ~33-35%/yr against the
    real EGX30's ~21% (data/market.py). Daily rebalancing of thinly traded names
    books bid-ask bounce as return, and the universe holds only survivors. Use
    it only to compare strategies built from the SAME universe; use
    data.market.returns() for "the market".
    """
    r = returns_wide.copy()
    if start is not None:
        r = r[r.index > start]
    return r.mean(axis=1, skipna=True)
