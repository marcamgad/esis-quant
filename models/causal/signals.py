"""
Layer 5 signal construction (spec 5.3): assemble a real (ticker, date)
panel from the database, estimate the causal effect of sentiment (Double
ML) and of momentum (2SLS IV) on 5-day forward returns while controlling
for factor/macro confounders, then build the per-stock causal_signal:

    causal_sentiment_i = theta_DML * sentiment_score_i
    causal_momentum_i  = theta_IV  * momentum_score_i
    causal_signal_i    = 0.5 * causal_sentiment_i + 0.5 * causal_momentum_i   (spec)

and cross-sectionally z-score it. The signal isolates the component of
return genuinely attributable to sentiment/momentum, net of confounding by
factor exposures and macro state.

HONESTY NOTE ON DATA: daily aggregated sentiment (daily_sentiment) is empty
until Layer 4's pipeline has run against a populated news_articles table
(currently ~10 real articles). When sentiment is unavailable the DML step
reports theta_sentiment = 0.0 with an `insufficient` flag and the causal
signal is carried entirely by the momentum-IV component -- an explicit,
logged degradation, not a silent fabrication. Momentum, forward returns and
all confounders come from real, densely-populated tables (daily_ohlcv,
factor_exposures, macro_daily), so the momentum-IV estimate is real.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection
from models.causal.double_ml import double_ml, naive_ols_effect
from models.causal.iv_estimator import two_stage_least_squares

logger = logging.getLogger("esis_quant.causal.signals")

# Confounders used to de-bias the treatment effects. beta_momentum is
# deliberately excluded from the momentum equations (it is the momentum-
# factor loading, mechanically related to the momentum characteristic we
# treat as D); the remaining betas + macro deltas are legitimate controls.
CONFOUNDER_BETAS = ["beta_market", "beta_liquidity", "beta_egp_sensitivity",
                    "beta_oil_sensitivity", "idiosyncratic_vol"]


def _load_prices(tickers):
    with get_connection() as conn:
        placeholders = ",".join(["?"] * len(tickers))
        # adjusted_close aliased to close: momentum and forward returns must
        # be computed on the split/dividend-adjusted series (see SYS-CRIT-01).
        df = pd.read_sql(
            f"SELECT ticker, date, adjusted_close AS close FROM daily_ohlcv WHERE ticker IN ({placeholders}) "
            "ORDER BY ticker, date",
            conn, params=tuple(tickers), parse_dates=["date"],
        )
    return df


def _load_factor_exposures(tickers):
    cols = ", ".join(CONFOUNDER_BETAS)
    with get_connection() as conn:
        placeholders = ",".join(["?"] * len(tickers))
        df = pd.read_sql(
            f"SELECT ticker, date, {cols} FROM factor_exposures WHERE ticker IN ({placeholders})",
            conn, params=tuple(tickers), parse_dates=["date"],
        )
    return df


def _load_macro():
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT date, usd_egp, brent_oil_usd FROM macro_daily ORDER BY date",
            conn, parse_dates=["date"],
        )
    df["usd_egp_chg"] = df["usd_egp"].pct_change(20)
    df["brent_chg"] = df["brent_oil_usd"].pct_change(20)
    return df[["date", "usd_egp_chg", "brent_chg"]]


def _load_sentiment(tickers):
    """Daily aggregated sentiment (Layer 4). Empty DataFrame if none yet."""
    with get_connection() as conn:
        placeholders = ",".join(["?"] * len(tickers))
        df = pd.read_sql(
            f"SELECT ticker, date, sentiment_score FROM daily_sentiment WHERE ticker IN ({placeholders})",
            conn, params=tuple(tickers), parse_dates=["date"],
        )
    return df


def _load_regime():
    with get_connection() as conn:
        df = pd.read_sql("SELECT date, regime FROM regime_history ORDER BY date", conn, parse_dates=["date"])
    return df


def _compute_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: 5-day forward return (target Y) and 12-1 month momentum
    characteristic (Jegadeesh-Titman: cumulative return from t-252 to t-21).
    """
    h = config.CAUSAL_FORWARD_RETURN_HORIZON
    lb = config.CAUSAL_MOMENTUM_LOOKBACK_DAYS
    skip = config.CAUSAL_MOMENTUM_SKIP_DAYS
    out = []
    for ticker, df in prices.groupby("ticker"):
        df = df.sort_values("date").reset_index(drop=True)
        close = df["close"]
        fwd_ret = close.shift(-h) / close - 1
        # momentum known AT date t: return over [t-lb, t-skip]
        momentum = close.shift(skip) / close.shift(lb) - 1
        out.append(pd.DataFrame({
            "ticker": ticker, "date": df["date"],
            "fwd_ret": fwd_ret.values, "momentum": momentum.values,
        }))
    return pd.concat(out, ignore_index=True)


def build_causal_panel(tickers=None) -> pd.DataFrame:
    """Merge price features, factor exposures, macro deltas, sentiment and
    regime into one long (ticker, date) panel. Adds the cross-sectional
    leave-one-out mean momentum and a lagged-momentum-x-BULL instrument used
    by the momentum IV. Returns rows sorted by (date, ticker).
    """
    tickers = tickers or [t["ticker"] for t in config.EGX_SEED_TICKERS]
    prices = _load_prices(tickers)
    if prices.empty:
        raise ValueError("build_causal_panel: no price data -- run Layer 1 collectors first")

    panel = _compute_price_features(prices)
    panel = panel.merge(_load_factor_exposures(tickers), on=["ticker", "date"], how="left")
    panel = panel.merge(_load_macro(), on="date", how="left")

    sent = _load_sentiment(tickers)
    if not sent.empty:
        panel = panel.merge(sent, on=["ticker", "date"], how="left")
    else:
        panel["sentiment_score"] = np.nan

    regime = _load_regime()
    panel = panel.merge(regime, on="date", how="left")

    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # --- Instruments for the momentum 2SLS (both strictly PREDETERMINED) ---
    # Z1: the PRIOR, NON-OVERLAPPING formation-window momentum. The current
    #     momentum characteristic D is the return over [t-(skip+lookback),
    #     t-skip]; the instrument is the return over the immediately
    #     preceding, disjoint window [t-(skip+2*lookback), t-(skip+lookback)]
    #     -- i.e. "last year's momentum". Two design points, both fixing the
    #     original spec's flaws:
    #       * it is the stock's OWN past return, not the cross-sectional mean
    #         momentum (the market factor), so it does not affect individual
    #         returns through a market channel (fixes MODEL-SPEC-01's
    #         exclusion violation), and
    #       * its window does not overlap D's, so it is not the near-tautology
    #         a small lag of D would be (a 5-day lag of a 12-month momentum is
    #         ~97% the same series, giving a meaninglessly huge first-stage F).
    #     Relevance rests on momentum persistence across adjacent years;
    #     exclusion on last year's momentum affecting the t->t+5 return only
    #     through the current momentum characteristic -- an assumption, stated
    #     plainly, but far cleaner than either original instrument.
    # prior-window momentum = momentum shifted by one full lookback window
    # (momentum itself already encodes the [skip, skip+lookback] window, so
    # shifting it back by `lookback` trading days yields the prior, disjoint
    # window's momentum without recomputing from prices).
    lb = config.CAUSAL_MOMENTUM_LOOKBACK_DAYS
    panel["mom_prior"] = panel.groupby("ticker")["momentum"].shift(lb)
    # Z2: prior-window momentum interacted with a LAGGED BULL-regime dummy
    #     (regime as of t - CAUSAL_IV_LAG_DAYS, not t). The prior version
    #     interacted with the CONTEMPORANEOUS regime, which the HMM builds
    #     from day-t market returns -- an endogenous instrument
    #     (MODEL-SPEC-02). Lagging the dummy makes it predetermined while
    #     preserving the spec-5.2 "lagged x regime dummy" instrument form.
    panel["regime_lag"] = panel.groupby("ticker")["regime"].shift(config.CAUSAL_IV_LAG_DAYS)
    panel["bull_dummy_lag"] = (panel["regime_lag"] == "BULL").astype(float)
    panel["mom_prior_x_bull"] = panel["mom_prior"] * panel["bull_dummy_lag"]

    return panel


def _confounder_matrix(df: pd.DataFrame, extra_cols=None) -> np.ndarray:
    cols = list(CONFOUNDER_BETAS) + ["usd_egp_chg", "brent_chg"] + list(extra_cols or [])
    return df[cols].to_numpy(dtype=float), cols


def _panel_purge(df: pd.DataFrame) -> int:
    """Cross-fitting purge width (in stacked-panel rows) that spans the
    forward-return horizon in CALENDAR terms. The DML input is a panel
    sorted by (date, ticker), so one calendar day is ~median-cross-section
    rows; purging only `horizon` flat rows would cover a fraction of a day
    and leave the overlapping-return leak intact. Scaling by the typical
    daily width makes the purge remove roughly `horizon` days of neighbours
    around each validation block (STAT-LEAK-03, panel form).
    """
    daily = df.groupby("date").size()
    width = int(daily.median()) if len(daily) else 1
    return max(1, width) * config.CAUSAL_FORWARD_RETURN_HORIZON


def run_causal_estimation(panel: pd.DataFrame) -> dict:
    """Run the two causal estimators on the panel:

      * DML for the sentiment effect (theta_DML) on 5-day forward returns,
        controlling for factor + macro confounders. Falls back to
        theta=0 / insufficient when sentiment data is unavailable.
      * 2SLS IV for the momentum effect (theta_IV), instrumenting the
        momentum characteristic with cross-sectional LOO mean momentum and
        lagged-momentum-x-BULL, controlling for the same confounders.

    Returns a diagnostics dict consumed by construct_causal_signal and the
    Layer 5 charts.
    """
    results = {}

    # ---- DML: sentiment effect ----
    sent_cols = CONFOUNDER_BETAS + ["usd_egp_chg", "brent_chg"]
    sent_df = panel.dropna(subset=["fwd_ret", "sentiment_score"] + sent_cols)
    if len(sent_df) >= config.CAUSAL_MIN_OBS:
        X, xcols = _confounder_matrix(sent_df)
        dml = double_ml(sent_df["fwd_ret"].to_numpy(), sent_df["sentiment_score"].to_numpy(), X,
                        purge=_panel_purge(sent_df))
        dml["regime"] = sent_df["regime"].to_numpy()
        results["dml_sentiment"] = dml
        logger.info("DML sentiment: theta=%.5f se=%.5f t=%.2f (n=%d)",
                    dml["theta"], dml["se"], dml["t_stat"], dml["n"])
    else:
        results["dml_sentiment"] = {"theta": 0.0, "se": np.nan, "t_stat": np.nan,
                                    "n": len(sent_df), "insufficient": True,
                                    "naive": {"theta": 0.0, "se": np.nan, "t_stat": np.nan}}
        logger.warning("DML sentiment: only %d rows with sentiment -- theta forced to 0.0", len(sent_df))

    # ---- DML: momentum effect (real-data causal estimate, for comparison) ----
    mom_cols = CONFOUNDER_BETAS + ["usd_egp_chg", "brent_chg"]
    mom_df = panel.dropna(subset=["fwd_ret", "momentum"] + mom_cols)
    if len(mom_df) >= config.CAUSAL_MIN_OBS:
        Xm, _ = _confounder_matrix(mom_df)
        dml_m = double_ml(mom_df["fwd_ret"].to_numpy(), mom_df["momentum"].to_numpy(), Xm,
                          purge=_panel_purge(mom_df))
        dml_m["regime"] = mom_df["regime"].to_numpy()
        results["dml_momentum"] = dml_m
        logger.info("DML momentum: theta=%.5f se=%.5f t=%.2f (n=%d)",
                    dml_m["theta"], dml_m["se"], dml_m["t_stat"], dml_m["n"])

    # ---- 2SLS IV: momentum effect ----
    iv_df = panel.dropna(subset=["fwd_ret", "momentum", "mom_prior", "mom_prior_x_bull"] + mom_cols)
    if len(iv_df) >= config.CAUSAL_MIN_OBS:
        Xiv, _ = _confounder_matrix(iv_df)
        instrument_names = ["mom_prior", "mom_prior_x_bull"]
        Z = iv_df[instrument_names].to_numpy(dtype=float)
        iv = two_stage_least_squares(iv_df["fwd_ret"].to_numpy(), iv_df["momentum"].to_numpy(), Z, Xiv)
        # per-instrument first-stage F (each instrument alone, controls included),
        # for the "instrument strength" chart
        from models.causal.iv_estimator import first_stage_f
        ones = np.ones((len(iv_df), 1))
        iv["per_instrument_f"] = {
            name: first_stage_f(iv_df["momentum"].to_numpy(), Z[:, [j]],
                                np.column_stack([ones, Xiv]))["f_stat"]
            for j, name in enumerate(instrument_names)
        }
        results["iv_momentum"] = iv
        logger.info("2SLS momentum: theta=%.5f se=%.5f t=%.2f first-stage F=%.2f (n=%d)",
                    iv["theta"], iv["se"], iv["t_stat"], iv["first_stage_f"]["f_stat"], iv["n"])
    else:
        results["iv_momentum"] = {"theta": 0.0, "se": np.nan, "t_stat": np.nan,
                                  "first_stage_f": {"f_stat": np.nan, "weak": True},
                                  "sargan": {"statistic": np.nan, "df": 0}, "n": len(iv_df),
                                  "insufficient": True}
        logger.warning("2SLS momentum: only %d usable rows -- theta forced to 0.0", len(iv_df))

    return results


def construct_causal_signal(panel: pd.DataFrame, results: dict) -> pd.DataFrame:
    """Build per-(ticker, date) causal_signal from the estimated effects and
    the stock characteristics, then cross-sectionally z-score each day.
    Returns columns: ticker, date, causal_sentiment, causal_momentum,
    causal_signal_raw, causal_signal (z-scored).
    """
    theta_sent = results.get("dml_sentiment", {}).get("theta", 0.0) or 0.0
    theta_iv = results.get("iv_momentum", {}).get("theta", 0.0) or 0.0
    w_s = config.CAUSAL_SIGNAL_SENTIMENT_WEIGHT
    w_m = config.CAUSAL_SIGNAL_MOMENTUM_WEIGHT

    df = panel.copy()
    df["causal_sentiment"] = theta_sent * df["sentiment_score"].fillna(0.0)
    df["causal_momentum"] = theta_iv * df["momentum"]
    df["causal_signal_raw"] = w_s * df["causal_sentiment"] + w_m * df["causal_momentum"]

    out = df.dropna(subset=["causal_signal_raw"]).copy()
    # cross-sectional z-score within each date
    g = out.groupby("date")["causal_signal_raw"]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    out["causal_signal"] = ((out["causal_signal_raw"] - mean) / std).fillna(0.0)
    return out[["ticker", "date", "causal_sentiment", "causal_momentum",
                "causal_signal_raw", "causal_signal"]]


def _fit_period_thetas(train_panel: pd.DataFrame):
    """Estimate (theta_sent, theta_iv) from a training subset only. Fast:
    the sentiment DML short-circuits to 0.0 when there aren't enough dated
    sentiment rows (the current state everywhere), and the momentum IV is a
    closed-form 2SLS. theta_iv is None when momentum can't be identified on
    the subset (handled by the caller's fallback).
    """
    conf = CONFOUNDER_BETAS + ["usd_egp_chg", "brent_chg"]
    theta_sent = 0.0
    sent_df = train_panel.dropna(subset=["fwd_ret", "sentiment_score"] + conf)
    if len(sent_df) >= config.CAUSAL_MIN_OBS:
        X, _ = _confounder_matrix(sent_df)
        theta_sent = double_ml(sent_df["fwd_ret"].to_numpy(), sent_df["sentiment_score"].to_numpy(),
                               X, purge=_panel_purge(sent_df))["theta"]
    theta_iv = None
    iv_df = train_panel.dropna(subset=["fwd_ret", "momentum", "mom_prior", "mom_prior_x_bull"] + conf)
    if len(iv_df) >= config.CAUSAL_MIN_OBS:
        Xiv, _ = _confounder_matrix(iv_df)
        Z = iv_df[["mom_prior", "mom_prior_x_bull"]].to_numpy(dtype=float)
        theta_iv = two_stage_least_squares(iv_df["fwd_ret"].to_numpy(), iv_df["momentum"].to_numpy(), Z, Xiv)["theta"]
    return theta_sent, theta_iv


def construct_causal_signal_walkforward(panel: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward causal signal (Phase 1 leak fix): the causal effect theta
    applied to rows in calendar year Y is estimated using ONLY data strictly
    before Y (expanding window), never the pooled 2016-2026 panel. This
    removes the look-ahead of a single global theta -- a 2018 row's signal can
    no longer borrow a coefficient partly fit on 2024-2026 (the TFT's own test
    span).

    HONEST NOTE (verified): while theta_sent == 0 (no historical sentiment
    density), causal_signal reduces to the cross-sectional z-score of momentum,
    which is INVARIANT to the positive scalar theta_iv -- so this fix currently
    produces byte-identical values to the pooled version and did not affect any
    past result. It matters only once sentiment coverage makes theta_sent
    nonzero and the sentiment/momentum mix ratio starts to depend on theta.
    """
    panel = panel.sort_values("date").copy()
    years = sorted(pd.DatetimeIndex(panel["date"]).year.unique())
    theta_by_year = {yr: _fit_period_thetas(panel[panel["date"] < pd.Timestamp(f"{yr}-01-01")])
                     for yr in years}
    # fallback theta_iv for early years with too little prior history; it only
    # rescales the momentum term, which the per-date z-score cancels, so any
    # positive constant is harmless (no look-ahead introduced).
    fallback_iv = next((ti for _, (ts, ti) in sorted(theta_by_year.items()) if ti is not None), 1.0)
    row_year = pd.DatetimeIndex(panel["date"]).year
    ts_row = np.array([theta_by_year[y][0] for y in row_year])
    ti_row = np.array([theta_by_year[y][1] if theta_by_year[y][1] is not None else fallback_iv for y in row_year])

    w_s, w_m = config.CAUSAL_SIGNAL_SENTIMENT_WEIGHT, config.CAUSAL_SIGNAL_MOMENTUM_WEIGHT
    panel["causal_sentiment"] = ts_row * panel["sentiment_score"].fillna(0.0).to_numpy()
    panel["causal_momentum"] = ti_row * panel["momentum"].to_numpy()
    panel["causal_signal_raw"] = w_s * panel["causal_sentiment"] + w_m * panel["causal_momentum"]

    out = panel.dropna(subset=["causal_signal_raw"]).copy()
    g = out.groupby("date")["causal_signal_raw"]
    std = g.transform("std").replace(0.0, np.nan)
    out["causal_signal"] = ((out["causal_signal_raw"] - g.transform("mean")) / std).fillna(0.0)
    logger.info("construct_causal_signal_walkforward: %d rows, per-year thetas=%s",
                len(out), {y: (round(ts, 4), round(ti, 4) if ti is not None else None)
                           for y, (ts, ti) in theta_by_year.items()})
    return out[["ticker", "date", "causal_sentiment", "causal_momentum", "causal_signal_raw", "causal_signal"]]


def store_causal_signals(signal_df: pd.DataFrame) -> int:
    """Upsert causal_signal into the signals table (PK ticker, date),
    preserving any existing alpha_score/regime on matching rows. Returns the
    number of rows written.
    """
    rows = [
        (r["ticker"], r["date"].strftime("%Y-%m-%d"), float(r["causal_signal"]))
        for _, r in signal_df.iterrows()
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO signals (ticker, date, causal_signal) VALUES (?, ?, ?)
               ON CONFLICT(ticker, date) DO UPDATE SET causal_signal=excluded.causal_signal""",
            rows,
        )
    logger.info("store_causal_signals: wrote %d causal_signal rows", len(rows))
    return len(rows)


def run(tickers=None, store: bool = True) -> dict:
    """End-to-end Layer 5: build panel -> estimate causal effects ->
    construct + (optionally) store causal_signal. Returns
    {'panel', 'results', 'signals'} for downstream charts.
    """
    panel = build_causal_panel(tickers)
    # Full-panel estimation is kept for the DIAGNOSTIC charts only (first-stage
    # F, Sargan, naive-vs-causal) -- it never feeds the stored feature.
    results = run_causal_estimation(panel)
    # The stored causal_signal (which feeds Layer 6) uses WALK-FORWARD thetas
    # so no future data leaks into any training-period row (Phase 1 fix).
    signals = construct_causal_signal_walkforward(panel)
    if store and not signals.empty:
        store_causal_signals(signals)
    return {"panel": panel, "results": results, "signals": signals}
