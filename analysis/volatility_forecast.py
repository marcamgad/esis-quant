"""
Volatility forecasting -- the one prediction problem in this system worth trying.

WHY VOLATILITY AND NOT RETURNS. Five independent selection tests on this data
have now returned four nulls and one significant NEGATIVE result (multi-timeframe
momentum, -20.9%/yr, permutation p = 0.010). The standing conclusion is that the
data is the constraint, not the estimator, so a stronger estimator pointed at
returns finds formulas rather than signal.

Volatility is a different problem in a way that matters:

  * it is PERSISTENT -- high-volatility periods cluster, in every market ever
    studied, which is why GARCH works and return forecasting does not;
  * the system ALREADY DEPENDS ON IT. Minimum-variance optimisation with mu = 0
    is entirely a covariance problem, so the covariance estimate IS the strategy.
    Nothing else it computes matters as much;
  * it is cheaply FALSIFIABLE out of sample, on a target that is observed rather
    than inferred.

THE BASELINE IS ESTABLISHED BEFORE ANY SEARCH RUNS. That ordering is the whole
methodology. The system currently estimates volatility as a 180-session trailing
standard deviation; if a fitted model cannot beat that on held-out data, it has
found nothing, however good its training error looks. Baselines are computed
here, in the same file, on the same folds, so the comparison cannot quietly
become a comparison against a weaker strawman later.

WHAT "BETTER" MEANS. Two tests, and a model must pass BOTH:
  1. lower forecast error on held-out data than every baseline;
  2. a portfolio built from its covariance realises LOWER volatility than one
     built from the trailing estimate.
Test 1 without test 2 is an academic result. Test 2 is the one that pays.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.volatility_forecast")

TRADING_DAYS = 252
HORIZON = 21          # forecast the next ~month of realised volatility
MIN_HISTORY = 250     # sessions required before a name can enter the panel

# Maximum share of non-moving days in the feature window. A name quieter than
# this is not low-volatility, it is infrequently priced, and its realised vol
# measures the trading calendar rather than the business. Set at 15% to match
# the staleness filter already used for universe construction, so the two cannot
# disagree about which names are real.
MAX_ZERO_FRAC = 0.15


def build_panel(rets: pd.DataFrame, horizon: int = HORIZON,
                tickers=None, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """(date, ticker) rows of volatility features and the FORWARD realised vol.

    Features are all backward-looking as of `date`; the target is the realised
    volatility of the `horizon` sessions AFTER it. Any leakage here would make
    every number downstream meaningless, so the target is built by shifting the
    feature window forward rather than by slicing around a point.

    Feature choice follows the volatility literature rather than invention:
      vol_21 / vol_63 / vol_126  -- realised vol at three speeds; persistence at
                                    multiple horizons is the HAR model's core idea
      vol_ratio                  -- short/long vol, i.e. is vol rising or falling
      abs_ret_5                  -- recent shock magnitude
      neg_ret_21                 -- share of down days (the leverage effect: vol
                                    rises more after falls than after rises)
      range_proxy                -- mean |daily return|, a scale estimator far
                                    less outlier-sensitive than the s.d.
      zero_frac                  -- share of non-moving days. NOT a vol predictor
                                    but a data-quality feature: a name that does
                                    not reprice looks calm and is not, which is
                                    exactly how CRST reached 21% of a live book.
    """
    cols = list(rets.columns) if tickers is None else [t for t in tickers if t in rets.columns]
    rows = []
    for t in cols:
        s = rets[t].dropna()
        if len(s) < min_history + horizon:
            continue
        v21 = s.rolling(21).std() * np.sqrt(TRADING_DAYS)
        v63 = s.rolling(63).std() * np.sqrt(TRADING_DAYS)
        v126 = s.rolling(126).std() * np.sqrt(TRADING_DAYS)
        v180 = s.rolling(180).std() * np.sqrt(TRADING_DAYS)      # the incumbent
        absr5 = s.abs().rolling(5).mean() * np.sqrt(TRADING_DAYS)
        rng = s.abs().rolling(21).mean() * np.sqrt(TRADING_DAYS)
        neg = (s < 0).rolling(21).mean()
        zero = (s == 0).rolling(63).mean()
        # forward realised vol: shift(-horizon) so row t sees only t+1..t+horizon
        fwd = s.rolling(horizon).std().shift(-horizon) * np.sqrt(TRADING_DAYS)
        df = pd.DataFrame({"vol_21": v21, "vol_63": v63, "vol_126": v126,
                           "vol_180": v180, "abs_ret_5": absr5, "range_proxy": rng,
                           "neg_ret_21": neg, "zero_frac": zero, "target": fwd})
        df["ticker"] = t
        df["date"] = s.index
        rows.append(df.dropna())
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    # A target of exactly zero means the name did not move for a whole month --
    # a data artefact, not a volatility observation. Keeping them would teach any
    # model that stale names are safe, which is the single most expensive wrong
    # belief available in this dataset.
    panel = panel[panel["target"] > 1e-6]

    # STALE NAMES POISON THIS PANEL, and the first run proved it: built over all
    # 199 tickers, every baseline correlated 0.09-0.15 with realised vol, when
    # equity volatility forecasts normally reach 0.5-0.7. The cause is the same
    # defect that let CRST sit at 21% of a live book -- a name that reprices one
    # day in three has a "volatility" that is mostly an artefact of when it last
    # traded, and no model can forecast an artefact. Dropping rows where either
    # the feature window or the target window is substantially non-trading is a
    # data-quality filter, not a performance tweak.
    panel = panel[panel["zero_frac"] <= MAX_ZERO_FRAC]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def time_split(panel: pd.DataFrame, train_frac: float = 0.7):
    """Split by DATE, never at random.

    A random split puts tomorrow in the training set and yesterday in the test
    set for the same stock, and volatility is persistent enough that this alone
    manufactures an impressive score from nothing.
    """
    dates = np.sort(panel["date"].unique())
    cut = dates[int(len(dates) * train_frac)]
    return panel[panel["date"] < cut].copy(), panel[panel["date"] >= cut].copy(), cut


# --------------------------------------------------------------------------
# Baselines. Computed FIRST, on the same rows, so the bar cannot move.
# --------------------------------------------------------------------------

def baselines(df: pd.DataFrame) -> dict:
    """Forecasts a fitted model has to beat to have found anything.

      vol_180     what the system uses today -- the number to beat
      vol_21      naive persistence; a strong baseline in volatility work
      vol_63      medium-horizon persistence
      har_like    0.5*vol_21 + 0.3*vol_63 + 0.2*vol_126, a fixed-weight stand-in
                  for the HAR model. No fitting, so it cannot overfit -- and if
                  a searched formula cannot beat a fixed blend of its own
                  inputs, the search contributed nothing.
      mean_train  the unconditional mean. If nothing beats THIS, the exercise is
                  over before it starts.
    """
    out = {"vol_180": df["vol_180"].to_numpy(),
           "vol_21": df["vol_21"].to_numpy(),
           "vol_63": df["vol_63"].to_numpy(),
           "har_like": (0.5 * df["vol_21"] + 0.3 * df["vol_63"] + 0.2 * df["vol_126"]).to_numpy()}
    return out


def score(pred: np.ndarray, actual: np.ndarray) -> dict:
    """Error metrics. Several, because they disagree and the disagreement is
    informative: RMSE is dominated by the few most volatile names, MAE is not,
    and QLIKE is the standard loss in the volatility literature precisely
    because it penalises under-forecasting asymmetrically -- which is the error
    that actually hurts a risk model."""
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    ok = np.isfinite(pred) & np.isfinite(actual) & (actual > 0) & (pred > 0)
    p, a = pred[ok], actual[ok]
    if len(p) < 10:
        return {"n": int(len(p)), "rmse": np.nan, "mae": np.nan, "qlike": np.nan, "corr": np.nan}
    # QLIKE divides by the forecast variance, so a forecast near zero sends the
    # loss to infinity and the mean then measures one outlier instead of skill.
    # The first run of this returned 3.5e12 to 4.3e15 -- numbers that look like
    # a result and are actually a division. Flooring at 1% annualised vol is far
    # below any real EGX name and removes the singularity without touching the
    # comparison.
    VOL_FLOOR = 0.01
    pq = np.maximum(p, VOL_FLOOR)
    aq = np.maximum(a, VOL_FLOOR)
    return {"n": int(len(p)),
            "rmse": float(np.sqrt(np.mean((p - a) ** 2))),
            "mae": float(np.mean(np.abs(p - a))),
            # QLIKE on variances, the standard form
            "qlike": float(np.mean(np.log(pq ** 2) + (aq ** 2) / (pq ** 2))),
            "corr": float(np.corrcoef(p, a)[0, 1])}


def evaluate_baselines(test: pd.DataFrame) -> pd.DataFrame:
    a = test["target"].to_numpy()
    return pd.DataFrame({k: score(v, a) for k, v in baselines(test).items()}).T


def export_for_typedgp(train: pd.DataFrame, path: str, max_rows: int = 4000,
                       features=("vol_21", "vol_63", "vol_126", "abs_ret_5",
                                 "range_proxy", "neg_ret_21"), seed: int = 0,
                       log_target: bool = False, log_features: bool = False) -> dict:
    """
    `log_target` FIXES A REAL FAILURE, it is not a preprocessing nicety.

    Fitting the raw target under MAE, the search returned `tanh(v126)` at MAE
    0.1235 against a baseline of 0.1948 -- a 37% "improvement" that was entirely
    an artefact. tanh saturates at 1.0 while the target reaches 13.67 with a
    median of 0.384, so the formula says "predict the input for calm names and
    1.0 for every volatile one". MAE barely punishes a catastrophic miss on the
    tail, so capping the tail WINS on that metric -- and the tail is the only
    part a risk model exists for. The Pareto front confirmed it by ranking
    `tanh(tanh(v126))`, i.e. more capping, higher still, with a bare constant
    (the median) already close to the baseline.

    Realised volatility is approximately lognormal. On log(vol) the target is
    near-symmetric, the tail that made capping profitable is gone, and MAE on a
    symmetric target is close to rank-preserving -- which is what the covariance
    estimator actually needs. The fix is to the target, not to the metric.
    """
    """Write whitespace-separated `x1 .. xn y` rows for TypedGP's --data.

    Subsampled because a genetic search evaluates every candidate expression
    against every row, so cost is linear in rows and 100k of them buys accuracy
    the noise floor cannot support. Sampled uniformly at random ACROSS the
    training window rather than truncated, so the sample keeps the period's full
    range of volatility regimes instead of one calm stretch of it.

    `zero_frac` and `vol_180` are deliberately withheld: the first is a
    data-quality flag rather than a predictor, and the second is the incumbent
    baseline -- handing the search the answer it is supposed to beat would make
    any improvement unfalsifiable.
    """
    rng = np.random.default_rng(seed)
    d = train.dropna(subset=list(features) + ["target"])
    d = d[(d[list(features)] > 0).all(axis=1) & (d["target"] > 0)] if (log_target or log_features) else d
    if len(d) > max_rows:
        d = d.iloc[rng.choice(len(d), max_rows, replace=False)]
    with open(path, "w", encoding="utf-8") as fh:
        for r in d.itertuples():
            vals = [getattr(r, f) for f in features]
            if log_features:
                vals = [np.log(v) for v in vals]
            y = np.log(r.target) if log_target else r.target
            fh.write(" ".join(f"{v:.6f}" for v in vals) + f" {y:.6f}\n")
    return {"path": path, "rows": len(d), "vars": list(features),
            "log_target": log_target, "log_features": log_features}


def run(rets: pd.DataFrame = None, train_frac: float = 0.7) -> dict:
    """Build the panel, split by date, and report what the baselines achieve."""
    if rets is None:
        from backtest.engine import load_price_returns
        rets = load_price_returns()
    panel = build_panel(rets)
    if panel.empty:
        return {"ok": False, "reason": "no usable rows"}
    train, test, cut = time_split(panel, train_frac)
    return {"ok": True, "panel_rows": len(panel), "train_rows": len(train),
            "test_rows": len(test), "split_date": str(pd.Timestamp(cut).date()),
            "n_tickers": panel["ticker"].nunique(),
            "baselines": evaluate_baselines(test), "train": train, "test": test}
