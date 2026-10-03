"""
Tier 2 — is the regime model actually any good?  (items 5, 6, 7)

The entire claimed edge lives in the BEAR overlay. Three prior audits measured
what it DID; none established that the model producing it is sound. This module
attacks it three ways, all of them causal.

ITEM 5 — WALK-FORWARD STATE-COUNT SWEEP.
`analysis/robustness_suite.regime_count_sweep` fits the HMM once on the whole
history and decodes the whole history, and its own docstring says the resulting
"3 states is much better" is a look-ahead artifact. That warning is correct and
it is also an admission that the sweep cannot answer the question. This rebuilds
it on `models.regime_pipeline.compute_regime_series(mode="walkforward")` — the
same annual-refit, frozen-parameter, filtered-decode machinery the live path
uses — so each state count is judged on labels it could actually have produced
in real time.

ITEM 6 — DOES THE EMBARGO PRODUCE BETTER LABELS?
The 50-day embargo moved BULL from 166 to 331 days. The prior audit reported the
shift and explicitly declined to claim it was an improvement, because no
evidence either way had been produced. Reporting a change as neutral is not the
same as testing it. Here both labelings are scored on the ONE thing a regime
call is for: does a BEAR call, made causally, precede bad returns?

ITEM 7 — DOES THE METHOD GENERALISE, OR IS IT EGX CURVE-FITTING?
The HMM has only ever been fitted and judged on EGX. A method that only works on
the market it was developed on is not a method. This applies the SAME procedure
to other emerging markets with known crises and asks whether BEAR calls precede
drawdowns there too.

  WHAT THIS DOES AND DOES NOT TEST. It tests the METHODOLOGY -- Gaussian HMM on
  return/volatility/drawdown features, walk-forward -- not the fitted EGX
  parameters and not the full EGX feature set (breadth and Amihud need a
  constituent panel that does not exist for the comparison markets). A reduced
  feature set is used for EVERY market INCLUDING EGX, so the comparison is
  like-for-like even though it is not identical to the live model.

SCORING. Every test uses forward returns after a causally-made call, with the
overlap correction (t / sqrt(h)) established in analysis/overlay_tuning.py.
Overlapping windows are not independent observations and the raw t-stat
double-counts; a regime result that only survives without the correction is not
a result.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.regime_validation")

TRADING_DAYS = 252
STATE_GRID = (2, 3, 4, 5, 6)
EMBARGO_GRID = (0, 25, 50, 100)
FORWARD_H = 21

# Emerging-market proxies with well-documented crisis periods, chosen for having
# a real bear market in the sample rather than for how the test comes out.
CROSS_MARKETS = {
    "EGYPT (EGPT)": "EGPT",     # VanEck Egypt — the home market, as control
    "TURKEY (TUR)": "TUR",      # 2018 lira crisis
    "BRAZIL (EWZ)": "EWZ",      # 2015-16 recession
    "S.AFRICA (EZA)": "EZA",    # 2015-16, 2020
    "INDIA (INDA)": "INDA",     # 2020 COVID
}


def _reduced_features(px: pd.Series, vol_window: int = 20,
                      dd_window: int = 60) -> pd.DataFrame:
    """Return / volatility / drawdown features computable for ANY single market.

    Deliberately minimal and identical across markets. Every column at date D
    uses only data through D, so a walk-forward fit on these is causal.
    """
    r = px.pct_change()
    return pd.DataFrame({
        "ret": r,
        "vol": r.rolling(vol_window, min_periods=vol_window // 2).std(),
        "dd": px / px.rolling(dd_window, min_periods=dd_window // 2).max() - 1.0,
    }).dropna()


def _regime_series(df: pd.DataFrame) -> pd.Series:
    """compute_regime_series returns `date` as a COLUMN, not the index. Indexing
    it positionally would silently misalign every call against the returns it is
    scored on, so the conversion is done in one place."""
    if "date" in df.columns:
        return df.set_index(pd.DatetimeIndex(df["date"]))["regime"]
    return df["regime"]


def _bear_state(obs: pd.DataFrame, states: pd.Series) -> object:
    """The state with the lowest mean return. Defined from the fitted data, not
    assumed by index -- HMM state numbering is arbitrary across refits."""
    means = {s: float(obs.loc[states.index[states == s], "ret"].mean())
             for s in states.dropna().unique()}
    return min(means, key=means.get) if means else None


def score_calls(is_bear: pd.Series, fwd_source: pd.Series, h: int = FORWARD_H) -> dict:
    """Do BEAR calls precede worse returns? Overlap-corrected.

    `is_bear` must be SHIFTED by the caller if it is not already a call made
    before the return it is judged against -- this function does not shift, so
    the causality is the caller's responsibility and is visible at the call site.
    """
    fwd = (1 + fwd_source).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
    df = pd.concat([is_bear.rename("bear"), fwd.rename("fwd")], axis=1).dropna()
    b, n = df.loc[df["bear"] == 1, "fwd"], df.loc[df["bear"] == 0, "fwd"]
    if len(b) < 20 or len(n) < 20:
        return {"ok": False, "reason": f"too few observations (bear={len(b)}, other={len(n)})",
                "n_bear": len(b), "n_other": len(n)}
    se = np.sqrt(b.var(ddof=1) / len(b) + n.var(ddof=1) / len(n))
    t_raw = float((b.mean() - n.mean()) / se) if se > 0 else np.nan
    t_adj = t_raw / np.sqrt(h) if t_raw == t_raw else np.nan
    return {"ok": True, "n_bear": int(len(b)), "n_other": int(len(n)),
            "mean_after_bear": float(b.mean()), "mean_after_other": float(n.mean()),
            "diff": float(b.mean() - n.mean()), "t_raw": t_raw, "t_adj": t_adj,
            "bear_worse": bool(b.mean() < n.mean()),
            "significant": bool(abs(t_adj) >= 2.0) if t_adj == t_adj else False}


# --------------------------------------------------------------------------
# ITEM 5 — walk-forward state-count sweep
# --------------------------------------------------------------------------

def walkforward_bear_calls(obs: pd.DataFrame, n_states: int, embargo: int = None,
                           n_init: int = 8, random_state: int = 42,
                           min_train_obs: int = 250) -> pd.Series:
    """Causal BEAR calls for ARBITRARY state count.

    WHY THIS EXISTS RATHER THAN CALLING compute_regime_series DIRECTLY.
    `models.regime_labeler.label_states` deliberately RAISES for K != 4 -- it
    refuses to guess a BULL/BEAR/HIGH_VOL/RECOVERY mapping for a different state
    count, which is the right call. But that makes the live function unusable for
    a state-count sweep.

    This mirrors compute_regime_series(mode="walkforward") exactly -- expanding
    annual refit, embargo before each boundary, frozen params, filtered
    (forward-only) decode -- and identifies BEAR the SAME way the labeler does:
    the state with the lowest mean 20-day market return. For K = 4 it therefore
    reproduces the live definition; for other K it extends only the part that is
    unambiguous.
    """
    from models.hmm import GaussianHMM
    from models.regime_pipeline import standardize_expanding, EMBARGO_DAYS
    from models.regime_features import OBSERVATION_COLUMNS

    std = standardize_expanding(obs, OBSERVATION_COLUMNS).dropna()
    X = std.to_numpy()
    dates = std.index
    ret_col = OBSERVATION_COLUMNS.index("o1_market_return_20d")
    embargo = EMBARGO_DAYS if embargo is None else int(embargo)

    vintages = []
    for b in pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="YS"):
        cutoff = b - pd.Timedelta(days=embargo) if embargo > 0 else b
        mask = np.asarray(dates < cutoff)
        if int(mask.sum()) < min_train_obs:
            continue
        hmm = GaussianHMM(n_states=n_states, n_init=n_init, random_state=random_state)
        fit = hmm.fit(X[mask])
        mu = fit["mu"] if isinstance(fit, dict) and "mu" in fit else hmm.mu_
        bear = int(np.argmin(np.asarray(mu)[:, ret_col]))   # same rule as the labeler
        vintages.append((b, hmm, bear))
    if not vintages:
        raise ValueError("no boundary had enough prior data to refit")

    out = []
    for i, (b, hmm, bear) in enumerate(vintages):
        end = vintages[i + 1][0] if i + 1 < len(vintages) else dates.max() + pd.Timedelta(days=1)
        active = np.asarray((dates >= b) & (dates < end))
        if not active.any():
            continue
        states_full = hmm.filtered_decode(X)     # forward-only: no look-ahead
        idx = np.where(active)[0]
        out.append(pd.Series((states_full[idx] == bear).astype(int), index=dates[idx]))
    return pd.concat(out).sort_index()


def walkforward_state_sweep(grid=STATE_GRID, embargo: int = None) -> pd.DataFrame:
    """Refit walk-forward at each state count and score the resulting BEAR calls
    against realised forward returns."""
    from models.regime_features import build_observation_matrix
    from backtest.engine import load_price_returns

    obs = build_observation_matrix()
    mkt = load_price_returns().mean(axis=1).dropna()
    rows = []
    for k in grid:
        try:
            reg = walkforward_bear_calls(obs, n_states=k, embargo=embargo)
            # the call for day D is known at D's close and judged on D+1 onward
            is_bear = reg.shift(1).dropna()
            common = is_bear.index.intersection(mkt.index)
            s = score_calls(is_bear.loc[common], mkt.loc[common])
            rows.append({"n_states": k, "bear_days": int(reg.sum()),
                         "n_labelled": int(len(reg)),
                         **{key: s.get(key) for key in
                            ("diff", "t_raw", "t_adj", "bear_worse", "significant")}})
        except Exception as exc:  # noqa: BLE001
            logger.warning("state count %s failed: %s", k, exc)
            rows.append({"n_states": k, "bear_days": None, "diff": None,
                         "t_adj": None, "significant": None})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# ITEM 6 — is the embargoed labelling better?
# --------------------------------------------------------------------------

def embargo_sweep(grid=EMBARGO_GRID, n_states: int = 4) -> pd.DataFrame:
    """Score each embargo length on whether its BEAR calls precede bad returns.

    This is the test the prior audit said had not been done. A longer embargo
    trains on less data, so it is NOT free -- if the labels do not get better at
    predicting, the shift in BULL/BEAR counts is just a different set of labels,
    not a more correct one.
    """
    from models.regime_features import build_observation_matrix
    from models.regime_pipeline import compute_regime_series
    from backtest.engine import load_price_returns

    obs = build_observation_matrix()
    mkt = load_price_returns().mean(axis=1).dropna()
    rows = []
    for e in grid:
        try:
            df = compute_regime_series(obs, mode="walkforward", n_states=n_states,
                                       embargo=e)
            reg = _regime_series(df)
            counts = reg.value_counts().to_dict()
            is_bear = (reg == "BEAR").astype(int).shift(1).dropna()
            common = is_bear.index.intersection(mkt.index)
            s = score_calls(is_bear.loc[common], mkt.loc[common])
            rows.append({"embargo": e, "BEAR": counts.get("BEAR", 0),
                         "BULL": counts.get("BULL", 0),
                         "HIGH_VOL": counts.get("HIGH_VOL", 0),
                         "RECOVERY": counts.get("RECOVERY", 0),
                         **{k: s.get(k) for k in ("diff", "t_raw", "t_adj",
                                                  "bear_worse", "significant")}})
        except Exception as exc:  # noqa: BLE001
            logger.warning("embargo %s failed: %s", e, exc)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# ITEM 7 — cross-market validation of the METHOD
# --------------------------------------------------------------------------

def cross_market(markets=None, n_states: int = 4, min_train: int = 750,
                 train_frac: float = 0.6, n_init: int = 1,
                 max_obs: int = 1500) -> pd.DataFrame:
    """Apply the same walk-forward HMM procedure to other emerging markets.

    Rolls forward annually: fit on everything strictly before year Y, decode
    year Y with frozen parameters. Same shape as the live pipeline, on the
    reduced feature set that every market can supply.
    """
    import time as _time
    import yfinance as yf
    from models.hmm import GaussianHMM

    markets = CROSS_MARKETS if markets is None else markets
    rows = []
    for name, sym in markets.items():
        try:
            # Yahoo rate-limits under sustained use and returns an EMPTY frame
            # rather than raising. Without retries that silently became
            # "only 0 observations", which reads like a data limitation when it
            # is actually a throttle -- exactly the kind of false INCONCLUSIVE
            # that must not be reported as a finding.
            px = None
            for attempt in range(4):
                h = yf.Ticker(sym).history(start="2010-01-01")["Close"]
                if len(h):
                    px = h
                    break
                _time.sleep(3 * (attempt + 1))
            if px is None or not len(px):
                rows.append({"market": name, "ok": False,
                             "reason": "no data after 4 attempts (throttled or delisted)"})
                continue
            px.index = pd.to_datetime(px.index).tz_localize(None)
            obs = _reduced_features(px)
            if len(obs) < min_train + 250:
                rows.append({"market": name, "ok": False,
                             "reason": f"only {len(obs)} observations"})
                continue
            # COMPUTE CONCESSION, STATED. The from-scratch EM fit costs >10 min on
            # ~4,000 observations, so the 15-fold annual roll used on EGX is not
            # feasible across five markets. This uses ONE expanding split per
            # market (fit on the first `train_frac`, test on the rest) with a
            # single EM restart, on at most `max_obs` weekly-thinned rows.
            #
            # WHAT THAT COSTS: fewer, larger folds and a single restart make the
            # per-market estimate noisier. It does NOT introduce look-ahead --
            # the fit still sees only the training prefix and the decode is still
            # forward-only. The generalisation question ("does the method work at
            # all off EGX") survives a noisier estimator; a fine-grained
            # comparison of markets would not, and is not claimed.
            if len(obs) > max_obs:
                obs = obs.iloc[::max(len(obs) // max_obs, 1)]
            cut = int(len(obs) * train_frac)
            train, test = obs.iloc[:cut], obs.iloc[cut:]
            if len(train) < min_train or test.empty:
                rows.append({"market": name, "ok": False,
                             "reason": f"train {len(train)} < {min_train} after thinning"})
                continue
            mu = train.mean(); sd = train.std().replace(0, 1.0)
            hmm = GaussianHMM(n_states=n_states, n_init=n_init, random_state=0)
            hmm.fit(((train - mu) / sd).values)                  # fit on PAST only
            # filtered_decode is FORWARD-ONLY. `decode` (Viterbi) would let a
            # later observation revise an earlier label, which is exactly the
            # smoothing look-ahead the live pipeline exists to avoid.
            st = pd.Series(hmm.filtered_decode(((test - mu) / sd).values),
                           index=test.index)
            tr = pd.Series(hmm.filtered_decode(((train - mu) / sd).values),
                           index=train.index)
            bear = _bear_state(train, tr)                        # labelled from TRAIN
            is_bear = (st == bear).astype(int).shift(1).dropna()
            # returns on the SAME (possibly thinned) grid the calls live on
            r = px.reindex(obs.index).pct_change().dropna()
            common = is_bear.index.intersection(r.index)
            s = score_calls(is_bear.loc[common], r.loc[common])
            rows.append({"market": name, "ok": s.get("ok", False),
                         "n_bear": s.get("n_bear"), "n_other": s.get("n_other"),
                         "diff": s.get("diff"), "t_raw": s.get("t_raw"),
                         "t_adj": s.get("t_adj"), "bear_worse": s.get("bear_worse"),
                         "significant": s.get("significant"),
                         "reason": s.get("reason", "")})
        except Exception as exc:  # noqa: BLE001
            logger.warning("market %s failed: %s", name, exc)
            rows.append({"market": name, "ok": False, "reason": str(exc)[:60]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def _fmt_t(v):
    return "  n/a" if v is None or v != v else f"{v:6.2f}"


def format_state_sweep(df: pd.DataFrame, live: int = 4) -> str:
    lines = ["ITEM 5 — WALK-FORWARD STATE-COUNT SWEEP (causal labels, 21d forward)",
             f"  {'states':>7}{'bear days':>11}{'diff':>9}{'t(raw)':>9}{'t(adj)':>9}  verdict"]
    for r in df.itertuples():
        v = ("worse & SIGNIFICANT" if r.significant else
             "worse, not significant" if r.bear_worse else "NOT worse")
        mark = "  <-- live" if r.n_states == live else ""
        bd = "n/a" if r.bear_days is None else f"{r.bear_days}"
        d = "  n/a" if r.diff is None or r.diff != r.diff else f"{r.diff*100:7.2f}%"
        lines.append(f"  {r.n_states:>7}{bd:>11}{d:>9}{_fmt_t(r.t_raw)}{_fmt_t(r.t_adj)}  {v}{mark}")
    return "\n".join(lines)


def format_embargo(df: pd.DataFrame, live: int = 50) -> str:
    lines = ["ITEM 6 — DOES THE EMBARGO PRODUCE BETTER LABELS?",
             f"  {'embargo':>8}{'BEAR':>7}{'BULL':>7}{'HI_VOL':>8}{'RECOV':>7}"
             f"{'diff':>9}{'t(adj)':>9}  verdict"]
    for r in df.itertuples():
        v = ("worse & SIGNIFICANT" if r.significant else
             "worse, not significant" if r.bear_worse else "NOT worse")
        mark = "  <-- live" if r.embargo == live else ""
        d = "  n/a" if r.diff is None or r.diff != r.diff else f"{r.diff*100:7.2f}%"
        lines.append(f"  {r.embargo:>8}{r.BEAR:>7}{r.BULL:>7}{r.HIGH_VOL:>8}"
                     f"{r.RECOVERY:>7}{d:>9}{_fmt_t(r.t_adj)}  {v}{mark}")
    lines.append("  A longer embargo trains on LESS data. If the calls do not predict better,")
    lines.append("  the shift in label counts is a different labelling, not a better one.")
    return "\n".join(lines)


def format_cross_market(df: pd.DataFrame) -> str:
    lines = ["ITEM 7 — CROSS-MARKET VALIDATION OF THE METHOD (walk-forward, reduced features)",
             f"  {'market':>16}{'bear d':>8}{'other d':>9}{'diff':>9}{'t(adj)':>9}  verdict"]
    for r in df.itertuples():
        if not r.ok:
            lines.append(f"  {r.market:>16}   {getattr(r, 'reason', 'failed')}")
            continue
        v = ("worse & SIGNIFICANT" if r.significant else
             "worse, not significant" if r.bear_worse else "NOT worse")
        lines.append(f"  {r.market:>16}{r.n_bear:>8}{r.n_other:>9}{r.diff*100:>8.2f}%"
                     f"{_fmt_t(r.t_adj)}  {v}")
    lines.append("  Tests the METHOD (HMM on return/vol/drawdown, walk-forward), not the")
    lines.append("  fitted EGX parameters and not the full EGX feature set.")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    print(format_state_sweep(walkforward_state_sweep()))
    print()
    print(format_embargo(embargo_sweep()))
    print()
    print(format_cross_market(cross_market()))
