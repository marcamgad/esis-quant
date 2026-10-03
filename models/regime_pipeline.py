"""
Layer 2 orchestration: fit the Gaussian HMM on Layer 1 data, decode
regimes, label them, and persist regime_history / regime_statistics.

Assumes the DB already has enough history loaded (see main.py run_layer2,
which backfills via Layer 1's DataCollector before calling this).
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection
from models.hmm import GaussianHMM
from models.regime_features import build_observation_matrix, load_returns_panel, OBSERVATION_COLUMNS
from models.regime_labeler import label_states, label_columns_for_gamma

logger = logging.getLogger("esis_quant.regime_pipeline")


def standardize(df: pd.DataFrame, columns: list):
    """Z-score standardize the given columns. Necessary here because the
    raw features live on wildly different scales (e.g. Amihud illiquidity
    ~1e-9 vs breadth ~0.5) -- an unstandardized covariance matrix would be
    dominated by whichever feature happens to have the largest raw
    magnitude, not whichever is actually most informative. Standardizing
    per-column preserves each column's ranking (order-preserving linear
    transform), so the regime_labeler heuristic (argmax/argmin on a
    column) gives the same answer either way.
    """
    means = df[columns].mean()
    stds = df[columns].std().replace(0, 1.0)
    standardized = (df[columns] - means) / stds
    return standardized.to_numpy(), means, stds


def standardize_expanding(df: pd.DataFrame, columns: list, min_periods: int = 60) -> pd.DataFrame:
    """Point-in-time z-score standardization: each row t is scaled using the
    mean/std of the series UP TO AND INCLUDING t only (expanding window),
    never the full-sample statistics. This removes the look-ahead bias of
    global standardization -- under global scaling the z-score at date t
    embeds the future mean/variance of the features, which is unavailable in
    live trading (STAT-LEAK-01).

    Rows before `min_periods` observations are NaN (not enough history to
    scale) and are meant to be dropped by the caller. Like the global
    version this is an order-preserving per-column transform, so the
    regime_labeler argmax/argmin heuristics are unaffected within any given
    cross-section-free column comparison.
    """
    expanding = df[columns].expanding(min_periods=min_periods)
    means = expanding.mean()
    stds = expanding.std().replace(0, 1.0)
    return (df[columns] - means) / stds


def compute_regime_durations(regime_series: pd.Series) -> dict:
    """Average run-length (in trading days) of consecutive same-label
    stretches, per regime label. Uses explicit predecessor comparison
    (labels[1:] != labels[:-1]) rather than np.roll, which compares
    labels[0] against labels[-1] and requires an error-prone manual
    fixup for the first element.
    """
    labels = regime_series.to_numpy()
    if len(labels) == 0:
        return {}
    changes = np.concatenate([[True], labels[1:] != labels[:-1]])
    run_id = np.cumsum(changes)
    runs = (pd.DataFrame({"label": labels, "run_id": run_id})
            .groupby("run_id")["label"]
            .agg(["first", "size"]))
    return runs.groupby("first")["size"].mean().to_dict()


def compute_regime_statistics(obs_df: pd.DataFrame, regime_series: pd.Series, returns_panel: pd.DataFrame,
                               hmm: GaussianHMM, label_map: dict) -> pd.DataFrame:
    """Descriptive stats per regime label, computed from the RAW
    (non-standardized) data grouped by decoded regime -- not from the
    HMM's internal standardized emission parameters, so the numbers here
    are directly interpretable (e.g. mean_return is an actual mean daily
    return, not a z-score).
    """
    durations = compute_regime_durations(regime_series)
    label_by_state = label_map
    state_by_label = {v: k for k, v in label_by_state.items()}

    aligned_returns = returns_panel.reindex(obs_df.index)

    rows = []
    for regime in ["BULL", "BEAR", "HIGH_VOL", "RECOVERY"]:
        mask = regime_series == regime
        daily_rets = obs_df.loc[mask, "market_proxy_daily_return"]
        mean_return = float(daily_rets.mean()) if mask.any() else None
        volatility = float(daily_rets.std() * np.sqrt(252)) if mask.sum() > 1 else None

        sub_returns = aligned_returns.loc[mask]
        if mask.sum() > 1 and sub_returns.shape[1] > 1:
            corr = sub_returns.corr()
            upper = corr.to_numpy()[np.triu_indices(len(corr), k=1)]
            avg_corr = float(np.nanmean(upper)) if len(upper) else None
        else:
            avg_corr = None

        state_idx = state_by_label[regime]
        transitions = {
            f"transition_to_{other.lower()}": float(hmm.A_[state_idx, state_by_label[other]])
            for other in ["BULL", "BEAR", "HIGH_VOL", "RECOVERY"]
        }

        rows.append({
            "regime": regime,
            "mean_return": mean_return,
            "volatility": volatility,
            "avg_cross_sectional_correlation": avg_corr,
            "avg_duration_days": durations.get(regime),
            **transitions,
        })

    return pd.DataFrame(rows).set_index("regime")


def _store_regime_history(regime_series: pd.Series, gamma: np.ndarray, gamma_columns: list):
    col_to_state = {col: k for k, col in enumerate(gamma_columns)}
    rows = []
    for i, (date, regime) in enumerate(regime_series.items()):
        rows.append((
            date.strftime("%Y-%m-%d"),
            regime,
            float(gamma[i, col_to_state["prob_bull"]]),
            float(gamma[i, col_to_state["prob_bear"]]),
            float(gamma[i, col_to_state["prob_high_vol"]]),
            float(gamma[i, col_to_state["prob_recovery"]]),
        ))
    with get_connection() as conn:
        # Regime detection is a full-history recompute, not an incremental
        # append: clear the table first so dates that the current fit no
        # longer produces (e.g. warm-up rows dropped by expanding-window
        # standardization) don't linger as stale rows from a previous run.
        # DELETE + INSERT run in one transaction (get_connection commits on
        # exit), so a failure rolls back to the prior table rather than
        # leaving it empty.
        conn.execute("DELETE FROM regime_history")
        conn.executemany(
            """INSERT INTO regime_history
               (date, regime, prob_bull, prob_bear, prob_high_vol, prob_recovery)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 regime=excluded.regime,
                 prob_bull=excluded.prob_bull,
                 prob_bear=excluded.prob_bear,
                 prob_high_vol=excluded.prob_high_vol,
                 prob_recovery=excluded.prob_recovery""",
            rows,
        )


def _store_regime_history_df(regime_df: pd.DataFrame):
    """Store a regime frame (from compute_regime_series) including its
    param_vintage_date. Full-recompute semantics: clear then insert, so dates
    the current (walk-forward) run no longer produces don't linger stale.
    """
    rows = [
        (pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), r["regime"],
         float(r["prob_bull"]), float(r["prob_bear"]), float(r["prob_high_vol"]),
         float(r["prob_recovery"]), pd.Timestamp(r["param_vintage_date"]).strftime("%Y-%m-%d"))
        for _, r in regime_df.iterrows()
    ]
    with get_connection() as conn:
        conn.execute("DELETE FROM regime_history")
        conn.executemany(
            """INSERT INTO regime_history
               (date, regime, prob_bull, prob_bear, prob_high_vol, prob_recovery, param_vintage_date)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )


def _store_regime_statistics(stats_df: pd.DataFrame):
    with get_connection() as conn:
        for regime, row in stats_df.iterrows():
            conn.execute(
                """INSERT INTO regime_statistics
                   (regime, mean_return, volatility, avg_cross_sectional_correlation, avg_duration_days,
                    transition_to_bull, transition_to_bear, transition_to_high_vol, transition_to_recovery)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(regime) DO UPDATE SET
                     mean_return=excluded.mean_return, volatility=excluded.volatility,
                     avg_cross_sectional_correlation=excluded.avg_cross_sectional_correlation,
                     avg_duration_days=excluded.avg_duration_days,
                     transition_to_bull=excluded.transition_to_bull, transition_to_bear=excluded.transition_to_bear,
                     transition_to_high_vol=excluded.transition_to_high_vol,
                     transition_to_recovery=excluded.transition_to_recovery""",
                (
                    regime, row["mean_return"], row["volatility"], row["avg_cross_sectional_correlation"],
                    row["avg_duration_days"], row["transition_to_bull"], row["transition_to_bear"],
                    row["transition_to_high_vol"], row["transition_to_recovery"],
                ),
            )


def _warn_if_labels_look_inconsistent(regime_series: pd.Series, stats_df: pd.DataFrame):
    """The regime_labeler heuristic picks each label from a handful of
    ranking rules (return extremes, then transition-into-from-bear, then
    an elimination step) -- it has no way to check afterward whether the
    result actually matches each label's textbook definition. This is a
    real, recurring limitation of unsupervised HMM fitting on real data:
    a 4-state model doesn't always carve reality into exactly the four
    archetypes a hand-designed taxonomy assumes (found twice now: first
    RECOVERY absorbing a persistent "calm market" state, then separately
    HIGH_VOL landing on the single LOWEST-volatility state of the four
    once the true high-vol state got claimed by BEAR). Rather than patch
    one label at a time as new mismatches turn up, this checks all four
    against their defining characteristic every run and logs every
    violation, so a future mismatch on a different label surfaces loudly
    instead of silently shipping a mislabeled-but-plausible-looking chart.
    """
    if not {"BULL", "BEAR", "HIGH_VOL", "RECOVERY"}.issubset(stats_df.index):
        return  # a regime with zero decoded days has no row to check

    share = regime_series.value_counts(normalize=True)
    violations = []

    bull_vol_rank = stats_df["volatility"].rank(ascending=False)["BULL"]
    if stats_df.loc["BULL", "mean_return"] <= 0:
        violations.append("BULL has non-positive mean_return, contradicting 'trending up'")
    if bull_vol_rank == 1:
        violations.append("BULL has the HIGHEST volatility of all four regimes, contradicting 'low volatility'")

    if stats_df["mean_return"].idxmin() != "BEAR":
        violations.append(
            f"BEAR does not have the lowest mean_return ('{stats_df['mean_return'].idxmin()}' does instead)"
        )

    if stats_df["volatility"].idxmax() != "HIGH_VOL":
        violations.append(
            f"HIGH_VOL does not have the highest volatility ('{stats_df['volatility'].idxmax()}' does instead)"
        )

    recovery_share = share.get("RECOVERY", 0.0)
    recovery_duration = stats_df.loc["RECOVERY", "avg_duration_days"]
    if recovery_share > 0.40 or recovery_duration > 60:
        violations.append(
            f"RECOVERY accounts for {recovery_share * 100:.0f}% of days with "
            f"~{recovery_duration:.0f}-day average runs -- looks like a persistent "
            "'calm/normal market' state, not a short transitional bottoming-out phase"
        )

    if violations:
        logger.warning(
            "Regime label consistency check found %d mismatch(es) against each label's "
            "textbook definition: %s. Treat regime labels as approximate -- check "
            "regime_statistics directly (mean_return, volatility, avg_duration_days) for "
            "each state's actual empirical character rather than trusting the name alone.",
            len(violations), "; ".join(violations),
        )


def _fit_and_label(X_train: np.ndarray, n_states: int, n_init: int, random_state: int):
    """Fit a GaussianHMM on X_train and return (hmm, label_map)."""
    hmm = GaussianHMM(n_states=n_states, n_init=n_init, random_state=random_state, tol=1e-6, n_iter=200)
    hmm.fit(X_train)
    return hmm, label_states(hmm.mu_, hmm.A_)


# FROZEN VINTAGES (Stage 0, 2026-10-03).
#
# A walk-forward vintage is fit once a year, on data strictly before its Jan-1
# boundary, and never changes after that. The daily job nevertheless refit every
# vintage since 2017 every day -- 3-8 minutes of EM for parameters that are, by
# construction, constant. On a laptop that was long enough to be killed: the
# daily runs of 23, 24 and 28 Sep died inside this step, regime_history stopped
# at 17 Sep, and no BEAR signal could have reached the trade list.
#
# Each fitted vintage is now stored as JSON and reused. The key includes a hash
# of the exact training matrix, so a vintage refits only if its own training
# data actually changed (a backfill, a universe change, a ticker-subset study),
# never because a new day arrived. clear_vintages() forces a full refit.
# Off by default in compute_regime_series; the live paths opt in.
_VINTAGE_KEY = ("boundary", "n_states", "n_init", "random_state", "embargo",
                "features", "n_train", "data_hash")


def _ensure_vintage_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_vintages (
            boundary TEXT NOT NULL, n_states INTEGER NOT NULL, n_init INTEGER NOT NULL,
            random_state INTEGER NOT NULL, embargo INTEGER NOT NULL, features TEXT NOT NULL,
            n_train INTEGER NOT NULL, data_hash TEXT NOT NULL,
            params_json TEXT NOT NULL, fitted_at TEXT,
            PRIMARY KEY (boundary, n_states, n_init, random_state, embargo, features,
                         n_train, data_hash))""")


def clear_vintages() -> int:
    with get_connection() as conn:
        _ensure_vintage_table(conn)
        return int(conn.execute("DELETE FROM regime_vintages").rowcount or 0)


def _cached_fit(boundary, X_train: np.ndarray, n_states: int, n_init: int,
                random_state: int, embargo: int, features: str = None):
    import hashlib
    import json
    features = features or ",".join(OBSERVATION_COLUMNS)
    digest = hashlib.sha1(np.ascontiguousarray(np.round(X_train, 8)).tobytes()).hexdigest()
    key = (pd.Timestamp(boundary).strftime("%Y-%m-%d"), int(n_states), int(n_init),
           int(random_state), int(embargo), features, int(len(X_train)), digest)
    where = " AND ".join(f"{k} = ?" for k in _VINTAGE_KEY)
    with get_connection() as conn:
        _ensure_vintage_table(conn)
        row = conn.execute(f"SELECT params_json FROM regime_vintages WHERE {where}", key).fetchone()
    if row:
        p = json.loads(row[0])
        hmm = GaussianHMM(n_states=n_states, n_init=n_init, random_state=random_state, tol=1e-6, n_iter=200)
        hmm.pi_, hmm.A_ = np.array(p["pi"]), np.array(p["A"])
        hmm.mu_, hmm.sigma_ = np.array(p["mu"]), np.array(p["sigma"])
        return hmm, label_states(hmm.mu_, hmm.A_)
    hmm, label_map = _fit_and_label(X_train, n_states, n_init, random_state)
    params = json.dumps({"pi": hmm.pi_.tolist(), "A": hmm.A_.tolist(),
                         "mu": hmm.mu_.tolist(), "sigma": hmm.sigma_.tolist()})
    with get_connection() as conn:
        conn.execute(f"INSERT OR REPLACE INTO regime_vintages ({', '.join(_VINTAGE_KEY)}, "
                     "params_json, fitted_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (*key, params, pd.Timestamp.now().isoformat(timespec="seconds")))
    return hmm, label_map


def _regime_frame(dates, states, proba, label_map, vintage):
    """Assemble a regime DataFrame from decoded states + (T,K) state-order
    probabilities, mapping HMM state indices to BULL/BEAR/HIGH_VOL/RECOVERY.
    """
    state_for_label = {lab: st for st, lab in label_map.items()}
    df = pd.DataFrame({"date": dates,
                       "regime": [label_map[s] for s in states],
                       "param_vintage_date": vintage})
    for lab, col in [("BULL", "prob_bull"), ("BEAR", "prob_bear"),
                     ("HIGH_VOL", "prob_high_vol"), ("RECOVERY", "prob_recovery")]:
        df[col] = proba[:, state_for_label[lab]]
    return df


# EMBARGO on the walk-forward refit (audit deferred item P4.12).
#
# WHAT IT IS AND IS NOT FIXING -- stated precisely, because it would be easy to
# claim more than it does. The walk-forward refit is ALREADY causal: it trains
# on `dates < b` only, freezes the parameters, and decodes forward with a
# filtered (forward-only) pass. No future observation enters the fit, so there
# is no look-ahead to remove.
#
# What the embargo addresses is DEPENDENCE, not leakage. The observation
# features are rolling windows -- breadth uses a 50-day moving average, Amihud a
# 20-day one -- so the observation at the boundary date b is computed from a
# window that overlaps the training period. Training and "out-of-sample" rows
# adjacent to the boundary are therefore not independent, which flatters any
# out-of-sample claim even though nothing leaked.
#
# The embargo drops the last EMBARGO_DAYS training observations before each
# boundary, so the fit never sees the raw data feeding the first out-of-sample
# feature windows. Set to the longest feature lookback in
# models/regime_features (the 50-day breadth MA), rounded up.
EMBARGO_DAYS = 50


def compute_regime_series(obs_df: pd.DataFrame, mode: str = "walkforward", n_states: int = 4,
                          n_init: int = 8, random_state: int = 42, min_train_obs: int = 250,
                          embargo: int = None, use_cache: bool = False) -> pd.DataFrame:
    """Produce the regime series under one of three inference regimes, so the
    Phase-A gate can measure exactly how much of any regime-timing edge is
    real versus a look-ahead artifact:

      "smoothed"    : full-history EM fit + Viterbi/forward-backward posterior.
                      Descriptive; NOT causal (each label peeks at the whole
                      sequence). This is the historical default.
      "filtered"    : full-history EM fit, but FILTERED (forward-only) decode
                      -- isolates the smoothing look-ahead while holding the
                      parameter fit fixed.
      "walkforward" : expanding-window EM REFIT (annual) + filtered decode --
                      the genuinely point-in-time signal. At each Jan-1
                      boundary with >= min_train_obs prior observations, the
                      HMM is refit on data strictly before that boundary and
                      used (frozen) to filter forward until the next boundary.
                      Rows before the first valid boundary have no causal
                      regime and are dropped. param_vintage_date records which
                      parameter fit produced each row, so any historical call
                      is auditable for what was knowable at the time.

    Standardization is always expanding-window (point-in-time); the fit itself
    uses only the training slice. Returns a DataFrame:
    [date, regime, prob_bull, prob_bear, prob_high_vol, prob_recovery,
     param_vintage_date].
    """
    std_df = standardize_expanding(obs_df, OBSERVATION_COLUMNS).dropna()
    X = std_df.to_numpy()
    dates = std_df.index

    if mode in ("smoothed", "filtered"):
        hmm, label_map = _fit_and_label(X, n_states, n_init, random_state)
        if mode == "smoothed":
            states = hmm.decode(X)[0]
            proba = hmm.predict_proba(X)
        else:
            states = hmm.filtered_decode(X)
            proba = hmm.filtered_proba(X)
        return _regime_frame(dates, states, proba, label_map, vintage=dates[-1])

    if mode != "walkforward":
        raise ValueError(f"unknown mode {mode!r}")

    boundaries = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="YS")
    frames = []
    vintages = []
    embargo = EMBARGO_DAYS if embargo is None else int(embargo)
    for b in boundaries:
        # embargo: exclude the last `embargo` observations before the boundary,
        # whose rolling feature windows overlap the first out-of-sample rows
        cutoff = b - pd.Timedelta(days=embargo) if embargo > 0 else b
        train_mask = np.asarray(dates < cutoff)
        if int(train_mask.sum()) >= min_train_obs:
            if use_cache:
                hmm_b, label_b = _cached_fit(b, X[train_mask], n_states, n_init, random_state,
                                             embargo)
            else:
                hmm_b, label_b = _fit_and_label(X[train_mask], n_states, n_init, random_state)
            vintages.append((b, hmm_b, label_b))
    if not vintages:
        raise ValueError("compute_regime_series(walkforward): no boundary had enough prior data to refit")

    for i, (b, hmm_b, label_b) in enumerate(vintages):
        active_end = vintages[i + 1][0] if i + 1 < len(vintages) else (dates.max() + pd.Timedelta(days=1))
        active = np.asarray((dates >= b) & (dates < active_end))
        if not active.any():
            continue
        # filter causally over the FULL series with this vintage's frozen
        # params; rows in [b, active_end) use only obs up to their own date and
        # params fit on data strictly before b -> no look-ahead.
        proba_full = hmm_b.filtered_proba(X)
        states_full = hmm_b.filtered_decode(X)
        idx = np.where(active)[0]
        frames.append(_regime_frame(dates[idx], states_full[idx], proba_full[idx], label_b, vintage=b))

    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def run_regime_detection(tickers=None, n_states: int = 4, n_init: int = 8, random_state: int = 42,
                         mode: str = "walkforward", describe: bool = True):
    """End-to-end Layer 2: build observations from the DB (Layer 1), fit the
    HMM, persist regime_history (point-in-time, walk-forward by default) and
    the descriptive regime_statistics. `mode` is passed to compute_regime_series
    ("walkforward"|"filtered"|"smoothed") -- production uses "walkforward" so
    stored regimes are causal (see the A1 gate). Returns a summary dict.

    describe=False skips the full-history smoothed fit behind regime_statistics.
    That fit is descriptive only, never feeds a trade, and was the other
    multi-minute step in the daily job; the daily job runs it at most weekly.
    """
    obs_df_full = build_observation_matrix(tickers)
    if not describe:
        regime_df = compute_regime_series(obs_df_full, mode=mode, n_states=n_states,
                                          n_init=n_init, random_state=random_state,
                                          use_cache=True)
        _store_regime_history_df(regime_df)
        logger.info("Regime detection [%s, no describe]: %d stored days (%s..%s)", mode,
                    len(regime_df), regime_df["date"].min().date(), regime_df["date"].max().date())
        return {"n_days": len(regime_df), "mode": mode, "regime_df": regime_df,
                "last_date": regime_df["date"].max()}
    obs_df = obs_df_full
    if len(obs_df) < 100:
        logger.warning(
            "Only %d observation rows available -- a 4-state, %d-feature "
            "Gaussian HMM has %d free covariance parameters per state and "
            "will be poorly identified with this little data. Consider a "
            "longer backfill.",
            len(obs_df), len(OBSERVATION_COLUMNS),
            len(OBSERVATION_COLUMNS) * (len(OBSERVATION_COLUMNS) + 1) // 2,
        )

    # Point-in-time (expanding-window) standardization: no future feature
    # statistics leak into the scaling of past observations. Drop the
    # warm-up rows that lack enough history to standardize, and realign
    # obs_df so every downstream index (decoded regimes, gamma, stats) maps
    # to the same trimmed set of dates.
    std_df = standardize_expanding(obs_df, OBSERVATION_COLUMNS).dropna()
    obs_df = obs_df.loc[std_df.index]
    X = std_df.to_numpy()

    # Full-history smoothed fit: used ONLY for the DESCRIPTIVE regime_statistics
    # (mean return / vol / duration per regime over the completed history) --
    # a legitimately backward-looking summary. It is NOT what gets stored as
    # the trading signal.
    hmm = GaussianHMM(n_states=n_states, n_init=n_init, random_state=random_state, tol=1e-6, n_iter=200)
    hmm.fit(X)
    decoded, best_loglik = hmm.decode(X)
    label_map = label_states(hmm.mu_, hmm.A_)
    regime_series_smoothed = pd.Series([label_map[s] for s in decoded], index=obs_df.index, name="regime")
    returns_panel = load_returns_panel(tickers)
    stats_df = compute_regime_statistics(obs_df, regime_series_smoothed, returns_panel, hmm, label_map)

    # PRODUCTION regime_history uses the point-in-time series (walk-forward
    # refit + filtered decode by default) -- the A1 gate showed the naive
    # smoothed/full-fit version overstated the regime-timing edge (Sharpe 1.64
    # -> 1.42 once made honest), so anything feeding a trading decision must
    # read this, not the smoothed series.
    regime_df = compute_regime_series(obs_df_full, mode=mode, n_states=n_states,
                                      n_init=n_init, random_state=random_state)
    _store_regime_history_df(regime_df)
    _store_regime_statistics(stats_df)

    regime_df = regime_df.set_index("date")
    regime_series = regime_df["regime"]
    gamma_columns = ["prob_bull", "prob_bear", "prob_high_vol", "prob_recovery"]
    gamma = regime_df[gamma_columns].to_numpy()
    obs_aligned = obs_df_full.loc[obs_df_full.index.isin(regime_df.index)]
    logger.info("Regime detection [%s]: %d stored days (%s..%s), regime counts=%s",
                mode, len(regime_df), regime_df.index.min().date(), regime_df.index.max().date(),
                regime_series.value_counts().to_dict())
    _warn_if_labels_look_inconsistent(regime_series_smoothed, stats_df)

    return {
        "n_days": len(regime_df),
        "loglik": best_loglik,
        "mode": mode,
        "regime_counts": regime_series.value_counts().to_dict(),
        "obs_df": obs_aligned,
        "regime_series": regime_series,
        "regime_df": regime_df.reset_index(),
        "gamma": gamma,
        "gamma_columns": gamma_columns,
        "hmm": hmm,
        "label_map": label_map,
        "stats_df": stats_df,
    }
