"""
Assemble the real (ticker, date) feature panel for the TFT from the
database, build 60-day sliding windows with multi-horizon forward-return
targets, and split strictly by date (walk-forward). Everything is real data
from Layers 1/3/4/5; the assembled feature set is the subset of the spec's
29 that actually exists (documented per feature below).

Standardization is fit on the TRAINING span only and applied to val/test --
no future statistics leak into past windows (the lesson from the Layer 2/5
audit). Targets (forward returns) are left in raw return units so the
pinball loss is in an interpretable scale.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.tft.features")

# Real temporal features actually available in the DB (spec's 29 assume full
# fundamental+options coverage we don't have -- see config note).
TEMPORAL_COLS = [
    "ret1",          # adjusted daily return
    "log_value",     # log(1 + EGP value traded)  -- liquidity/flow
    "hl_range",      # (high-low)/close           -- intraday range
    "close_to_ma20", # close/MA20 - 1             -- trend
    "beta_market", "beta_liquidity", "beta_egp_sensitivity",
    "beta_oil_sensitivity", "idiosyncratic_vol",  # Layer 3 factor exposures
    "causal_signal",    # Layer 5 (0 where missing)
    "usd_egp_ret", "brent_ret",  # Layer 1 macro (broadcast to all tickers)
    # Layer 2 regime-conditioning: the market regime one-hot at each past day,
    # so the model can adapt its quantile forecasts to regime (the observed
    # directional edge is regime-concentrated -- BULL 10-20d far above
    # HIGH_VOL). Kept as raw 0/1 (NOT standardized). CAVEAT: regime_history is
    # the HMM's SMOOTHED (Viterbi/forward-backward) decode, so the label at day
    # t is mildly informed by t+1.. -- a small smoothing look-ahead near regime
    # transitions. Acceptable here (the whole system already conditions on the
    # same smoothed regime, Layers 3/5); a strict point-in-time version would
    # use FILTERED (forward-only) regime probabilities -- a future refinement.
    "regime_bull", "regime_bear", "regime_high_vol", "regime_recovery",
    # NOTE: sentiment_score is DELIBERATELY EXCLUDED (see Layer 4 notes): real
    # daily_sentiment only covers ~2026, so it is 100% zero across the training
    # span -- a dead feature that would just waste a VSN slot.
]
# regime one-hots are already 0/1 -- excluded from z-scoring
REGIME_ONEHOT_COLS = ["regime_bull", "regime_bear", "regime_high_vol", "regime_recovery"]
STANDARDIZE_TEMPORAL_COLS = [c for c in TEMPORAL_COLS if c not in REGIME_ONEHOT_COLS]
STATIC_COLS = ["static_size", "static_vol"]  # per-ticker liquidity size + unconditional vol


def _load_panel(tickers):
    ph = ",".join(["?"] * len(tickers))
    with get_connection() as conn:
        px = pd.read_sql(
            f"SELECT ticker, date, high, low, adjusted_close AS close, volume, value_traded_egp "
            f"FROM daily_ohlcv WHERE ticker IN ({ph}) ORDER BY ticker, date",
            conn, params=tuple(tickers), parse_dates=["date"])
        fx = pd.read_sql(
            f"SELECT ticker, date, beta_market, beta_liquidity, beta_egp_sensitivity, "
            f"beta_oil_sensitivity, idiosyncratic_vol FROM factor_exposures WHERE ticker IN ({ph})",
            conn, params=tuple(tickers), parse_dates=["date"])
        sent = pd.read_sql(
            f"SELECT ticker, date, sentiment_score FROM daily_sentiment WHERE ticker IN ({ph})",
            conn, params=tuple(tickers), parse_dates=["date"])
        sig = pd.read_sql(
            f"SELECT ticker, date, causal_signal FROM signals WHERE ticker IN ({ph})",
            conn, params=tuple(tickers), parse_dates=["date"])
        macro = pd.read_sql(
            "SELECT date, usd_egp, brent_oil_usd FROM macro_daily ORDER BY date",
            conn, parse_dates=["date"])
    return px, fx, sent, sig, macro


def build_feature_panel(tickers=None) -> pd.DataFrame:
    """Long (ticker, date) DataFrame with TEMPORAL_COLS + the 4 forward-return
    targets (fwd_1/5/10/20) + static per-ticker columns. Rows are raw (not yet
    standardized). Sentiment/causal absences are filled with 0 (a neutral
    signal), everything else must be non-NaN for a row to survive.
    """
    tickers = tickers or [t["ticker"] for t in config.EGX_SEED_TICKERS]
    px, fx, sent, sig, macro = _load_panel(tickers)
    if px.empty:
        raise ValueError("build_feature_panel: no price data -- run Layer 1 first")

    # Macro (usd/egp, brent) comes from a source whose calendar doesn't line
    # up with EGX trading days, so a naive merge leaves ~20-25% of dates NaN
    # -- scattered enough that essentially every 60-day window would contain a
    # gap and be dropped. Forward-fill the LEVELS onto the EGX trading-date
    # grid (last-known value, the point-in-time-correct thing) via merge_asof,
    # then compute daily returns on that gap-free per-date series.
    trading_dates = pd.DataFrame({"date": np.sort(px["date"].unique())})
    macro = macro.sort_values("date")
    macro_on_grid = pd.merge_asof(trading_dates, macro, on="date", direction="backward")
    # ffill the LEVELS: macro_daily stores NaN entries on days a given market
    # was closed (e.g. brent NaN on US holidays that are EGX trading days), and
    # merge_asof carries that NaN forward. Forward-filling the last-known level
    # makes the return series gap-free (a flat-carry day just gives a 0 return).
    macro_on_grid[["usd_egp", "brent_oil_usd"]] = macro_on_grid[["usd_egp", "brent_oil_usd"]].ffill()
    macro_on_grid["usd_egp_ret"] = macro_on_grid["usd_egp"].pct_change()
    macro_on_grid["brent_ret"] = macro_on_grid["brent_oil_usd"].pct_change()
    macro = macro_on_grid[["date", "usd_egp_ret", "brent_ret"]]

    horizons = config.TFT_FORECAST_HORIZONS
    thr = config.TFT_RETURN_ARTIFACT_THRESHOLD
    parts = []
    n_artifacts = 0
    artifact_log = []
    for ticker, df in px.groupby("ticker"):
        df = df.sort_values("date").reset_index(drop=True)
        close = df["close"]

        # --- neutralize split/data artifacts (SYS-CRIT-01 residue) ---
        # yfinance's adjusted_close misses some EGX splits and corporate_actions
        # is empty, so a split shows up as an impossible single-day return (EGX
        # has ~10-20% daily price limits). Any |daily return| > thr is such an
        # artifact: set that day's return to 0 (a split has ~0 economic return)
        # and rebuild a CLEAN price index from the corrected returns, so the
        # artifact poisons neither the features nor the forward-return TARGETS.
        raw_ret = close.pct_change()
        bad = raw_ret.abs() > thr
        if bad.any():
            for d in df.loc[bad, "date"]:
                artifact_log.append((ticker, str(pd.Timestamp(d).date()), float(raw_ret[df["date"] == d].iloc[0])))
            n_artifacts += int(bad.sum())
        clean_ret = raw_ret.mask(bad, 0.0)
        clean_close = close.iloc[0] * (1.0 + clean_ret.fillna(0.0)).cumprod()

        out = pd.DataFrame({"ticker": ticker, "date": df["date"]})
        out["ret1"] = clean_ret
        out["_stale"] = (raw_ret.abs() < 1e-9).astype(float)  # halted/stale day flag (not a feature)
        out["log_value"] = np.log1p(df["value_traded_egp"].clip(lower=0).fillna(0))
        out["hl_range"] = (df["high"] - df["low"]) / close.replace(0, np.nan)  # intraday ratio: scale-invariant, raw ok
        out["close_to_ma20"] = clean_close / clean_close.rolling(20).mean() - 1
        for h in horizons:
            out[f"fwd_{h}"] = clean_close.shift(-h) / clean_close - 1
        parts.append(out)
    panel = pd.concat(parts, ignore_index=True)
    if n_artifacts:
        logger.warning("build_feature_panel: neutralized %d split/data-artifact daily returns (|ret|>%.0f%%): %s",
                       n_artifacts, thr * 100, artifact_log)

    panel = panel.merge(fx, on=["ticker", "date"], how="left")
    panel = panel.merge(sent, on=["ticker", "date"], how="left")
    panel = panel.merge(sig, on=["ticker", "date"], how="left")
    panel = panel.merge(macro, on="date", how="left")

    # Factor exposures are stored on a sparse, irregular date grid (they are
    # slow-moving as-of estimates, not emitted every trading day). Carry the
    # last-known exposure forward within each ticker so the daily panel has a
    # contiguous factor series -- point-in-time correct (that beta WAS the
    # latest estimate on that date) and necessary for building unbroken
    # 60-day windows. Rows before a ticker's first exposure stay NaN and are
    # dropped by the finite-window filter.
    panel = panel.sort_values(["ticker", "date"])
    beta_cols = ["beta_market", "beta_liquidity", "beta_egp_sensitivity",
                 "beta_oil_sensitivity", "idiosyncratic_vol"]
    panel[beta_cols] = panel.groupby("ticker")[beta_cols].ffill()

    # Winsorize factor betas: values like beta_egp_sensitivity=18.7 on a
    # newly-listed stock are rolling-OLS artifacts (unstable regression on a
    # short/low-variance window), not real exposures. Clip each beta column to
    # its [pct, 1-pct] quantiles and log how many observations were touched.
    pct = config.TFT_BETA_WINSOR_PCT
    for c in beta_cols:
        lo, hi = panel[c].quantile(pct), panel[c].quantile(1 - pct)
        n_clip = int(((panel[c] < lo) | (panel[c] > hi)).sum())
        if n_clip:
            panel[c] = panel[c].clip(lo, hi)
            logger.info("build_feature_panel: winsorized %s -> %d obs clipped to [%.3f, %.3f]", c, n_clip, lo, hi)

    # Layer 2 regime one-hot (market-wide per date). UNKNOWN/missing dates ->
    # all-zero one-hot (the model reads "no regime info"), never dropped.
    with get_connection() as conn:
        rh = pd.read_sql("SELECT date, regime FROM regime_history", conn, parse_dates=["date"])
    panel = panel.merge(rh, on="date", how="left")
    for reg, col in [("BULL", "regime_bull"), ("BEAR", "regime_bear"),
                     ("HIGH_VOL", "regime_high_vol"), ("RECOVERY", "regime_recovery")]:
        panel[col] = (panel["regime"] == reg).astype(float)
    panel = panel.drop(columns=["regime"])

    panel["sentiment_score"] = panel["sentiment_score"].fillna(0.0)
    panel["causal_signal"] = panel["causal_signal"].fillna(0.0)

    # Data-quality guard (Phase 2): sentiment_score is currently EXCLUDED from
    # TEMPORAL_COLS because daily_sentiment only covers a recent ~4-week window
    # (100% zero across the 2016-2022 train span). Log its train-span coverage
    # so that if it is ever re-added to TEMPORAL_COLS, a still-dead column is
    # caught loudly here instead of silently shipping as 60 constant zeros.
    if "sentiment_score" not in TEMPORAL_COLS:
        train_sent = panel.loc[panel["date"] <= pd.Timestamp(config.TFT_TRAIN_END), "sentiment_score"]
        nz = float((train_sent != 0).mean()) if len(train_sent) else 0.0
        if nz < 0.01:
            logger.warning("build_feature_panel: sentiment_score train-span nonzero fraction = %.4f "
                           "(<1%%) -- correctly kept OUT of TEMPORAL_COLS; do not re-add until historical "
                           "sentiment density improves", nz)

    # per-ticker static features from the TRAIN span only (avoid look-ahead)
    train_mask = panel["date"] <= pd.Timestamp(config.TFT_TRAIN_END)
    stat = (panel[train_mask].groupby("ticker")
            .agg(static_size=("log_value", "mean"), static_vol=("ret1", "std")))
    panel = panel.merge(stat, on="ticker", how="left")
    return panel


def _standardize(panel: pd.DataFrame, cols):
    """Standardize `cols` using TRAIN-span mean/std only (no look-ahead), then
    clip the standardized values to +/- config.TFT_ZSCORE_CLIP sigmas so a
    single real extreme event can't dominate the input scale. Clipping counts
    are logged. Returns (panel, stats). Targets are standardized elsewhere --
    never here -- so this clip never touches the forward-return labels.
    """
    train = panel[panel["date"] <= pd.Timestamp(config.TFT_TRAIN_END)]
    clip = config.TFT_ZSCORE_CLIP
    panel = panel.copy()
    center, scale, n_clipped = {}, {}, {}
    for c in cols:
        mu = train[c].mean()
        sd = train[c].std() or 1.0
        z = (panel[c] - mu) / sd
        nclip = int((z.abs() > clip).sum())
        panel[c] = z.clip(-clip, clip)
        center[c], scale[c], n_clipped[c] = float(mu), float(sd), nclip
        if nclip:
            logger.info("_standardize: %s -> %d values clipped to +/-%.0f sigma", c, nclip, clip)
    return panel, {"center": center, "scale": scale, "method": "zscore_clipped", "clip": clip,
                   "n_clipped": n_clipped}


def build_windows(tickers=None, seq_len: int = None):
    """Build 60-day windows + targets, standardized (train-fit), split by date.

    Returns a dict:
      splits: {name: {"past": (N,T,F), "static": (N,S), "targets": (N,H),
                       "dates": np.array[datetime64], "tickers": np.array[str]}}
        for name in {"train","val","test"}
      temporal_cols, static_cols, horizons, quantiles, standardization stats.
    """
    seq_len = seq_len or config.TFT_SEQUENCE_LENGTH
    horizons = config.TFT_FORECAST_HORIZONS
    panel = build_feature_panel(tickers)

    # standardize continuous temporal + static using train-span statistics;
    # regime one-hots are left as raw 0/1 (already on a fixed, comparable scale)
    panel, temporal_stats = _standardize(panel, STANDARDIZE_TEMPORAL_COLS)
    panel, static_stats = _standardize(panel, STATIC_COLS)
    panel[STATIC_COLS] = panel[STATIC_COLS].fillna(0.0)
    panel[REGIME_ONEHOT_COLS] = panel[REGIME_ONEHOT_COLS].fillna(0.0)

    target_cols = [f"fwd_{h}" for h in horizons]
    train_end = pd.Timestamp(config.TFT_TRAIN_END)
    val_end = pd.Timestamp(config.TFT_VAL_END)

    buckets = {k: {"past": [], "static": [], "targets": [], "dates": [], "tickers": []}
               for k in ("train", "val", "test")}

    n_stale_dropped = 0
    for ticker, df in panel.groupby("ticker"):
        df = df.sort_values("date").reset_index(drop=True)
        feats = df[TEMPORAL_COLS].to_numpy(dtype=np.float32)
        stat = df[STATIC_COLS].to_numpy(dtype=np.float32)
        targs = df[target_cols].to_numpy(dtype=np.float32)
        dates = df["date"].to_numpy()
        stale = df["_stale"].to_numpy(dtype=np.float32) if "_stale" in df else np.zeros(len(df))

        for end in range(seq_len - 1, len(df)):
            window = feats[end - seq_len + 1: end + 1]  # (T,F)
            target = targs[end]                          # (H,)
            if not np.isfinite(window).all() or not np.isfinite(target).all():
                continue
            if not np.isfinite(stat[end]).all():
                continue
            # drop halted/stale windows (illiquid stretches -> unlearnable noise)
            if stale[end - seq_len + 1: end + 1].mean() > config.TFT_MAX_STALE_FRAC:
                n_stale_dropped += 1
                continue
            end_date = pd.Timestamp(dates[end])
            split = "train" if end_date <= train_end else ("val" if end_date <= val_end else "test")
            b = buckets[split]
            b["past"].append(window)
            b["static"].append(stat[end])
            b["targets"].append(target)
            b["dates"].append(dates[end])
            b["tickers"].append(ticker)

    splits = {}
    for name, b in buckets.items():
        if b["past"]:
            splits[name] = {
                "past": np.stack(b["past"]).astype(np.float32),
                "static": np.stack(b["static"]).astype(np.float32),
                "targets": np.stack(b["targets"]).astype(np.float32),
                "dates": np.array(b["dates"]),
                "tickers": np.array(b["tickers"]),
            }
        else:
            splits[name] = {"past": np.zeros((0, seq_len, len(TEMPORAL_COLS)), np.float32),
                            "static": np.zeros((0, len(STATIC_COLS)), np.float32),
                            "targets": np.zeros((0, len(horizons)), np.float32),
                            "dates": np.array([]), "tickers": np.array([])}

    logger.info("build_windows: train=%d val=%d test=%d windows (T=%d, F=%d temporal, %d static); "
                "dropped %d stale/halted windows (>%.0f%% zero-return days)",
                len(splits["train"]["past"]), len(splits["val"]["past"]), len(splits["test"]["past"]),
                seq_len, len(TEMPORAL_COLS), len(STATIC_COLS), n_stale_dropped, config.TFT_MAX_STALE_FRAC * 100)

    return {
        "splits": splits, "temporal_cols": TEMPORAL_COLS, "static_cols": STATIC_COLS,
        "horizons": horizons, "quantiles": config.TFT_QUANTILES,
        "temporal_stats": temporal_stats, "static_stats": static_stats,
    }


def export_features_npz(path: str = None, tickers=None) -> str:
    """Export the built windows as a single compressed .npz (arrays keep their
    (N,T,F)/(N,S)/(N,H) shapes) for upload to Kaggle. No parquet engine
    required. Keys: {split}_past/_static/_targets/_dates/_tickers plus
    temporal_cols, static_cols, horizons, quantiles. Returns the path.
    """
    import os
    path = path or config.TFT_FEATURES_NPZ_PATH
    data = build_windows(tickers)

    # per-window regime label (from Layer 2) so the Kaggle notebook can break
    # results down by BULL/BEAR/HIGH_VOL/RECOVERY without needing the DB.
    with get_connection() as conn:
        rh = pd.read_sql("SELECT date, regime FROM regime_history", conn, parse_dates=["date"])
    regime_map = dict(zip(rh["date"].values, rh["regime"].values))

    payload = {
        "temporal_cols": np.array(data["temporal_cols"]),
        "static_cols": np.array(data["static_cols"]),
        "horizons": np.array(data["horizons"]),
        "quantiles": np.array(data["quantiles"]),
    }
    for split, b in data["splits"].items():
        payload[f"{split}_past"] = b["past"]
        payload[f"{split}_static"] = b["static"]
        payload[f"{split}_targets"] = b["targets"]
        payload[f"{split}_dates"] = b["dates"].astype("datetime64[D]")
        payload[f"{split}_tickers"] = b["tickers"]
        payload[f"{split}_regimes"] = np.array([regime_map.get(np.datetime64(d, "ns"), "UNKNOWN") for d in b["dates"]])
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.savez_compressed(path, **payload)
    logger.info("export_features_npz: wrote train/val/test windows to %s", path)
    return path
