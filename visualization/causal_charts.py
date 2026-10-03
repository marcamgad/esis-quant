"""
Layer 5 causal-analysis chart (spec 5.4): a 2x3 dark-theme figure
summarizing the Double ML + IV causal signal extraction. Built on the
shared theme in visualization/theme.py.

Panels (spec 5.4):
  (1,1) Naive OLS vs Causal (DML) treatment effect, with 95% CIs + bias.
  (1,2) First-stage instrument strength (per-instrument F, threshold F=10).
  (1,3) DML partialling-out scatter: D-tilde vs Y-tilde, slope = theta,
        colored by regime.
  (2,1) Information Coefficient at 5d/10d/20d: naive momentum vs causal signal.
  (2,2) Lasso regularization path for the confounder controls (Y ~ X).
  (2,3) Causal-signal distribution by market regime (violin).

DATA-HONESTY: with no daily sentiment yet, the treatment shown in (1,1)/(1,3)
is MOMENTUM (the estimand with real data). Titles/annotations say so
explicitly rather than plotting an all-zero sentiment effect as if it were
informative.
"""

import logging
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from visualization.theme import BG_COLOR, TEXT_COLOR, REGIME_COLORS, style_axis
from models.causal.lasso import lasso_path, lasso_cv
from models.causal import signals as causal_signals

logger = logging.getLogger("esis_quant.viz.causal")

_H_LIST = [5, 10, 20]


def _pick_treatment_result(results: dict):
    """Prefer the momentum DML result (real data); fall back to sentiment if
    that is the one that actually ran. Returns (label, dml_result_dict).
    """
    if results.get("dml_momentum") and not results["dml_momentum"].get("insufficient"):
        return "Momentum", results["dml_momentum"]
    return "Sentiment", results.get("dml_sentiment", {})


def _ci(theta, se, z=1.96):
    if se is None or not np.isfinite(se):
        return 0.0
    return z * se


def _panel_naive_vs_causal(ax, label, dml):
    naive = dml.get("naive", {"theta": 0.0, "se": np.nan})
    xs = ["Naive OLS", "Causal (DML)"]
    thetas = [naive.get("theta", 0.0), dml.get("theta", 0.0)]
    errs = [_ci(naive.get("theta", 0.0) and naive.get("se"), naive.get("se")), _ci(dml.get("theta"), dml.get("se"))]
    colors = ["#e67e22", "#2ecc71"]
    ax.bar(xs, thetas, yerr=errs, color=colors, capsize=6, edgecolor=TEXT_COLOR, linewidth=0.5)
    bias = thetas[0] - thetas[1]
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.6, alpha=0.5)
    ax.set_title(f"OLS vs Causal (DML) {label} Effect\nconfounding bias = {bias:+.4f}", fontsize=11)
    ax.set_ylabel("effect on 5d fwd return")
    style_axis(ax)


def _panel_instrument_strength(ax, iv):
    per = iv.get("per_instrument_f", {})
    if not per:
        joint = iv.get("first_stage_f", {}).get("f_stat", np.nan)
        per = {"joint": joint}
    names = list(per.keys())
    fvals = [per[n] for n in names]
    colors = ["#2ecc71" if f >= config.CAUSAL_WEAK_IV_F_THRESHOLD else "#e74c3c" for f in fvals]
    ax.bar(names, fvals, color=colors, edgecolor=TEXT_COLOR, linewidth=0.5)
    ax.axhline(config.CAUSAL_WEAK_IV_F_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.2,
               label=f"weak-IV threshold (F={config.CAUSAL_WEAK_IV_F_THRESHOLD:g})")
    ax.set_yscale("log")
    ax.set_title("Instrument Strength (First-Stage F)", fontsize=11)
    ax.set_ylabel("F-statistic (log scale)")
    ax.legend(facecolor=BG_COLOR, edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.tick_params(axis="x", labelrotation=15)
    style_axis(ax)


def _panel_dml_scatter(ax, label, dml):
    d = dml.get("D_tilde"); y = dml.get("Y_tilde")
    if d is None or y is None:
        ax.text(0.5, 0.5, "no DML residuals", ha="center", va="center", color=TEXT_COLOR, transform=ax.transAxes)
        style_axis(ax); return
    regimes = dml.get("regime")
    # subsample for a legible scatter
    n = len(d)
    idx = np.random.default_rng(0).choice(n, size=min(n, 4000), replace=False)
    if regimes is not None:
        for reg, col in REGIME_COLORS.items():
            m = regimes[idx] == reg
            if m.any():
                ax.scatter(d[idx][m], y[idx][m], s=4, alpha=0.3, color=col, label=reg)
    else:
        ax.scatter(d[idx], y[idx], s=4, alpha=0.3, color="#3498db")
    theta = dml.get("theta", 0.0)
    xline = np.linspace(np.nanpercentile(d, 1), np.nanpercentile(d, 99), 50)
    ax.plot(xline, theta * xline, color=TEXT_COLOR, linewidth=1.5, label=f"slope theta={theta:.4f}")
    ax.set_title(f"DML Partialling-Out ({label})", fontsize=11)
    ax.set_xlabel("D-tilde (residualized treatment)")
    ax.set_ylabel("Y-tilde (residualized return)")
    ax.legend(facecolor=BG_COLOR, edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR, fontsize=7, markerscale=2)
    style_axis(ax)


def _mean_ic(panel: pd.DataFrame, signal_col: str, horizon: int) -> float:
    """Average cross-sectional Pearson IC between `signal_col` at date t and
    the h-day forward return, computed per date then averaged over dates.
    """
    fwd_col = f"_fwd_{horizon}"
    if fwd_col not in panel.columns:
        return np.nan
    ics = []
    for _, g in panel.dropna(subset=[signal_col, fwd_col]).groupby("date"):
        if len(g) >= 5 and g[signal_col].std() > 0 and g[fwd_col].std() > 0:
            ics.append(np.corrcoef(g[signal_col], g[fwd_col])[0, 1])
    return float(np.nanmean(ics)) if ics else np.nan


def _panel_ic(ax, panel, signals_df):
    merged = panel.merge(signals_df[["ticker", "date", "causal_signal"]], on=["ticker", "date"], how="left")
    naive_ic = [_mean_ic(merged, "momentum", h) for h in _H_LIST]
    causal_ic = [_mean_ic(merged, "causal_signal", h) for h in _H_LIST]
    x = np.arange(len(_H_LIST)); w = 0.38
    ax.bar(x - w / 2, naive_ic, w, label="naive momentum", color="#e67e22", edgecolor=TEXT_COLOR, linewidth=0.5)
    ax.bar(x + w / 2, causal_ic, w, label="causal signal", color="#2ecc71", edgecolor=TEXT_COLOR, linewidth=0.5)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.6, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{h}d" for h in _H_LIST])
    ax.set_title("Information Coefficient: Naive vs Causal", fontsize=11)
    ax.set_ylabel("mean cross-sectional IC")
    ax.legend(facecolor=BG_COLOR, edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    style_axis(ax)


def _panel_lasso_path(ax, panel):
    cols = causal_signals.CONFOUNDER_BETAS + ["usd_egp_chg", "brent_chg"]
    df = panel.dropna(subset=["fwd_ret"] + cols)
    if len(df) < config.CAUSAL_MIN_OBS:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", color=TEXT_COLOR, transform=ax.transAxes)
        style_axis(ax); return
    X = df[cols].to_numpy(dtype=float); y = df["fwd_ret"].to_numpy()
    path = lasso_path(X, y)
    best_lambda = lasso_cv(X, y)["best_lambda"]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(cols)))
    for j, name in enumerate(cols):
        ax.plot(path["lambdas"], path["coefs"][:, j], color=cmap[j], label=name, linewidth=1.2)
    ax.axvline(best_lambda, color=TEXT_COLOR, linestyle="--", linewidth=1.2, label=f"CV lambda={best_lambda:.4g}")
    ax.set_xscale("log")
    ax.set_title("Lasso Regularization Path (Confounders)", fontsize=11)
    ax.set_xlabel("lambda (log scale)"); ax.set_ylabel("standardized coefficient")
    ax.legend(facecolor=BG_COLOR, edgecolor=TEXT_COLOR, labelcolor=TEXT_COLOR, fontsize=6, ncol=2)
    style_axis(ax)


def _panel_signal_by_regime(ax, panel, signals_df):
    merged = signals_df.merge(panel[["ticker", "date", "regime"]], on=["ticker", "date"], how="left")
    order = [r for r in ["BULL", "BEAR", "HIGH_VOL", "RECOVERY"] if r in merged["regime"].unique()]
    data = [merged.loc[merged["regime"] == r, "causal_signal"].dropna().to_numpy() for r in order]
    data = [d for d in data if len(d) > 0]
    order = [r for r, d in zip(order, [merged.loc[merged["regime"] == r, "causal_signal"].dropna().to_numpy() for r in order]) if len(d) > 0]
    if not data:
        ax.text(0.5, 0.5, "no regime-tagged signals", ha="center", va="center", color=TEXT_COLOR, transform=ax.transAxes)
        style_axis(ax); return
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, r in zip(parts["bodies"], order):
        pc.set_facecolor(REGIME_COLORS.get(r, "#888")); pc.set_alpha(0.6)
    if "cmeans" in parts:
        parts["cmeans"].set_color(TEXT_COLOR)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order, rotation=15)
    ax.set_title("Causal Signal Distribution by Regime", fontsize=11)
    ax.set_ylabel("causal_signal (z-scored)")
    style_axis(ax)


def generate_causal_charts(run_output: dict = None, tickers=None, save_path: str = None) -> str:
    """Build the Layer 5 causal-analysis figure. Runs the Layer 5 pipeline
    if `run_output` (the dict from models.causal.signals.run) isn't supplied.
    Returns the saved PNG path.
    """
    if run_output is None:
        run_output = causal_signals.run(tickers=tickers, store=False)
    panel = run_output["panel"].copy()
    results = run_output["results"]
    signals_df = run_output["signals"]

    # precompute forward returns at the IC horizons (for the IC panel)
    panel = _attach_forward_returns(panel, tickers, _H_LIST)

    label, dml = _pick_treatment_result(results)
    iv = results.get("iv_momentum", {})

    save_path = save_path or os.path.join(config.CHARTS_DIR, "causal_analysis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.patch.set_facecolor(BG_COLOR)

    _panel_naive_vs_causal(axes[0, 0], label, dml)
    _panel_instrument_strength(axes[0, 1], iv)
    _panel_dml_scatter(axes[0, 2], label, dml)
    _panel_ic(axes[1, 0], panel, signals_df)
    _panel_lasso_path(axes[1, 1], panel)
    _panel_signal_by_regime(axes[1, 2], panel, signals_df)

    fig.suptitle("ESIS-QUANT Layer 5: Causal Signal Extraction (Double ML + IV) - Egyptian Exchange",
                 color=TEXT_COLOR, fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("generate_causal_charts: saved %s", save_path)
    return save_path


def _attach_forward_returns(panel: pd.DataFrame, tickers, horizons) -> pd.DataFrame:
    """Add _fwd_h columns (h-day forward return) per ticker from close prices."""
    tickers = tickers or [t["ticker"] for t in config.EGX_SEED_TICKERS]
    prices = causal_signals._load_prices(tickers)
    parts = []
    for ticker, df in prices.groupby("ticker"):
        df = df.sort_values("date").reset_index(drop=True)
        row = {"ticker": ticker, "date": df["date"]}
        for h in horizons:
            row[f"_fwd_{h}"] = (df["close"].shift(-h) / df["close"] - 1).values
        parts.append(pd.DataFrame(row))
    fwd = pd.concat(parts, ignore_index=True)
    base = panel.drop(columns=[c for c in panel.columns if c.startswith("_fwd_")], errors="ignore")
    return base.merge(fwd, on=["ticker", "date"], how="left")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(generate_causal_charts())
