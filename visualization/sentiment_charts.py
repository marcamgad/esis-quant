"""
Layer 4 sentiment chart. Adapted from the original 6-panel spec to what's
actually available:
  (1,1) Market-wide sentiment index (unchanged)
  (1,2) Sentiment-by-TICKER heatmap, not by sector -- Layer 1 has no
        sector data for any ticker (see config.EGX_SEED_TICKERS docstring)
  (2,1) Sentiment vs forward-return scatter with regime color + OLS line
        (manual OLS via data.linalg, manual normal-approximation p-value
        -- not an exact Student-t p-value, see _ols_with_ci docstring)
  (2,2) News volume by stock, last 7 days (unchanged)
  (3,1) Calibration curve, before/after manual from-scratch temperature
        scaling -- computed against whatever real trained model + held-out
        predictions exist; with the current tiny/sparse dataset this is a
        proof-of-pipeline exercise, not a claim of good calibration
  (3,2) Sentiment IC decay across 1d/2d/3d/5d/10d/20d horizons (unchanged)

HONESTY NOTE: several panels will show "insufficient data" placeholders
rather than fabricated curves, because as of this build there are only a
handful of real scraped news articles and no deep sentiment history --
see models/sentiment/dataset.py module docstring.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
from data.database import get_connection
from data.linalg import ols_fit
from visualization.theme import BG_COLOR, TEXT_COLOR, REGIME_COLORS, style_axis

logger = logging.getLogger("esis_quant.sentiment_charts")

IC_HORIZONS = [1, 2, 3, 5, 10, 20]


def _insufficient_data_panel(ax, title: str, reason: str):
    style_axis(ax)
    ax.set_title(title, color=TEXT_COLOR)
    ax.text(0.5, 0.5, f"Insufficient data:\n{reason}", ha="center", va="center",
            color=TEXT_COLOR, fontsize=10, transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _ols_with_ci(x: np.ndarray, y: np.ndarray):
    """Manual OLS y = a + b*x with the slope's standard error, t-stat, and
    a NORMAL-APPROXIMATION two-sided p-value (2*(1-Phi(|t|))), via
    math.erf -- not an exact Student-t p-value (that needs the incomplete
    beta function, excessive machinery for a chart annotation). This
    slightly understates p-values at small N; fine for a diagnostic
    annotation, not for a formal hypothesis test at small samples.
    Returns dict(slope, intercept, se_slope, t_stat, p_value, y_pred, ci_lo, ci_hi).
    """
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta = ols_fit(X, y)
    residuals = y - X @ beta
    dof = max(n - 2, 1)
    sigma2 = (residuals @ residuals) / dof
    XtX_inv = np.linalg.inv(X.T @ X + 1e-12 * np.eye(2))
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    se_slope = se[1]
    t_stat = beta[1] / se_slope if se_slope > 0 else 0.0

    import math
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    x_grid = np.linspace(x.min(), x.max(), 100)
    X_grid = np.column_stack([np.ones(100), x_grid])
    y_pred = X_grid @ beta
    pred_var = sigma2 * np.einsum("ij,jk,ik->i", X_grid, XtX_inv, X_grid)
    se_pred = np.sqrt(np.maximum(pred_var, 0))
    ci_lo, ci_hi = y_pred - 1.96 * se_pred, y_pred + 1.96 * se_pred

    return {"slope": beta[1], "intercept": beta[0], "se_slope": se_slope, "t_stat": t_stat,
            "p_value": p_value, "x_grid": x_grid, "y_pred": y_pred, "ci_lo": ci_lo, "ci_hi": ci_hi}


def fit_temperature(logits: np.ndarray, labels: np.ndarray, lr: float = 0.05, n_steps: int = 200) -> float:
    """Temperature scaling (Guo et al. 2017): find scalar T > 0 minimizing
    NLL of softmax(logits/T) on held-out data, via gradient descent on
    log(T) (keeps T positive without a constrained optimizer).
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([log_T], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        optimizer.step()
    return float(torch.exp(log_T).item())


def reliability_curve(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10):
    """Per-bin (mean confidence, empirical accuracy) pairs for a
    reliability diagram. Bins with zero samples are omitted (NaN).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf.append(confidences[mask].mean())
        bin_acc.append(correctness[mask].mean())
    return np.array(bin_conf), np.array(bin_acc)


def information_coefficient(sentiment: pd.Series, forward_returns: pd.Series) -> float:
    """IC = Pearson correlation between today's sentiment score and the
    N-day-ahead forward return, computed manually (no scipy.stats).
    """
    aligned = pd.concat([sentiment, forward_returns], axis=1).dropna()
    if len(aligned) < 3:
        return np.nan
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _load_data():
    with get_connection() as conn:
        sentiment = pd.read_sql("SELECT * FROM daily_sentiment", conn, parse_dates=["date"])
        prices = pd.read_sql("SELECT ticker, date, adjusted_close AS close FROM daily_ohlcv ORDER BY ticker, date", conn, parse_dates=["date"])
        regimes = pd.read_sql("SELECT date, regime FROM regime_history", conn, parse_dates=["date"])
        news = pd.read_sql("SELECT published_at, tickers_mentioned FROM news_articles WHERE tickers_mentioned != ''",
                            conn, parse_dates=["published_at"])
    return sentiment, prices, regimes, news


def plot_sentiment_analysis(eval_result: dict = None, output_path: str = None) -> str:
    """eval_result: optional dict from models.sentiment.trainer.evaluate(),
    plus raw val logits/labels if calibration plotting is wanted (keys
    "val_logits", "val_labels" -- not produced by evaluate() itself, pass
    them separately if available). Everything else is read from the DB.
    """
    sentiment, prices, regimes, news = _load_data()
    output_path = output_path or os.path.join(config.CHARTS_DIR, "sentiment_analysis.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("ESIS-QUANT: Sentiment Analysis (Layer 4)", color=TEXT_COLOR, fontsize=14)

    # (1,1) Market-wide sentiment index
    ax = axes[0, 0]
    style_axis(ax)
    if sentiment.empty:
        _insufficient_data_panel(ax, "EGX Market Sentiment Index", "no daily_sentiment rows yet")
    else:
        market_index = sentiment.groupby("date")["sentiment_score"].mean()
        colors = np.where(market_index >= 0, "#2ecc71", "#e74c3c")
        ax.bar(market_index.index, market_index, color=colors, width=0.8)
        if len(market_index) >= 20:
            ax.plot(market_index.index, market_index.rolling(20).mean(), color="#f0f0f0", linewidth=1.2, label="20d avg")
            ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR)
        ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
        ax.set_title("EGX Market Sentiment Index", color=TEXT_COLOR)

    # (1,2) Sentiment by TICKER heatmap (substitutes sector -- no sector data, see module docstring)
    ax = axes[0, 1]
    style_axis(ax)
    if sentiment.empty:
        _insufficient_data_panel(ax, "Sentiment by Ticker Heatmap (Last 60 Days)", "no daily_sentiment rows yet")
    else:
        recent = sentiment[sentiment["date"] >= sentiment["date"].max() - pd.Timedelta(days=60)]
        pivot = recent.pivot_table(index="ticker", columns="date", values="sentiment_score")
        im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, color=TEXT_COLOR, fontsize=7)
        ax.set_xticks([])
        ax.set_title("Sentiment by Ticker Heatmap (Last 60 Days)", color=TEXT_COLOR)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2,1) Sentiment vs forward returns scatter
    ax = axes[1, 0]
    style_axis(ax)
    merged = sentiment.merge(prices, on=["ticker", "date"], how="inner").sort_values(["ticker", "date"])
    if not merged.empty:
        merged["fwd_return_5d"] = merged.groupby("ticker")["close"].shift(-5) / merged["close"] - 1
        merged = merged.merge(regimes, on="date", how="left")
    plot_data = merged.dropna(subset=["sentiment_score", "fwd_return_5d"]) if not merged.empty else pd.DataFrame()

    if len(plot_data) < 5:
        _insufficient_data_panel(ax, "Sentiment Predictability: Score vs Forward Return",
                                  f"only {len(plot_data)} (sentiment, forward-return) pairs available")
    else:
        for regime, color in REGIME_COLORS.items():
            sub = plot_data[plot_data["regime"] == regime]
            if len(sub):
                ax.scatter(sub["sentiment_score"], sub["fwd_return_5d"], color=color, label=regime, alpha=0.7, s=20)
        fit = _ols_with_ci(plot_data["sentiment_score"].to_numpy(), plot_data["fwd_return_5d"].to_numpy())
        ax.plot(fit["x_grid"], fit["y_pred"], color="#f0f0f0", linewidth=1.5)
        ax.fill_between(fit["x_grid"], fit["ci_lo"], fit["ci_hi"], color="#f0f0f0", alpha=0.2)
        corr = plot_data["sentiment_score"].corr(plot_data["fwd_return_5d"])
        ax.set_title(f"Sentiment Predictability: Score vs Forward Return (r={corr:.2f}, p={fit['p_value']:.3f})",
                     color=TEXT_COLOR, fontsize=10)
        ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=7)

    # (2,2) News volume by stock, last 7 days
    ax = axes[1, 1]
    style_axis(ax)
    if news.empty:
        _insufficient_data_panel(ax, "News Volume by Stock -- Last 7 Days", "no tagged news articles yet")
    else:
        cutoff = news["published_at"].max() - pd.Timedelta(days=7)
        recent_news = news[news["published_at"] >= cutoff]
        counts = recent_news["tickers_mentioned"].str.split(",").explode().value_counts().head(20)
        if counts.empty:
            _insufficient_data_panel(ax, "News Volume by Stock -- Last 7 Days", "no ticker mentions in last 7 days")
        else:
            latest_sentiment = sentiment.sort_values("date").groupby("ticker")["sentiment_score"].last()
            colors = [plt.cm.RdYlGn(0.5 + 0.5 * latest_sentiment.get(t, 0.0)) for t in counts.index]
            ax.bar(counts.index, counts.values, color=colors)
            ax.tick_params(axis="x", rotation=45, colors=TEXT_COLOR)
            ax.set_title("News Volume by Stock -- Last 7 Days", color=TEXT_COLOR)

    # (3,1) Calibration curve, before/after temperature scaling
    ax = axes[2, 0]
    style_axis(ax)
    if eval_result is not None and "val_logits" in eval_result and len(eval_result.get("val_logits", [])) >= 20:
        logits = np.asarray(eval_result["val_logits"])
        labels = np.asarray(eval_result["val_labels"])
        probs_before = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        conf_before = probs_before.max(axis=1)
        pred_before = probs_before.argmax(axis=1)
        correct_before = (pred_before == labels).astype(float)

        T = fit_temperature(logits, labels)
        probs_after = torch.softmax(torch.tensor(logits) / T, dim=-1).numpy()
        conf_after = probs_after.max(axis=1)

        bc_before, ba_before = reliability_curve(conf_before, correct_before)
        bc_after, ba_after = reliability_curve(conf_after, correct_before)
        ax.plot([0, 1], [0, 1], "--", color=TEXT_COLOR, alpha=0.5, label="perfect calibration")
        ax.plot(bc_before, ba_before, "o-", color="#e74c3c", label="before scaling")
        ax.plot(bc_after, ba_after, "o-", color="#2ecc71", label=f"after scaling (T={T:.2f})")
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
        ax.set_title("Sentiment Model Calibration", color=TEXT_COLOR)
    else:
        _insufficient_data_panel(ax, "Sentiment Model Calibration",
                                  "no validation logits passed -- run models.sentiment.trainer.evaluate() "
                                  "on a real held-out set first")

    # (3,2) Sentiment alpha decay curve
    ax = axes[2, 1]
    style_axis(ax)
    if merged.empty:
        _insufficient_data_panel(ax, "Sentiment Signal Alpha Decay", "no (sentiment, price) overlap yet")
    else:
        ics = []
        for h in IC_HORIZONS:
            fwd = merged.groupby("ticker")["close"].shift(-h) / merged["close"] - 1
            ic = information_coefficient(merged["sentiment_score"], fwd)
            ics.append(ic)
        if all(np.isnan(ics)):
            _insufficient_data_panel(ax, "Sentiment Signal Alpha Decay", "too few samples per horizon for a correlation")
        else:
            ax.plot([str(h) + "d" for h in IC_HORIZONS], ics, "o-", color="#3498db")
            ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
            ax.set_title("Sentiment Signal Alpha Decay", color=TEXT_COLOR)
            ax.set_ylabel("Information Coefficient")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)

    logger.info("Saved sentiment analysis chart to %s", output_path)
    return output_path
