"""
Layer 6 TFT chart (spec 6.4): a 3x2 dark-theme figure summarizing the
Temporal Fusion Transformer's probabilistic forecasts and interpretability,
computed from the trained checkpoint over the real TEST split.

Panels:
  (1,1) Probabilistic forecast for a representative stock: realized 5-day
        forward return vs predicted median + 50% (Q25-Q75) and 90% (Q05-Q95)
        bands across test dates.
  (1,2) Temporal variable importance: mean VSN selection weight per feature.
  (2,1) Temporal attention profile: mean attention from the forecast position
        back over the 60-day lookback (memory structure).
  (2,2) Quantile calibration: nominal vs empirical coverage (diagonal = ideal).
  (3,1) Directional accuracy by horizon, vs the 50% random baseline.
  (3,2) Sample stock probabilistic forecasts (4 stocks).
"""

import logging
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from visualization.theme import BG_COLOR, TEXT_COLOR, style_axis
from models.tft.model import TemporalFusionTransformer
from models.tft.features import build_windows
from models.tft.trainer import directional_accuracy, quantile_coverage

logger = logging.getLogger("esis_quant.viz.tft")


def load_tft(checkpoint_path: str = None):
    """Load a trained TFT + its metadata from a checkpoint (the format saved
    by the training run / Kaggle notebook). Returns (model, checkpoint dict).
    """
    checkpoint_path = checkpoint_path or config.TFT_MODEL_CHECKPOINT_PATH
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = TemporalFusionTransformer(
        n_temporal=ckpt["n_temporal"], n_static=ckpt["n_static"],
        horizons=ckpt["horizons"], quantiles=ckpt["quantiles"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _predict(model, past, static, batch=512):
    preds, tw, attn = [], [], []
    with torch.no_grad():
        for i in range(0, len(past), batch):
            p = torch.tensor(past[i:i + batch]); s = torch.tensor(static[i:i + batch])
            pr, interp = model(p, s)
            preds.append(pr.numpy())
            tw.append(interp["temporal_weights"].numpy())
            attn.append(interp["attention"].numpy())
    return np.concatenate(preds), np.concatenate(tw), np.concatenate(attn)


def _forecast_panel(ax, dates, actual, preds, q, title):
    i05, i25, i50, i75, i95 = [q.index(x) for x in (0.05, 0.25, 0.5, 0.75, 0.95)]
    order = np.argsort(dates)
    d = dates[order]
    ax.fill_between(d, preds[order, i05], preds[order, i95], color="#3498db", alpha=0.2, label="90% (Q05-Q95)")
    ax.fill_between(d, preds[order, i25], preds[order, i75], color="#3498db", alpha=0.4, label="50% (Q25-Q75)")
    ax.plot(d, preds[order, i50], color="#f0f0f0", linewidth=1.0, label="median")
    ax.plot(d, actual[order], color="#2ecc71", linewidth=0.8, alpha=0.8, label="realized")
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.5, alpha=0.4)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=7)
    style_axis(ax)


def generate_tft_charts(checkpoint_path: str = None, save_path: str = None,
                        horizon_for_forecast: int = 5) -> str:
    """Build the Layer 6 figure from the trained checkpoint over the real
    test split. Returns the saved PNG path.
    """
    model, ckpt = load_tft(checkpoint_path)
    q = ckpt["quantiles"]; horizons = ckpt["horizons"]
    temporal_cols = ckpt["temporal_cols"]
    hidx = horizons.index(horizon_for_forecast) if horizon_for_forecast in horizons else 0

    data = build_windows()
    te = data["splits"]["test"]
    if len(te["past"]) == 0:
        te = data["splits"]["val"]
    preds, tw, attn = _predict(model, te["past"], te["static"])
    targets, dates, tickers = te["targets"], te["dates"], te["tickers"]

    save_path = save_path or os.path.join(config.CHARTS_DIR, "tft_analysis.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(22, 18))
    fig.patch.set_facecolor(BG_COLOR)

    # (1,1) probabilistic forecast for the most-traded representative stock
    rep = max(set(tickers.tolist()), key=lambda t: (tickers == t).sum())
    m = tickers == rep
    _forecast_panel(axes[0, 0], dates[m], targets[m, hidx], preds[m, hidx, :], q,
                    f"TFT Probabilistic {horizon_for_forecast}d Forecast - {rep}")

    # (1,2) temporal variable importance (mean VSN weight per feature, over time+samples)
    ax = axes[0, 1]
    importance = tw.mean(axis=(0, 1))  # (n_temporal,)
    order = np.argsort(importance)
    ax.barh(np.array(temporal_cols)[order], importance[order], color="#3498db", edgecolor=TEXT_COLOR, linewidth=0.4)
    ax.set_title("TFT Variable Importance (Temporal)", color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel("mean selection weight")
    style_axis(ax)

    # (2,1) temporal attention profile: mean attention from last position over lags
    ax = axes[1, 0]
    last_attn = attn[:, -1, :].mean(axis=0)  # (T,) attention from forecast pos to each past step
    lags = np.arange(len(last_attn) - 1, -1, -1)  # 59..0 (0 = most recent)
    ax.plot(lags, last_attn, color="#e67e22", linewidth=1.5)
    ax.fill_between(lags, 0, last_attn, color="#e67e22", alpha=0.25)
    ax.invert_xaxis()
    ax.set_title("Temporal Self-Attention (forecast position -> past lag)", color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel("trading days ago"); ax.set_ylabel("mean attention")
    style_axis(ax)

    # (2,2) quantile calibration
    ax = axes[1, 1]
    cov = quantile_coverage(preds, targets, q)
    ax.plot([0, 1], [0, 1], "--", color=TEXT_COLOR, alpha=0.5, label="perfect")
    ax.plot(q, cov, "o-", color="#2ecc71", label="empirical")
    for qq, cc in zip(q, cov):
        ax.annotate(f"{cc:.2f}", (qq, cc), color=TEXT_COLOR, fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title("TFT Quantile Calibration", color=TEXT_COLOR, fontsize=11)
    ax.set_xlabel("nominal quantile"); ax.set_ylabel("empirical coverage")
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    style_axis(ax)

    # (3,1) directional accuracy by horizon
    ax = axes[2, 0]
    da = directional_accuracy(preds, targets, q.index(0.5))
    colors = ["#2ecc71" if a >= 0.5 else "#e74c3c" for a in da]
    ax.bar([f"{h}d" for h in horizons], da, color=colors, edgecolor=TEXT_COLOR, linewidth=0.4)
    ax.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1.0, label="random (50%)")
    ax.set_ylim(0.3, max(0.6, da.max() + 0.05))
    ax.set_title("Directional Accuracy by Horizon", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("accuracy"); ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    style_axis(ax)

    # (3,2) sample stock forecasts (4 distinct tickers)
    ax = axes[2, 1]
    ax.axis("off")
    uniq = [t for t in dict.fromkeys(tickers.tolist())][:4]
    sub = fig.add_gridspec(2, 2, left=0.55, right=0.98, top=0.30, bottom=0.04, hspace=0.5, wspace=0.3)
    for k, tk in enumerate(uniq):
        a = fig.add_subplot(sub[k // 2, k % 2])
        mm = tickers == tk
        _forecast_panel(a, dates[mm], targets[mm, hidx], preds[mm, hidx, :], q, tk)
        a.legend_.remove() if a.get_legend() else None
    axes[2, 1].set_title("Sample Stock Forecasts", color=TEXT_COLOR, fontsize=11)

    fig.suptitle("ESIS-QUANT Layer 6: Temporal Fusion Transformer - Multi-Horizon Quantile Forecasts",
                 color=TEXT_COLOR, fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("generate_tft_charts: saved %s", save_path)
    return save_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(generate_tft_charts())
