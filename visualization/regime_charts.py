"""
Layer 2 regime chart. Adapted from the original 5-subplot spec to what
Layer 1 can actually supply:
  1. Market-proxy price level (NOT the real EGX30 -- see
     models/regime_features module docstring) with regime-colored background
  2. Regime probability stacked area
  3. Realized volatility (20d annualized) with 20%/40% reference lines
  4. Market breadth (% of universe above 50d MA)
  5. USD/EGP -- CBE rate-decision markers from the spec are omitted since
     Layer 1's CBE scraper is not implemented (cbe.org.eg blocks plain
     HTTP requests; see data/collectors.py scrape_cbe_decisions).
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from visualization.theme import REGIME_COLORS, BG_COLOR, TEXT_COLOR, style_axis as _style_axis

logger = logging.getLogger("esis_quant.regime_charts")


def _shade_regimes(ax, regime_series: pd.Series):
    """Draw axvspan blocks for contiguous same-regime runs."""
    labels = regime_series.to_numpy()
    dates = regime_series.index
    run_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[run_start]:
            ax.axvspan(dates[run_start], dates[i - 1], color=REGIME_COLORS.get(labels[run_start], "#888888"),
                       alpha=0.15, linewidth=0)
            run_start = i


def plot_regime_analysis(result: dict, output_path: str = None) -> str:
    """result: the dict returned by models.regime_pipeline.run_regime_detection.
    Saves a 5-panel PNG and returns its path.
    """
    obs_df = result["obs_df"]
    regime_series = result["regime_series"]
    gamma = result["gamma"]
    gamma_columns = result["gamma_columns"]

    output_path = output_path or os.path.join(config.CHARTS_DIR, "regime_analysis.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(
        5, 1, figsize=(20, 16), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5, 1.5]},
    )
    fig.patch.set_facecolor(BG_COLOR)

    # --- Panel 1: market proxy level + regime shading ---------------------
    ax = axes[0]
    _style_axis(ax)
    _shade_regimes(ax, regime_series)
    ax.plot(obs_df.index, obs_df["market_proxy_level"], color="#f0f0f0", linewidth=1.2)
    ax.set_ylabel("Market-proxy level (base=100)")
    ax.set_title(
        "ESIS-QUANT: Market Regime Analysis -- Egyptian Exchange\n"
        "(market-proxy index built from seed-ticker universe; NOT the real EGX30 -- see regime_features docstring)",
        fontsize=12,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in REGIME_COLORS.values()]
    ax.legend(handles, REGIME_COLORS.keys(), loc="upper left", facecolor=BG_COLOR, labelcolor=TEXT_COLOR, ncol=4)

    # --- Panel 2: regime probability stacked area --------------------------
    ax = axes[1]
    _style_axis(ax)
    name_map = {"prob_bull": "BULL", "prob_bear": "BEAR", "prob_high_vol": "HIGH_VOL", "prob_recovery": "RECOVERY"}
    stack_colors = [REGIME_COLORS[name_map[c]] for c in gamma_columns]
    ax.stackplot(obs_df.index, gamma.T, colors=stack_colors, labels=[name_map[c] for c in gamma_columns], alpha=0.8)
    ax.set_ylabel("P(regime)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", facecolor=BG_COLOR, labelcolor=TEXT_COLOR, ncol=4, fontsize=8)

    # --- Panel 3: realized volatility ---------------------------------------
    ax = axes[2]
    _style_axis(ax)
    ax.plot(obs_df.index, obs_df["o2_realized_vol_20d"], color="#f39c12", linewidth=1.0)
    ax.axhline(0.20, color=TEXT_COLOR, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(0.40, color=TEXT_COLOR, linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_ylabel("20d realized vol (ann.)")

    # --- Panel 4: breadth -----------------------------------------------------
    ax = axes[3]
    _style_axis(ax)
    breadth = obs_df["o3_breadth"].to_numpy()
    ax.plot(obs_df.index, breadth, color="#9b59b6", linewidth=1.0)
    ax.axhline(0.5, color=TEXT_COLOR, linestyle="--", alpha=0.4, linewidth=0.8)
    half = np.full_like(breadth, 0.5)
    ax.fill_between(
        obs_df.index, half, breadth,
        where=(breadth >= 0.5), color="#2ecc71", alpha=0.3, interpolate=True,
    )
    ax.fill_between(
        obs_df.index, half, breadth,
        where=(breadth < 0.5), color="#e74c3c", alpha=0.3, interpolate=True,
    )
    ax.set_ylabel("Breadth (% > 50d MA)")
    ax.set_ylim(0, 1)

    # --- Panel 5: USD/EGP -----------------------------------------------------
    ax = axes[4]
    _style_axis(ax)
    ax.plot(obs_df.index, obs_df["usd_egp"], color="#1abc9c", linewidth=1.0)
    ax.set_ylabel("USD/EGP")
    ax.set_xlabel("Date")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)

    logger.info("Saved regime analysis chart to %s", output_path)
    return output_path
