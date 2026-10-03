"""
Layer 3 factor-model chart. Adapted from the original 4x2, 8-panel spec
to what's actually available:
  (1,1) Cumulative factor returns (5 factors, not 9 -- see
        models/factor_returns module docstring)
  (1,2) Factor correlation heatmap
  (2,1) Rolling 252d factor Sharpe ratios
  (2,2) Fama-MacBeth risk premia with Newey-West 95% CI
  (3,1) Regime-conditional factor Sharpe (wires Layer 2's regime_history)
  (3,2) Current cross-sectional factor exposure distribution (box plot)
  (4,1) Alpha score evolution, most recent top 10 stocks
  (4,2) Current top-10 alpha scores (bar) -- substitutes for the spec's
        "Information ratio by sector" panel, since Layer 1 has no sector
        data for any ticker (see config.EGX_SEED_TICKERS docstring)
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from models.factor_model import BETA_COLUMNS
from models.factor_returns import FACTOR_NAMES
from visualization.theme import BG_COLOR, TEXT_COLOR, REGIME_COLORS, style_axis

logger = logging.getLogger("esis_quant.factor_charts")

BETA_TO_FACTOR = {"beta_market": "MKT", "beta_momentum": "MOM", "beta_liquidity": "LIQ",
                  "beta_egp_sensitivity": "EGP", "beta_oil_sensitivity": "OIL"}
FACTOR_COLORS = {"MKT": "#3498db", "MOM": "#2ecc71", "LIQ": "#e67e22", "EGP": "#9b59b6", "OIL": "#e74c3c"}


def _rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean / std) * np.sqrt(252)


def plot_factor_analysis(factor_returns: pd.DataFrame, exposures_df: pd.DataFrame, premia_df: pd.DataFrame,
                          regime_history: pd.DataFrame, alpha_scores_df: pd.DataFrame,
                          output_path: str = None) -> str:
    output_path = output_path or os.path.join(config.CHARTS_DIR, "factor_analysis.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(24, 20))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("ESIS-QUANT: Factor Model Analysis (5 of 9 spec factors -- see factor_returns docstring)",
                 color=TEXT_COLOR, fontsize=14)

    # (1,1) Cumulative factor returns
    ax = axes[0, 0]
    style_axis(ax)
    for factor in FACTOR_NAMES:
        cumulative = (1 + factor_returns[factor]).cumprod()
        ax.plot(factor_returns.index, cumulative, label=factor, color=FACTOR_COLORS[factor], linewidth=1.0)
    ax.set_title("Cumulative Factor Returns", color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR)

    # (1,2) Factor correlation heatmap
    ax = axes[0, 1]
    style_axis(ax)
    corr = factor_returns[FACTOR_NAMES].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FACTOR_NAMES)))
    ax.set_yticks(range(len(FACTOR_NAMES)))
    ax.set_xticklabels(FACTOR_NAMES, color=TEXT_COLOR)
    ax.set_yticklabels(FACTOR_NAMES, color=TEXT_COLOR)
    for i in range(len(FACTOR_NAMES)):
        for j in range(len(FACTOR_NAMES)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Factor Correlation Matrix", color=TEXT_COLOR)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (2,1) Rolling 252d factor Sharpe
    ax = axes[1, 0]
    style_axis(ax)
    for factor in FACTOR_NAMES:
        sharpe = _rolling_sharpe(factor_returns[factor])
        ax.plot(factor_returns.index, sharpe, label=factor, color=FACTOR_COLORS[factor], linewidth=1.0)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    ax.set_title("Rolling 252-Day Factor Sharpe Ratios", color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR)

    # (2,2) Fama-MacBeth premia with Newey-West 95% CI
    ax = axes[1, 1]
    style_axis(ax)
    factor_rows = premia_df.loc[BETA_COLUMNS]
    y_pos = np.arange(len(BETA_COLUMNS))
    ci95 = 1.96 * factor_rows["se"]
    colors = ["#2ecc71" if (p > 0 and abs(t) > 1.96) else ("#e74c3c" if p < 0 and abs(t) > 1.96 else "#7f8c8d")
              for p, t in zip(factor_rows["premium"], factor_rows["t_stat"])]
    ax.barh(y_pos, factor_rows["premium"], xerr=ci95, color=colors, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([BETA_TO_FACTOR[c] for c in BETA_COLUMNS], color=TEXT_COLOR)
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8)
    ax.set_title(f"Fama-MacBeth Monthly Risk Premia (95% CI, {int(factor_rows['n_months'].iloc[0])} months)",
                 color=TEXT_COLOR)

    # (3,1) Regime-conditional factor Sharpe
    ax = axes[2, 0]
    style_axis(ax)
    merged = factor_returns.join(regime_history.set_index("date")["regime"], how="inner")
    regimes = ["BULL", "BEAR", "HIGH_VOL", "RECOVERY"]
    width = 0.15
    for i, factor in enumerate(FACTOR_NAMES):
        sharpes = []
        for regime in regimes:
            sub = merged.loc[merged["regime"] == regime, factor]
            sharpes.append((sub.mean() / sub.std()) * np.sqrt(252) if len(sub) > 5 and sub.std() > 0 else 0.0)
        ax.bar(np.arange(len(regimes)) + i * width, sharpes, width, label=factor, color=FACTOR_COLORS[factor])
    tick_positions = np.arange(len(regimes)) + width * (len(FACTOR_NAMES) - 1) / 2
    ax.set_xticks(tick_positions, labels=regimes)
    ax.tick_params(axis="x", colors=TEXT_COLOR)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    ax.set_title("Factor Sharpe Ratio by Regime", color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # (3,2) Current cross-sectional factor exposure distribution
    ax = axes[2, 1]
    style_axis(ax)
    latest_date = exposures_df["date"].max()
    latest = exposures_df[exposures_df["date"] == latest_date]
    box_data = [latest[c].dropna().to_numpy() for c in BETA_COLUMNS]
    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=[BETA_TO_FACTOR[c] for c in BETA_COLUMNS])
    for patch, factor in zip(bp["boxes"], [BETA_TO_FACTOR[c] for c in BETA_COLUMNS]):
        patch.set_facecolor(FACTOR_COLORS[factor])
        patch.set_alpha(0.6)
    ax.set_title(f"Cross-Sectional Factor Exposure Distribution ({latest_date.date()})", color=TEXT_COLOR)

    # (4,1) Alpha score evolution, top 10 most recent
    ax = axes[3, 0]
    style_axis(ax)
    latest_alpha_date = alpha_scores_df["date"].max()
    top10 = (alpha_scores_df[alpha_scores_df["date"] == latest_alpha_date]
             .nlargest(10, "alpha_score")["ticker"].tolist())
    earliest_date = alpha_scores_df["date"].min()
    lookback_start = max(
        latest_alpha_date - pd.Timedelta(days=90),
        earliest_date,
    )
    for ticker in top10:
        sub = alpha_scores_df[(alpha_scores_df["ticker"] == ticker) & (alpha_scores_df["date"] >= lookback_start)]
        ax.plot(sub["date"], sub["alpha_score"], label=ticker, linewidth=1.0)
    ax.axhline(0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    actual_days = (latest_alpha_date - lookback_start).days
    ax.set_title(
        f"Alpha Score Evolution -- Current Top 10 "
        f"(last {actual_days}d as of {latest_alpha_date.date()})",
        color=TEXT_COLOR,
    )
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=7, ncol=2)

    # (4,2) Current top-10 alpha scores (substitutes for "IR by sector" -- no sector data, see module docstring)
    ax = axes[3, 1]
    style_axis(ax)
    latest_scores = (alpha_scores_df[alpha_scores_df["date"] == latest_alpha_date]
                     .nlargest(10, "alpha_score").sort_values("alpha_score"))
    ax.barh(latest_scores["ticker"], latest_scores["alpha_score"], color="#2ecc71")
    ax.set_title(f"Current Top 10 Alpha Scores ({latest_alpha_date.date()})", color=TEXT_COLOR)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close(fig)

    logger.info("Saved factor analysis chart to %s", output_path)
    return output_path
