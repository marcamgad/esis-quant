"""
Shared dark-theme styling for all ESIS-QUANT charts (extracted from
regime_charts.py once factor_charts.py needed the same helpers -- keeps
one canonical definition instead of copy-pasted styling constants
silently drifting apart between chart modules).
"""

REGIME_COLORS = {
    "BULL": "#2ecc71",
    "BEAR": "#e74c3c",
    "HIGH_VOL": "#e67e22",
    "RECOVERY": "#3498db",
}

BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_ALPHA = 0.2


def style_axis(ax):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.grid(True, alpha=GRID_ALPHA)
