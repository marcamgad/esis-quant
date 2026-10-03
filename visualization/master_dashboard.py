"""
Master dashboard (Phase D1): one figure presenting the REAL, cost-adjusted
ESIS-QUANT system -- min-variance core + walk-forward BEAR regime overlay --
against its benchmarks, with risk metrics and attribution. Reuses the shared
dark theme in visualization/theme.py. Nothing here is a mock-up; every panel
is computed from portfolio.system.run_full_system on the live database.
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
from portfolio.system import run_full_system

logger = logging.getLogger("esis_quant.viz.dashboard")


def _nav(r):
    return (1 + r).cumprod()


def generate_master_dashboard(result: dict = None, save_path: str = None) -> str:
    result = result or run_full_system()
    ret = result["returns"]
    metrics = result["metrics"]
    latest = result["latest"]
    save_path = save_path or os.path.join(config.CHARTS_DIR, "master_dashboard.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor(BG_COLOR)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1], hspace=0.35, wspace=0.25)

    # (row 1) NAV curves, full width
    ax = fig.add_subplot(gs[0, :]); style_axis(ax)
    ax.plot(_nav(ret["equal_weight"]).index, _nav(ret["equal_weight"]), color="#888", lw=1.2, label="equal-weight (market)")
    ax.plot(_nav(ret["core"]).index, _nav(ret["core"]), color="#3498db", lw=1.2, label="min-variance core")
    ax.plot(_nav(ret["full_system"]).index, _nav(ret["full_system"]), color="#2ecc71", lw=1.8, label="FULL SYSTEM (core + BEAR overlay)")
    ax.set_yscale("log")
    ax.set_title("Cumulative NAV (log scale) — net of 0.55% round-trip costs", color=TEXT_COLOR, fontsize=13)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    # (2,0) drawdown comparison
    ax = fig.add_subplot(gs[1, 0]); style_axis(ax)
    for r, c, lab in [(ret["equal_weight"], "#888", "market"), (ret["full_system"], "#2ecc71", "full system")]:
        nav = _nav(r); dd = nav / nav.cummax() - 1
        ax.fill_between(dd.index, dd * 100, 0, color=c, alpha=0.4, label=lab)
    ax.set_title("Drawdown (%)", color=TEXT_COLOR, fontsize=11)
    ax.legend(facecolor=BG_COLOR, labelcolor=TEXT_COLOR, fontsize=8)

    # (2,1) regime timeline + exposure
    ax = fig.add_subplot(gs[1, 1]); style_axis(ax)
    regime = result["regime"]; exp = result["exposure"]
    ax.plot(exp.index, exp, color="#f0f0f0", lw=1.0, label="exposure")
    ax.fill_between(exp.index, exp, 0, color="#2ecc71", alpha=0.15)
    for reg_name, col in REGIME_COLORS.items():
        mask = regime.reindex(exp.index) == reg_name
        ax.fill_between(exp.index, 0, 1, where=mask.to_numpy(), color=col, alpha=0.12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Regime (shaded) + book exposure", color=TEXT_COLOR, fontsize=11)

    # (2,2) attribution waterfall
    ax = fig.add_subplot(gs[1, 2]); style_axis(ax)
    bh = result["attribution"]["brinson"]["total"]
    te = result["attribution"]["timing"]
    bars = {"allocation": bh.get("allocation", 0)*100, "selection": bh.get("selection", 0)*100,
            "interaction": bh.get("interaction", 0)*100, "regime\ntiming": te["timing_contrib"]*100}
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bars.values()]
    ax.bar(list(bars.keys()), list(bars.values()), color=colors, edgecolor=TEXT_COLOR, lw=0.5)
    ax.axhline(0, color=TEXT_COLOR, lw=0.6)
    ax.set_title("Attribution — cumulative contribution (%)", color=TEXT_COLOR, fontsize=11)
    ax.tick_params(axis="x", labelsize=8)

    # (3,0)+(3,1) metrics table
    ax = fig.add_subplot(gs[2, :2]); ax.axis("off")
    rows = ["Ann. return", "Sharpe", "Sortino", "Max drawdown", "Calmar", "DD duration (d)", "CF VaR 95%"]
    keys = ["ann_return", "sharpe", "sortino", "max_drawdown", "calmar", "dd_duration", "cornish_fisher_var95"]
    cols = [("Full system", "full_system"), ("Min-var core", "core"), ("Equal-weight", "equal_weight")]
    table = [[r] + [_fmt(k, metrics[ck][k]) for _, ck in cols] for r, k in zip(rows, keys)]
    t = ax.table(cellText=table, colLabels=["Metric"] + [c for c, _ in cols], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#444"); cell.set_facecolor(BG_COLOR)
        cell.set_text_props(color=("#2ecc71" if (c == 1 and r > 0) else TEXT_COLOR),
                            weight=("bold" if r == 0 else "normal"))

    # (3,2) live state
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off"); ax.set_facecolor(BG_COLOR)
    reg = latest["regime"]; rc = REGIME_COLORS.get(reg, "#888")
    lines = [
        ("As of", str(pd.Timestamp(latest["date"]).date())),
        ("Regime", reg),
        ("Book exposure", f"{latest['exposure']*100:.0f}%"),
        ("Circuit breaker", "ARMED" if latest["breaker_active"] else "OK"),
        ("Current drawdown", f"{latest['current_drawdown']*100:.1f}%"),
    ]
    ax.text(0.5, 1.0, "LIVE STATE", color=TEXT_COLOR, fontsize=13, ha="center", va="top", weight="bold")
    for i, (k, v) in enumerate(lines):
        y = 0.8 - i * 0.16
        ax.text(0.05, y, k, color="#aaa", fontsize=11, va="center")
        ax.text(0.95, y, v, color=(rc if k == "Regime" else TEXT_COLOR), fontsize=12, va="center", ha="right", weight="bold")

    fig.suptitle("ESIS-QUANT — Egyptian Stock Intelligence System  ·  risk-managed passive core (walk-forward, cost-adjusted)",
                 color=TEXT_COLOR, fontsize=15, y=0.98)
    fig.savefig(save_path, dpi=200, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)
    logger.info("generate_master_dashboard: saved %s", save_path)
    return save_path


def _fmt(key, val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if key in ("ann_return", "max_drawdown", "cornish_fisher_var95"):
        return f"{val*100:+.1f}%" if key != "cornish_fisher_var95" else f"{val*100:.2f}%"
    if key == "dd_duration":
        return f"{int(val)}"
    return f"{val:.2f}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(generate_master_dashboard())
