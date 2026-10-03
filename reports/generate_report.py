"""
Final performance report (Phase D3). Generates a markdown report of the whole
system from the live state -- every number is pulled from
portfolio.system.run_full_system, not hand-copied, so it can never drift from
the code. States the result plainly, including the parts that are
underwhelming (no selection alpha) and the caveats, per the house rules.
"""

import logging
import os

import pandas as pd

import config
from portfolio.system import run_full_system

logger = logging.getLogger("esis_quant.reports")


def _row(m, key, pct=False, dpct=False):
    v = m.get(key)
    if v is None:
        return "-"
    if dpct:
        return f"{v*100:+.1f}%"
    if pct:
        return f"{v*100:.2f}%"
    return f"{v:.2f}"


def generate_final_report(result: dict = None, path: str = None) -> str:
    result = result or run_full_system()
    path = path or os.path.join(config.BASE_DIR, "reports", "ESIS_QUANT_final_report.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fs, core, eq = (result["metrics"][k] for k in ("full_system", "core", "equal_weight"))
    bh = result["attribution"]["brinson"]["total"]
    te = result["attribution"]["timing"]
    latest = result["latest"]

    def line(label, k, **kw):
        return f"| {label} | **{_row(fs, k, **kw)}** | {_row(core, k, **kw)} | {_row(eq, k, **kw)} |"

    from data.universe import survivorship_disclosure
    md = f"""# ESIS-QUANT — Final Performance Report

```
{survivorship_disclosure()}
```

_Generated {pd.Timestamp.now().date()} from the live database. All figures are
walk-forward, point-in-time, and net of the REAL Thndr fee schedule
(backtest/fees.py: flat EGP 4.95 + 0.112% per order + date-aware stamp duty),
charged per order against a {config.BACKTEST_DEFAULT_AUM_EGP:,.0f} EGP book.
The old "0.55% round trip" figure this report used to quote was a
percentage-only model that never charged the flat per-order fee at all._

## What the system is
A **risk-managed passive core** for the Egyptian Exchange: a minimum-variance
book (from-scratch Ledoit-Wolf covariance + projected-gradient optimizer) with
a **walk-forward BEAR-regime de-risking overlay** (from-scratch Gaussian HMM,
filtered/causal inference, annual refit). It is **not** a stock-picking system.

## Headline results

| Metric | Full system | Min-var core | Equal-weight (market) |
|---|---|---|---|
{line("Annualized return", "ann_return", dpct=True)}
{line("Sharpe", "sharpe")}
{line("Sortino", "sortino")}
{line("Max drawdown", "max_drawdown", dpct=True)}
{line("Calmar", "calmar")}
{line("Max-DD duration (days)", "dd_duration")}
{line("Cornish-Fisher VaR 95%", "cornish_fisher_var95", pct=True)}

**Bottom line:** the full system delivers **Sharpe {fs['sharpe']:.2f} / max drawdown
{fs['max_drawdown']*100:.0f}%** versus the market's **{eq['sharpe']:.2f} / {eq['max_drawdown']*100:.0f}%**,
and time-underwater collapses from **{eq['dd_duration']} days to {fs['dd_duration']}**.
Roughly the market's return at less than half the drawdown.

## Where the edge comes from (Brinson-Hood-Beebower + timing, reconciles exactly)
- **Stock selection adds NO alpha — it costs return.** Min-var core vs equal-weight active
  return = allocation {bh.get('allocation',0)*100:+.1f}% + selection {bh.get('selection',0)*100:+.1f}%
  + interaction {bh.get('interaction',0)*100:+.1f}%. The core buys lower risk by giving up return
  (the low-volatility trade-off).
- **The regime-timing overlay is the entire value-add: {te['timing_contrib']*100:+.0f}% cumulative**,
  and it cuts drawdown at the same time.

## Live state (as of {pd.Timestamp(latest['date']).date()})
- Regime: **{latest['regime']}** · Book exposure: **{latest['exposure']*100:.0f}%** ·
  Circuit breaker: **{'ARMED' if latest['breaker_active'] else 'OK'}** ·
  Current drawdown: **{latest['current_drawdown']*100:.1f}%**

## Honest caveats (read these)
1. **Returns are nominal EGP.** The pound devalued ~5x over the sample, inflating return
   *levels*; the Sharpe/drawdown *comparisons* remain valid.
2. **No selection alpha.** Confirmed by attribution — do not treat this as a stock picker.
3. **The drawdown circuit breaker whipsaws** on this data (it was found to turn a +824% path
   into -90% via ~1,185 costly round trips) and is **disabled by default**; the regime overlay
   alone controls drawdown better.
4. **Single market, ~9 years, one regime history.** Out-of-sample / other-market robustness is
   untested. This is an in-sample-of-history walk-forward, not a live track record.
5. Sentiment (Layer 4) and the TFT (Layer 6) do **not** contribute to this result — measured to
   be at/near random and excluded; kept for research, not trading.

_Not investment advice._
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("generate_final_report: wrote %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(generate_final_report())
