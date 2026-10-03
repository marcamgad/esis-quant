"""
What is the BEAR overlay actually worth once cash stops earning zero?

THE QUESTION. The system's only measured edge is "go to cash in BEAR". The
backtest credited that cash with 0% return. Egypt's overnight rate ran roughly
8%-27% across the sample and the book sits in cash ~30% of days, so the 0%
assumption systematically understates the one leg that carries the edge.

THE METHOD, AND WHY IT IS A SWEEP AND NOT A POINT ESTIMATE. The CBE rate series
in data/rates.py is a RECONSTRUCTION from published MPC decisions, not fetched
data (no free feed exists). Reporting a single number off an unverified series
would be exactly the kind of false precision this project's audit exists to
prevent. So the headline output is a SENSITIVITY SWEEP: the same backtest re-run
with cash earning 0%, 5%, 10%, 15%, 20%, 25%, plus the reconstructed schedule.
If the conclusion holds across that whole range, it does not depend on the
reconstruction being right -- and that is the only claim worth making.

REAL TERMS MATTER MORE HERE THAN ANYWHERE ELSE. High EGP rates are largely
compensation for devaluation. Nominal cash yield therefore flatters the result;
the USD column is the honest one, and it is reported alongside every row. A
20% EGP yield while the pound falls 20% is roughly flat in real terms -- but
that is still enormously better than 0%, which is a guaranteed real loss.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.cash_yield")

SWEEP_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def _summarize(returns: pd.Series) -> dict:
    from analysis import real_returns as rr
    r = returns.dropna()
    ann = (1 + r).prod() ** (252 / len(r)) - 1 if len(r) else float("nan")
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    nav = (1 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1).min())
    usd = rr.to_hard_currency(r)
    u = usd.dropna()
    usd_ann = (1 + u).prod() ** (252 / len(u)) - 1 if len(u) else float("nan")
    usd_sd = u.std(ddof=1)
    usd_sharpe = float(u.mean() / usd_sd * np.sqrt(252)) if usd_sd > 0 else float("nan")
    return {"cagr": float(ann), "sharpe": sharpe, "max_dd": mdd,
            "usd_cagr": float(usd_ann), "usd_sharpe": usd_sharpe}


def run_sweep(rates=None, use_reconstructed: bool = True, lookback: int = 180,
              rebalance_days: int = 21) -> pd.DataFrame:
    """Re-run the full system at each assumed cash rate. One row per rate."""
    import config
    from data.database import get_connection
    from data.universe import load_tradable
    from data.rates import daily_cash_return
    from backtest.engine import BacktestEngine, load_price_returns
    from portfolio.system import _min_variance_fn
    from risk.overlay import apply_overlay

    rates = SWEEP_RATES if rates is None else rates
    from backtest.fees import book_pct_per_side
    COST = book_pct_per_side(config.BACKTEST_DEFAULT_AUM_EGP)

    rets = load_price_returns()
    trad = load_tradable()
    with get_connection() as conn:
        reg = pd.read_sql("SELECT date, regime FROM regime_history", conn,
                          parse_dates=["date"]).set_index("date")["regime"]

    eng = BacktestEngine(rebalance_days=rebalance_days, cost_per_turnover=COST)
    core = eng.run_portfolio(_min_variance_fn, rets, lookback=lookback, tradable=trad)
    core_ret = core["net_returns"]
    regime = reg.reindex(core_ret.index).ffill()

    rows = []
    scenarios = [(f"{r*100:.0f}% flat", dict(flat_rate=r)) for r in rates]
    if use_reconstructed:
        scenarios.append(("CBE reconstructed", dict()))

    for label, kw in scenarios:
        cash_daily = daily_cash_return(core_ret.index, **kw)
        ov = apply_overlay(core_ret, regime, bear_scale=0.0,
                           cost_per_turnover=COST, cash_daily=cash_daily)
        s = _summarize(ov["managed_returns"])
        s["scenario"] = label
        s["avg_cash_rate"] = float(((1 + cash_daily) ** 252 - 1).mean())
        s["pct_days_in_cash"] = float((ov["exposure"] < 0.5).mean())
        rows.append(s)

    # the un-overlaid core, for reference (always fully invested -> no cash leg)
    c = _summarize(core_ret)
    c.update({"scenario": "CORE (no overlay)", "avg_cash_rate": float("nan"),
              "pct_days_in_cash": 0.0})
    rows.append(c)
    return pd.DataFrame(rows)[["scenario", "avg_cash_rate", "pct_days_in_cash",
                               "cagr", "sharpe", "max_dd", "usd_cagr", "usd_sharpe"]]


def format_sweep(df: pd.DataFrame) -> str:
    lines = ["CASH-YIELD SENSITIVITY -- what the BEAR overlay is worth when cash earns something",
             "(nominal EGP and real USD side by side; USD is the honest column)",
             f"  {'scenario':<20}{'cash%':>7}{'in-cash':>9}{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>8}"
             f"{'usdCAGR':>9}{'usdShrp':>9}"]
    for r in df.itertuples():
        cr = f"{r.avg_cash_rate*100:.1f}" if r.avg_cash_rate == r.avg_cash_rate else "  -"
        lines.append(f"  {r.scenario:<20}{cr:>7}{r.pct_days_in_cash*100:>8.0f}%"
                     f"{r.cagr*100:>8.1f}%{r.sharpe:>8.2f}{r.max_dd*100:>7.1f}%"
                     f"{r.usd_cagr*100:>8.1f}%{r.usd_sharpe:>9.2f}")
    lines.append("  NOTE: the CBE row uses a RECONSTRUCTED, unverified rate schedule "
                 "(data/rates.py) --")
    lines.append("  trust the SHAPE across the flat rows, not that single row's precision.")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    import warnings
    warnings.filterwarnings("ignore")
    print(format_sweep(run_sweep()))
