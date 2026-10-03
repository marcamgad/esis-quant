"""
Part 4 -- no-trade band: is a threshold rule better than rebalancing every 21
days, under the REAL fee schedule?

WHY THIS AND NOT RL. The RL audit concluded that rebalance TIMING is the only
genuinely sequential decision in this system -- made sequential by the flat
EGP 2 per-order fee, which makes cost lumpy and path-dependent. It also
concluded that this problem has a well-known near-optimal solution that needs
no learning: a NO-TRADE BAND (Constantinides). Trade only when the book has
drifted far enough from target that the tracking error is worth the fee;
otherwise do nothing.

This is the honest first step. If a three-line threshold rule captures most of
the available benefit, then RL was never needed -- and that is a result worth
having before spending weeks on a policy network fitted to ~136 decisions.

THE RULE. At each session, compute drift = ||w_current - w_target||_1 (total
absolute weight difference). Rebalance only if drift > band. Otherwise hold and
let the book drift. band = 0 reproduces daily rebalancing; a very large band
means never trade.

COSTS ARE THE REAL SCHEDULE, NOT A PROXY. Every rebalance is charged per order
through backtest/fees.py: flat EGP + percentage + date-aware stamp duty. That
is the whole point -- a percentage-only cost model cannot express why waiting
is valuable, because it makes 10 small trades cost the same as 1 big one.

WHAT WOULD MAKE THIS WORTH ADOPTING: higher net CAGR than the 21-day cadence at
equal-or-lower drawdown, and a band width where that holds across a RANGE rather
than at one lucky value. A result that only works at one band is a fitted
parameter, not a rule.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger("esis_quant.no_trade_band")

TRADING_DAYS = 252
DEFAULT_BANDS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80)


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    nav = (1 + r).cumprod()
    sd = r.std(ddof=1)
    return {"cagr": float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1),
            "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan,
            "max_dd": float((nav / nav.cummax() - 1).min())}


def simulate(band: float, aum_egp: float = 100_000.0, lookback: int = 180,
             rebalance_days: int = 21, max_names: int = 25) -> dict:
    """Run the min-variance book with a drift-triggered rebalance rule.

    The TARGET is recomputed on the same cadence the live system uses; the BAND
    decides whether to actually trade toward it. band=0 with rebalance_days=21
    reproduces the current behaviour exactly, which is the control.
    """
    from data.universe import load_tradable
    from backtest.engine import BacktestEngine, load_price_returns
    from portfolio.system import _min_variance_fn
    from backtest.fees import one_side_cost

    rets = load_price_returns()
    trad = load_tradable()
    eng = BacktestEngine(rebalance_days=rebalance_days, min_names=5)
    trad_map = (trad[trad["is_tradable"] == 1].groupby("date")["ticker"].apply(set)
                if not trad.empty else None)

    dates = rets.index
    decision_days = list(dates[lookback::rebalance_days])
    w_cur = pd.Series(dtype=float)
    out = pd.Series(0.0, index=dates)
    n_rebal, total_cost_egp = 0, 0.0

    for i, T in enumerate(decision_days):
        window = rets.loc[:T].tail(lookback)
        cols = eng._eligible_names(window, T, trad_map)
        if len(cols) < 5:
            continue
        w_t = _min_variance_fn(window[cols].fillna(0.0))
        w_t = w_t[w_t > 1e-6].nlargest(max_names)
        if w_t.empty:
            continue
        w_t = w_t / w_t.sum()

        names = w_cur.index.union(w_t.index)
        drift = float((w_t.reindex(names, fill_value=0.0) - w_cur.reindex(names, fill_value=0.0)).abs().sum())

        cost_frac = 0.0
        if w_cur.empty or drift > band:
            dw = (w_t.reindex(names, fill_value=0.0) - w_cur.reindex(names, fill_value=0.0)).abs()
            c = sum(one_side_cost(float(d) * aum_egp, as_of=T,
                                  spread_per_side=0.0)["total_egp"]
                    for d in dw if d > 0)
            cost_frac = c / aum_egp
            total_cost_egp += c
            w_cur = w_t
            n_rebal += 1

        end = decision_days[i + 1] if i + 1 < len(decision_days) else dates[-1]
        hold = dates[(dates > T) & (dates <= end)]
        if len(hold) == 0:
            continue
        daily = rets.loc[hold, w_cur.index].mul(w_cur, axis=1).sum(axis=1, min_count=1).fillna(0.0)
        daily.iloc[0] -= cost_frac
        out.loc[hold] = daily.values
        # let weights drift with realized returns -- that drift is what the band
        # is deciding about, so it must not be reset silently
        gr = (1 + rets.loc[hold, w_cur.index].fillna(0.0)).prod()
        w_cur = (w_cur * gr)
        w_cur = w_cur / w_cur.sum()

    r = out[dates > decision_days[0]] if decision_days else out
    s = _stats(r)
    s.update({"band": band, "n_rebalances": n_rebal,
              "cost_egp": total_cost_egp, "cost_pct_of_aum": total_cost_egp / aum_egp})
    return s


def sweep(bands=DEFAULT_BANDS, aum_egp: float = 100_000.0) -> pd.DataFrame:
    return pd.DataFrame([simulate(b, aum_egp=aum_egp) for b in bands])


def format_sweep(df: pd.DataFrame, aum: float) -> str:
    lines = [f"NO-TRADE BAND vs FIXED 21-DAY CADENCE  (AUM {aum:,.0f} EGP, real fee schedule)",
             f"  {'band':>7}{'rebalances':>12}{'total cost':>13}{'cost %':>9}"
             f"{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}"]
    for r in df.itertuples():
        lines.append(f"  {r.band:>7.2f}{r.n_rebalances:>12}{r.cost_egp:>12,.0f}"
                     f"{r.cost_pct_of_aum*100:>8.1f}%{r.cagr*100:>8.1f}%"
                     f"{r.sharpe:>8.2f}{r.max_dd*100:>8.1f}%")
    base = df[df["band"] == 0.0]
    if not base.empty:
        b = base.iloc[0]
        best = df.loc[df["cagr"].idxmax()]
        lines.append("  " + "-" * 60)
        lines.append(f"  control (band 0, always rebalance): CAGR {b.cagr*100:.1f}%, "
                     f"cost {b.cost_pct_of_aum*100:.1f}% of AUM")
        lines.append(f"  best band {best['band']:.2f}: CAGR {best['cagr']*100:.1f}% "
                     f"({(best['cagr']-b.cagr)*100:+.1f}pp), "
                     f"{int(b.n_rebalances - best['n_rebalances'])} fewer rebalances")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    import sys
    aum = float(sys.argv[1]) if len(sys.argv) > 1 else 100_000.0
    df = sweep(aum_egp=aum)
    print(format_sweep(df, aum))
