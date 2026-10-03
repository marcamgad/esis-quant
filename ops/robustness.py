"""
Cross-market robustness test.

The EGX result (min-variance core + cash-in-BEAR overlay cuts drawdown ~half)
was selected on EGX history. The only way to know if the METHOD generalizes --
versus being an EGX artifact / in-sample luck -- is to run the identical
methodology on markets it was never tuned on and see whether the SAME effect
appears: (a) min-variance beats equal-weight risk-adjusted, and (b) the BEAR
overlay cuts drawdown without wrecking return.

This harness reuses the exact core math from the production system
(models.hmm.GaussianHMM with filtered/causal inference, Ledoit-Wolf covariance,
the mean-variance optimizer, backtest.engine.run_portfolio, risk.overlay,
backtest.analytics). What it re-implements market-agnostically: the regime
FEATURES (built from the basket itself -- no EGX-specific macro like usd/egp)
and a small economic regime LABELLER mirroring models.regime_labeler's logic.
Everything stays walk-forward and point-in-time; nothing here peeks ahead.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from models.hmm import GaussianHMM
from models.regime_pipeline import standardize_expanding
from portfolio.covariance import ledoit_wolf_shrinkage
from portfolio.mv_optimizer import mean_variance_weights
from backtest.engine import BacktestEngine
from backtest import analytics as an
from risk.overlay import apply_overlay

logger = logging.getLogger("esis_quant.robustness")

COST = 0.00275
FEATURES = ["ret20", "vol20", "breadth"]


def fetch_panel(tickers, start="2015-01-01") -> pd.DataFrame:
    """Wide (date x ticker) daily returns from Adj Close, coverage-filtered
    exactly like the production loader (a date needs >=50% of listed names)."""
    data = yf.download(tickers, start=start, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all").sort_index()
    # coverage filter (mirror data.calendar): drop stray non-session dates
    listed = data.ffill().notna() & data.bfill().notna()
    cov = (data.notna() & listed).sum(axis=1) / listed.sum(axis=1).replace(0, np.nan)
    data = data.loc[cov >= 0.50]
    return data.pct_change()


def market_features(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Market-level regime features from the basket itself: 20d market return,
    20d annualized vol, and breadth (fraction of names above their 50d MA).
    All are strictly backward-looking at each date."""
    mkt = returns_wide.mean(axis=1, skipna=True)
    price = (1 + returns_wide.fillna(0)).cumprod()
    ma50 = price.rolling(50, min_periods=25).mean()
    breadth = (price > ma50).sum(axis=1) / price.notna().sum(axis=1)
    feat = pd.DataFrame({
        "ret20": mkt.rolling(20, min_periods=10).sum(),
        "vol20": mkt.rolling(20, min_periods=10).std() * np.sqrt(252),
        "breadth": breadth,
    })
    return feat.dropna()


def _label_states(mu: np.ndarray) -> dict:
    """Economic labels for 4 HMM states (mirrors models.regime_labeler logic on
    the [ret20, vol20, breadth] feature order): lowest-return=BEAR,
    highest-vol among the rest=HIGH_VOL, highest-return among the rest=BULL,
    remainder=RECOVERY."""
    ret, vol = mu[:, 0], mu[:, 1]
    order = list(range(len(mu)))
    bear = int(np.argmin(ret))
    rest = [s for s in order if s != bear]
    high_vol = int(rest[np.argmax(vol[rest])])
    rest2 = [s for s in rest if s != high_vol]
    bull = int(rest2[np.argmax(ret[rest2])])
    recovery = [s for s in rest2 if s != bull][0]
    return {bear: "BEAR", high_vol: "HIGH_VOL", bull: "BULL", recovery: "RECOVERY"}


def walkforward_regimes(features: pd.DataFrame, min_train: int = 500) -> pd.Series:
    """Point-in-time regime labels: annual walk-forward refit + filtered
    (forward-only) decode, exactly the production design. Rows before the first
    refit boundary with enough history are dropped."""
    std = standardize_expanding(features, FEATURES).dropna()
    X = std.to_numpy()
    dates = std.index
    boundaries = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="YS")
    vintages = []
    for b in boundaries:
        mask = np.asarray(dates < b)
        if mask.sum() >= min_train:
            hmm = GaussianHMM(n_states=4, n_init=4, random_state=42, tol=1e-6, n_iter=200)
            hmm.fit(X[mask])
            vintages.append((b, hmm, _label_states(hmm.mu_)))
    if not vintages:
        raise ValueError("walkforward_regimes: not enough history to refit")

    out = pd.Series(index=dates, dtype=object)
    for i, (b, hmm, lab) in enumerate(vintages):
        end = vintages[i + 1][0] if i + 1 < len(vintages) else dates.max() + pd.Timedelta(days=1)
        active = np.asarray((dates >= b) & (dates < end))
        if not active.any():
            continue
        dec = hmm.filtered_decode(X)
        idx = np.where(active)[0]
        out.iloc[idx] = [lab[s] for s in dec[idx]]
    return out.dropna()


def _min_var_fn(window: pd.DataFrame) -> pd.Series:
    lw = ledoit_wolf_shrinkage(window.values)
    w = mean_variance_weights(np.zeros(window.shape[1]), lw["sigma"], max_weight=0.15)["weights"]
    return pd.Series(w, index=window.columns)


def _eq_fn(window: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / window.shape[1], index=window.columns)


def run_market(name: str, tickers: list, start: str = "2015-01-01") -> dict:
    """Run the full methodology on one market and return the comparison."""
    rets = fetch_panel(tickers, start)
    logger.info("[%s] %d names, %s..%s, %d sessions", name, rets.shape[1],
                rets.index.min().date(), rets.index.max().date(), len(rets))
    regimes = walkforward_regimes(market_features(rets))

    eng = BacktestEngine(rebalance_days=21, cost_per_turnover=COST)
    core = eng.run_portfolio(_min_var_fn, rets, lookback=180, tradable=None)
    eqw = eng.run_portfolio(_eq_fn, rets, lookback=180, tradable=None)
    core_ret = core["net_returns"]
    reg_d = regimes.reindex(core_ret.index).ffill()
    managed = apply_overlay(core_ret, reg_d, bear_scale=0.0, cost_per_turnover=COST)["managed_returns"]

    def m(r):
        s = an.summary(r)
        return {"sharpe": s["sharpe"], "mdd": s["max_drawdown"], "ann": s["ann_return"], "calmar": s["calmar"]}

    bear_frac = float((reg_d == "BEAR").mean())
    return {"name": name, "n_names": rets.shape[1], "bear_frac": bear_frac,
            "equal_weight": m(eqw["net_returns"]), "core": m(core_ret), "full_system": m(managed)}


# Baskets chosen for LONG, clean history on different markets (not tuned on).
MARKETS = {
    "US large-cap": ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "WMT", "HD", "DIS",
                     "CSCO", "PFE", "VZ", "INTC", "CVX", "MRK", "BA", "CAT", "MCD", "NKE",
                     "IBM", "GE", "AXP", "GS", "MMM", "T", "WFC", "UNH"],
    "India large-cap": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
                        "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
                        "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "POWERGRID.NS"],
}


def _print_result(r: dict):
    print(f"\n=== {r['name']}  ({r['n_names']} names, BEAR {r['bear_frac']*100:.0f}% of days) ===", flush=True)
    for k in ["equal_weight", "core", "full_system"]:
        d = r[k]
        print(f"  {k:14s} Sharpe {d['sharpe']:+.2f}  MDD {d['mdd']*100:6.1f}%  "
              f"Calmar {d['calmar']:+.2f}  Ann {d['ann']*100:+.1f}%", flush=True)


def run_all(markets: dict = None) -> pd.DataFrame:
    markets = markets or MARKETS
    rows = []
    for name, tickers in markets.items():
        try:
            r = run_market(name, tickers)
            rows.append(r)
            _print_result(r)   # print AS EACH market finishes, so partial results survive
        except Exception as exc:  # noqa: BLE001 -- one market failing shouldn't kill the rest
            logger.error("run_market(%s) failed: %s", name, exc)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_all()
    print("\nROBUSTNESS_DONE", flush=True)
