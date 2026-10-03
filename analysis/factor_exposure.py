"""
Is the "alpha" actually a factor tilt?  (audit 2026-08-05, P6.22)

THE QUESTION. The core book is min-variance with mu=0. It has beaten equal-weight
on Sharpe across the sample. Two explanations fit that equally well:

  (a) the covariance estimate is genuinely good, or
  (b) min-variance systematically buys low-beta, low-volatility names, and the
      low-volatility anomaly is a well-documented factor that has paid off in
      many markets for decades.

(b) is not alpha. It is a known risk premium available to anyone who sorts on
volatility, and it would be dishonest to bill it as skill. The audit scored
Section 28 🟡 precisely because nobody had checked.

METHOD. Regress the strategy's daily excess returns on factors built from THIS
repo's own price data -- no external factor library, because none exists for
EGX and inventing one would be worse than not testing:

  MKT  equal-weighted return of the tradable universe (the only market proxy
       available; config.py records that no reliable EGX30 history could be
       sourced from Yahoo under any symbol tried)
  SIZE small-minus-big by trailing traded VALUE, the only size proxy available
       without market-cap data (value traded correlates with size but is not
       size -- stated so the proxy is never mistaken for the real factor)
  MOM  winners-minus-losers on 12-1 month returns (Jegadeesh-Titman formation:
       252-day lookback skipping the most recent 21 days)
  VOL  low-minus-high trailing 60-day volatility -- the factor min-variance is
       most likely to be loading on, and therefore the one that matters here

READING THE RESULT. If alpha is insignificant once VOL is included, the strategy
is a low-volatility tilt with extra steps. If alpha survives, something else is
going on. Either answer is publishable; only hiding the question is not.

Newey-West standard errors (lag 5) because daily portfolio returns are
autocorrelated and OLS errors would overstate significance -- the same mistake
the overlay timing test made before its overlap correction.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.factor_exposure")

TRADING_DAYS = 252
NW_LAG = 5


def _newey_west_se(X: np.ndarray, resid: np.ndarray, lag: int = NW_LAG) -> np.ndarray:
    """HAC standard errors. Daily portfolio returns are autocorrelated; plain OLS
    errors would make every loading look more significant than it is."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (resid[:, None] * X).T @ (resid[:, None] * X)
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        u = resid[L:, None] * X[L:]
        v = resid[:-L, None] * X[:-L]
        G = u.T @ v
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def build_factors(rets: pd.DataFrame = None, vol_window: int = 60,
                  mom_lookback: int = 252, mom_skip: int = 21) -> pd.DataFrame:
    """Long-short factor returns from this repo's own price panel.

    Every factor is built from information available STRICTLY BEFORE the day it
    prices: the sort is on data through T-1 and the return earned is T's.
    """
    from backtest.engine import load_price_returns, load_value_traded
    rets = load_price_returns() if rets is None else rets
    vt = load_value_traded()

    mkt = rets.mean(axis=1, skipna=True)

    def _ls(rank_df: pd.DataFrame, high_minus_low: bool) -> pd.Series:
        """Top-third minus bottom-third, rebalanced daily on lagged ranks."""
        r = rank_df.shift(1)                      # sort on yesterday, earn today
        out = pd.Series(index=rets.index, dtype=float)
        for d in rets.index:
            if d not in r.index:
                continue
            row = r.loc[d].dropna()
            if len(row) < 6:
                continue
            n = max(len(row) // 3, 1)
            hi = row.nlargest(n).index
            lo = row.nsmallest(n).index
            today = rets.loc[d]
            a, b = today.reindex(hi).mean(), today.reindex(lo).mean()
            if a == a and b == b:
                out.loc[d] = (a - b) if high_minus_low else (b - a)
        return out

    vol = rets.rolling(vol_window, min_periods=vol_window // 2).std()
    mom = (1 + rets).rolling(mom_lookback - mom_skip, min_periods=100).apply(
        np.prod, raw=True).shift(mom_skip) - 1
    size_proxy = vt.reindex(index=rets.index, columns=rets.columns).rolling(
        20, min_periods=10).mean() if not vt.empty else None

    f = {"MKT": mkt,
         "VOL": _ls(vol, high_minus_low=False),      # LOW minus high volatility
         "MOM": _ls(mom, high_minus_low=True)}       # winners minus losers
    if size_proxy is not None:
        f["SIZE"] = _ls(size_proxy, high_minus_low=False)   # SMALL minus big
    return pd.DataFrame(f).dropna(how="all")


def regress(strategy: pd.Series, factors: pd.DataFrame = None) -> dict:
    """Regress strategy returns on the factors. Alpha is the intercept,
    annualised, with a Newey-West t-stat."""
    factors = build_factors() if factors is None else factors
    df = pd.concat([strategy.rename("y"), factors], axis=1).dropna()
    if len(df) < 100:
        return {"ok": False, "reason": f"only {len(df)} overlapping observations"}
    y = df["y"].values
    names = [c for c in df.columns if c != "y"]
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in names])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = _newey_west_se(X, resid)
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"ok": True, "n": len(df), "labels": ["ALPHA"] + names,
            "coef": beta, "t": t,
            "alpha_annual": float(beta[0] * TRADING_DAYS),
            "alpha_t": float(t[0]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan}


def format_result(res: dict, label: str) -> str:
    if not res.get("ok"):
        return f"{label}: {res.get('reason')}"
    lines = [f"FACTOR REGRESSION — {label}  (n={res['n']}, R²={res['r2']:.3f}, Newey-West lag {NW_LAG})",
             f"  {'factor':>8}{'loading':>10}{'t-stat':>9}"]
    for name, c, t in zip(res["labels"], res["coef"], res["t"]):
        shown = c * TRADING_DAYS if name == "ALPHA" else c
        lines.append(f"  {name:>8}{shown:>10.3f}{t:>9.2f}")
    sig = abs(res["alpha_t"]) >= 2.0
    lines.append(f"  annualised alpha {res['alpha_annual']*100:+.2f}%  "
                 f"t={res['alpha_t']:.2f}  ->  "
                 f"{'SIGNIFICANT' if sig else 'NOT distinguishable from factor exposure'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from portfolio.system import run_full_system
    r = run_full_system()
    f = build_factors()
    for key in ("core", "full_system"):
        print(format_result(regress(r["returns"][key], f), key))
        print()
