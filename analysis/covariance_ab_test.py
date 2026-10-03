"""
Does a better volatility FORECAST produce a better PORTFOLIO?

WHY THIS TEST EXISTS SEPARATELY FROM THE FORECAST TEST. `analysis.
volatility_forecast` established that a fixed HAR-like blend of 21/63/126-day
volatility beats the system's 180-session trailing estimate on held-out data
(Spearman 0.392 vs 0.275, MAE 0.195 vs 0.220). That is a statement about
forecast accuracy, and forecast accuracy is necessary but NOT sufficient: the
optimiser does not consume a volatility, it consumes a covariance matrix, and it
is entirely possible to predict volatility better and allocate worse. Only a
portfolio built from the estimate can settle it.

TypedGP is deliberately absent from this comparison. Its best formula scored
Spearman 0.3676 -- identical to six decimal places to `vol_63` alone, because
`log(tanh(x))` is monotonic and cannot reorder anything -- so it never cleared
the baseline and has nothing to contribute here.

THE DESIGN, and why it isolates one thing. A covariance matrix factors as
D @ C @ D, with D the diagonal of volatilities and C the correlation matrix.
Both arms of this test use the SAME correlation matrix from the SAME window and
differ only in D. So any difference in realised portfolio volatility is
attributable to the volatility estimate and to nothing else -- not to a
different lookback, not to a different universe, not to a different optimiser.

Everything is walk-forward: weights at date t are built only from data before t,
and scored on the returns after it.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.covariance_ab")

TRADING_DAYS = 252
CORR_WINDOW = 180     # correlation lookback, identical in both arms
REBALANCE = 21
MAX_WEIGHT = 0.20     # same cap the live optimiser uses
MIN_NAMES = 25


def har_like(rets: pd.Series) -> float:
    """0.5*vol_21 + 0.3*vol_63 + 0.2*vol_126 -- fixed weights, nothing fitted."""
    v21 = rets.tail(21).std() * np.sqrt(TRADING_DAYS)
    v63 = rets.tail(63).std() * np.sqrt(TRADING_DAYS)
    v126 = rets.tail(126).std() * np.sqrt(TRADING_DAYS)
    if not np.isfinite(v21) or not np.isfinite(v63) or not np.isfinite(v126):
        return np.nan
    return 0.5 * v21 + 0.3 * v63 + 0.2 * v126


def trailing_180(rets: pd.Series) -> float:
    """What the system uses today."""
    return rets.tail(180).std() * np.sqrt(TRADING_DAYS)


def _min_var_weights(cov: np.ndarray, max_w: float = MAX_WEIGHT) -> np.ndarray:
    """Long-only minimum variance with a weight cap, by projected gradient.

    Deliberately simple and IDENTICAL for both arms. A cleverer solver would not
    make the comparison fairer -- it would only add a second thing that differs
    between the two runs, which is exactly what this test is designed to avoid.
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(600):
        grad = cov @ w
        w = w - 0.5 * grad / (np.linalg.norm(grad) + 1e-12) * 0.02
        w = np.clip(w, 0.0, max_w)
        s = w.sum()
        if s <= 0:
            return np.full(n, 1.0 / n)
        w = w / s
    return w


def run(rets: pd.DataFrame = None, start_after: int = 400, rebalance: int = REBALANCE,
        max_weight: float = MAX_WEIGHT, max_zero_frac: float = 0.15) -> dict:
    """Walk-forward A/B of the two volatility estimates inside one optimiser."""
    if rets is None:
        from backtest.engine import load_price_returns
        rets = load_price_returns()

    dates = rets.index
    out = {"har": [], "trail": [], "equal": [], "dates": []}
    n_rebal = 0

    for i in range(start_after, len(dates) - rebalance, rebalance):
        hist = rets.iloc[:i]
        fwd = rets.iloc[i:i + rebalance]

        # Universe: enough history, not stale, and actually trading forward.
        live = []
        for c in rets.columns:
            h = hist[c].tail(CORR_WINDOW)
            if h.notna().sum() < CORR_WINDOW * 0.8:
                continue
            if (h.fillna(0) == 0).mean() > max_zero_frac:
                continue
            if fwd[c].notna().sum() < rebalance // 2:
                continue
            live.append(c)
        if len(live) < MIN_NAMES:
            continue

        win = hist[live].tail(CORR_WINDOW).fillna(0.0)
        corr = win.corr().to_numpy()
        if not np.all(np.isfinite(corr)):
            continue

        v_har = np.array([har_like(hist[c].dropna()) for c in live])
        v_trl = np.array([trailing_180(hist[c].dropna()) for c in live])
        ok = np.isfinite(v_har) & np.isfinite(v_trl) & (v_har > 0) & (v_trl > 0)
        if ok.sum() < MIN_NAMES:
            continue
        live = [c for c, k in zip(live, ok) if k]
        corr = corr[np.ix_(ok, ok)]
        v_har, v_trl = v_har[ok], v_trl[ok]

        fwd_r = fwd[live].fillna(0.0).to_numpy()

        for label, vol in (("har", v_har), ("trail", v_trl)):
            D = np.diag(vol)
            cov = D @ corr @ D
            w = _min_var_weights(cov, max_weight)
            out[label].append(fwd_r @ w)

        out["equal"].append(fwd_r @ np.full(len(live), 1.0 / len(live)))
        out["dates"].append(dates[i])
        n_rebal += 1

    if n_rebal < 8:
        return {"ok": False, "reason": f"only {n_rebal} rebalances"}

    res = {"ok": True, "n_rebalances": n_rebal,
           "first": str(pd.Timestamp(out["dates"][0]).date()),
           "last": str(pd.Timestamp(out["dates"][-1]).date())}
    series = {}
    for label in ("har", "trail", "equal"):
        r = np.concatenate(out[label])
        series[label] = r
        eq = np.cumprod(1.0 + r)
        res[label] = {
            "realised_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
            "mean_ann": float(r.mean() * TRADING_DAYS),
            "max_dd": float((eq / np.maximum.accumulate(eq) - 1.0).min()),
            "n_days": int(len(r)),
        }
    # The question is whether the difference is bigger than the noise in it.
    d = series["har"] - series["trail"]
    res["paired"] = {
        "vol_reduction_pp": 100.0 * (res["trail"]["realised_vol"] - res["har"]["realised_vol"]),
        "vol_reduction_pct": 100.0 * (1.0 - res["har"]["realised_vol"] / res["trail"]["realised_vol"]),
        "return_diff_ann_pp": 100.0 * (res["har"]["mean_ann"] - res["trail"]["mean_ann"]),
        "daily_diff_t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if d.std(ddof=1) > 0 else 0.0,
    }
    return res


def format_report(r: dict) -> str:
    if not r.get("ok"):
        return f"COVARIANCE A/B — {r.get('reason')}"
    L = ["=" * 68,
         "COVARIANCE A/B — does a better vol forecast build a better book?",
         "=" * 68,
         f"  walk-forward, {r['n_rebalances']} rebalances, {r['first']} to {r['last']}",
         "  identical correlation matrix and optimiser in both arms;",
         "  ONLY the volatility estimate differs.", "",
         f"  {'arm':22}{'realised vol':>14}{'ann return':>13}{'max DD':>10}"]
    for label, name in (("trail", "vol_180 (incumbent)"), ("har", "har_like blend"),
                        ("equal", "equal weight (ref)")):
        d = r[label]
        L.append(f"  {name:22}{d['realised_vol']*100:>13.2f}%{d['mean_ann']*100:>12.2f}%"
                 f"{d['max_dd']*100:>9.1f}%")
    p = r["paired"]
    L += ["",
          f"  volatility reduction: {p['vol_reduction_pp']:+.2f} pp "
          f"({p['vol_reduction_pct']:+.1f}%)",
          f"  return difference:    {p['return_diff_ann_pp']:+.2f} pp/yr "
          f"(paired t = {p['daily_diff_t']:+.2f})",
          "",
          "  Lower realised volatility is the PASS condition. A return difference",
          "  is not evidence either way: mu = 0 in both arms, so neither is trying",
          "  to earn more, and any gap is noise the test cannot attribute."]
    return "\n".join(L)
