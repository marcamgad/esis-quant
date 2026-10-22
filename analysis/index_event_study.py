"""
EGX30 index inclusion / exclusion event study.

HYPOTHESIS AND WHY IT IS MORE PROMISING THAN NEWS OR MOMENTUM. Funds that track
EGX30 must buy added stocks and sell removed ones on a known date. That is a
mechanical, non-informational flow, which is the kind of effect that can survive
in a thin market where information-based edges do not.

PRE-REGISTERED DESIGN (fixed in this docstring before any result was seen):

  events       data/egx_index_events.py; one per (ticker, review), +1 added,
               -1 removed.
  PRESSURE     entry: close of the first session strictly AFTER the announcement.
               exit:  close of the first session after the effective date (or one
               session after entry, whichever is later). Prediction: added stocks
               beat the market, removed stocks lag it.
  REVERSAL     entry: that exit close. exit: 20 sessions later. Prediction: the
               pressure fades -- added stocks lag, removed stocks recover.
  score        predicted sign x (stock return - equal-weight market return).
  baseline     the same tickers on random valid sessions at least 20 sessions from
               any index event for that ticker, same window lengths, same predicted
               signs; 2,000 draws. One-sided p = share of placebo means >= real.
  tests        PRIMARY: pressure (all events) and reversal (all events), Bonferroni
               across those 2. The added-only and removed-only splits are reported
               but are descriptive, not tests.
  verdict      PASS: n >= 20, mean score > 0, corrected p < 0.05.

  run-up       the 20 sessions BEFORE the announcement are reported descriptively.
               If added stocks already rose there, the change was anticipated
               (brokers publish forecasts, e.g. Pharos in July 2018), which is
               useful to know but is not scored.

REPLICATION (registered 2026-10-03, after the events below were recovered and
BEFORE any of them was measured). data/egx_index_events.RECOVERED_2026_10_03
holds 35 events from 5 reviews (Jan 2021, Jul 2021, Jan 2022, Jan 2023, Jul
2025) that were NOT in the original test. They are an out-of-sample check:
  test     PRESSURE window only (the one that passed), same entry/exit, same
           score, same 2,000-draw placebo, ONE test so no Bonferroni.
  verdict  REPLICATES: n >= 20, mean > 0, p < 0.05, and the cluster check (all 5
           reviews present, majority of review means positive, positive without
           the largest review). Otherwise FAILS TO REPLICATE.
  also     additions-only typical-price implementability, descriptive.
A failure here outweighs the original pass: these events were chosen by nothing
but what free sources happened to keep.

BATCH B (registered 2026-10-03, after batch A FAILED and before B was measured).
RECOVERED_2026_10_03_B: 20 events from Jan 2015, Jan 2016, Jan 2017 (Al Mal
archive). Same test as batch A. With 3 reviews it cannot meet the 5-review cluster
rule, so its best possible verdict is SUPPORTIVE (n >= 20, mean > 0, p < 0.05)
or NOT SUPPORTIVE. It is reported beside A and NEVER overturns A's verdict:
looking for more data after a failure is exactly how a dead effect gets revived.

LIMITS. About 70 measurable events across 11 reviews, 2017-2026, some reviews
partial. Before September 2026 announcements came only 1-4 days before the
effective date, so the pressure window is short. The equal-weight market
benchmark includes the stocks themselves; with ~200 names the effect is small.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.index_event_study")

N_PLACEBO = 2000
EXCLUSION = 20
REVERSAL_H = 20
RUNUP_H = 20
MIN_EVENTS = 20
MIN_REVIEWS = 5
ALPHA = 0.05


def _panel():
    from backtest.engine import load_price_returns
    rets = load_price_returns()
    return rets, rets.mean(axis=1, skipna=True)


def _window_ar(s: np.ndarray, m: np.ndarray, start: int, end: int):
    """Abnormal return over sessions (start, end]; None if the stock mostly did not trade."""
    if start < 0 or end >= len(s) or end <= start:
        return None
    w, mm = s[start + 1:end + 1], m[start + 1:end + 1]
    if np.isfinite(w).sum() < max(1, int(0.6 * (end - start))):
        return None
    return float(np.prod(1 + np.nan_to_num(w)) - np.prod(1 + np.nan_to_num(mm)))


def positions(idx: pd.DatetimeIndex, announced, effective):
    ann = idx.searchsorted(pd.Timestamp(announced), side="right")   # first session after announcement
    eff = idx.searchsorted(pd.Timestamp(effective), side="left")     # first session on/after effective
    exit_p = max(ann + 1, eff + 1)
    return ann, eff, exit_p


def measure(events: pd.DataFrame, rets: pd.DataFrame, market: pd.Series) -> pd.DataFrame:
    idx = rets.index
    m = market.to_numpy()
    rows = []
    for e in events.itertuples():
        if e.ticker not in rets.columns:
            continue
        s = rets[e.ticker].to_numpy()
        ann, eff, exit_p = positions(idx, e.announced, e.effective)
        if ann >= len(idx):
            continue
        rec = {"ticker": e.ticker, "announced": e.announced, "effective": e.effective,
               "side": e.side, "pressure_len": exit_p - ann}
        rec["pressure_ar"] = _window_ar(s, m, ann, exit_p)
        rec["reversal_ar"] = _window_ar(s, m, exit_p, exit_p + REVERSAL_H)
        rec["runup_ar"] = _window_ar(s, m, ann - 1 - RUNUP_H, ann - 1)
        rec["pressure_score"] = None if rec["pressure_ar"] is None else e.side * rec["pressure_ar"]
        rec["reversal_score"] = None if rec["reversal_ar"] is None else -e.side * rec["reversal_ar"]
        rows.append(rec)
    return pd.DataFrame(rows)


def placebo(measured: pd.DataFrame, events: pd.DataFrame, rets: pd.DataFrame, market: pd.Series,
            window: str, n_draws: int = N_PLACEBO, seed: int = 0) -> np.ndarray:
    """Mean-score distribution on random sessions, same tickers, lengths and signs."""
    rng = np.random.default_rng(seed)
    idx = rets.index
    m = market.to_numpy()
    col = f"{window}_score"
    d = measured.dropna(subset=[col])
    if d.empty:
        return np.array([])
    ev_pos = {}
    for e in events.itertuples():
        ev_pos.setdefault(e.ticker, []).append(idx.searchsorted(pd.Timestamp(e.effective)))
    pools, lens, signs, ticks = {}, [], [], []
    for r in d.itertuples():
        t = r.ticker
        if t not in pools:
            valid = np.where(np.isfinite(rets[t].to_numpy()))[0]
            mask = np.ones(len(valid), bool)
            for p in ev_pos.get(t, []):
                mask &= np.abs(valid - p) > EXCLUSION
            pools[t] = valid[mask]
        L = r.pressure_len if window == "pressure" else REVERSAL_H
        sign = r.side if window == "pressure" else -r.side
        ticks.append(t); lens.append(int(L)); signs.append(sign)
    series = {t: rets[t].to_numpy() for t in set(ticks)}
    out = np.empty(n_draws)
    for k in range(n_draws):
        vals = []
        for t, L, sg in zip(ticks, lens, signs):
            pool = pools[t]
            pool = pool[pool + L < len(idx)]
            if len(pool) == 0:
                continue
            start = int(pool[rng.integers(len(pool))])
            ar = _window_ar(series[t], m, start, start + L)
            if ar is not None:
                vals.append(sg * ar)
        out[k] = np.mean(vals) if vals else np.nan
    return out[np.isfinite(out)]


def run(seed: int = 0, index: str = "EGX30") -> dict:
    """`index`: 'EGX30' or 'EGX70'. The EGX70 run uses the same pre-registered
    windows, baseline and verdict rule; only the event list differs."""
    from data.egx_index_events import events, events70
    ev = pd.DataFrame(events70() if index == "EGX70" else events())
    rets, market = _panel()
    meas = measure(ev, rets, market)

    results = []
    for window in ("pressure", "reversal"):
        col = f"{window}_score"
        d = meas.dropna(subset=[col])
        n = len(d)
        mean = float(d[col].mean()) if n else np.nan
        pm = placebo(meas, ev, rets, market, window, seed=seed) if n >= 5 else np.array([])
        p = float((np.sum(pm >= mean) + 1) / (len(pm) + 1)) if len(pm) else np.nan
        p_corr = min(1.0, p * 2) if np.isfinite(p) else np.nan
        # CLUSTER CHECK (added 2026-09-14, after the EGX70 run). Events in one
        # review share dates and market conditions, so the placebo's independence
        # assumption overstates the evidence when there are few reviews. The first
        # EGX70 run "passed" at p=0.003 on 69 events from only 3 reviews, and one
        # review supplied nearly all of it (+10.6% vs +0.3% and +0.6%). A PASS now
        # also requires: at least MIN_REVIEWS reviews, a majority of review-level
        # means positive, and a positive mean after dropping the single largest
        # review. The EGX30 result was checked against this rule when it was added.
        by_rev = d.groupby("effective")[col].mean() if n else pd.Series(dtype=float)
        n_rev = len(by_rev)
        share_pos = float((by_rev > 0).mean()) if n_rev else np.nan
        drop_top = (float(d[d["effective"] != by_rev.idxmax()][col].mean())
                    if n_rev > 1 else np.nan)
        cluster_ok = bool(n_rev >= MIN_REVIEWS and share_pos > 0.5 and
                          np.isfinite(drop_top) and drop_top > 0)
        if n < MIN_EVENTS:
            verdict = "INSUFFICIENT"
        elif mean > 0 and p_corr < ALPHA:
            verdict = "PASS" if cluster_ok else "NOT ROBUST"
        else:
            verdict = "NULL"
        rec = {"window": window, "n": n, "mean_score": mean,
               "reviews": n_rev, "reviews_positive": share_pos, "mean_without_top_review": drop_top,
               "hit_rate": float((d[col] > 0).mean()) if n else np.nan,
               "placebo_mean": float(pm.mean()) if len(pm) else np.nan,
               "p": p, "p_corrected": p_corr, "verdict": verdict}
        for side, lab in ((1, "added"), (-1, "removed")):
            ds = d[d["side"] == side]
            rec[f"{lab}_n"] = len(ds)
            rec[f"{lab}_mean_ar"] = float(ds[f"{window}_ar"].mean()) if len(ds) else np.nan
        results.append(rec)

    runup = {lab: float(meas[meas["side"] == s]["runup_ar"].dropna().mean())
             for s, lab in ((1, "added"), (-1, "removed"))}
    return {"index": index, "events": len(ev), "measured": len(meas), "table": pd.DataFrame(results),
            "runup": runup, "detail": meas}


def replication(seed: int = 0, batch: str = "A") -> dict:
    """Pre-registered out-of-sample check on the events recovered 2026-10-03.
    batch 'A' (2021-2025, the registered test) or 'B' (2015-2017, descriptive)."""
    from data.egx_index_events import events, RECOVERED_2026_10_03, RECOVERED_2026_10_03_B
    keys = RECOVERED_2026_10_03 if batch == "A" else RECOVERED_2026_10_03_B
    allev = pd.DataFrame(events())
    key = list(zip(allev["announced"], allev["ticker"], allev["side"]))
    new = allev[[k in keys for k in key]].reset_index(drop=True)
    rets, market = _panel()
    meas = measure(new, rets, market)
    d = meas.dropna(subset=["pressure_score"])
    n = len(d)
    mean = float(d["pressure_score"].mean()) if n else np.nan
    # placebo exclusions use ALL known events for each ticker, not just the new ones
    pm = placebo(meas, allev, rets, market, "pressure", seed=seed) if n >= 5 else np.array([])
    p = float((np.sum(pm >= mean) + 1) / (len(pm) + 1)) if len(pm) else np.nan
    by_rev = d.groupby("effective")["pressure_score"].mean()
    drop_top = float(d[d["effective"] != by_rev.idxmax()]["pressure_score"].mean()) if len(by_rev) > 1 else np.nan
    cluster_ok = bool(len(by_rev) >= MIN_REVIEWS and (by_rev > 0).mean() > 0.5 and drop_top > 0)
    ok = n >= MIN_EVENTS and mean > 0 and p < ALPHA and cluster_ok
    side = {lab: (int((d["side"] == s).sum()), float(d[d["side"] == s]["pressure_ar"].mean()))
            for s, lab in ((1, "added"), (-1, "removed"))}
    return {"events": len(new), "n": n, "mean_score": mean, "median": float(d["pressure_score"].median()),
            "hit": float((d["pressure_score"] > 0).mean()), "placebo_mean": float(pm.mean()) if len(pm) else np.nan,
            "p": p, "by_review": by_rev.round(4).to_dict(), "without_top_review": drop_top,
            "sides": side, "detail": d,
            "verdict": ("REPLICATES" if ok else "FAILS TO REPLICATE") if batch == "A" else
                       ("SUPPORTIVE" if n >= MIN_EVENTS and mean > 0 and p < ALPHA else "NOT SUPPORTIVE")}


def implementability(index: str = "EGX30", delays=(0, 1), slippage: float = 0.0057,
                     sizes=(1000, 2000, 5000)) -> pd.DataFrame:
    """Can the ADDITIONS leg actually be captured? (added 2026-10-03; descriptive,
    not part of the pre-registered test).

    The test enters at the CLOSE of the first session after the announcement
    and exits at the CLOSE one session after the effective date. A person buys
    DURING a session at roughly the typical price (H+L+C)/3 and pays ~0.57%
    above it (measured on the live account). EGX has no short selling, so only
    added stocks can be traded. `delay` = extra sessions before acting.
    """
    from data.egx_index_events import events, events70
    from backtest.engine import load_fill_ratio
    from backtest.fees import round_trip_cost
    ev = pd.DataFrame(events70() if index == "EGX70" else events())
    ev = ev[ev["side"] == 1]
    rets, market = _panel()
    ratio = load_fill_ratio(list(rets.columns)).reindex(index=rets.index, columns=rets.columns).fillna(1.0)
    idx, m = rets.index, market.to_numpy()
    rows = []
    for e in ev.itertuples():
        if e.ticker not in rets.columns:
            continue
        ann, eff, exit_p = positions(idx, e.announced, e.effective)
        s = rets[e.ticker].to_numpy()
        tp = ratio[e.ticker].to_numpy()
        for d in delays:
            b, x = ann + d, exit_p + d
            if x >= len(idx) or b >= x:
                continue
            w = s[b + 1:x + 1]
            if np.isfinite(w).sum() < max(1, int(0.6 * (x - b))):
                continue
            # bought at typical(b), held closes b..x, sold at typical(x)
            stock = (1.0 / tp[b]) * np.prod(1 + np.nan_to_num(w)) * tp[x] - 1.0
            mkt = np.prod(1 + np.nan_to_num(m[b + 1:x + 1])) - 1.0
            close_ar = _window_ar(s, m, ann, exit_p)
            rows.append({"ticker": e.ticker, "effective": e.effective, "delay": d,
                         "close_model_ar": close_ar, "typical_ar": stock - mkt,
                         "stock_ret": stock})
    d = pd.DataFrame(rows)
    out = []
    for dl, g in d.groupby("delay"):
        rec = {"delay_sessions": dl, "n": len(g), "close_model": g["close_model_ar"].mean(),
               "typical_fill": g["typical_ar"].mean(), "median": g["typical_ar"].median(),
               "hit": float((g["typical_ar"] > 0).mean())}
        for p in sizes:
            cost = round_trip_cost(p)["total_egp"] / p + 2 * slippage
            rec[f"net_{p}"] = g["typical_ar"].mean() - cost
            rec[f"net_abs_{p}"] = g["stock_ret"].mean() - cost      # what the account actually makes
        out.append(rec)
    return pd.DataFrame(out)


def format_report(res: dict) -> str:
    t = res["table"]
    pct = lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "  —"
    L = [f"{res.get('index', 'EGX30')} INDEX CHANGE EVENT STUDY",
         f"events on file {res['events']} | measurable {res['measured']}",
         "score = predicted sign x (stock - market). Pressure predicts added up / removed down;",
         "reversal predicts the opposite over the next 20 sessions. PASS: n>=20, Bonferroni p<0.05.", ""]
    for r in t.to_dict("records"):
        L.append(f"{r['window'].upper():9} n={r['n']:<3} score {pct(r['mean_score'])}  hit {r['hit_rate']*100:.0f}%  "
                 f"placebo {pct(r['placebo_mean'])}  p(corr) {r['p_corrected']:.3f}  -> {r['verdict']}")
        L.append(f"          reviews {r['reviews']} | positive {r['reviews_positive']*100:.0f}% | "
                 f"mean without largest review {pct(r['mean_without_top_review'])}")
        L.append(f"          added   n={r['added_n']:<3} stock-market {pct(r['added_mean_ar'])}")
        L.append(f"          removed n={r['removed_n']:<3} stock-market {pct(r['removed_mean_ar'])}")
    L += ["", "run-up, 20 sessions before announcement (descriptive):",
          f"  added {pct(res['runup']['added'])} | removed {pct(res['runup']['removed'])}"]
    return "\n".join(L)


def main() -> int:
    import sys
    logging.basicConfig(level=logging.WARNING)
    index = sys.argv[1].upper() if len(sys.argv) > 1 else "EGX30"
    print(format_report(run(index=index)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
