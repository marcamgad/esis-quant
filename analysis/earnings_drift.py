"""
Post-earnings drift on EGX -- does the market under-react to reported earnings?

THE IDEA (Ball & Brown 1968; Bernard & Thomas 1989). Prices keep drifting in
the direction of an earnings surprise for weeks after the announcement. It is
the most replicated anomaly in finance, and it should be STRONGER in a thin,
retail-heavy, under-covered market like EGX. If any information effect exists
here, this is the most likely one.

PRE-REGISTERED DESIGN (written 2026-10-03, before any result was computed).

  SURPRISE  seasonal random walk, no analyst forecasts exist for EGX:
              SUE_q = (E_q - E_{q-4}) / sd(E_j - E_{j-4}) over the 8 prior quarters
            (at least 4 prior changes required). E = quarterly net income.
  EVENTS    A (PRIMARY): every quarterly result in the statement data, entered
              at the close of the first session strictly after `available_from`
              (period end + 90 days for Q1-Q3, + 120 for Q4). Deliberately LATE:
              if drift survives a 1-2 month delayed entry it is not an artefact
              of announcement-day timing.
            B (SECONDARY, small): results reported in the press (news_figures,
              rules extraction, confidence >= 0.6, EGP), entered at the close of
              the first session strictly after publication. Surprise sign = the
              year-on-year change against the statement figure of the same
              quarter a year earlier.
  SIGNAL    events with |SUE| >= 0.5 (A) or any nonzero change (B); predicted
            sign = sign(SUE).
  SCORE     predicted sign x (stock return - equal-weight market return) over
            sessions (entry, entry + h].
  HORIZONS  PRIMARY h = 60; secondary h = 20. Bonferroni x2.
  BASELINE  same tickers and signs on random valid sessions at least 10 sessions
            from any of that ticker's events; 500 draws; one-sided p.
  PASS      n >= 20, mean score > 0, corrected p < 0.05, AND the cluster check:
            >= 5 calendar quarters, a majority of quarter-level means positive,
            positive after dropping the single largest quarter.
  REPORTED, NOT TESTED: long-only top-SUE (SUE >= 1) excess return, gross and
            after round-trip fees at a 1,000 EGP position; positive vs negative
            surprise legs separately.

AMENDMENT -- DESIGN B2 (registered 2026-10-03, after A and B were run and BEFORE
any archive article was fetched). B had only 16-17 events. B2 scales it with the
Al Mal archive (results articles found by headline keywords in the sitemap,
2016-2026, each with its own publication timestamp):
  EVENTS    archive + live press results (news_figures, confidence >= 0.6, EGP,
            kind = result, metric net_profit / net_loss), first report per
            (ticker, period, year), entered at the close of the first session
            strictly after publication.
  SIGN      the year-on-year comparison THE ARTICLE ITSELF reports: sign of the
            stated % change, else sign(value - stated prior value), else the
            statement figure a year earlier (as in B). No sign -> excluded.
  TEST      identical to A: horizons 20 and 60 (60 primary), score, 500-draw
            placebo, Bonferroni x2, PASS rule with the quarter cluster check.
  The point of B2: entry ~39 days EARLIER than A. If drift is an under-reaction
  to news, B2 should be STRONGER than A; if B2 is weaker, A's tail is something
  else (e.g. index or liquidity events that happen to follow results).

KNOWN LIMITS, stated now:
  * statement values come from stockanalysis TODAY: restatements leak a little
    future into old quarters (small; quarterly NI is rarely restated by much);
  * survivorship: only names still listed are in the statement data;
  * Q4 is sometimes derived (FY minus 9M) by the data vendor.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.earnings_drift")

HORIZONS = (20, 60)
PRIMARY = 60
SUE_MIN = 0.5
N_PLACEBO = 500
EXCLUSION = 10
MIN_EVENTS = 20
MIN_QUARTERS = 5
ALPHA = 0.05


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------
def statement_events() -> pd.DataFrame:
    """Design A: SUE from quarterly statements, entry after available_from."""
    with get_connection() as conn:
        q = pd.read_sql("SELECT ticker, fiscal_period, period_end, available_from, value FROM fundamentals "
                        "WHERE statement='income' AND frequency='quarterly' AND line_item='Net Income' "
                        "AND value IS NOT NULL AND period_end IS NOT NULL", conn,
                        parse_dates=["period_end", "available_from"])
    q = q[q["fiscal_period"].str.match(r"^Q[1-4] ")].drop_duplicates(["ticker", "period_end"])
    rows = []
    for t, g in q.sort_values("period_end").groupby("ticker"):
        g = g.set_index("period_end")
        e = g["value"]
        # seasonal lag by DATE, not position: a missing quarter must not shift the comparison
        prior = e.reindex(e.index - pd.DateOffset(years=1)).values
        chg = pd.Series(e.values - prior, index=e.index)
        for i, (pe, c) in enumerate(chg.items()):
            hist = chg.iloc[max(0, i - 8):i].dropna()
            if np.isnan(c) or len(hist) < 4 or hist.std() <= 0:
                continue
            rows.append({"ticker": t, "period_end": pe, "event_date": g.loc[pe, "available_from"],
                         "sue": float(c / hist.std()), "e": float(e.loc[pe]), "e_prior": float(prior[i]),
                         "design": "A"})
    return pd.DataFrame(rows)


def press_events_b2() -> pd.DataFrame:
    """Design B2: sign from the article's own comparison (see the amendment)."""
    with get_connection() as conn:
        nf = pd.read_sql("SELECT ticker, value_m, pct_change, prior_m, period, fiscal_year, period_end, "
                         "published_at, first_seen_at FROM news_figures WHERE kind='result' "
                         "AND confidence >= 0.6 AND metric IN ('net_profit','net_loss') "
                         "AND period IN ('Q1','Q2','Q3','Q4','H1','9M','FY') AND fiscal_year IS NOT NULL "
                         "AND (currency IS NULL OR currency='EGP')", conn)
        st = pd.read_sql("SELECT ticker, fiscal_period, value FROM fundamentals WHERE statement='income' "
                         "AND frequency='quarterly' AND line_item='Net Income' AND value IS NOT NULL", conn)
    if nf.empty:
        return pd.DataFrame()
    st["label"], st["year"] = st["fiscal_period"].str[:2], st["fiscal_period"].str[-4:].astype(int)
    st_prior = st.set_index(["ticker", "label", "year"])["value"]
    signs = []
    for r in nf.itertuples():
        s = 0.0
        if pd.notna(r.pct_change) and r.pct_change != 0:
            s = np.sign(r.pct_change)
        elif pd.notna(r.prior_m) and pd.notna(r.value_m) and r.value_m != r.prior_m:
            s = np.sign(r.value_m - r.prior_m)
        elif pd.notna(r.value_m) and (r.ticker, r.period, int(r.fiscal_year) - 1) in st_prior.index:
            s = np.sign(r.value_m - st_prior.loc[(r.ticker, r.period, int(r.fiscal_year) - 1)])
        signs.append(s)
    nf["sue"] = signs
    nf["event_date"] = pd.to_datetime(nf["published_at"].fillna(nf["first_seen_at"]), errors="coerce",
                                      utc=True).dt.tz_localize(None)
    nf = nf[(nf["sue"] != 0) & nf["event_date"].notna()]
    nf = nf.sort_values("event_date").drop_duplicates(["ticker", "period", "fiscal_year"])
    return pd.DataFrame({"ticker": nf["ticker"], "period_end": pd.to_datetime(nf["period_end"]),
                         "event_date": nf["event_date"], "sue": nf["sue"], "e": nf["value_m"],
                         "e_prior": nf["prior_m"], "design": "B2"})


def press_events() -> pd.DataFrame:
    """Design B: press-reported results vs the statement figure a year earlier."""
    with get_connection() as conn:
        nf = pd.read_sql("SELECT ticker, metric, value_m, period, fiscal_year, period_end, published_at, "
                         "first_seen_at FROM news_figures WHERE kind='result' AND confidence >= 0.6 "
                         "AND metric IN ('net_profit','net_loss') AND period IN ('Q1','Q2','Q3','Q4') "
                         "AND fiscal_year IS NOT NULL AND value_m IS NOT NULL "
                         "AND (currency IS NULL OR currency='EGP')", conn)
        st = pd.read_sql("SELECT ticker, fiscal_period, value FROM fundamentals WHERE statement='income' "
                         "AND frequency='quarterly' AND line_item='Net Income' AND value IS NOT NULL", conn)
    if nf.empty:
        return pd.DataFrame()
    st["label"] = st["fiscal_period"].str[:2]
    st["year"] = st["fiscal_period"].str[-4:].astype(int)
    nf["prior_year"] = nf["fiscal_year"].astype(int) - 1
    m = nf.merge(st, left_on=["ticker", "period", "prior_year"], right_on=["ticker", "label", "year"], how="inner")
    m["event_date"] = pd.to_datetime(m["published_at"].fillna(m["first_seen_at"]), errors="coerce", utc=True) \
        .dt.tz_localize(None)
    m["change"] = m["value_m"] - m["value"]
    m = m[(m["change"] != 0) & m["event_date"].notna()]
    m = m.sort_values("event_date").drop_duplicates(["ticker", "period", "fiscal_year"])   # first report only
    return pd.DataFrame({"ticker": m["ticker"], "period_end": pd.to_datetime(m["period_end"]),
                         "event_date": m["event_date"], "sue": np.sign(m["change"]),
                         "e": m["value_m"], "e_prior": m["value"], "design": "B"})


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
# BENCHMARK CHECK (2026-10-04, not a new test). With the real EGX30 instead of the
# registered equal-weight market, A h=60 reads +3.31% (p .004) -- but bad-news
# stocks ALSO beat the EGX30 by +4.4%: that is the ~13%/yr gap between the average
# stock and the index, not drift. Good-minus-bad spread is unchanged (~+3%).
# The registered (equal-weight) result stands.
def _panel():
    from backtest.engine import load_price_returns
    r = load_price_returns()
    n = r.notna().sum(axis=1)
    mkt = r.where(r.abs() < 0.25).mean(axis=1).where(n >= 30)
    return r, mkt


def _ar(s, m, start, h):
    """Abnormal return over (start, start+h]; None if mostly untraded."""
    end = start + h
    if start < 0 or end >= len(s):
        return None
    w, mm = s[start + 1:end + 1], m[start + 1:end + 1]
    if np.isfinite(w).sum() < 0.6 * h:
        return None
    return float(np.prod(1 + np.nan_to_num(w)) - np.prod(1 + np.nan_to_num(mm)))


def measure(ev: pd.DataFrame, r: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    idx, m = r.index, mkt.to_numpy()
    out = []
    for e in ev.itertuples():
        if e.ticker not in r:
            continue
        i = idx.searchsorted(pd.Timestamp(e.event_date).normalize(), side="right")   # strictly after
        if i >= len(idx):
            continue
        s = r[e.ticker].to_numpy()
        rec = e._asdict()
        rec.pop("Index", None)
        rec["entry_pos"] = int(i)
        for h in HORIZONS:
            ar = _ar(s, m, i, h)
            rec[f"ar_{h}"] = ar
            rec[f"score_{h}"] = None if ar is None else float(np.sign(e.sue) * ar)
        out.append(rec)
    return pd.DataFrame(out)


def placebo(meas: pd.DataFrame, r: pd.DataFrame, mkt: pd.Series, h: int, n: int = N_PLACEBO, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = mkt.to_numpy()
    d = meas.dropna(subset=[f"score_{h}"])
    ev_pos = d.groupby("ticker")["entry_pos"].apply(np.array).to_dict()
    pools, series = {}, {}
    for t in d["ticker"].unique():
        s = r[t].to_numpy()
        valid = np.where(np.isfinite(s))[0]
        valid = valid[valid + h < len(s)]
        mask = np.ones(len(valid), bool)
        for p in ev_pos.get(t, []):
            mask &= np.abs(valid - p) > EXCLUSION
        pools[t], series[t] = valid[mask], s
    ticks, signs = d["ticker"].to_numpy(), np.sign(d["sue"].to_numpy())
    means = np.empty(n)
    for k in range(n):
        vals = []
        for t, sg in zip(ticks, signs):
            pool = pools[t]
            if len(pool) == 0:
                continue
            ar = _ar(series[t], m, int(pool[rng.integers(len(pool))]), h)
            if ar is not None:
                vals.append(sg * ar)
        means[k] = np.mean(vals) if vals else np.nan
    return means[np.isfinite(means)]


def test(ev: pd.DataFrame, r, mkt, label: str, seed: int = 0) -> pd.DataFrame:
    meas = measure(ev, r, mkt)
    rows = []
    for h in HORIZONS:
        d = meas.dropna(subset=[f"score_{h}"])
        n = len(d)
        mean = float(d[f"score_{h}"].mean()) if n else np.nan
        pm = placebo(meas, r, mkt, h, seed=seed) if n >= 5 else np.array([])
        p = float((np.sum(pm >= mean) + 1) / (len(pm) + 1)) if len(pm) else np.nan
        pc = min(1.0, p * len(HORIZONS)) if p == p else np.nan
        d = d.assign(q=pd.to_datetime(d["event_date"]).dt.to_period("Q"))
        byq = d.groupby("q")[f"score_{h}"].mean()
        drop = float(d[d["q"] != byq.idxmax()][f"score_{h}"].mean()) if len(byq) > 1 else np.nan
        cluster_ok = len(byq) >= MIN_QUARTERS and (byq > 0).mean() > 0.5 and drop == drop and drop > 0
        verdict = ("INSUFFICIENT" if n < MIN_EVENTS else
                   ("PASS" if cluster_ok else "NOT ROBUST") if (mean > 0 and pc < ALPHA) else "NULL")
        pos, neg = d[d["sue"] > 0], d[d["sue"] < 0]
        rows.append({"design": label, "h": h, "primary": h == PRIMARY, "n": n, "mean_score": mean,
                     "hit": float((d[f"score_{h}"] > 0).mean()) if n else np.nan,
                     "placebo": float(pm.mean()) if len(pm) else np.nan, "p_corr": pc,
                     "quarters": len(byq), "q_positive": float((byq > 0).mean()) if len(byq) else np.nan,
                     "without_top_q": drop, "good_news_ar": float(pos[f"ar_{h}"].mean()) if len(pos) else np.nan,
                     "bad_news_ar": float(neg[f"ar_{h}"].mean()) if len(neg) else np.nan, "verdict": verdict})
    return pd.DataFrame(rows), meas


def long_only(meas: pd.DataFrame, h: int = PRIMARY, sue_min: float = 1.0, position_egp: float = 1000.0) -> dict:
    from backtest.fees import round_trip_cost
    d = meas[(meas["sue"] >= sue_min)].dropna(subset=[f"ar_{h}"])
    if d.empty:
        return {"n": 0}
    fee = round_trip_cost(position_egp)["total_egp"] / position_egp
    return {"n": len(d), "gross": float(d[f"ar_{h}"].mean()), "median": float(d[f"ar_{h}"].median()),
            "fee_round_trip": fee, "net": float(d[f"ar_{h}"].mean() - fee),
            "hit": float((d[f"ar_{h}"] > 0).mean())}


def run(seed: int = 0) -> dict:
    r, mkt = _panel()
    a = statement_events()
    a = a[a["sue"].abs() >= SUE_MIN]
    b = press_events()
    ta, ma = test(a, r, mkt, "A statements (late entry)", seed)
    out = {"A": ta, "A_detail": ma, "A_long_only": long_only(ma), "n_events_A": len(a)}
    if len(b):
        tb, mb = test(b, r, mkt, "B press (timely entry)", seed)
        out.update({"B": tb, "B_detail": mb, "B_long_only": long_only(mb, sue_min=1.0), "n_events_B": len(b)})
    b2 = press_events_b2()
    if len(b2):
        t2, m2 = test(b2, r, mkt, "B2 press+archive (timely)", seed)
        out.update({"B2": t2, "B2_detail": m2, "B2_long_only": long_only(m2, sue_min=1.0), "n_events_B2": len(b2)})
    return out


def format_report(res: dict) -> str:
    pct = lambda x: "   n/a" if x != x else f"{x:+.2%}"
    L = ["=" * 78, "POST-EARNINGS DRIFT — pre-registered (see module docstring)", "=" * 78]
    for k in ("A", "B", "B2"):
        if k not in res:
            continue
        for r in res[k].to_dict("records"):
            L.append(f"{r['design']:28} h={r['h']:<3}{'*' if r['primary'] else ' '} n={r['n']:<5} "
                     f"score {pct(r['mean_score'])} hit {r['hit']:.0%} placebo {pct(r['placebo'])} "
                     f"p(corr) {r['p_corr']:.3f} -> {r['verdict']}")
            L.append(f"{'':32}quarters {r['quarters']} ({r['q_positive']:.0%} positive), "
                     f"without largest {pct(r['without_top_q'])} | good news {pct(r['good_news_ar'])}, "
                     f"bad news {pct(r['bad_news_ar'])}")
        lo = res.get(f"{k}_long_only", {})
        if lo.get("n"):
            L.append(f"{'':32}long-only SUE>=1 (descriptive): n={lo['n']} gross {pct(lo['gross'])} "
                     f"(median {pct(lo['median'])}), fees {lo['fee_round_trip']:.2%} -> net {pct(lo['net'])}")
        L.append("")
    L.append("* = primary horizon. PASS needs n>=20, corrected p<0.05 and the quarter cluster check.")
    return "\n".join(L)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    print(format_report(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
