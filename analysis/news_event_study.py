"""
News event study -- stage 4: did the stage 3 calls turn out right?

THE QUESTION. Stage 3 labels each news item favourable / neutral / unfavourable
for a company. This module measures what the stock actually did afterwards and
asks whether the labels beat chance. Until an event type passes here, its label
is a documented opinion and must not reach the optimiser.

DESIGN, FIXED BEFORE ANY RESULT WAS SEEN (pre-registered in this docstring):

  events     stage 3 calls, deduplicated to one per (ticker, event type, 3-day
             window), low-confidence calls excluded, neutral calls excluded from
             the directional test.
  entry      the close of the FIRST valid trading session strictly AFTER the
             publication date. Headlines carry a date but rarely a reliable time,
             and a same-day entry would score a move that had already happened by
             the time anyone could read the story. The move from the last close
             before publication to entry is reported separately ("gap") and never
             counted as skill.
  return     abnormal return = stock return minus the equal-weight market return
             over the same sessions, so a rising market does not make every
             favourable call look right.
  horizons   1, 5 and 20 sessions after entry. 5 is the PRIMARY horizon; 1 and 20
             are reported but are not used for the verdict, so there is one test
             per event type, not three.
  score      direction-adjusted abnormal return: direction x abnormal return. A
             right call is positive whichever way it pointed.
  baseline   placebo events: the same tickers on random valid dates that are at
             least 20 sessions away from any real event for that ticker, carrying
             the same directions. 2,000 draws. p = share of placebo means >= the
             real mean (one-sided: the question is whether calls are RIGHT).
  verdict    PASS only if n >= 20 distinct events AND mean score > 0 AND the
             Bonferroni-corrected p < 0.05 across every event type tested.
             Everything else is INSUFFICIENT (n < 20), NULL, or WRONG (mean score
             significantly negative in the other tail).

WHY THIS STRICT. Every selection method tried on this data -- momentum, rank
persistence, regime-conditional factors, TypedGP -- produced an attractive
in-sample number first. The corrections are what separated them from signal.

KNOWN LIMITATION, STATED UP FRONT. Almost every item was first seen by this
system on 2026-09-13, because Google News back-filled months of history on the
first run. This study therefore uses PUBLICATION dates to reconstruct what the
calls would have been. A live system only ever acts on first-seen time, and there
is no forward data for that yet. A pass here is necessary, not sufficient.
"""

import logging

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.news_event_study")

HORIZONS = (1, 5, 20)
PRIMARY = 5
MIN_EVENTS = 20
N_PLACEBO = 2000
PLACEBO_EXCLUSION = 20
ALPHA = 0.05


def load_events(method: str = "rules_v1", min_confidence: str = "medium") -> pd.DataFrame:
    """Deduplicated, confidence-filtered stage 3 calls."""
    from analysis.news_impact import dedupe_events
    allowed = {"high": ("high",), "medium": ("high", "medium"), "low": ("high", "medium", "low")}[min_confidence]
    with get_connection() as conn:
        df = pd.read_sql(
            "SELECT ticker, substr(published_at, 1, 10) AS date, event_type, direction, strength, "
            "confidence, link_method, news_id FROM news_assessments "
            "WHERE method = ? AND published_at IS NOT NULL AND event_type NOT IN ('other', 'earnings_mixed') "
            "AND confidence IN (%s)" % ",".join("?" * len(allowed)),
            conn, params=(method, *allowed))
    if df.empty:
        return df
    return dedupe_events(df).reset_index(drop=True)


def _panel():
    from backtest.engine import load_price_returns
    rets = load_price_returns()                       # calendar-filtered sessions
    market = rets.mean(axis=1, skipna=True)
    return rets, market


def forward_abnormal(rets: pd.DataFrame, market: pd.Series, ticker: str, date,
                     horizons=HORIZONS) -> dict:
    """Abnormal returns after the first session strictly after `date`.

    Returns {} when the ticker has no usable prices there. A session with a
    missing return for the stock is treated as missing, not zero: filling with
    zero would manufacture 'no reaction' for a stock that simply did not trade.
    """
    if ticker not in rets.columns:
        return {}
    idx = rets.index
    pos = idx.searchsorted(pd.Timestamp(date), side="right")   # strictly after
    if pos >= len(idx):
        return {}
    s = rets[ticker]
    out = {"entry": idx[pos]}
    # gap: last close before publication -> entry close (reported, never scored)
    if pos >= 1 and pd.notna(s.iloc[pos]):
        out["gap"] = float(s.iloc[pos] - market.iloc[pos])
    for h in horizons:
        end = pos + h
        if end >= len(idx):
            continue
        w = s.iloc[pos + 1:end + 1]
        m = market.iloc[pos + 1:end + 1]
        if w.notna().sum() < max(1, int(0.6 * h)):
            continue
        r_stock = float((1 + w.fillna(0)).prod() - 1)
        r_mkt = float((1 + m.fillna(0)).prod() - 1)
        out[f"ar_{h}"] = r_stock - r_mkt
    return out


def measure(events: pd.DataFrame, rets=None, market=None) -> pd.DataFrame:
    if rets is None or market is None:
        rets, market = _panel()
    rows = []
    for r in events.itertuples():
        fa = forward_abnormal(rets, market, r.ticker, r.date)
        if f"ar_{PRIMARY}" not in fa:
            continue
        row = {"ticker": r.ticker, "date": r.date, "event_type": r.event_type,
               "direction": int(r.direction), "strength": int(r.strength),
               "confidence": r.confidence, "entry": fa["entry"], "gap": fa.get("gap")}
        for h in HORIZONS:
            row[f"ar_{h}"] = fa.get(f"ar_{h}")
            if fa.get(f"ar_{h}") is not None:
                row[f"score_{h}"] = int(r.direction) * fa[f"ar_{h}"]
        rows.append(row)
    return pd.DataFrame(rows)


def placebo_means(measured: pd.DataFrame, rets: pd.DataFrame, market: pd.Series,
                  n_draws: int = N_PLACEBO, seed: int = 0) -> np.ndarray:
    """Distribution of the mean score for random dates with the same tickers and
    directions. One pool of candidate dates per ticker, excluding +/-20 sessions
    around every real event for that ticker."""
    rng = np.random.default_rng(seed)
    idx = rets.index
    last_ok = len(idx) - PRIMARY - 1
    pools = {}
    for t, g in measured.groupby("ticker"):
        if t not in rets.columns:
            continue
        valid = np.where(rets[t].notna().to_numpy())[0]
        valid = valid[(valid > 0) & (valid < last_ok)]
        ev_pos = idx.searchsorted(pd.to_datetime(g["date"]).to_numpy(), side="right")
        mask = np.ones(len(valid), bool)
        for p in ev_pos:
            mask &= np.abs(valid - p) > PLACEBO_EXCLUSION
        pool = valid[mask]
        if len(pool) >= 10:
            # precompute placebo abnormal returns at the primary horizon
            s = rets[t].to_numpy()
            m = market.to_numpy()
            ars = []
            for p in pool:
                w = s[p + 1:p + 1 + PRIMARY]
                mm = m[p + 1:p + 1 + PRIMARY]
                if np.isfinite(w).sum() < int(0.6 * PRIMARY):
                    continue
                ars.append(np.prod(1 + np.nan_to_num(w)) - np.prod(1 + np.nan_to_num(mm)))
            if len(ars) >= 10:
                pools[t] = np.asarray(ars)
    usable = measured[measured["ticker"].isin(pools)]
    if usable.empty:
        return np.array([])
    tick = usable["ticker"].to_numpy()
    dirs = usable["direction"].to_numpy()
    out = np.empty(n_draws)
    for k in range(n_draws):
        draws = np.array([pools[t][rng.integers(len(pools[t]))] for t in tick])
        out[k] = float(np.mean(dirs * draws))
    return out


def verdict(n: int, mean: float, p_corr: float, p_low_corr: float) -> str:
    if n < MIN_EVENTS:
        return "INSUFFICIENT"
    if mean > 0 and p_corr < ALPHA:
        return "PASS"
    if mean < 0 and p_low_corr < ALPHA:
        return "WRONG"
    return "NULL"


def run(min_confidence: str = "medium", seed: int = 0) -> dict:
    events = load_events(min_confidence=min_confidence)
    rets, market = _panel()
    m = measure(events, rets, market)
    directional = m[m["direction"] != 0] if not m.empty else m

    groups = [("ALL directional calls", directional)]
    for et, g in directional.groupby("event_type"):
        groups.append((et, g))
    n_tests = max(1, sum(1 for name, g in groups if len(g) >= MIN_EVENTS))

    rows = []
    for name, g in groups:
        n = len(g)
        rec = {"group": name, "n": n}
        if n:
            for h in HORIZONS:
                col = f"score_{h}"
                vals = g[col].dropna() if col in g else pd.Series(dtype=float)
                rec[f"mean_score_{h}d"] = float(vals.mean()) if len(vals) else np.nan
            rec["hit_rate_5d"] = float((g[f"score_{PRIMARY}"] > 0).mean())
            rec["mean_gap"] = float(g["dir_gap"].mean()) if "dir_gap" in g else float(
                (g["direction"] * g["gap"]).dropna().mean()) if g["gap"].notna().any() else np.nan
        if n >= 5:
            pm = placebo_means(g, rets, market, seed=seed)
            real = rec.get(f"mean_score_{PRIMARY}d", np.nan)
            if len(pm):
                p_hi = float((np.sum(pm >= real) + 1) / (len(pm) + 1))
                p_lo = float((np.sum(pm <= real) + 1) / (len(pm) + 1))
                rec.update({"placebo_mean": float(pm.mean()), "p": p_hi,
                            "p_corrected": min(1.0, p_hi * n_tests),
                            "p_low_corrected": min(1.0, p_lo * n_tests)})
        rec["verdict"] = verdict(n, rec.get(f"mean_score_{PRIMARY}d", np.nan) if n else np.nan,
                                 rec.get("p_corrected", 1.0), rec.get("p_low_corrected", 1.0))
        rows.append(rec)

    table = pd.DataFrame(rows)
    _store(m, table)
    return {"events_loaded": len(events), "events_measured": len(m),
            "directional": len(directional), "tests": n_tests, "table": table, "measured": m}


def _store(measured: pd.DataFrame, table: pd.DataFrame):
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS news_event_scores (
            run_at TEXT, grp TEXT, n INTEGER, mean_score_1d REAL, mean_score_5d REAL,
            mean_score_20d REAL, hit_rate_5d REAL, placebo_mean REAL, p REAL,
            p_corrected REAL, verdict TEXT)""")
        for r in table.to_dict("records"):
            conn.execute("INSERT INTO news_event_scores VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (now, r["group"], r["n"], r.get("mean_score_1d"), r.get("mean_score_5d"),
                          r.get("mean_score_20d"), r.get("hit_rate_5d"), r.get("placebo_mean"),
                          r.get("p"), r.get("p_corrected"), r["verdict"]))


def format_report(res: dict) -> str:
    t = res["table"]
    L = ["NEWS EVENT STUDY — do stage 3 calls beat random dates?",
         f"events loaded {res['events_loaded']} | with prices after them {res['events_measured']} | "
         f"directional {res['directional']} | event types tested {res['tests']}",
         "score = direction x (stock return - market return); 5-session horizon decides; "
         f"PASS needs n>={MIN_EVENTS} and Bonferroni p<{ALPHA}", ""]
    hdr = f"{'group':24}{'n':>5}{'1d':>8}{'5d':>8}{'20d':>8}{'hit5d':>7}{'placebo':>9}{'p(corr)':>9}  verdict"
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in t.to_dict("records"):
        def pct(x):
            return f"{x*100:+.2f}%" if x is not None and pd.notna(x) else "   —"
        L.append(f"{r['group'][:24]:24}{r['n']:>5}{pct(r.get('mean_score_1d')):>8}{pct(r.get('mean_score_5d')):>8}"
                 f"{pct(r.get('mean_score_20d')):>8}"
                 f"{(str(round(r['hit_rate_5d']*100))+'%') if pd.notna(r.get('hit_rate_5d', np.nan)) else '—':>7}"
                 f"{pct(r.get('placebo_mean')):>9}"
                 f"{(format(r['p_corrected'], '.3f') if pd.notna(r.get('p_corrected', np.nan)) else '—'):>9}  {r['verdict']}")
    L += ["", "Uses publication dates to reconstruct history; almost all items were first seen on 2026-09-13,",
          "so no live forward record exists yet. A PASS is necessary, not sufficient."]
    return "\n".join(L)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    res = run()
    print(format_report(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
