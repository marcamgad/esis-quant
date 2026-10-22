"""
Agent 7 -- RESEARCH LAB. A queue runner for pre-registered hypotheses.

WHY A LAB AND NOT AN IDEA GENERATOR. Selection has been tested six ways on this
data: four null, one significantly negative (-20.9%/yr), the news study null.
On a market like that, a process that generates hundreds of variants is
guaranteed to find a false positive. The lab is safe only because:

  1. A hypothesis is REGISTERED (spec + pass criteria, timestamped) before it
     runs. Nothing is tested that was not written down first.
  2. EVERY trial ever run is logged in research_trials, and each result is
     judged by the Deflated Sharpe Ratio with N = all trials so far plus the
     documented prior trials. The bar rises with every idea tried.
  3. It runs on a SANDBOX COPY of the database, from the research package, which
     the live-path tripwire test bans from the trading path. A passing trial
     cannot reach a ticket; it earns a forward paper period, nothing more.

GRAMMAR. factor template x lookback x holding period, long-only (EGX has no
short selling): buy the top quintile of the point-in-time liquid universe,
equal weight, hold, rebalance; judged as EXCESS over the equal-weight universe,
net of the real fee schedule at the account's size.

  mom(L)     trailing L-session return, skipping the latest 5 sessions
  rev(L)     minus the trailing L-session return
  lowvol(L)  minus the L-session volatility
  illiq(L)   Amihud illiquidity: mean |return| / value traded over L
  liq(L)     minus Amihud (the most liquid names)

PASS (fixed here, before any trial ran):
  DSR > 0.95  AND  net excess > 0  AND  positive in >= 60% of calendar years
  AND  positive in both halves of the sample.

    py -3.11 -m research.lab grid          # register + run the standard grid
    py -3.11 -m research.lab report
"""

import json
import logging
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd

logger = logging.getLogger("esis_quant.research_lab")

# Trials run BEFORE this log existed (documented in reports/ and memory): the
# regime-FM, rank persistence, Fama-MacBeth variants, causal/double-ML, the
# multi-timeframe composite and its inverse, sentiment, TFT, TypedGP, news
# event types. 20 is a LOWER bound; the true count is higher.
PRIOR_TRIALS = 20
TEMPLATES = ("mom", "rev", "lowvol", "illiq", "liq")
LOOKBACKS = (20, 60, 120, 250)
HOLDS = (20, 60)
QUANTILE = 0.2
BOOK_EGP = 12_000.0
CRITERIA = {"dsr": 0.95, "min_excess": 0.0, "min_year_share": 0.6, "both_halves": True}


# --------------------------------------------------------------------------
# sandbox
# --------------------------------------------------------------------------
def sandbox() -> str:
    """Copy the live DB once per process and point this process at the copy."""
    import config
    if getattr(config, "_LAB_SANDBOX", None):
        return config.DB_PATH
    dst = os.path.join(tempfile.gettempdir(), "esis_lab_sandbox.db")
    src = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    out = sqlite3.connect(dst)
    src.backup(out)
    out.close()
    src.close()
    config.DB_PATH, config._LAB_SANDBOX = dst, True
    return dst


def _trial_db():
    """The trial log lives in the REAL database (it must persist across runs);
    it is the only thing the lab writes there."""
    import config
    path = getattr(config, "_LAB_REAL_DB", None) or config.DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS research_trials (
        id INTEGER PRIMARY KEY AUTOINCREMENT, registered_at TEXT, spec_json TEXT, criteria_json TEXT,
        ran_at TEXT, n_obs INTEGER, ann_excess REAL, sharpe REAL, dsr REAL, n_trials INTEGER,
        year_share REAL, half1 REAL, half2 REAL, verdict TEXT)""")
    return conn


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _panels():
    from backtest.engine import load_price_returns, load_value_traded
    from data.pit_universe import build_pit_universe
    r = load_price_returns()
    r = r[r.index >= "2015-01-01"]
    vt = load_value_traded().reindex(index=r.index, columns=r.columns)
    pit = build_pit_universe()
    uni = (pit.pivot_table(index="date", columns="ticker", values="in_universe", aggfunc="max")
              .reindex(index=r.index, columns=r.columns).fillna(0).astype(bool))
    return r, vt, uni


def factor(template: str, L: int, r: pd.DataFrame, vt: pd.DataFrame) -> pd.DataFrame:
    """Score per (date, ticker), using data up to and including that date only."""
    if template in ("mom", "rev"):
        cum = np.log1p(r.fillna(0)).rolling(L).sum().shift(5)
        return cum if template == "mom" else -cum
    if template == "lowvol":
        return -r.rolling(L, min_periods=L // 2).std()
    amihud = (r.abs() / vt.replace(0, np.nan)).rolling(L, min_periods=L // 2).mean()
    return amihud if template == "illiq" else -amihud


def backtest(score: pd.DataFrame, r: pd.DataFrame, uni: pd.DataFrame, hold: int,
             book: float = BOOK_EGP, fill: dict = None) -> pd.Series:
    """Daily net excess return of the top-quintile basket over the equal-weight universe.

    `fill`: None = trades at the signal close (the first 40 trials). Or
    {"ratio": load_fill_ratio(), "lag": 2, "slippage": 0.0057} = the measured
    execution of the live account (backtest.engine.fill_adjustment)."""
    from backtest.fees import one_side_cost
    from backtest.engine import fill_adjustment
    dates = r.index
    out = pd.Series(0.0, index=dates)
    held = pd.Index([])
    for i in range(0, len(dates) - 1, hold):
        d = dates[i]
        s = score.loc[d][uni.loc[d]].dropna()
        if len(s) < 10:
            continue
        k = max(2, int(len(s) * QUANTILE))
        pick = s.nlargest(k).index
        names = uni.loc[d][uni.loc[d]].index
        window = dates[i + 1:i + 1 + hold]
        basket = r.loc[window, pick].fillna(0).mean(axis=1)
        bench = r.loc[window, names].fillna(0).mean(axis=1)
        # fees: names entering or leaving pay one side each, at the real per-order cost
        changed = len(set(pick) ^ set(held))
        per_name = book / len(pick)
        cost = changed * one_side_cost(per_name)["total_egp"] / book
        ex = basket - bench
        ex.iloc[0] -= cost
        if fill is not None and len(window):
            names = pick.union(held)
            dw = (pd.Series(1.0 / len(pick), index=pick).reindex(names, fill_value=0.0)
                  - (pd.Series(1.0 / len(held), index=held).reindex(names, fill_value=0.0)
                     if len(held) else 0.0))
            ex = ex + fill_adjustment(dw, window, r, fill["ratio"], lag=fill.get("lag", 1),
                                      slippage=fill.get("slippage", 0.0)).values
        out.loc[window] = ex.values
        held = pick
    return out[out.index >= dates[250]]          # skip the warm-up the longest lookback needs


# --------------------------------------------------------------------------
# registration and evaluation
# --------------------------------------------------------------------------
def register(spec: dict) -> int:
    conn = _trial_db()
    cur = conn.execute("INSERT INTO research_trials (registered_at, spec_json, criteria_json) VALUES (?,?,?)",
                       (pd.Timestamp.now().isoformat(timespec="seconds"), json.dumps(spec, sort_keys=True),
                        json.dumps(CRITERIA)))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return int(rid)


def evaluate(ex: pd.Series, n_trials: int, trial_sharpes) -> dict:
    from analysis.deflated_sharpe import deflated_sharpe
    ex = ex.dropna()
    yrs = len(ex) / 245
    ann = float((1 + ex).prod() ** (1 / yrs) - 1) if yrs > 0 else float("nan")
    sh = float(ex.mean() / ex.std() * np.sqrt(245)) if ex.std() > 0 else 0.0
    dsr = deflated_sharpe(ex, n_trials=n_trials, trial_sharpes=trial_sharpes)["dsr"]
    by_year = (1 + ex).groupby(ex.index.year).prod() - 1
    half = len(ex) // 2
    h1, h2 = float(ex.iloc[:half].mean() * 245), float(ex.iloc[half:].mean() * 245)
    share = float((by_year > 0).mean())
    passed = (dsr > CRITERIA["dsr"] and ann > CRITERIA["min_excess"]
              and share >= CRITERIA["min_year_share"] and h1 > 0 and h2 > 0)
    return {"n_obs": len(ex), "ann_excess": ann, "sharpe": sh, "dsr": float(dsr), "year_share": share,
            "half1": h1, "half2": h2, "verdict": "CANDIDATE (forward paper only)" if passed else "NULL"}


def run_pending() -> pd.DataFrame:
    """Run every registered trial that has not run yet."""
    import config
    config._LAB_REAL_DB = config.DB_PATH
    sandbox()
    conn = _trial_db()
    pending = pd.read_sql("SELECT id, spec_json FROM research_trials WHERE ran_at IS NULL", conn)
    done = pd.read_sql("SELECT sharpe FROM research_trials WHERE ran_at IS NOT NULL", conn)
    conn.close()
    if pending.empty:
        return pending
    r, vt, uni = _panels()
    results, sharpes = [], done["sharpe"].dropna().tolist()
    series = {}
    ratio = None
    for p in pending.itertuples():
        spec = json.loads(p.spec_json)
        fill = None
        if spec.get("fill") == "typical":
            if ratio is None:
                from backtest.engine import load_fill_ratio
                ratio = load_fill_ratio()
            fill = {"ratio": ratio, "lag": spec.get("fill_lag", 2), "slippage": spec.get("fill_slippage", 0.0)}
        ex = backtest(factor(spec["template"], spec["lookback"], r, vt), r, uni, spec["hold"], fill=fill)
        series[p.id] = ex
        sharpes.append(float(ex.mean() / ex.std() * np.sqrt(245)) if ex.std() > 0 else 0.0)
    # judge every trial against ALL trials logged, including this batch
    n_total = PRIOR_TRIALS + len(done) + len(pending)
    conn = _trial_db()
    for p in pending.itertuples():
        ev = evaluate(series[p.id], n_total, sharpes)
        conn.execute("UPDATE research_trials SET ran_at=?, n_obs=?, ann_excess=?, sharpe=?, dsr=?, n_trials=?, "
                     "year_share=?, half1=?, half2=?, verdict=? WHERE id=?",
                     (pd.Timestamp.now().isoformat(timespec="seconds"), ev["n_obs"], ev["ann_excess"],
                      ev["sharpe"], ev["dsr"], n_total, ev["year_share"], ev["half1"], ev["half2"],
                      ev["verdict"], int(p.id)))
        results.append({"id": p.id, **json.loads(p.spec_json), **ev})
    conn.commit()
    conn.close()
    return pd.DataFrame(results)


def grid(execution: str = "close") -> pd.DataFrame:
    """Register the standard grid (once per execution model) and run it.

    execution="measured" re-runs every hypothesis with the live account's
    measured execution (fills 2 sessions late at the typical price + 0.57%).
    These are NEW trials and count toward the deflation N: a second look at the
    same ideas is still a second look."""
    conn = _trial_db()
    have = {r[0] for r in conn.execute("SELECT spec_json FROM research_trials")}
    conn.close()
    for t in TEMPLATES:
        for L in LOOKBACKS:
            for h in HOLDS:
                spec = {"template": t, "lookback": L, "hold": h, "quantile": QUANTILE,
                        "universe": "pit_top50", "book_egp": BOOK_EGP}
                if execution == "measured":
                    spec.update({"fill": "typical", "fill_lag": 2, "fill_slippage": 0.0057})
                if json.dumps(spec, sort_keys=True) not in have:
                    register(spec)
    return run_pending()


def report() -> str:
    conn = _trial_db()
    t = pd.read_sql("SELECT * FROM research_trials WHERE ran_at IS NOT NULL ORDER BY sharpe DESC", conn)
    conn.close()
    if t.empty:
        return "no trials run yet"
    spec = pd.json_normalize(t["spec_json"].map(json.loads))
    t = pd.concat([t.drop(columns=["spec_json", "criteria_json"]), spec], axis=1)
    n = int(t["n_trials"].max())
    L = ["=" * 78, f"RESEARCH LAB — {len(t)} trials logged (+{PRIOR_TRIALS} prior) | deflation N = {n}", "=" * 78,
         f"  candidates: {int(t['verdict'].str.startswith('CANDIDATE').sum())} | "
         f"positive net excess: {int((t['ann_excess'] > 0).sum())} of {len(t)}", ""]
    for r in t.head(12).itertuples():
        L.append(f"  #{r.id:<3} {r.template:6} L={r.lookback:<3} hold={r.hold:<2}  excess {r.ann_excess:+6.1%}/yr  "
                 f"Sharpe {r.sharpe:+.2f}  DSR {r.dsr:.2f}  yrs+ {r.year_share:.0%}  "
                 f"halves {r.half1:+.1%}/{r.half2:+.1%}  -> {r.verdict}")
    L.append("  ...")
    for r in t.tail(4).itertuples():
        L.append(f"  #{r.id:<3} {r.template:6} L={r.lookback:<3} hold={r.hold:<2}  excess {r.ann_excess:+6.1%}/yr  "
                 f"Sharpe {r.sharpe:+.2f}  DSR {r.dsr:.2f}  -> {r.verdict}")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    if argv[:1] == ["grid"]:
        grid("measured" if "--measured" in argv else "close")
    elif argv[:1] == ["run"]:
        run_pending()
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
