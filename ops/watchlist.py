"""
Watchlist: stocks that became INTERESTING recently, and only while they are.

WHAT IT IS. Every refresh scans the tradable universe for fresh, dated
triggers -- something that happened, not something that is merely true:

  group      trigger                         source
  earnings   press: profit up >= 25% y/y     news_figures (confidence >= 0.6)
             statement: quarterly NI >= +30% fundamentals, newly available
  dividend   a cash dividend announced       news_figures
  price      near the 52-week low while the  prices + fundamentals
             latest quarter is profitable
             and growing; or a 52-week high
             on >= 2x normal value traded
  attention  value traded >= 3x its median;  prices, news_links
             >= 3 linked news items in 3 days
  analyst    added by hand / by Claude with  watchlist_triggers
             a written reason

A stock stays only while it has a positive trigger newer than TTL_DAYS
(default 7 days, about 5 sessions). With no fresh trigger it expires on its
own -- nothing lingers. Cautions (illiquid, high leverage, loss-making, a recent
rights issue/split, a big profit drop) lower the score and are always shown;
negative equity, funds and untradable names are excluded outright.

WHAT IT IS NOT. Not a recommendation and not a forecast. Every method this
system has tested for picking stocks failed out of sample (memory: Phase 2
selection, momentum, earnings drift, index additions). The watchlist is a
reading list with the evidence attached. To find out whether it is worth more
than that, every new entry is RECORDED AT THE TIME with its price, the EGX30
level and its reasons (watchlist_calls), and score_calls() later compares it,
at 20 / 60 / 250 sessions, with the EGX30, cash, and the average stock on the
same dates. Roughly 70 independent calls are needed before that comparison
means anything.

SIZE ARITHMETIC. min_position() gives the smallest whole number of shares at
which the all-in round trip (Thndr fees + 0.57% slippage each side) is at most
MAX_COST_PCT of the position. That is arithmetic about fees, not a suggestion
to buy.

WRITE BOUNDARY (engine-enforced, ops/agent_guard): only the four watchlist
tables. Reads may run on a read-only connection (ESIS_DB_READONLY=1).

    py -3.11 -m ops.watchlist            # refresh + show
    py -3.11 -m ops.watchlist score      # how past entries did
"""

import json
import math
import sqlite3
import sys

import numpy as np
import pandas as pd

import config
from data.database import get_connection
from ops import agent_guard as ag

AGENT = "watchlist"
WRITABLE = {"watchlist", "watchlist_triggers", "watchlist_calls", "research_notes", "system_signals",
            "valuation_cache"}

TTL_DAYS = 7
MAX_ENTRIES = 12
SLIPPAGE = 0.0057                 # per side, measured on the live account
MAX_COST_PCT = 0.03               # round trip, all-in, as a share of the position
MIN_VALUE_TRADED = 250_000        # EGP/day median; below this a 2k order moves the price
HORIZONS = (20, 60, 250)
CALL_COOLDOWN_DAYS = 60           # a re-entry within this period is not a new call
EXCLUDED_TYPES = {"fund", "fx_class"}
FINANCIAL_SECTORS = {"Banks", "Non-bank Financial Services"}   # D/E is not meaningful there

GROUP_WEIGHT = {"earnings": 2.0, "dividend": 1.0, "price": 1.5, "attention": 1.0, "analyst": 2.0}
CAUTION_WEIGHT = {"illiquid": 1.5, "high_leverage": 1.0, "loss_making": 1.5,
                  "recent_capital_change": 1.0, "profit_drop": 1.5, "stale_financials": 0.5}


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def ensure_tables():
    """Schema setup, through the same guard: it can create only its own tables."""
    con = _writer()
    try:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker TEXT PRIMARY KEY, status TEXT, score REAL, first_added TEXT,
            last_trigger TEXT, expires TEXT, reasons TEXT, cautions TEXT, updated_at TEXT);
        CREATE TABLE IF NOT EXISTS watchlist_triggers (
            ticker TEXT, grp TEXT, kind TEXT, event_date TEXT, detail TEXT, ref TEXT,
            detected_at TEXT, PRIMARY KEY (ticker, kind, event_date, ref));
        CREATE TABLE IF NOT EXISTS watchlist_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, call_date TEXT, price REAL,
            egx30 REAL, score REAL, reasons TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS system_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, signal_date TEXT, ticker TEXT, action TEXT,
            shares INTEGER, price REAL, value_egp REAL, egx30 REAL, rule TEXT, reasons TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS valuation_cache (
            ticker TEXT, fetched_on TEXT, pe REAL, pb REAL, div_yield REAL, earnings_yield REAL,
            PRIMARY KEY (ticker, fetched_on));
        CREATE TABLE IF NOT EXISTS research_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, author TEXT, note TEXT, created_at TEXT);
        """)
        con.commit()
    finally:
        con.close()


def _writer():
    """A read-write connection that can touch ONLY the watchlist tables."""
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    return ag.guarded(con, WRITABLE, AGENT)


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def _universe(as_of) -> pd.DataFrame:
    with get_connection() as conn:
        d = conn.execute("SELECT MAX(date) FROM tradable_universe WHERE date <= ?", (str(as_of.date()),)).fetchone()[0]
        u = pd.read_sql("SELECT t.ticker, s.name_english, s.sector, s.security_type FROM tradable_universe t "
                        "LEFT JOIN stocks s ON s.ticker = t.ticker WHERE t.date = ? AND t.is_tradable = 1",
                        conn, params=(d,))
        held = {r[0] for r in conn.execute("SELECT ticker FROM user_positions WHERE shares > 0")}
    u = u[~u["security_type"].isin(EXCLUDED_TYPES)].copy()
    u["held"] = u["ticker"].isin(held)
    return u.set_index("ticker")


def _prices(tickers, as_of, days: int = 420) -> dict:
    start = (as_of - pd.Timedelta(days=days)).date()
    with get_connection() as conn:
        px = pd.read_sql("SELECT ticker, date, close, adjusted_close, value_traded_egp FROM daily_ohlcv "
                         "WHERE date >= ? AND date <= ? AND close > 0", conn,
                         params=(str(start), str(as_of.date())), parse_dates=["date"])
    px = px[px["ticker"].isin(tickers)].drop_duplicates(["ticker", "date"], keep="last")
    adj = px.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    from data.ca_factors import adjust_panel
    return {"adj": adjust_panel(adj),
            "close": px.pivot(index="date", columns="ticker", values="close").sort_index(),
            "value": px.pivot(index="date", columns="ticker", values="value_traded_egp").sort_index()}


MAX_QUARTER_AGE_DAYS = 200      # a "latest quarter" older than this is not current
SMALL_BASE = 3.0                # growth beyond +300% is almost always a tiny prior year


def _growth(ticker: str, as_of) -> dict:
    """Latest knowable quarter vs a year earlier, or {} when it is stale or
    the comparison is meaningless (prior year zero or a loss)."""
    from data.fundamentals import quarterly_yoy
    q = quarterly_yoy(ticker, str(as_of.date()))
    if not q or q.get("change") is None or (q.get("prior_value") or 0) <= 0:
        return {}
    if (as_of - pd.Timestamp(q["period_end"])).days > MAX_QUARTER_AGE_DAYS:
        return {}
    g = q["change"]
    q["label"] = (f"more than tripled from a small base ({q['prior_value']:,.0f} -> {q['value']:,.0f})"
                  if g > SMALL_BASE else f"{g:+.0%} y/y")
    return q


# --------------------------------------------------------------------------
# triggers: (ticker, grp, kind, event_date, detail, ref)
# --------------------------------------------------------------------------
def _news_triggers(u, as_of, since) -> list:
    out = []
    with get_connection() as conn:
        # The event date is when the article was PUBLISHED. first_seen_at is when this
        # system fetched it -- for the Al Mal archive that is October 2026 for an
        # article from 2023, which once put 2023 results on the list as "news".
        # Items with no publication date only count if first seen inside the window.
        pub = "COALESCE(i.published_at, f.first_seen_at)"
        f = pd.read_sql(f"SELECT f.ticker, f.kind, f.metric, f.pct_change, f.period, f.fiscal_year, f.news_id, "
                        f"{pub} AS event_at, i.headline FROM news_figures f JOIN news_items i ON i.id = f.news_id "
                        f"WHERE f.confidence >= 0.6 AND {pub} > ? AND {pub} <= ? AND f.first_seen_at <= ?",
                        conn, params=(since.isoformat(), as_of.isoformat(), as_of.isoformat()))
        # attention: only articles that NAME the company (text/code links), not the
        # search results fetched for it ('query' links are mostly unrelated pages),
        # counted against the company's own normal rate.
        pubi = "COALESCE(i.published_at, i.first_seen_at)"
        links = pd.read_sql(f"SELECT l.ticker, i.id, {pubi} AS event_at FROM news_links l JOIN news_items i "
                            f"ON i.id = l.news_id WHERE l.method IN ('text', 'code') "
                            f"AND {pubi} > ? AND {pubi} <= ? AND i.first_seen_at <= ?", conn,
                            params=((as_of - pd.Timedelta(days=93)).isoformat(), as_of.isoformat(),
                                    as_of.isoformat()))
    f = f[f["ticker"].isin(u.index)]
    for r in f.itertuples():
        day = str(r.event_at)[:10]
        per = " ".join(x for x in (r.period if isinstance(r.period, str) else "",
                                    str(int(r.fiscal_year)) if pd.notna(r.fiscal_year) else "") if x)
        head = (r.headline or "")[:120]
        if pd.notna(r.fiscal_year) and int(r.fiscal_year) < as_of.year - 1:
            continue
        if r.kind == "result" and r.metric == "net_profit" and pd.notna(r.pct_change):
            if r.pct_change >= 25:
                out.append((r.ticker, "earnings", "press_profit_growth", day,
                            f"press: net profit {r.pct_change:+.0f}% {per} -- \"{head}\"", f"news:{r.news_id}"))
            elif r.pct_change <= -25:
                out.append((r.ticker, "caution", "profit_drop", day,
                            f"press: net profit {r.pct_change:+.0f}% {per}", f"news:{r.news_id}"))
        elif r.kind == "dividend":
            out.append((r.ticker, "dividend", "dividend_announced", day, f"press: dividend -- \"{head}\"",
                        f"news:{r.news_id}"))
    links = links[links["ticker"].isin(u.index)]
    links["event_at"] = pd.to_datetime(links["event_at"], errors="coerce", utc=True).dt.tz_localize(None)
    cut = as_of - pd.Timedelta(days=3)
    recent = links[links["event_at"] > cut].groupby("ticker")["id"].nunique()
    base = links[links["event_at"] <= cut].groupby("ticker")["id"].nunique() / 90 * 3   # usual 3-day count
    for t, n in recent[recent >= 3].items():
        usual = float(base.get(t, 0.0))
        if n >= 3 * max(usual, 1.0):
            out.append((t, "attention", "news_cluster", str(as_of.date()),
                        f"{n} articles naming it in 3 days (usual ~{usual:.1f})", f"cluster:{as_of.date()}"))
    return out


def _statement_triggers(u, as_of, since) -> list:
    from data.fundamentals import quarterly_yoy
    with get_connection() as conn:
        fresh = pd.read_sql("SELECT DISTINCT ticker, available_from FROM fundamentals WHERE frequency = 'quarterly' "
                            "AND line_item = 'Net Income' AND available_from > ? AND available_from <= ?",
                            conn, params=(str(since.date()), str(as_of.date())))
    out = []
    for r in fresh[fresh["ticker"].isin(u.index)].itertuples():
        q = _growth(r.ticker, as_of)
        if q and q["value"] > 0 and q["change"] >= 0.30:
            out.append((r.ticker, "earnings", "statement_profit_growth", r.available_from,
                        f"statement: net income {q['label']} ({q['period']})", f"stmt:{q['period']}"))
    return out


def _price_triggers(u, px, as_of) -> list:
    from data.fundamentals import quarterly_yoy
    adj, val = px["adj"], px["value"]
    if adj.empty:
        return []
    last = adj.index[-1]
    win = adj[adj.index > last - pd.Timedelta(days=365)]
    out = []
    for t in u.index:
        if t not in win or win[t].dropna().shape[0] < 150:
            continue
        s = win[t].dropna()
        p, lo, hi = float(s.iloc[-1]), float(s.min()), float(s.max())
        v = val[t].dropna() if t in val else pd.Series(dtype=float)
        med = float(v.tail(60).median()) if len(v) >= 20 else np.nan
        v_today = float(v.iloc[-1]) if len(v) and v.index[-1] == last else np.nan
        day = str(last.date())
        if np.isfinite(med) and med > 0 and np.isfinite(v_today) and v_today >= 3 * med:
            out.append((t, "attention", "value_surge", day,
                        f"value traded {v_today / 1e6:,.1f}M EGP = {v_today / med:.1f}x its 60-day median",
                        f"vol:{day}"))
        if p >= hi * 0.999 and np.isfinite(med) and np.isfinite(v_today) and v_today >= 2 * med:
            out.append((t, "price", "new_52w_high", day, f"closed at a 52-week high ({p:,.2f}) on "
                        f"{v_today / med:.1f}x normal value", f"hi:{day}"))
        if p <= lo * 1.10:
            q = _growth(t, as_of)
            if q and q["value"] > 0 and q["change"] > 0:
                out.append((t, "price", "near_52w_low_profitable", day,
                            f"{(p / lo - 1):+.0%} above its 52-week low while the latest quarter "
                            f"({q['period']}) grew: {q['label']}", f"lo:{day}"))
    return out


def _cautions(t, row, px, as_of) -> list:
    from data.fundamentals import leverage_as_of, quarterly_yoy
    c = []
    v = px["value"][t].dropna().tail(60) if t in px["value"] else pd.Series(dtype=float)
    med = float(v.median()) if len(v) >= 20 else 0.0
    if med < MIN_VALUE_TRADED:
        c.append(("illiquid", f"median value traded {med:,.0f} EGP/day"))
    lev = leverage_as_of(t, str(as_of.date()))
    if lev.get("negative_equity"):
        c.append(("negative_equity", "shareholders' equity <= 0"))
    elif row.get("sector") not in FINANCIAL_SECTORS and (lev.get("debt_to_equity") or 0) > 2:
        c.append(("high_leverage", f"debt/equity {lev['debt_to_equity']:.1f}"))
    q = quarterly_yoy(t, str(as_of.date()))
    if not q or (as_of - pd.Timestamp(q["period_end"])).days > MAX_QUARTER_AGE_DAYS:
        c.append(("stale_financials", f"no quarter in the last {MAX_QUARTER_AGE_DAYS} days"
                                      + (f" (latest {q['period']})" if q else "")))
    elif q["value"] <= 0:
        c.append(("loss_making", f"latest quarter ({q['period']}) net income {q['value']:,.0f}"))
    with get_connection() as conn:
        ca = conn.execute("SELECT date, action_type FROM corporate_actions WHERE ticker = ? AND date > ? "
                          "AND date <= ? AND action_type IN ('SPLIT_CORROBORATED','SPLIT')",
                          (t, str((as_of - pd.Timedelta(days=45)).date()), str(as_of.date()))).fetchall()
    if ca:
        c.append(("recent_capital_change", f"split / rights issue / merger on {ca[-1][0]}"))
    return c


# --------------------------------------------------------------------------
# refresh
# --------------------------------------------------------------------------
def refresh(as_of=None, ttl_days: int = TTL_DAYS) -> dict:
    """Detect triggers, then rebuild the active list. Idempotent."""
    ensure_tables()
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    since = as_of - pd.Timedelta(days=ttl_days)
    u = _universe(as_of)
    px = _prices(list(u.index), as_of)
    found = (_news_triggers(u, as_of, since) + _statement_triggers(u, as_of, since)
             + _price_triggers(u, px, as_of))
    con = _writer()
    try:
        for t, grp, kind, day, detail, ref in found:
            ag.run(con, "INSERT OR IGNORE INTO watchlist_triggers VALUES (?,?,?,?,?,?,?)",
                   (t, grp, kind, day, detail, ref, _now()), AGENT)
        con.commit()
        trig = pd.read_sql("SELECT * FROM watchlist_triggers WHERE event_date > ? AND event_date <= ?", con,
                           params=(str(since.date()), str(as_of.date())))
        prev = {r[0]: r[1] for r in con.execute("SELECT ticker, status FROM watchlist")}
        entries = []
        for t, g in trig.groupby("ticker"):
            if t not in u.index:
                continue
            pos = g[g["grp"] != "caution"]
            if pos.empty:
                continue
            cautions = _cautions(t, u.loc[t].to_dict(), px, as_of)
            cautions += [("profit_drop", d) for d in g[g["grp"] == "caution"]["detail"]]
            if any(k == "negative_equity" for k, _ in cautions):
                continue
            score = sum(GROUP_WEIGHT.get(gr, 1.0) for gr in pos["grp"].unique())
            score -= sum(CAUTION_WEIGHT.get(k, 0.0) for k in {k for k, _ in cautions})
            if score <= 0:
                continue
            last = pd.Timestamp(pos["event_date"].max())
            entries.append({"ticker": t, "score": round(score, 2), "last_trigger": str(last.date()),
                            "expires": str((last + pd.Timedelta(days=ttl_days)).date()),
                            "reasons": pos.sort_values("event_date", ascending=False)[["grp", "detail"]]
                            .drop_duplicates().values.tolist(),
                            "cautions": [list(x) for x in dict(cautions).items()]})
        entries = sorted(entries, key=lambda e: -e["score"])[:MAX_ENTRIES]
        active = {e["ticker"] for e in entries}
        new_calls = []
        for e in entries:
            first = con.execute("SELECT first_added FROM watchlist WHERE ticker = ?", (e["ticker"],)).fetchone()
            ag.run(con, "INSERT INTO watchlist VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET "
                        "status='active', score=excluded.score, last_trigger=excluded.last_trigger, "
                        "expires=excluded.expires, reasons=excluded.reasons, cautions=excluded.cautions, "
                        "updated_at=excluded.updated_at, first_added=CASE WHEN watchlist.status='active' "
                        "THEN watchlist.first_added ELSE excluded.first_added END",
                   (e["ticker"], "active", e["score"], str(as_of.date()), e["last_trigger"], e["expires"],
                    json.dumps(e["reasons"], ensure_ascii=False), json.dumps(e["cautions"], ensure_ascii=False),
                    _now()), AGENT)
            if prev.get(e["ticker"]) != "active":
                recent = con.execute("SELECT 1 FROM watchlist_calls WHERE ticker = ? AND call_date > ?",
                                     (e["ticker"], str((as_of - pd.Timedelta(days=CALL_COOLDOWN_DAYS)).date()))
                                     ).fetchone()
                if not recent:
                    new_calls.append(e)
        expired = [t for t, s in prev.items() if s == "active" and t not in active]
        for t in expired:
            ag.run(con, "UPDATE watchlist SET status='expired', updated_at=? WHERE ticker=?", (_now(), t), AGENT)
        from data import market
        lv = market.level()
        lv = lv[lv.index <= as_of]
        egx = float(lv.iloc[-1]) if len(lv) else None
        for e in new_calls:
            s = px["adj"][e["ticker"]].dropna()
            if s.empty:
                continue
            ag.run(con, "INSERT INTO watchlist_calls (ticker, call_date, price, egx30, score, reasons, created_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                   (e["ticker"], str(s.index[-1].date()), float(s.iloc[-1]), egx, e["score"],
                    json.dumps(e["reasons"], ensure_ascii=False), _now()), AGENT)
        con.commit()
    finally:
        con.close()
    return {"as_of": str(as_of.date()), "triggers_found": len(found), "active": len(entries),
            "new_calls": [e["ticker"] for e in new_calls], "expired": expired}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def active() -> pd.DataFrame:
    ensure_tables()
    with get_connection() as conn:
        w = pd.read_sql("SELECT * FROM watchlist WHERE status = 'active' ORDER BY score DESC", conn)
    for col in ("reasons", "cautions"):
        w[col] = w[col].apply(lambda s: json.loads(s) if s else [])
    return w


def last_close(ticker: str):
    with get_connection() as conn:
        r = conn.execute("SELECT date, close FROM daily_ohlcv WHERE ticker = ? AND close > 0 "
                         "ORDER BY date DESC LIMIT 1", (ticker,)).fetchone()
    return (r[0], float(r[1])) if r else (None, None)


def min_position(ticker: str, max_cost_pct: float = MAX_COST_PCT) -> dict:
    """Smallest whole number of shares whose all-in round trip <= max_cost_pct."""
    from backtest.fees import round_trip_cost
    day, p = last_close(ticker)
    if not p:
        return {}
    cost = lambda v: round_trip_cost(v)["pct_of_order"] + 2 * SLIPPAGE
    if cost(1e9) > max_cost_pct:
        return {"ticker": ticker, "price": p, "price_date": day, "possible": False}
    lo, hi = p, 1e9
    while cost(lo) > max_cost_pct and hi - lo > 1:      # cost falls as the order grows (flat fee)
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cost(mid) > max_cost_pct else (lo, mid)
        if cost(lo) <= max_cost_pct:
            break
    v = lo if cost(lo) <= max_cost_pct else hi
    n = max(1, math.ceil(v / p))
    return {"ticker": ticker, "price": p, "price_date": day, "possible": True, "shares": n,
            "value_egp": n * p, "round_trip_pct": cost(n * p), "max_cost_pct": max_cost_pct,
            "one_share_round_trip_pct": cost(p)}


def score_calls(as_of=None) -> dict:
    """Every recorded entry vs EGX30, cash and the average stock, at 20/60/250 sessions."""
    from backtest.engine import load_price_returns
    from data import market
    from data.rates import cash_rate_series
    ensure_tables()
    with get_connection() as conn:
        calls = pd.read_sql("SELECT * FROM watchlist_calls ORDER BY call_date", conn, parse_dates=["call_date"])
    if calls.empty:
        return {"calls": 0, "table": calls, "note": "no calls recorded yet"}
    r = load_price_returns()
    r = r[r.index >= calls["call_date"].min() - pd.Timedelta(days=5)]
    ew = r.clip(-0.5, 1.0).mean(axis=1)
    egx = market.returns(r.index).fillna(0)
    cash = (1 + cash_rate_series(r.index, haircut=0.0)) ** (1 / 245) - 1
    rows = []
    for c in calls.itertuples():
        i0 = r.index.searchsorted(c.call_date, side="right")       # returns AFTER the call day
        rec = {"ticker": c.ticker, "call_date": str(c.call_date.date()), "score": c.score}
        for h in HORIZONS:
            if i0 + h > len(r.index) or c.ticker not in r:
                continue
            sl = slice(i0, i0 + h)
            stock = float(np.prod(1 + r[c.ticker].iloc[sl].fillna(0)) - 1)
            rec[f"ret_{h}"] = stock
            rec[f"vs_egx30_{h}"] = stock - float(np.prod(1 + egx.iloc[sl]) - 1)
            rec[f"vs_avg_stock_{h}"] = stock - float(np.prod(1 + ew.iloc[sl].fillna(0)) - 1)
            rec[f"vs_cash_{h}"] = stock - float(np.prod(1 + cash.iloc[sl]) - 1)
        rows.append(rec)
    t = pd.DataFrame(rows)
    summary = {}
    for h in HORIZONS:
        col = f"vs_avg_stock_{h}"
        if col in t and t[col].notna().any():
            d = t.dropna(subset=[col])
            summary[h] = {"n": len(d), "vs_egx30": float(d[f"vs_egx30_{h}"].mean()),
                          "vs_avg_stock": float(d[col].mean()), "vs_cash": float(d[f"vs_cash_{h}"].mean()),
                          "hit_vs_avg_stock": float((d[col] > 0).mean())}
    return {"calls": len(calls), "summary": summary, "table": t,
            "note": "inconclusive until ~70 independent calls have matured"}


def add_manual(ticker: str, reason: str, author: str = "claude") -> dict:
    """A hand-written trigger (group 'analyst'); it expires like any other."""
    ensure_tables()
    ticker = ticker.strip().upper()
    day = str(pd.Timestamp.now().date())
    con = _writer()
    try:
        ag.run(con, "INSERT OR REPLACE INTO watchlist_triggers VALUES (?,?,?,?,?,?,?)",
               (ticker, "analyst", "analyst_note", day, f"{author}: {reason[:400]}", f"manual:{author}", _now()),
               AGENT)
        ag.run(con, "INSERT INTO research_notes (ticker, author, note, created_at) VALUES (?,?,?,?)",
               (ticker, author, reason[:4000], _now()), AGENT)
        con.commit()
    finally:
        con.close()
    return {"ticker": ticker, "trigger_date": day, "expires_unless_renewed": str(
        (pd.Timestamp(day) + pd.Timedelta(days=TTL_DAYS)).date())}


def add_note(ticker: str, note: str, author: str = "claude") -> dict:
    ensure_tables()
    con = _writer()
    try:
        ag.run(con, "INSERT INTO research_notes (ticker, author, note, created_at) VALUES (?,?,?,?)",
               (ticker.strip().upper(), author, note[:4000], _now()), AGENT)
        con.commit()
    finally:
        con.close()
    return {"ticker": ticker.strip().upper(), "saved": True}


def format_active(w: pd.DataFrame = None) -> str:
    w = active() if w is None else w
    if w.empty:
        return "Watchlist is empty: nothing had a fresh trigger in the last %d days." % TTL_DAYS
    L = [f"WATCHLIST — {len(w)} names with a trigger in the last {TTL_DAYS} days "
         "(evidence, not recommendations)"]
    for r in w.itertuples():
        L.append(f"\n{r.ticker}  score {r.score:.1f}  in since {r.first_added}, expires {r.expires} unless renewed")
        for grp, detail in r.reasons:
            L.append(f"   + [{grp}] {detail}")
        for kind, detail in r.cautions:
            L.append(f"   ! [{kind}] {detail}")
        m = min_position(r.ticker)
        if m.get("possible"):
            L.append(f"   size arithmetic: {m['shares']:,} shares x {m['price']:,.2f} = {m['value_egp']:,.0f} EGP "
                     f"keeps the round trip at {m['round_trip_pct']:.2%} (1 share alone: "
                     f"{m['one_share_round_trip_pct']:.0%})")
    L.append("\nThe system has not found a stock-picking method that beats the EGX30 or cash out of sample. "
             "Every entry is recorded and scored later (py -3.11 -m ops.watchlist score).")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    import logging
    logging.disable(logging.WARNING)
    if argv[:1] == ["score"]:
        s = score_calls()
        print({k: v for k, v in s.items() if k != "table"})
    else:
        print(refresh())
        print(format_active())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
