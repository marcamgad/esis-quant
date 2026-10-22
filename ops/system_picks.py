"""
System picks: the rule that turns the watchlist into BUY / TRIM / SELL / HOLD.

RULE v2 (fixed 2026-10-17; replaces v1 of the same day, which had recorded
signals but none scored). Any change becomes v3 and its score starts over.
A long-term rule: own profitable, growing, sensibly financed, reasonably
priced businesses; size them so fees stay small and no single name or sector
dominates; sell when the business breaks, not when the price wobbles.

  BUY   a watchlist name (fresh trigger, last 7 days) passing ALL gates:
          G1 latest quarter (<= 200 days old) profitable
          G2 it grew y/y, and not only "from a small base"
          G3 debt/equity <= 1.5 (financials exempt)
          G4 operating cash flow positive (latest annual)
          G5 median value traded >= 1,000,000 EGP/day
          G6 no split / rights issue / merger in 45 days
          G7 positive equity
          G8 P/E between 0 and 15 (earnings yield >= 6.7%)
          G9 P/B <= 4 (financials: <= 2.5)
        ranked by watchlist score, then profit growth. SIZE = smallest whole
        number of shares with an all-in round trip <= 3%. The BUY must also fit
        the BOOK: after it, the name is <= 20% and its sector <= 30% of the account,
        and there is cash for it (available cash + proceeds of today's SELLs).
        A BUY that passes every gate but has no cash is listed as WAIT.
  SELL  a held name whose last 12 months of net income is a LOSS, or whose last
        TWO quarters were both losses, or with negative equity, or debt/equity
        > 3 (non-financials). One bad quarter alone is not a sell (that was v1).
  TRIM  a held name above 30% of the account: sell down to 20%, if the sale is
        large enough to be fee-efficient.
  HOLD  everything else held.

Every BUY / SELL / TRIM is written to system_signals with price and the EGX30
level; scorecard() scores the current rule's BUYs at 20 / 60 / 250 sessions.

    py -3.11 -m ops.system_picks          # today's signals (and record them)
    py -3.11 -m ops.system_picks score
"""

import json
import math
import re
import sys

import numpy as np
import pandas as pd

from data.database import get_connection
from ops import agent_guard as ag
from ops import watchlist as w

RULE = "v2-2026-10-17"
MAX_BUYS = 3
MAX_DE, SELL_DE = 1.5, 3.0
MIN_VALUE = 1_000_000
CAPITAL_CHANGE_DAYS = 45
REPEAT_DAYS = 60
MAX_PE, MAX_PB, MAX_PB_FIN = 15.0, 4.0, 2.5
NAME_CAP, SECTOR_CAP = 0.20, 0.30            # same limits the challenger enforces
TRIM_ABOVE = 0.30
SELL_COST = 0.0072 + 0.0057                  # ~fees + slippage on a sale, for proceeds


# --------------------------------------------------------------------------
# valuation (stockanalysis statistics page: computed in one currency by the vendor;
# raw statements are not, because several EGX companies report in US dollars)
# --------------------------------------------------------------------------
def valuation(ticker: str) -> dict:
    w.ensure_tables()
    today = str(pd.Timestamp.now().date())
    with get_connection() as conn:
        r = conn.execute("SELECT pe, pb, div_yield, earnings_yield FROM valuation_cache WHERE ticker = ? "
                         "AND fetched_on = ?", (ticker, today)).fetchone()
    if r:
        return {"pe": r[0], "pb": r[1], "div_yield": r[2], "earnings_yield": r[3]}
    from data.fundamentals import _fetch
    try:
        txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ",
                                         _fetch(f"https://stockanalysis.com/quote/egx/{ticker}/statistics/")))
    except Exception:  # noqa: BLE001
        return {}

    def num(label):
        m = re.search(re.escape(label) + r"\s+(-?[\d,.]+)(%?)", txt)
        if not m:
            return None
        v = float(m.group(1).replace(",", ""))
        return v / 100 if m.group(2) else v
    out = {"pe": num("PE Ratio"), "pb": num("PB Ratio"), "div_yield": num("Dividend Yield"),
           "earnings_yield": num("Earnings Yield")}
    con = w._writer()
    try:
        ag.run(con, "INSERT OR REPLACE INTO valuation_cache VALUES (?,?,?,?,?,?)",
               (ticker, today, out["pe"], out["pb"], out["div_yield"], out["earnings_yield"]), w.AGENT)
        con.commit()
    finally:
        con.close()
    return out


# --------------------------------------------------------------------------
# facts and gates
# --------------------------------------------------------------------------
def _quarters(t, d) -> pd.DataFrame:
    from data.fundamentals import as_of
    q = as_of(t, str(d.date()), "income", frequency="quarterly")
    q = q[(q["line_item"] == "Net Income") & (q["fiscal_period"] != "TTM")].dropna(subset=["value"])
    return q.drop_duplicates("period_end").sort_values("period_end", ascending=False)


def _annual_ocf(t, d):
    from data.fundamentals import as_of
    cf = as_of(t, str(d.date()), "cashflow", frequency="annual")
    cf = cf[cf["line_item"] == "Operating Cash Flow"].dropna(subset=["value"])
    return None if cf.empty else float(cf.sort_values("period_end").iloc[-1]["value"])


def _facts(t, sector, px, d, with_valuation=True) -> dict:
    from data.fundamentals import leverage_as_of, quarterly_yoy
    q = quarterly_yoy(t, str(d.date()))
    fresh = bool(q) and (d - pd.Timestamp(q["period_end"])).days <= w.MAX_QUARTER_AGE_DAYS
    qs = _quarters(t, d)
    last4 = qs.head(4)
    span_ok = len(last4) == 4 and (pd.Timestamp(last4["period_end"].iloc[0]) -
                                   pd.Timestamp(last4["period_end"].iloc[-1])).days <= 300
    lev = leverage_as_of(t, str(d.date()))
    v = px["value"][t].dropna().tail(60) if t in px["value"] else pd.Series(dtype=float)
    with get_connection() as conn:
        ca = conn.execute("SELECT MAX(date) FROM corporate_actions WHERE ticker = ? AND date > ? AND date <= ? "
                          "AND action_type IN ('SPLIT_CORROBORATED','SPLIT')",
                          (t, str((d - pd.Timedelta(days=CAPITAL_CHANGE_DAYS)).date()), str(d.date()))).fetchone()[0]
    return {"q": q if fresh else {}, "growth": w._growth(t, d), "de": lev.get("debt_to_equity"),
            "financial": sector in w.FINANCIAL_SECTORS, "negative_equity": bool(lev.get("negative_equity")),
            "ocf": _annual_ocf(t, d), "value": float(v.median()) if len(v) >= 20 else 0.0, "capital_change": ca,
            "ttm": float(last4["value"].sum()) if span_ok else None,
            "last2": [float(x) for x in qs["value"].head(2)], "last2_periods": list(qs["fiscal_period"].head(2)),
            "val": valuation(t) if with_valuation else {}}


def _gates(f) -> list:
    """[(gate, passed, detail)] for the BUY rule."""
    q, g, val = f["q"], f["growth"], f.get("val") or {}
    pe, pb = val.get("pe"), val.get("pb")
    pb_max = MAX_PB_FIN if f["financial"] else MAX_PB
    return [
        ("G1 profitable latest quarter", bool(q) and q["value"] > 0,
         f"{q['period']} net income {q['value']:,.1f}" if q else "no quarter in the last 200 days"),
        ("G2 real growth", bool(g) and 0 < g["change"] <= w.SMALL_BASE, g["label"] if g else "no usable y/y comparison"),
        ("G3 leverage", f["financial"] or (f["de"] is not None and f["de"] <= MAX_DE),
         "financial company (exempt)" if f["financial"] else
         (f"debt/equity {f['de']:.2f}" if f["de"] is not None else "debt/equity unknown")),
        # a bank's operating cash flow swings with deposits and loans, not with its health
        ("G4 cash flow", f["financial"] or (f["ocf"] is not None and f["ocf"] > 0),
         "financial company (exempt)" if f["financial"] else
         (f"operating cash flow {f['ocf']:,.1f}" if f["ocf"] is not None else "no annual cash flow")),
        ("G5 liquidity", f["value"] >= MIN_VALUE, f"median value traded {f['value'] / 1e6:,.1f}M EGP/day"),
        ("G6 no recent capital change", f["capital_change"] is None,
         f"capital change on {f['capital_change']}" if f["capital_change"] else "none in 45 days"),
        ("G7 positive equity", not f["negative_equity"], "equity <= 0" if f["negative_equity"] else "ok"),
        ("G8 price vs earnings", pe is not None and 0 < pe <= MAX_PE,
         f"P/E {pe:.1f} (earnings yield {1 / pe:.1%})" if pe and pe > 0 else "no positive P/E"),
        ("G9 price vs book", pb is not None and pb <= pb_max,
         f"P/B {pb:.2f} (limit {pb_max})" if pb is not None else "no P/B"),
    ]


def _sell_reasons(f) -> list:
    why = []
    if f["ttm"] is not None and f["ttm"] < 0:
        why.append(f"last 12 months net income is a loss ({f['ttm']:,.1f})")
    if len(f["last2"]) == 2 and all(x < 0 for x in f["last2"]):
        why.append(f"two loss quarters in a row ({f['last2_periods'][1]} {f['last2'][1]:,.1f}, "
                   f"{f['last2_periods'][0]} {f['last2'][0]:,.1f})")
    if f["negative_equity"]:
        why.append("equity turned negative")
    if not f["financial"] and f["de"] is not None and f["de"] > SELL_DE:
        why.append(f"debt/equity {f['de']:.2f} > {SELL_DE}")
    return why


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------
def signals(as_of=None) -> dict:
    from ops.portfolio_tracker import available_cash, holdings
    d = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    u = w._universe(d)
    wl = w.active()
    pos, _ = holdings()
    cash = float(available_cash()["available"])
    held = set(pos["ticker"]) if not pos.empty else set()
    with get_connection() as conn:
        sectors = dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall())
    names = sorted(set(wl["ticker"]) | held)
    px = w._prices(names, d, days=120)
    total = float(pos["value"].sum() if not pos.empty else 0.0) + cash
    value = pos.set_index("ticker")["value"].to_dict() if not pos.empty else {}

    # held names: SELL / TRIM / HOLD
    sells, trims, holds = [], [], []
    for h in (pos.itertuples() if not pos.empty else []):
        f = _facts(h.ticker, sectors.get(h.ticker), px, d, with_valuation=False)
        why = _sell_reasons(f)
        if why:
            sells.append({"ticker": h.ticker, "shares": int(h.shares), "price": float(h.price), "why": why})
            continue
        wt = h.value / total if total else 0
        if wt > TRIM_ABOVE:
            excess = h.value - NAME_CAP * total
            n = math.ceil(excess / h.price)
            m = w.min_position(h.ticker)
            if m.get("possible") and n * h.price >= m["value_egp"]:
                trims.append({"ticker": h.ticker, "shares": n, "price": float(h.price),
                              "why": [f"{wt:.0%} of the account; selling {n:,} shares brings it to "
                                      f"{(h.value - n * h.price) / total:.0%} (cap {NAME_CAP:.0%})"]})
                continue
        holds.append({"ticker": h.ticker, "weight": wt,
                      "note": f"ttm net income {f['ttm']:,.1f}" if f["ttm"] is not None else "ttm unknown"})

    # cash after today's sells/trims (they settle T+2: a BUY funded by them waits for settlement)
    proceeds = sum(x["shares"] * x["price"] * (1 - SELL_COST) for x in sells + trims)
    funds = cash + proceeds
    for x in sells:
        value[x["ticker"]] = 0.0
    for x in trims:
        value[x["ticker"]] = value.get(x["ticker"], 0.0) - x["shares"] * x["price"]

    def sector_value(sec):
        return sum(v for t, v in value.items() if sectors.get(t) == sec)

    # BUY candidates: every gate, then the book
    cands, rejected = [], []
    for r in wl.itertuples():
        if r.ticker in held or r.ticker not in u.index:
            continue
        f = _facts(r.ticker, sectors.get(r.ticker), px, d)
        gates = _gates(f)
        if all(ok for _, ok, _ in gates):
            cands.append({"ticker": r.ticker, "score": r.score, "growth": f["growth"]["change"],
                          "triggers": [x[1] for x in r.reasons], "gates": gates, "val": f["val"]})
        else:
            rejected.append((r.ticker, [f"{g}: {detail}" for g, ok, detail in gates if not ok]))
    cands.sort(key=lambda b: (-b["score"], -b["growth"]))
    buys, waits, skipped = [], [], []
    for b in cands:
        if len(buys) >= MAX_BUYS:
            break
        m = w.min_position(b["ticker"])
        if not m.get("possible"):
            continue
        b["size"] = m
        sec = sectors.get(b["ticker"])
        name_w = m["value_egp"] / total if total else 1
        sec_w = (sector_value(sec) + m["value_egp"]) / total if total else 1
        if name_w > NAME_CAP:
            skipped.append((b["ticker"], f"even the minimum size would be {name_w:.0%} of the account"))
            continue
        if sec_w > SECTOR_CAP:
            skipped.append((b["ticker"], f"sector {sec} would reach {sec_w:.0%} (cap {SECTOR_CAP:.0%})"))
            continue
        if funds >= m["value_egp"]:
            b["funded_by"] = "cash" if cash >= m["value_egp"] else "proceeds of today's sells (after T+2)"
            buys.append(b)
            funds -= m["value_egp"]
            cash = max(0.0, cash - m["value_egp"])
            value[b["ticker"]] = m["value_egp"]
        else:
            b["needs"] = m["value_egp"] - funds
            waits.append(b)
    return {"as_of": str(d.date()), "rule": RULE, "account": total, "cash": float(available_cash()["available"]),
            "buy": buys, "wait": waits, "skipped": skipped, "rejected": rejected, "sell": sells, "trim": trims, "hold": holds}


def record(sig: dict) -> list:
    """Write today's BUY / SELL / TRIM; at most one per stock and action per REPEAT_DAYS."""
    from data import market
    w.ensure_tables()
    lv = market.level()
    egx = float(lv.iloc[-1]) if len(lv) else None
    con = w._writer()
    written = []
    try:
        since = str((pd.Timestamp(sig["as_of"]) - pd.Timedelta(days=REPEAT_DAYS)).date())
        for action, items in (("BUY", sig["buy"]), ("SELL", sig["sell"]), ("TRIM", sig["trim"])):
            for x in items:
                if con.execute("SELECT 1 FROM system_signals WHERE ticker = ? AND action = ? AND rule = ? "
                               "AND signal_date > ?", (x["ticker"], action, RULE, since)).fetchone():
                    continue
                size = x.get("size") or {}
                price = size.get("price") or x.get("price")
                shares = size.get("shares") or x.get("shares")
                ag.run(con, "INSERT INTO system_signals (signal_date, ticker, action, shares, price, value_egp, egx30, "
                            "rule, reasons, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (sig["as_of"], x["ticker"], action, shares, price, (shares or 0) * (price or 0), egx, RULE,
                        json.dumps(x.get("triggers") or x.get("why"), ensure_ascii=False), w._now()), w.AGENT)
                written.append((action, x["ticker"]))
        con.commit()
    finally:
        con.close()
    return written


def scorecard(rule: str = RULE) -> dict:
    """The rule's BUY signals vs EGX30, cash and the average stock."""
    from backtest.engine import load_price_returns
    from data import market
    from data.rates import cash_rate_series
    w.ensure_tables()
    with get_connection() as conn:
        s = pd.read_sql("SELECT * FROM system_signals WHERE action = 'BUY' AND rule = ? ORDER BY signal_date", conn,
                        params=(rule,), parse_dates=["signal_date"])
    if s.empty:
        return {"signals": 0, "rule": rule}
    r = load_price_returns()
    r = r[r.index >= s["signal_date"].min() - pd.Timedelta(days=5)]
    ew = r.clip(-0.5, 1.0).mean(axis=1)
    egx = market.returns(r.index).fillna(0)
    cash = (1 + cash_rate_series(r.index, haircut=0.0)) ** (1 / 245) - 1
    out = {"signals": len(s), "rule": rule}
    for h in w.HORIZONS:
        rows = []
        for x in s.itertuples():
            i0 = r.index.searchsorted(x.signal_date, side="right")
            if i0 + h > len(r.index) or x.ticker not in r:
                continue
            sl = slice(i0, i0 + h)
            st = float(np.prod(1 + r[x.ticker].iloc[sl].fillna(0)) - 1)
            rows.append({"vs_egx30": st - float(np.prod(1 + egx.iloc[sl]) - 1),
                         "vs_avg": st - float(np.prod(1 + ew.iloc[sl].fillna(0)) - 1),
                         "vs_cash": st - float(np.prod(1 + cash.iloc[sl]) - 1)})
        if rows:
            dd = pd.DataFrame(rows)
            out[h] = {"n": len(dd), **{k: float(dd[k].mean()) for k in dd},
                      "hit_vs_egx30": float((dd["vs_egx30"] > 0).mean())}
    return out


def format_signals(sig: dict) -> str:
    L = [f"SYSTEM SIGNALS {sig['as_of']} — rule {sig['rule']} (long-term) | account {sig['account']:,.0f} EGP, "
         f"cash {sig['cash']:,.0f}"]
    for x in sig["sell"]:
        L.append(f"\nSELL {x['ticker']}: all {x['shares']:,} shares (~{x['shares'] * x['price']:,.0f} EGP) — "
                 + "; ".join(x["why"]))
    for x in sig["trim"]:
        L.append(f"\nTRIM {x['ticker']}: sell {x['shares']:,} shares (~{x['shares'] * x['price']:,.0f} EGP) — "
                 + "; ".join(x["why"]))
    for b in sig["buy"]:
        m = b["size"]
        L.append(f"\nBUY {b['ticker']}: {m['shares']:,} shares at ~{m['price']:,.2f} = {m['value_egp']:,.0f} EGP "
                 f"(round trip {m['round_trip_pct']:.2%}; funded by {b['funded_by']})")
        for t in b["triggers"][:2]:
            L.append(f"   why now: {t}")
        L.append("   gates: " + " | ".join(detail for _, _, detail in b["gates"]))
    if not sig["buy"]:
        L.append("\nBUY: none today.")
    for b in sig["wait"]:
        m = b["size"]
        L.append(f"WAIT {b['ticker']}: passes every gate; {m['shares']:,} shares (~{m['value_egp']:,.0f} EGP) "
                 f"needs {b['needs']:,.0f} EGP more cash")
    for t, why in sig["skipped"]:
        L.append(f"SKIPPED {t}: {why}")
    if sig.get("rejected"):
        L.append("\nWatchlist names that failed a gate:")
        for t, fails in sig["rejected"]:
            L.append(f"   {t}: " + " | ".join(fails))
    if sig["hold"]:
        L.append("\nHOLD: " + ", ".join(f"{x['ticker']} ({x['weight']:.0%}, {x['note']})" for x in sig["hold"]))
    L.append("\nRule v2 is being scored from today against the EGX30 and cash (py -3.11 -m ops.system_picks score).")
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
        print(scorecard())
        return 0
    sig = signals()
    print(format_signals(sig))
    print("\nrecorded:", record(sig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
