"""
High-risk rule: bigger swings, looser quality bars, strict exits.

WHAT THE DATA SAYS FIRST (EGX 2016-2026, measured 2026-10-17). Sorting stocks by
volatility into fifths, next-month returns (annualised) were 19% / 36% / 42% /
39% / 36% from the calmest to the wildest. The calmest fifth lagged, but the
WILDEST fifth earned less than the middle: past a point, more risk did not pay
more here, and those figures are flattered by bid-ask bounce and survivorship
in exactly the most volatile small stocks. So this rule is a different BET,
not a better one, and it is scored separately from the careful rule
(ops/system_picks.py) so the two can be compared on the record.

RULE aggr-v1 (fixed 2026-10-17; any change becomes v2).
  BUY   a watchlist name (fresh trigger, last 7 days) that passes:
          A1 latest quarter (<= 200 days old) profitable
          A2 grew y/y -- growth from a small base IS allowed here
          A3 debt/equity <= 3 (financials exempt)
          A4 median value traded >= 300,000 EGP/day
          A5 no split / rights issue / merger in 45 days
          A6 positive equity
          A7 P/E between 0 and 40
          A8 HIGH RISK: 4-week volatility >= 12% (ops/forecast.py)
        ranked by watchlist score, then by 4-week volatility. Up to 2 a day.
        SIZE: the minimum fee-efficient position (round trip <= 3%), and no name
        above 10% of the account; the high-risk names together <= 30%.
  EXIT  for a stock bought on an aggr-v1 signal (price recorded at the signal):
          STOP   -20% from the signal price
          TARGET +40% from the signal price
          TIME   60 sessions after the signal
        whichever comes first. The careful rule's SELL tests apply as well.

Every signal is recorded in system_signals under its own rule name.

    py -3.11 -m ops.high_risk_picks
"""

import sys

import pandas as pd

from data.database import get_connection
from ops import system_picks as sp
from ops import watchlist as w

RULE = "aggr-v1-2026-10-17"
MAX_BUYS = 2
MAX_DE = 3.0
MIN_VALUE = 300_000
MAX_PE = 40.0
MIN_MONTH_VOL = 0.12
NAME_CAP, SLEEVE_CAP = 0.10, 0.30
STOP, TARGET, TIME_SESSIONS = -0.20, 0.40, 60


def _gates(f, month_vol) -> list:
    q, g, val = f["q"], f["growth"], f.get("val") or {}
    pe = val.get("pe")
    return [
        ("A1 profitable latest quarter", bool(q) and q["value"] > 0,
         f"{q['period']} net income {q['value']:,.1f}" if q else "no quarter in the last 200 days"),
        ("A2 growing", bool(g) and g["change"] > 0, g["label"] if g else "no usable y/y comparison"),
        ("A3 leverage", f["financial"] or (f["de"] is not None and f["de"] <= MAX_DE),
         "financial (exempt)" if f["financial"] else (f"debt/equity {f['de']:.2f}" if f["de"] is not None
                                                      else "debt/equity unknown")),
        ("A4 liquidity", f["value"] >= MIN_VALUE, f"median value traded {f['value'] / 1e6:,.2f}M EGP/day"),
        ("A5 no recent capital change", f["capital_change"] is None,
         f"capital change on {f['capital_change']}" if f["capital_change"] else "none in 45 days"),
        ("A6 positive equity", not f["negative_equity"], "equity <= 0" if f["negative_equity"] else "ok"),
        ("A7 P/E", pe is not None and 0 < pe <= MAX_PE, f"P/E {pe:.1f}" if pe and pe > 0 else "no positive P/E"),
        ("A8 high risk", month_vol is not None and month_vol >= MIN_MONTH_VOL,
         f"4-week volatility {month_vol:.1%}" if month_vol is not None else "volatility unknown"),
    ]


def _exits(held: pd.DataFrame, as_of: pd.Timestamp) -> list:
    """Stop / target / time for holdings that came from an aggr signal."""
    w.ensure_tables()
    with get_connection() as conn:
        s = pd.read_sql("SELECT ticker, signal_date, price FROM system_signals WHERE rule = ? AND action = 'BUY'",
                        conn, params=(RULE,), parse_dates=["signal_date"])
    if s.empty or held.empty:
        return []
    s = s.sort_values("signal_date").groupby("ticker").last()
    out = []
    for h in held.itertuples():
        if h.ticker not in s.index:
            continue
        x = s.loc[h.ticker]
        ret = h.price / x["price"] - 1
        sessions = len(pd.bdate_range(x["signal_date"], as_of, freq="C", weekmask="Sun Mon Tue Wed Thu")) - 1
        why = []
        if ret <= STOP:
            why.append(f"STOP: {ret:+.0%} since the signal at {x['price']:,.2f}")
        elif ret >= TARGET:
            why.append(f"TARGET: {ret:+.0%} since the signal at {x['price']:,.2f}")
        elif sessions >= TIME_SESSIONS:
            why.append(f"TIME: {sessions} sessions since the signal ({ret:+.0%})")
        if why:
            out.append({"ticker": h.ticker, "shares": int(h.shares), "price": float(h.price), "why": why})
    return out


def signals(as_of=None, careful: dict = None) -> dict:
    from ops.forecast import forecast
    from ops.portfolio_tracker import holdings
    d = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    careful = careful or sp.signals(as_of)
    u = w._universe(d)
    wl = w.active()
    pos, _ = holdings()
    held = set(pos["ticker"]) if not pos.empty else set()
    with get_connection() as conn:
        sectors = dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall())
    total = careful["account"]
    # cash left after the careful rule spends: the careful rule goes first
    funds = careful["cash"] + sum(x["shares"] * x["price"] * (1 - sp.SELL_COST)
                                  for x in careful["sell"] + careful["trim"])
    funds -= sum(b["size"]["value_egp"] for b in careful["buy"])
    taken = {b["ticker"] for b in careful["buy"]}
    exits = _exits(pos, d)
    sleeve_now = 0.0
    with get_connection() as conn:
        aggr_held = {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM system_signals WHERE rule = ? "
                                                "AND action = 'BUY'", (RULE,))} & held
    if not pos.empty:
        sleeve_now = float(pos[pos["ticker"].isin(aggr_held - {e["ticker"] for e in exits})]["value"].sum())
    px = w._prices(sorted(set(wl["ticker"])), d, days=120)
    cands, rejected = [], []
    for r in wl.itertuples():
        if r.ticker in held or r.ticker in taken or r.ticker not in u.index:
            continue
        f = sp._facts(r.ticker, sectors.get(r.ticker), px, d)
        fc = forecast(r.ticker)
        mv = fc.get("month_vol") if fc and "error" not in fc else None
        gates = _gates(f, mv)
        if all(ok for _, ok, _ in gates):
            cands.append({"ticker": r.ticker, "score": r.score, "month_vol": mv, "forecast": fc,
                          "triggers": [x[1] for x in r.reasons], "gates": gates})
        else:
            rejected.append((r.ticker, [f"{g}: {det}" for g, ok, det in gates if not ok]))
    cands.sort(key=lambda b: (-b["score"], -b["month_vol"]))
    buys, waits = [], []
    for b in cands:
        if len(buys) >= MAX_BUYS:
            break
        m = w.min_position(b["ticker"])
        if not m.get("possible"):
            continue
        b["size"] = m
        if m["value_egp"] / total > NAME_CAP:
            rejected.append((b["ticker"], [f"minimum size is {m['value_egp'] / total:.0%} of the account "
                                           f"(cap {NAME_CAP:.0%})"]))
            continue
        if (sleeve_now + m["value_egp"]) / total > SLEEVE_CAP:
            rejected.append((b["ticker"], [f"high-risk names would reach "
                                           f"{(sleeve_now + m['value_egp']) / total:.0%} (cap {SLEEVE_CAP:.0%})"]))
            continue
        if funds >= m["value_egp"]:
            buys.append(b)
            funds -= m["value_egp"]
            sleeve_now += m["value_egp"]
        else:
            b["needs"] = m["value_egp"] - max(funds, 0)
            waits.append(b)
    return {"as_of": str(d.date()), "rule": RULE, "account": total, "buy": buys, "wait": waits,
            "sell": exits, "trim": [], "rejected": rejected}


def format_signals(sig: dict) -> str:
    L = [f"HIGH-RISK SIGNALS {sig['as_of']} — rule {sig['rule']} | exits: stop {STOP:.0%}, "
         f"target {TARGET:+.0%}, {TIME_SESSIONS} sessions"]
    for x in sig["sell"]:
        L.append(f"\nSELL {x['ticker']}: all {x['shares']:,} shares — " + "; ".join(x["why"]))
    for b in sig["buy"]:
        m, f = b["size"], b["forecast"]
        q = f["quantiles"]
        L.append(f"\nBUY (HIGH RISK) {b['ticker']}: {m['shares']:,} shares at ~{m['price']:,.2f} = "
                 f"{m['value_egp']:,.0f} EGP")
        L.append(f"   4-week range (80%): {q['p10']:,.2f} to {q['p90']:,.2f} | chance of +10%: {f['p_up10']:.0%}, "
                 f"of -10%: {f['p_down10']:.0%} | stop at {m['price'] * (1 + STOP):,.2f}, "
                 f"target {m['price'] * (1 + TARGET):,.2f}")
        for t in b["triggers"][:2]:
            L.append(f"   why now: {t}")
        L.append("   gates: " + " | ".join(det for _, _, det in b["gates"]))
    if not sig["buy"]:
        L.append("\nBUY (HIGH RISK): none today.")
    for b in sig["wait"]:
        L.append(f"WAIT {b['ticker']}: passes every gate; needs {b['needs']:,.0f} EGP more cash")
    if sig["rejected"]:
        L.append("\nfailed a gate:")
        for t, fails in sig["rejected"]:
            L.append(f"   {t}: " + " | ".join(fails))
    return "\n".join(L)


def record(sig: dict) -> list:
    """Same ledger as the careful rule, under this rule's own name."""
    original = sp.RULE
    try:
        sp.RULE = RULE                      # record() tags rows with sp.RULE
        return sp.record({**sig, "trim": []})
    finally:
        sp.RULE = original


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    import logging
    logging.disable(logging.WARNING)
    sig = signals()
    print(format_signals(sig))
    print("\nrecorded:", record(sig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
