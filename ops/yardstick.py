"""
The cash yardstick: your book against doing nothing.

WHY. Seven ways to beat the market were tested here (selection, momentum,
regime timing, earnings drift, index additions, ...). None survived
out-of-sample. Meanwhile an Egyptian money-market fund pays ~19.5% a year with
daily liquidity, against ~14.5% inflation. So the honest benchmark for every
pound in the account is not the EGX30, it is that fund: a trade only adds value
if it beats what the same money would have earned sitting still.

THREE NUMBERS:
  1. RACE (exact, forward). From the date invested capital was last set, the
     money put in compounds at the cash yield. Book value minus that = what
     trading has added or cost so far. Deposits/withdrawals recorded with
     `portfolio_tracker deposit/withdraw` enter both sides on their dates.
  2. RACE SINCE THE FIRST TRADE (approximate). Deposits before 2026-10-03 were
     never recorded, so their dates are ESTIMATED from jumps in the daily totals
     (ESTIMATED_DEPOSITS). Shown with that label, never as a fact.
  3. HURDLE. What a single trade must earn just to tie with cash: the cash yield
     over the holding period plus the round-trip fee and slippage.
Real (after-inflation) versions of both sides are shown too.

Not advice; arithmetic on the account's own numbers.
"""

import logging

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.yardstick")

SLIPPAGE = 0.0057            # per side, measured on the live account (see backtest/engine.py)

# Before deposits were recorded (2026-10-03): first total seen, then the jump on
# 3 Sep 2026 (7,599 -> 10,841 with no trade large enough to explain it). The
# remainder of the 12,432 put in is placed there. Approximate by construction.
ESTIMATED_DEPOSITS = [("2026-08-05", 7553.06), ("2026-09-03", 12432.0 - 7553.06)]


def _inputs():
    from ops.allocator import INPUTS
    return INPUTS["cash_yield"][0], INPUTS["inflation"][0], INPUTS["cash_yield"][1]


def _grow(amount: float, start, end, rate: float) -> float:
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).days, 0)
    return amount * (1 + rate) ** (days / 365.0)


def book_value():
    """Broker total typed in today/yesterday if there is one, else the system's marks."""
    from ops.portfolio_tracker import holdings
    with get_connection() as conn:
        r = conn.execute("SELECT date, user_value FROM user_daily_snapshot WHERE user_value > 0 "
                         "ORDER BY date DESC LIMIT 1").fetchone()
    if r and (pd.Timestamp.now().normalize() - pd.Timestamp(r["date"])).days <= 3:
        return float(r["user_value"]), f"your broker total {r['date']}"
    pos, cash = holdings()
    v = float(pos["value"].sum() + cash) if not pos.empty else float(cash)
    return v, "system marks"


def _flows_after(as_of):
    with get_connection() as conn:
        try:
            f = pd.read_sql("SELECT date, amount FROM cash_flows WHERE recorded_at > ?", conn, params=(as_of,))
        except Exception:  # noqa: BLE001
            return []
    return list(f.itertuples(index=False))


def race(as_of=None) -> dict:
    """Book vs the same money in the cash fund, from the invested-capital date."""
    y, inf, y_date = _inputs()
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    with get_connection() as conn:
        r = conn.execute("SELECT as_of, invested FROM account_capital ORDER BY as_of DESC LIMIT 1").fetchone()
    if not r:
        return {}
    start, invested = pd.Timestamp(r["as_of"]), float(r["invested"])
    cash_alt = _grow(invested, start, now, y)
    put_in = invested
    for f in _flows_after(r["as_of"]):
        cash_alt += _grow(float(f.amount), f.date, now, y)
        put_in += float(f.amount)
    value, src = book_value()
    days = max((now - start).days, 0)
    infl = (1 + inf) ** (days / 365.0)
    return {"since": str(start.date()), "days": days, "put_in": put_in, "book": value, "book_source": src,
            "cash_alt": cash_alt, "trading_added": value - cash_alt,
            "book_real": value / infl - put_in, "cash_real": cash_alt / infl - put_in,
            "cash_yield": y, "inflation": inf, "yield_as_of": y_date}


def race_since_first_trade(as_of=None) -> dict:
    y, inf, _ = _inputs()
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    cash_alt = sum(_grow(a, d, now, y) for d, a in ESTIMATED_DEPOSITS)
    put_in = sum(a for _, a in ESTIMATED_DEPOSITS)
    value, src = book_value()
    return {"since": ESTIMATED_DEPOSITS[0][0], "put_in": put_in, "book": value, "cash_alt": cash_alt,
            "trading_added": value - cash_alt, "estimated": True}


def race_vs_egx30(as_of=None) -> dict:
    """The same (estimated) deposits put into the EGX30 instead: did picking
    stocks beat simply owning the index? Price index, so dividends are missing
    from the index side -- it is slightly flattering to your picks."""
    from data import market
    now = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    alt = 0.0
    for d, a in ESTIMATED_DEPOSITS:
        r = market.window_return(d, now)
        if not pd.notna(r):
            return {}
        alt += a * (1 + r)
    value, _ = book_value()
    return {"egx30_alt": alt, "book": value, "picks_added": value - alt}


def hurdle(position_egp: float = 2000.0, hold_sessions: int = 20) -> dict:
    """Return a trade needs just to tie with leaving the money in the cash fund."""
    from backtest.fees import round_trip_cost
    y, _, _ = _inputs()
    cash_part = (1 + y) ** (hold_sessions / 245.0) - 1          # ~245 EGX sessions a year
    fee_part = round_trip_cost(position_egp)["total_egp"] / position_egp
    return {"position": position_egp, "sessions": hold_sessions, "cash": cash_part,
            "fees": fee_part, "slippage": 2 * SLIPPAGE, "total": cash_part + fee_part + 2 * SLIPPAGE}


def format_section() -> str:
    L = ["\n[2c] VS DOING NOTHING  (your money in a cash fund instead)"]
    r = race()
    if not r:
        L.append("    invested capital not set — 'portfolio_tracker capital <amount>' to start the race.")
        return "\n".join(L)
    L.append(f"    cash fund {r['cash_yield']:.1%}/yr (as of {r['yield_as_of']}), inflation {r['inflation']:.1%}")
    L.append(f"    since {r['since']} ({r['days']} days, {r['put_in']:,.0f} EGP put in):")
    L.append(f"      your book {r['book']:,.0f}  ({r['book_source']})")
    L.append(f"      same money in the fund {r['cash_alt']:,.0f}")
    word = "ADDED" if r["trading_added"] >= 0 else "COST"
    L.append(f"      -> trading has {word} {abs(r['trading_added']):,.0f} EGP vs doing nothing")
    L.append(f"      after inflation: book {r['book_real']:+,.0f} EGP, fund {r['cash_real']:+,.0f} EGP")
    s = race_since_first_trade()
    L.append(f"    since the first trade ({s['since']}, ESTIMATED deposit dates): book {s['book']:,.0f} "
             f"vs fund {s['cash_alt']:,.0f} -> {s['trading_added']:+,.0f} EGP")
    m = race_vs_egx30()
    if m:
        L.append(f"    same deposits in an EGX30 index fund (price only, no dividends): {m['egx30_alt']:,.0f} "
                 f"-> your picks {m['picks_added']:+,.0f} EGP vs simply owning the index")
    for p in (1000, 2000, 5000):
        h = hurdle(p, 20)
        L.append(f"    a {p:,} EGP trade held 20 sessions must make {h['total']:+.2%} just to tie "
                 f"(cash {h['cash']:.2%} + fees {h['fees']:.2%} + slippage {h['slippage']:.2%})")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(format_section())
