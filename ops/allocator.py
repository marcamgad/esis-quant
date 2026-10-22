"""
Agent 6 -- ALLOCATOR. How much of the account belongs in EGX stocks at all.

THE QUESTION IT ANSWERS. Not WHICH stocks -- selection has no evidenced edge
here -- but how big the stock sleeve should be when cash pays a large, riskless
rate. In Sep 2026 Egyptian money-market funds pay ~19.5% with inflation at
14.5%: +4% real for no stock risk. The business case found the strategy loses
to bills whenever bills yield above ~19%. That finding never reached a
decision; this module turns it into one number.

THE RULE (deterministic, every input printed):
    excess   = historical equity return over cash, measured on this system's own
               data, SHRUNK 50% toward zero (survivorship: today's listings only;
               and a 10-year sample is short)
    drag     = what YOUR trading costs a year (fees paid, annualised from the ledger)
    sigma    = volatility of the book you actually hold (or the system's book)
    stocks % = clip( 0.5 * (excess - drag) / sigma^2 , 0, 100% )      [half-Kelly]

Half-Kelly because the excess estimate is uncertain; Kelly sizing on an
over-estimated edge is the standard way to lose money slowly. The rest sits in
a money-market fund (daily liquidity, no stock risk).

It never names a stock. It is the system's arithmetic, not personal advice.

    py -3.11 -m ops.allocator
"""

import logging
import sys

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.allocator")

AGENT = "allocator"
WRITABLE = {"allocation_log", "agent_findings"}

# Current market inputs, with source and date. The Sentinel flags them stale
# after INPUT_MAX_AGE_DAYS -- an allocator on a 6-month-old cash rate is wrong.
INPUTS = {
    "cash_yield": (0.195, "2026-10-03", "money-market funds ~19.4-22% 1y (Banque Misr 19.54% to Jul-2026; "
                                        "Credit Agricole MMF 3); CBE deposit 19.0% held 26 Sep 2026"),
    "inflation": (0.145, "2026-09-26", "urban headline, Aug 2026 (CBE MPC statement)"),
}
INPUT_MAX_AGE_DAYS = 60
SHRINK = 0.5
KELLY_FRACTION = 0.5


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS allocation_log (
        computed_at TEXT, scenario TEXT, excess REAL, drag REAL, sigma REAL,
        stock_share REAL, cash_yield REAL, inflation REAL, note TEXT)""")
    from ops.agent_guard import ensure_tables
    ensure_tables(conn)


def historical_excess(start: str = "2016-01-01") -> dict:
    """The EGX30 index (price only) vs the CBE deposit rate.

    Until 2026-10-04 this used the equal-weight mean of the system's universe,
    which returned 33% a year since 2016 against the real EGX30's ~22%: daily
    rebalancing of thinly traded names books bid-ask bounce as return, and the
    universe holds only companies that survived. That overstated the equity
    premium (+17%/yr vs ~+6%) and so the stock share. The equal-weight figure
    is kept in the output for comparison only. The EGX30 here excludes
    dividends (~3-5%/yr), which makes it conservative, not generous."""
    from backtest.engine import load_price_returns
    from data import market
    from data.rates import cash_rate_series
    lv = market.level()
    lv = lv[lv.index >= start]
    yrs = (lv.index[-1] - lv.index[0]).days / 365.25
    eq_cagr = float((lv.iloc[-1] / lv.iloc[0]) ** (1 / yrs) - 1)
    mret = lv.pct_change().dropna()
    cash = (1 + cash_rate_series(mret.index, haircut=0.0)) ** (1 / 245) - 1
    cash_cagr = float((1 + cash).prod() ** (1 / yrs) - 1)
    r = load_price_returns()
    r = r[r.index >= start]
    n = r.notna().sum(axis=1)
    ew = r.where(r.abs() < 0.25).mean(axis=1)[n >= 30].fillna(0)
    ew_cagr = float((1 + ew).prod() ** (245 / len(ew)) - 1)
    return {"equity_cagr": eq_cagr, "cash_cagr": cash_cagr, "excess": eq_cagr - cash_cagr,
            "equity_vol": float(mret.std() * np.sqrt(245)), "years": yrs,
            "benchmark": "EGX30 (price)", "equal_weight_cagr_for_reference": ew_cagr}


def book_sigma(lookback: int = 180) -> float:
    """Volatility of the book actually held (risk-weighted by current values)."""
    from backtest.engine import load_price_returns
    from ops.portfolio_tracker import holdings
    pos, cash = holdings()
    if pos.empty:
        return float("nan")
    r = load_price_returns(pos["ticker"].tolist()).tail(lookback).fillna(0)
    w = (pos.set_index("ticker")["value"] / (pos["value"].sum() + cash)).reindex(r.columns).fillna(0)
    return float((r @ w).std() * np.sqrt(245))


def system_sigma(lookback: int = 180) -> float:
    """Volatility of the system's own target book (min-variance)."""
    from backtest.engine import load_price_returns
    from ops.trade_list import latest_trade_list
    tl = latest_trade_list()
    tl = tl[tl["ticker"] != "__CASH__"]
    if tl.empty:
        return float("nan")
    w = tl.set_index("ticker")["weight"]
    r = load_price_returns(w.index.tolist()).tail(lookback).fillna(0)
    return float((r @ w.reindex(r.columns).fillna(0)).std() * np.sqrt(245))


def cost_drag(days: int = 30) -> dict:
    """Annualised fees from the ledger over the last `days`, as a share of the account."""
    from ops.portfolio_tracker import holdings
    pos, cash = holdings()
    value = float(pos["value"].sum() + cash) if not pos.empty else cash
    since = (pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat()
    from ops.portfolio_tracker import _ensure_fee_column
    with get_connection() as conn:
        _ensure_fee_column(conn)
        led = pd.read_sql("SELECT traded_at, shares, price, fee, price_source FROM user_trade_ledger "
                          "WHERE traded_at >= ? AND price_source != 'correction'", conn, params=(since,))
    if led.empty or value <= 0:
        return {"fees": 0.0, "turnover": 0.0, "drag": 0.0, "orders": 0}
    notional = float((led["shares"] * led["price"]).sum())
    fees = float(led["fee"].fillna(0).sum())
    est = led["fee"].fillna(0) == 0                     # orders booked without a fee: estimate it
    fees += float((4.95 + 0.00162 * (led.loc[est, "shares"] * led.loc[est, "price"])).sum())
    scale = 365.0 / days
    return {"fees": fees, "orders": len(led), "turnover": notional / value * scale,
            "drag": fees / value * scale}


def allocate(excess: float, drag: float, sigma: float) -> float:
    if not (sigma == sigma) or sigma <= 0:
        return float("nan")
    return float(np.clip(KELLY_FRACTION * (excess - drag) / sigma ** 2, 0.0, 1.0))


def run(extra_drag_override: float = None) -> dict:
    h = historical_excess()
    ex = SHRINK * h["excess"]
    cash_y, inf = INPUTS["cash_yield"][0], INPUTS["inflation"][0]
    cd = cost_drag()
    s_book, s_sys = book_sigma(), system_sigma()
    # the system's book trades ~monthly; its drag is the measured target turnover cost (~2%/yr)
    scen = [
        ("system book, low turnover", ex, 0.02, s_sys),
        ("your book, low turnover", ex, 0.02, s_book),
        ("your book, your current trading pace", ex,
         cd["drag"] if extra_drag_override is None else extra_drag_override, s_book),
    ]
    rows = [{"scenario": n, "excess": e, "drag": d, "sigma": s, "stock_share": allocate(e, d, s)}
            for n, e, d, s in scen]
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    from ops import agent_guard as ag
    findings = []
    for k, (v, d, src) in INPUTS.items():
        age = (pd.Timestamp.now() - pd.Timestamp(d)).days
        if age > INPUT_MAX_AGE_DAYS:
            findings.append({"severity": "medium", "kind": "stale_input", "subject": k,
                             "summary": f"allocator input {k}={v:.1%} is {age} days old ({src})",
                             "fix_hint": "update INPUTS in ops/allocator.py"})
    if cd["drag"] > ex:
        findings.append({"severity": "high", "kind": "costs_exceed_edge", "subject": "trading pace",
                         "summary": f"your trading costs ~{cd['drag']:.0%}/yr exceed the expected stock "
                                    f"excess over cash ({ex:.0%}/yr)",
                         "fix_hint": "fewer, larger trades; see ops/allocator.py"})
    with get_connection() as raw:
        _ensure(raw)
        conn = ag.guarded(raw, WRITABLE, AGENT)
        conn.executemany("INSERT INTO allocation_log VALUES (?,?,?,?,?,?,?,?,?)",
                         [(now, r["scenario"], r["excess"], r["drag"], r["sigma"], r["stock_share"],
                           cash_y, inf, None) for r in rows])
        ag.record_findings(conn, AGENT, findings)
    return {"history": h, "shrunk_excess": ex, "cash_yield": cash_y, "inflation": inf,
            "real_cash": (1 + cash_y) / (1 + inf) - 1, "costs": cd, "scenarios": rows}


def format_report(res: dict) -> str:
    h, c = res["history"], res["costs"]
    L = ["=" * 70, "ALLOCATOR — how much belongs in stocks (never which stocks)", "=" * 70,
         f"  cash (money-market) {res['cash_yield']:.1%} | inflation {res['inflation']:.1%} | "
         f"real cash {res['real_cash']:+.1%}",
         f"  EGX equal-weight {h['years']:.0f}y: {h['equity_cagr']:.1%}/yr vs cash {h['cash_cagr']:.1%}/yr "
         f"-> excess {h['excess']:+.1%}, used at half: {res['shrunk_excess']:+.1%}",
         f"  your last 30 days: {c['orders']} orders, fees ~{c['fees']:,.0f} EGP, turnover "
         f"{c['turnover']:.0f}x/yr -> cost drag {c['drag']:.0%}/yr", ""]
    for r in res["scenarios"]:
        share = "n/a" if r["stock_share"] != r["stock_share"] else f"{r['stock_share']:.0%}"
        L.append(f"  {r['scenario']:40} edge {r['excess']-r['drag']:+6.1%}  vol {r['sigma']:5.0%}  "
                 f"-> stocks {share:>5}")
    L += ["", "  Half-Kelly sizing on a shrunk edge. The system's arithmetic, not personal advice."]
    return "\n".join(L)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.WARNING)
    print(format_report(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
