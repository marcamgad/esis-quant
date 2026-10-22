"""
Agent 5 -- PRE-TRADE CHALLENGER. "The case against this trade", before you place it.

WHY. The money actually traded in this account is discretionary, and the
system's own limits were not applied to it: CRST reached 54% of the account
against a 20% single-name cap, 79% sat in two weak balance sheets, and 26
orders in two weeks cost more in fees than the account lost. None of those
facts was hidden -- they simply were not in front of you at the moment of the
order. This puts them there.

WHAT IT DOES. For a proposed buy (ticker, EGP amount) it checks, from facts only:
  position size after the trade  vs MAX_POSITION 20%      -> BLOCK
  sector weight after the trade  vs MAX_SECTOR 30%        -> BLOCK
  account drawdown guard (book_state)                     -> BLOCK when 'block'
  usable price history (>= 90 of 180 sessions)            -> WARN (risk unmeasurable)
  security type (fund, not a share)                       -> WARN
  order vs median daily traded value                      -> WARN above 5%
  fee as % of the order, and your recent trading pace     -> WARN
  re-entry: sold this name in the last 14 days            -> WARN
  run-up: +50% in 60 sessions / near 1-year high          -> INFO (momentum tested NEGATIVE here)
  fundamentals: losses, negative equity, leverage         -> WARN
  open findings on the name; unadjusted price events      -> WARN
  recent news calls on the name                           -> INFO
A BLOCK needs an override with a reason, which is logged. Every review is kept
in trade_reviews so that, over time, the outcomes of overridden trades can be
compared with the ones that passed and with cash.

It never says "buy". It says what is true about the order. Personal decisions
stay yours.

    py -3.11 -m ops.challenger CRST 2000
    py -3.11 -m ops.challenger CRST 2000 --override "earnings due, sized to exit fast"
    py -3.11 -m ops.challenger outcomes
"""

import json
import logging
import sys

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.challenger")

AGENT = "challenger"
WRITABLE = {"trade_reviews", "agent_findings"}
MAX_POSITION = 0.20
MAX_SECTOR = 0.30
MIN_USABLE = 90
LOOKBACK = 180
ADV_WARN = 0.05
REENTRY_DAYS = 14


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS trade_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT, reviewed_at TEXT, ticker TEXT, side TEXT,
        amount_egp REAL, price REAL, verdict TEXT, n_block INTEGER, n_warn INTEGER,
        checks_json TEXT, override_reason TEXT)""")
    from ops.agent_guard import ensure_tables
    ensure_tables(conn)


def _fee(amount: float) -> float:
    return 4.95 + 0.00162 * amount


def review(ticker: str, amount: float, side: str = "buy") -> dict:
    """All checks for one proposed order. Pure read; storing is separate."""
    from ops.portfolio_tracker import holdings
    from backtest.engine import load_price_returns
    ticker = ticker.upper()
    pos, cash = holdings()
    held = pos.set_index("ticker")["value"] if not pos.empty else pd.Series(dtype=float)
    account = float(held.sum() + cash)
    checks = []

    def add(level, name, text):
        checks.append({"level": level, "check": name, "text": text})

    with get_connection() as conn:
        row = conn.execute("SELECT * FROM stocks WHERE ticker=?", (ticker,)).fetchone()
        st = dict(row) if row else None                   # columns vary with the DB's age
        sectors = dict(conn.execute("SELECT ticker, sector FROM stocks").fetchall())
        px = pd.read_sql("SELECT date, close, volume, value_traded_egp FROM daily_ohlcv WHERE ticker=? "
                         "AND close > 0 ORDER BY date", conn, params=(ticker,), parse_dates=["date"])
    if st is None:
        add("WARN", "identity", f"{ticker} is not in the security master; nothing below can be checked")
    price = float(px["close"].iloc[-1]) if len(px) else float("nan")

    # 1. position and sector size after the trade
    new_val = float(held.get(ticker, 0.0)) + amount
    total_after = account if side == "buy" else account
    w = new_val / total_after if total_after > 0 else float("nan")
    lvl = "BLOCK" if w > MAX_POSITION else "OK"
    add(lvl, "position size", f"{ticker} would be {w:.0%} of the account (limit {MAX_POSITION:.0%})"
        + ("" if amount <= cash else f"; this needs {amount - cash:,.0f} EGP more than your {cash:,.0f} cash, "
                                      "so something else must be sold first"))
    sec = sectors.get(ticker)
    if sec:
        sec_val = sum(v for t, v in held.items() if sectors.get(t) == sec and t != ticker) + new_val
        ws = sec_val / total_after
        add("BLOCK" if ws > MAX_SECTOR else "OK", "sector",
            f"{sec}: {ws:.0%} of the account after the trade (limit {MAX_SECTOR:.0%})")

    # 2. account guard
    try:
        from risk.kill_switch import book_state
        b = book_state()
        add("BLOCK" if b["block_new_risk"] else ("WARN" if b["level"] == "warn" else "OK"), "account guard",
            f"account {b['drawdown']*100:+.1f}% from its peak" + (f" -- {b['reason']}" if b["reason"] else ""))
    except Exception:  # noqa: BLE001
        pass

    # 3. measurability, security type, liquidity
    r = load_price_returns([ticker]).tail(LOOKBACK)
    usable = int(r[ticker].notna().sum()) if ticker in r else 0
    add("WARN" if usable < MIN_USABLE else "OK", "price history",
        f"{usable} usable sessions of the last {LOOKBACK}" +
        (" -- its risk cannot be measured" if usable < MIN_USABLE else ""))
    if st and st.get("security_type") and st["security_type"] != "ordinary":
        add("WARN", "security type", f"{ticker} is a {st['security_type']}, not an ordinary share")
    adv = float(px["value_traded_egp"].tail(60).replace(0, np.nan).median()) if len(px) else float("nan")
    if adv == adv and adv > 0:
        add("WARN" if amount / adv > ADV_WARN else "OK", "liquidity",
            f"order is {amount / adv * 1e4:.2f} bp of the median daily value traded ({adv/1e6:,.1f}M EGP)")

    # 4. cost and pace
    fee_rt = 2 * _fee(amount) / amount
    from ops.portfolio_tracker import _ensure_fee_column
    with get_connection() as conn:
        _ensure_fee_column(conn)                          # old ledgers lack fee/price_source
        recent = pd.read_sql("SELECT traded_at, action, ticker, shares, price, fee FROM user_trade_ledger "
                             "WHERE traded_at >= ? AND price_source != 'correction'", conn,
                             params=((pd.Timestamp.now() - pd.Timedelta(days=30)).isoformat(),))
    add("WARN" if fee_rt > 0.01 else "OK", "fees",
        f"round-trip fees ~{2 * _fee(amount):,.0f} EGP = {fee_rt:.2%} of this order; "
        f"{len(recent)} orders in the last 30 days")
    sold = recent[(recent["ticker"] == ticker) & (recent["action"].str.lower() == "sell")]
    if len(sold) and (pd.Timestamp.now() - pd.to_datetime(sold["traded_at"]).max()).days <= REENTRY_DAYS:
        add("WARN", "re-entry", f"you sold {ticker} within the last {REENTRY_DAYS} days; buying back pays "
                                "two more fees for the same exposure")

    # 5. price behaviour (descriptive -- the momentum screen tested NEGATIVE on EGX)
    s = px.set_index("date")["close"]
    if len(s) > 60:
        run60 = s.iloc[-1] / s.iloc[-61] - 1
        y = s[s.index >= s.index[-1] - pd.Timedelta(days=365)]
        off = s.iloc[-1] / y.max() - 1
        big = int((s.pct_change().tail(120).abs() >= 0.09).sum())
        add("INFO", "price run",
            f"{run60:+.0%} in 60 sessions, {off:+.0%} from its 1-year high, {big} days moving 9%+ in 120 sessions"
            + ("; buying after large run-ups tested NEGATIVE on EGX (-20.9%/yr)" if run60 > 0.5 else ""))

    # 6. fundamentals (point-in-time: only what was public by today)
    try:
        from data.fundamentals import leverage_as_of, quarterly_yoy, as_of
        today = pd.Timestamp.now().date().isoformat()
        lev = leverage_as_of(ticker, today)
        if lev:
            if lev.get("negative_equity"):
                add("WARN", "balance sheet", "shareholders' equity is negative")
            elif lev.get("debt_to_equity") and lev["debt_to_equity"] > 2:
                add("WARN", "balance sheet", f"debt is {lev['debt_to_equity']:.1f}x equity ({lev['period_end']})")
            if lev.get("liabilities_to_assets") and lev["liabilities_to_assets"] > 0.85:
                add("WARN", "balance sheet", f"liabilities are {lev['liabilities_to_assets']:.0%} of assets")
        fy = as_of(ticker, today, "income")
        ni = fy[(fy["line_item"] == "Net Income") & fy["fiscal_period"].str.startswith("FY")].dropna(subset=["value"])
        streak = 0
        for v in ni.sort_values("period_end", ascending=False)["value"]:
            if v < 0:
                streak += 1
            else:
                break
        if streak:
            add("WARN", "profitability", f"{streak} consecutive loss-making year(s) on record")
        q = quarterly_yoy(ticker, today)
        if q.get("value") is not None and q["value"] < 0:
            add("WARN", "latest quarter", f"{q['period']} was a loss ({q['value']:,.1f})")
    except Exception as exc:  # noqa: BLE001
        logger.info("fundamentals check skipped: %s", exc)

    # 7. open findings, unconfirmed price events, recent news
    with get_connection() as conn:
        f = []
        try:
            f = conn.execute("SELECT severity, summary FROM agent_findings WHERE status='open' AND "
                             "(subject = ? OR subject LIKE ?)", (ticker, f"{ticker}@%")).fetchall()
        except Exception:  # noqa: BLE001
            pass
        for sev, summ in f:
            add("WARN", "open finding", f"[{sev}] {summ}")
        try:
            news = conn.execute("SELECT published_at, event_type, direction, rationale FROM news_assessments "
                                "WHERE ticker=? AND confidence IN ('high','medium') AND event_type != 'other' "
                                "AND published_at >= ? ORDER BY published_at DESC LIMIT 3",
                                (ticker, (pd.Timestamp.now() - pd.Timedelta(days=45)).isoformat())).fetchall()
        except Exception:  # noqa: BLE001
            news = []
        for d, et, di, why in news:
            add("INFO", "news", f"{str(d)[:10]} {et} ({'+' if di > 0 else '-' if di < 0 else '0'}): {why[:110]}")

    n_block = sum(c["level"] == "BLOCK" for c in checks)
    n_warn = sum(c["level"] == "WARN" for c in checks)
    return {"ticker": ticker, "side": side, "amount": amount, "price": price, "account": account,
            "cash": cash, "checks": checks, "n_block": n_block, "n_warn": n_warn,
            "verdict": "BLOCKED" if n_block else ("CAUTION" if n_warn else "CLEAR")}


def record(res: dict, override_reason: str = None) -> int:
    from ops import agent_guard as ag
    with get_connection() as raw:
        _ensure(raw)
        conn = ag.guarded(raw, WRITABLE, AGENT)
        cur = ag.run(conn, "INSERT INTO trade_reviews (reviewed_at, ticker, side, amount_egp, price, verdict, "
                           "n_block, n_warn, checks_json, override_reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (pd.Timestamp.now().isoformat(timespec="seconds"), res["ticker"], res["side"], res["amount"],
                      res["price"], res["verdict"] if not override_reason else "OVERRIDDEN", res["n_block"],
                      res["n_warn"], json.dumps(res["checks"], ensure_ascii=False), override_reason), AGENT)
        return int(cur.lastrowid)


def outcomes(horizons=(5, 20, 60)) -> pd.DataFrame:
    """What happened after each reviewed order: stock vs cash, by verdict.
    Over months this answers whether overrides beat the rules, cash, or neither."""
    from backtest.engine import load_price_returns
    from data.rates import cash_rate_series
    with get_connection() as conn:
        _ensure(conn)
        rv = pd.read_sql("SELECT * FROM trade_reviews", conn, parse_dates=["reviewed_at"])
    if rv.empty:
        return rv
    r = load_price_returns(rv["ticker"].unique().tolist())
    cash = (1 + cash_rate_series(r.index, haircut=0.0)) ** (1 / 245) - 1
    rows = []
    for x in rv.itertuples():
        if x.ticker not in r:
            continue
        s = r[x.ticker]
        i = s.index.searchsorted(x.reviewed_at.normalize(), side="right")
        rec = {"id": x.id, "ticker": x.ticker, "verdict": x.verdict}
        for h in horizons:
            if i + h < len(s):
                rec[f"ret_{h}"] = float((1 + s.iloc[i:i + h].fillna(0)).prod() - 1)
                rec[f"vs_cash_{h}"] = rec[f"ret_{h}"] - float((1 + cash.iloc[i:i + h]).prod() - 1)
        rows.append(rec)
    return pd.DataFrame(rows)


def format_memo(res: dict) -> str:
    L = ["=" * 70,
         f"CHALLENGER — {res['side'].upper()} {res['ticker']} for {res['amount']:,.0f} EGP"
         f" (last {res['price']:.3f})  ->  {res['verdict']}",
         "=" * 70, "  The case against this trade, from facts only:"]
    order = {"BLOCK": 0, "WARN": 1, "INFO": 2, "OK": 3}
    for c in sorted(res["checks"], key=lambda c: order[c["level"]]):
        L.append(f"  [{c['level']:5}] {c['check']:15} {c['text']}")
    if res["n_block"]:
        L += ["", "  BLOCKED by the system's own limits. To proceed anyway, record why:",
              f"    py -3.11 -m ops.challenger {res['ticker']} {res['amount']:.0f} --override \"your reason\""]
    L.append("  These are facts about the order, not advice. The decision is yours.")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if argv[:1] == ["outcomes"]:
        o = outcomes()
        print("no reviewed trades yet" if o.empty else o.to_string(index=False))
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 1
    reason = argv[argv.index("--override") + 1] if "--override" in argv else None
    res = review(argv[0], float(argv[1]), side="buy")
    rid = record(res, reason)
    print(format_memo(res))
    print(f"  (review #{rid} logged{' with override: ' + reason if reason else ''})")
    return 2 if res["n_block"] and not reason else 0


if __name__ == "__main__":
    raise SystemExit(main())
