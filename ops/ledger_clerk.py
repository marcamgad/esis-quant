"""
Agent 2 -- LEDGER CLERK. Keeps the trade ledger and your real holdings in agreement.

WHY. Positions and the ledger were two separate records that drifted: on
2026-10-03 they disagreed on 10 names -- seven exits never booked, OIH 1,685 in
the ledger against 1,693 at the broker, and three small leftovers from the old
ACCEPT path. A ledger that does not replay to your holdings cannot produce a
realised P&L, a cost basis, or a fee total anyone should trust.

HOW. The broker screen (what you send me: units, purchase value, market value)
is the truth for WHAT you hold. The ledger is the truth for HOW you got there.
The clerk replays the ledger, compares it to your holdings, and STAGES the
missing trades in ledger_staging:
  * a missing buy  -> its price is derived from your broker cost basis
  * a missing sell -> it needs the sale price from you (Thndr order history)
  * a tiny leftover from the old ACCEPT path -> a 'correction' row, no P&L
Nothing reaches user_trade_ledger until you approve it. Approval is your
action, not the agent's: the agent's own connection can write only
ledger_staging and agent_findings (enforced by the database engine).

    py -3.11 -m ops.ledger_clerk                      # reconcile, stage, show
    py -3.11 -m ops.ledger_clerk approve 7 2.05 6.20 2026-09-24   # id price [fee] [date]
    py -3.11 -m ops.ledger_clerk approve 7 2.05 6.20 2026-09-24 1693   # ... [shares]
    py -3.11 -m ops.ledger_clerk reject 7
    py -3.11 -m ops.ledger_clerk pnl                  # realised P&L from the ledger

Thndr contract notes (PDF) are the next intake: send one sample and a parser
for its fixed layout replaces typing these by hand.
"""

import logging
import sys

import pandas as pd

from data.database import get_connection
from ops import agent_guard as ag

logger = logging.getLogger("esis_quant.ledger_clerk")

AGENT = "ledger_clerk"
WRITABLE = {"ledger_staging", "agent_findings"}
# A leftover on a name you no longer hold is bookkeeping residue only if it is
# worth less than this at the last price. Judged by VALUE, not share count: the
# first version used "<= 10 shares" and called real exits of GSSC (2 sh, ~580 EGP)
# and GTWL (5 sh, ~1,190 EGP) residue.
CORRECTION_MAX_EGP = 100.0


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS ledger_staging (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, source TEXT, action TEXT,
        ticker TEXT, shares REAL, price REAL, fee REAL, trade_date TEXT, price_source TEXT,
        needs TEXT, status TEXT DEFAULT 'pending', note TEXT, decided_at TEXT)""")
    ag.ensure_tables(conn)


def replay(ledger: pd.DataFrame = None) -> pd.DataFrame:
    """Shares and average cost per ticker by replaying the ledger (average-cost
    method: a sell leaves the average unchanged). Fees enter the cost of a buy."""
    if ledger is None:
        with get_connection() as conn:
            ledger = pd.read_sql("SELECT * FROM user_trade_ledger ORDER BY traded_at, id", conn)
    book = {}
    for r in ledger.itertuples():
        sh, cost = book.get(r.ticker, (0.0, 0.0))
        fee = float(r.fee or 0.0)
        if str(r.action).lower() == "buy":
            new = sh + r.shares
            cost = (sh * cost + r.shares * (r.price or 0.0) + fee) / new if new > 0 else 0.0
            sh = new
        else:
            sh -= r.shares
        book[r.ticker] = (sh, cost)
    return pd.DataFrame([{"ticker": t, "shares": s, "avg_cost": c} for t, (s, c) in book.items()])


def differences() -> pd.DataFrame:
    with get_connection() as conn:
        pos = pd.read_sql("SELECT ticker, shares AS broker, avg_cost AS broker_cost FROM user_positions", conn)
    led = replay().rename(columns={"shares": "ledger", "avg_cost": "ledger_cost"})
    d = pos.merge(led, on="ticker", how="outer").fillna({"broker": 0.0, "ledger": 0.0})
    d["diff"] = d["broker"] - d["ledger"]
    return d[d["diff"].abs() > 1e-9].sort_values("ticker").reset_index(drop=True)


def stage(source: str = "reconcile") -> pd.DataFrame:
    """Stage one row per difference not already pending. Returns the pending queue."""
    d = differences()
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    findings = []
    with get_connection() as raw:
        _ensure(raw)
        conn = ag.guarded(raw, WRITABLE, AGENT)
        last_px = dict(conn.execute(
            "SELECT ticker, close FROM daily_ohlcv d WHERE close IS NOT NULL AND date = "
            "(SELECT MAX(date) FROM daily_ohlcv d2 WHERE d2.ticker = d.ticker AND close IS NOT NULL)"
        ).fetchall())
        pending = {(r[0], r[1]) for r in conn.execute(
            "SELECT ticker, action FROM ledger_staging WHERE status = 'pending'")}
        for r in d.itertuples():
            residue_egp = abs(r.ledger) * float(last_px.get(r.ticker) or float("inf"))
            if r.broker == 0 and r.ledger != 0 and residue_egp < CORRECTION_MAX_EGP:
                action, price, ps, needs = "correction", None, "correction", "confirm"
                note = (f"{r.ledger:g} share(s) (~{residue_egp:,.0f} EGP) left over; you hold none. "
                        "Approve to zero it without P&L.")
            elif r.diff > 0:
                action = "buy"
                # cost of the missing shares = broker total cost - ledger total cost
                total_b = r.broker * (r.broker_cost or 0.0)
                total_l = r.ledger * (r.ledger_cost or 0.0)
                price = (total_b - total_l) / r.diff if r.broker_cost == r.broker_cost else None
                ps, needs = "derived_from_broker_cost", "confirm"
                note = f"ledger {r.ledger:g} vs broker {r.broker:g}; price derived from your cost basis"
            else:
                action, price, ps, needs = "sell", None, None, "price"
                note = (f"ledger {r.ledger:g} vs broker {r.broker:g}; give the sale price "
                        "(and real share count if different) from Thndr's order history")
            findings.append({"severity": "high", "kind": "ledger_mismatch", "subject": r.ticker,
                             "summary": f"ledger {r.ledger:g} vs broker {r.broker:g} {r.ticker}",
                             "fix_hint": "py -3.11 -m ops.ledger_clerk"})
            if (r.ticker, action) in pending:
                continue
            ag.run(conn, "INSERT INTO ledger_staging (created_at, source, action, ticker, shares, price, "
                         "fee, trade_date, price_source, needs, note) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (now, source, action, r.ticker, abs(r.diff), price, None, None, ps, needs, note), AGENT)
        ag.record_findings(conn, AGENT, findings)
    return queue()


def queue() -> pd.DataFrame:
    with get_connection() as conn:
        _ensure(conn)
        return pd.read_sql("SELECT id, action, ticker, shares, price, fee, trade_date, needs, note "
                           "FROM ledger_staging WHERE status = 'pending' ORDER BY ticker", conn)


def approve(staging_id: int, price: float = None, fee: float = None, trade_date: str = None,
            shares: float = None) -> dict:
    """YOUR approval: write the staged trade into user_trade_ledger.

    Writes the ledger only. Positions and cash already come from your broker
    screen, so they are not touched -- the point is for the ledger to replay to them.
    """
    with get_connection() as conn:
        _ensure(conn)
        r = conn.execute("SELECT * FROM ledger_staging WHERE id = ? AND status = 'pending'",
                         (staging_id,)).fetchone()
        if not r:
            raise ValueError(f"no pending staged trade #{staging_id}")
        r = dict(r)
        shares = float(shares) if shares is not None else float(r["shares"])
        price = float(price) if price is not None else r["price"]
        if r["action"] == "sell" and price is None:
            raise ValueError("a sell needs its sale price: approve ID PRICE [FEE] [DATE] [SHARES]")
        when = pd.Timestamp(trade_date).isoformat(timespec="seconds") if trade_date else \
            pd.Timestamp.now().isoformat(timespec="seconds")
        if r["action"] == "correction":
            # zero the residue with the opposite side at zero price, labelled so
            # realised_pnl() skips it
            cur = replay().set_index("ticker")["shares"].get(r["ticker"], 0.0)
            row = (when, "sell" if cur > 0 else "buy", r["ticker"], abs(cur), 0.0, 0.0, "correction")
        else:
            # a price YOU typed is a real fill; an untouched derived price keeps its label
            src = r["price_source"] if (r["action"] == "buy" and price == r["price"]) else "broker"
            row = (when, r["action"], r["ticker"], shares, price, float(fee or 0.0), src or "broker")
        conn.execute("INSERT INTO user_trade_ledger (traded_at, action, ticker, shares, price, fee, "
                     "price_source) VALUES (?,?,?,?,?,?,?)", row)
        conn.execute("UPDATE ledger_staging SET status='approved', decided_at=?, price=?, fee=?, "
                     "trade_date=?, shares=? WHERE id=?",
                     (pd.Timestamp.now().isoformat(timespec="seconds"), price, fee, trade_date,
                      shares, staging_id))
    left = differences()
    still = left[left["ticker"] == r["ticker"]]
    return {"written": row, "ticker_reconciled": still.empty,
            "remaining": None if still.empty else float(still["diff"].iloc[0])}


def reject(staging_id: int, note: str = "rejected") -> None:
    with get_connection() as conn:
        _ensure(conn)
        conn.execute("UPDATE ledger_staging SET status='rejected', decided_at=?, note=? WHERE id=?",
                     (pd.Timestamp.now().isoformat(timespec="seconds"), note, staging_id))


def realised_pnl() -> pd.DataFrame:
    """Realised P&L per closed lot, average-cost method, fees included."""
    with get_connection() as conn:
        led = pd.read_sql("SELECT * FROM user_trade_ledger ORDER BY traded_at, id", conn)
    book, rows = {}, []
    for r in led.itertuples():
        sh, cost = book.get(r.ticker, (0.0, 0.0))
        fee = float(r.fee or 0.0)
        if str(r.action).lower() == "buy":
            new = sh + r.shares
            cost = (sh * cost + r.shares * (r.price or 0.0) + fee) / new if new > 0 else 0.0
            sh = new
        else:
            if r.price_source != "correction" and sh > 0:
                q = min(r.shares, sh)
                rows.append({"date": str(r.traded_at)[:10], "ticker": r.ticker, "shares": q,
                             "sell_price": r.price, "avg_cost": cost,
                             "pnl": q * (r.price - cost) - fee,
                             "assumed": r.price_source in ("assumed", "accepted")})
            sh -= r.shares
        book[r.ticker] = (sh, cost)
    return pd.DataFrame(rows)


def format_queue(q: pd.DataFrame, d: pd.DataFrame) -> str:
    L = ["=" * 66, f"LEDGER CLERK — {len(d)} difference(s) between the ledger and your holdings", "=" * 66]
    if q.empty:
        L.append("  ledger replays exactly to your holdings. Nothing to approve.")
        return "\n".join(L)
    for r in q.itertuples():
        px = f"{r.price:.4f}" if r.price == r.price and r.price is not None else "NEEDED"
        L.append(f"  #{r.id:<3} {r.action:10} {r.ticker:7} {r.shares:>9,.0f} sh  price {px:>8}   ({r.needs})")
        L.append(f"        {r.note}")
    L += ["", "  approve:  py -3.11 -m ops.ledger_clerk approve ID PRICE [FEE] [YYYY-MM-DD] [SHARES]",
          "  reject:   py -3.11 -m ops.ledger_clerk reject ID"]
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    cmd = argv[0] if argv else "reconcile"
    if cmd == "approve":
        a = argv[1:] + [None] * 5
        res = approve(int(a[0]), price=float(a[1]) if a[1] else None, fee=float(a[2]) if a[2] else None,
                      trade_date=a[3], shares=float(a[4]) if a[4] else None)
        print(f"written to ledger: {res['written']}")
        print("  ticker now reconciled" if res["ticker_reconciled"]
              else f"  still {res['remaining']:+g} shares apart")
    elif cmd == "reject":
        reject(int(argv[1]))
        print(f"#{argv[1]} rejected")
    elif cmd == "pnl":
        p = realised_pnl()
        if p.empty:
            print("no realised trades yet")
        else:
            print(p.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
            print(f"\nTOTAL realised {p['pnl'].sum():+,.2f} EGP "
                  f"({p['assumed'].sum()} row(s) use an ASSUMED price, not a real fill)")
    else:
        q = stage()
        print(format_queue(q, differences()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
