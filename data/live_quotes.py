"""
Delayed intraday quotes -- free, no account, no API key.

WHAT EXISTS (checked 2026-10-03). No licensed free real-time EGX feed exists for
an individual; EGX requires third-party display to be delayed. Two free sources
give usable delayed data:

  stocks   stockanalysis.com quote pages. Each page carries a quote object marked
           "Delayed Price" with last price, open, high, low, volume, previous
           close, market status (open/closed) and a timestamp, refreshed about
           every 10 minutes. Already the system's second reference source.
  EGX30    Yahoo ^CASE30, 1-minute bars, `exchangeDataDelayedBy` = 15 minutes.
           Yahoo does NOT carry intraday bars for individual EGX stocks -- only
           one daily bar -- so it cannot serve the stock leg.

Rejected: EGX's own site (blocks automated access), broker apps (terms), and
EGXBot / EGX.news (claim free real-time but name no licensed source).

WHAT IT IS FOR. Monitoring and execution checks: the book's value during the
session, the price you could actually trade at, and an early warning on a bad
day. NOT for signals -- the strategy decides once a day at the close, and these
rows never enter daily_ohlcv (a delayed scrape is not a closing price).

Every row stores the source's own quote timestamp next to the time we fetched
it, so the real delay is measured, never assumed.

    py -3.11 -m data.live_quotes            # one snapshot of your holdings + EGX30
    py -3.11 -m data.live_quotes --watch    # every 10 min while the market is open
"""

import logging
import re
import sys
import time

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.live_quotes")

SA_URL = "https://stockanalysis.com/quote/egx/{t}/"
POLL_SECONDS = 600            # the source refreshes about every 10 minutes
REQUEST_DELAY_S = 1.0         # politeness between tickers
CAIRO = "Africa/Cairo"
SESSION = ((10, 0), (14, 30))  # 10:00-14:15 trading, plus a margin for the last quote
DAY_MOVE_ALERT = -0.05        # book down 5% on the day: say so during the session

_QUOTE = re.compile(r"quote:\{(.*?)\}", re.S)
_PAIR = re.compile(r'(\w+):("(?:[^"\\]|\\.)*"|[^,]*)')


def parse_quote(html: str) -> dict:
    """The quote object embedded in a stockanalysis page, as a dict. Empty if absent."""
    m = _QUOTE.search(html)
    if not m:
        return {}
    raw = {}
    for k, v in _PAIR.findall(m.group(1)):
        v = v.strip()
        if v.startswith('"'):
            raw[k] = v[1:-1]
        elif v in ("true", "false"):
            raw[k] = v == "true"
        elif v in ("null", ""):
            raw[k] = None
        else:
            try:
                raw[k] = float(v)
            except ValueError:
                raw[k] = v
    if raw.get("p") is None:
        return {}
    ts = raw.get("ts")
    return {
        "price": raw.get("p"), "open": raw.get("o"), "high": raw.get("h"), "low": raw.get("l"),
        "volume": raw.get("v"), "prev_close": raw.get("cl"), "change_pct": raw.get("cp"),
        "market_state": raw.get("ms"), "label": raw.get("u"),
        "quote_time": (pd.Timestamp(int(ts), unit="ms", tz="UTC").tz_convert(CAIRO)
                       .tz_localize(None).isoformat(timespec="seconds") if ts else None),
    }


def fetch_stock(ticker: str) -> dict:
    from data.fundamentals import _fetch
    q = parse_quote(_fetch(SA_URL.format(t=ticker)))
    return {"ticker": ticker, "source": "stockanalysis", **q} if q else {}


def fetch_egx30() -> dict:
    """Last 1-minute bar of the EGX30 index from Yahoo."""
    import yfinance as yf
    h = yf.Ticker("^CASE30").history(period="5d", interval="1m")
    if h.empty:
        return {}
    last = h.iloc[-1]
    last_day = h.index.max().normalize()
    day = h[h.index.normalize() == last_day]
    before = h[h.index.normalize() < last_day]
    prev_close = float(before["Close"].iloc[-1]) if len(before) else None
    return {"ticker": "EGX30", "source": "yahoo", "price": float(last["Close"]),
            "open": float(day["Open"].iloc[0]), "high": float(day["High"].max()),
            "low": float(day["Low"].min()), "volume": float(day["Volume"].sum()),
            "prev_close": prev_close,
            "change_pct": (float(last["Close"]) / prev_close - 1) * 100 if prev_close else None,
            "market_state": None, "label": "1m bar, delayed 15 min",
            "quote_time": h.index.max().tz_convert(CAIRO).tz_localize(None).isoformat(timespec="seconds")}


def _ensure_table(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS intraday_quotes (
        ticker TEXT NOT NULL, quote_time TEXT, fetched_at TEXT NOT NULL, source TEXT,
        price REAL, open REAL, high REAL, low REAL, volume REAL, prev_close REAL,
        change_pct REAL, market_state TEXT, delay_min REAL,
        PRIMARY KEY (ticker, fetched_at))""")


def store(rows: list) -> int:
    cols = ["ticker", "quote_time", "fetched_at", "source", "price", "open", "high", "low",
            "volume", "prev_close", "change_pct", "market_state", "delay_min"]
    with get_connection() as conn:
        _ensure_table(conn)
        conn.executemany(f"INSERT OR REPLACE INTO intraday_quotes ({', '.join(cols)}) "
                         f"VALUES ({', '.join('?' * len(cols))})",
                         [tuple(r.get(c) for c in cols) for r in rows])
    return len(rows)


def holdings() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql("SELECT ticker, shares, avg_cost FROM user_positions", conn)


def snapshot(tickers=None) -> dict:
    """Fetch, timestamp and store one round of quotes; value the book on them."""
    pos = holdings()
    tickers = list(tickers) if tickers else pos["ticker"].tolist()
    now = pd.Timestamp.now(tz=CAIRO).tz_localize(None)
    rows, failed = [], []
    for t in tickers:
        try:
            q = fetch_stock(t)
            (rows if q else failed).append(q or t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("live quote %s: %s", t, exc)
            failed.append(t)
        time.sleep(REQUEST_DELAY_S)
    try:
        idx = fetch_egx30()
        if idx:
            rows.append(idx)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EGX30 intraday: %s", exc)
    for r in rows:
        r["fetched_at"] = now.isoformat(timespec="seconds")
        r["delay_min"] = ((now - pd.Timestamp(r["quote_time"])).total_seconds() / 60
                          if r.get("quote_time") else None)
    store(rows)

    q = pd.DataFrame([r for r in rows if r["ticker"] != "EGX30"])
    book = {}
    if not q.empty and not pos.empty:
        m = pos.merge(q, on="ticker", how="left")
        m["value"] = m["shares"] * m["price"]
        m["value_prev"] = m["shares"] * m["prev_close"]
        priced = m.dropna(subset=["value", "value_prev"])
        if len(priced):
            book = {"value": float(priced["value"].sum()),
                    "day_change": float(priced["value"].sum() / priced["value_prev"].sum() - 1),
                    "priced": len(priced), "held": len(m), "detail": m}
    return {"rows": rows, "failed": failed, "book": book, "fetched_at": now}


def format_snapshot(s: dict) -> str:
    L = [f"DELAYED QUOTES  fetched {s['fetched_at']:%Y-%m-%d %H:%M} Cairo"]
    for r in s["rows"]:
        chg = f"{r['change_pct']:+.2f}%" if r.get("change_pct") is not None else "   n/a"
        dl = f"{r['delay_min']:.0f} min old" if r.get("delay_min") is not None else "age ?"
        L.append(f"  {r['ticker']:7} {r['price']:>10,.2f}  {chg:>8}  {r.get('market_state') or '':6} "
                 f"{dl}  [{r['source']}]")
    if s["failed"]:
        L.append(f"  no free delayed quote: {', '.join(map(str, s['failed']))} "
                 "(not covered by stockanalysis; check Thndr)")
    b = s["book"]
    if b:
        when = "today" if market_open(pd.Timestamp(s["fetched_at"]).tz_localize(CAIRO)) else "last session"
        L.append(f"  YOUR BOOK ~{b['value']:,.0f} EGP for the priced names, "
                 f"{b['day_change']*100:+.2f}% {when} ({b['priced']}/{b['held']} holdings priced)")
        if b["day_change"] <= DAY_MOVE_ALERT:
            L.append(f"  !! book is down {b['day_change']*100:.1f}% today")
    L.append("  Delayed data for monitoring only. Check the live price in Thndr before any order.")
    return "\n".join(L)


def market_open(now=None) -> bool:
    now = now or pd.Timestamp.now(tz=CAIRO)
    if now.dayofweek in (4, 5):                       # Friday, Saturday
        return False
    (h0, m0), (h1, m1) = SESSION
    return (h0, m0) <= (now.hour, now.minute) <= (h1, m1)


def watch():
    """Poll while the market is open; exit after the session."""
    while True:
        if not market_open():
            print("market closed -- watch stopped")
            return
        s = snapshot()
        print(format_snapshot(s), flush=True)
        if s["book"] and s["book"]["day_change"] <= DAY_MOVE_ALERT:
            try:
                from ops.health import alert
                alert("ESIS-QUANT — book down today",
                      f"Book ~{s['book']['value']:,.0f} EGP, {s['book']['day_change']*100:+.1f}% today "
                      "(delayed quotes).")
            except Exception:  # noqa: BLE001
                pass
        time.sleep(POLL_SECONDS)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if "--watch" in argv:
        watch()
    else:
        print(format_snapshot(snapshot([a for a in argv if not a.startswith("--")] or None)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
