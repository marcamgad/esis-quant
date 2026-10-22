"""
Second price/reference source: stockanalysis.com's EGX listing.

WHY A SECOND SOURCE. Yahoo has been the only price feed, and this week showed
what that costs: it published nothing usable for five days (2026-08-13 to 08-17),
emitted a fabricated carried-forward session on 08-16 that reset the staleness
clock, and it carries no company names, no market cap and no index membership at
all. A single feed with no cross-check means a vendor error is indistinguishable
from a market move.

WHAT THIS ADDS THAT YAHOO DOES NOT HAVE
  * company NAME per ticker -- previously `stocks.name_english` was null for
    almost every name, which is why index-membership lists (published as company
    names, never as tickers) could not be mapped to the universe at all.
  * MARKET CAP -- never available before. Size is the missing axis: EGX30 and
    EGX70 EWI are both defined partly on free-float market cap, and without it
    any "large cap"/"mid cap" statement was guesswork.
  * an independent PRICE to cross-check Yahoo against.

WHAT IT IS NOT. It is a page scrape of a third-party site, not an exchange feed.
The prices are whatever that page last rendered and carry no timestamp, so they
are NOT stored into daily_ohlcv and never become marks -- they are used to
CROSS-CHECK Yahoo and to flag disagreement. Broker values still outrank
everything (see ops.portfolio_tracker.set_broker_mark), then Yahoo, and this
source is advisory only. Storing an untimestamped scrape as a price bar would
recreate the exact failure mode the 2026-08-18 calendar fix was written to stop.
"""

import logging
import re
import urllib.request

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.stockanalysis")

LIST_URL = "https://stockanalysis.com/list/egyptian-stock-exchange/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# The page is server-rendered Svelte: one <tr> per stock, with the ticker inside
# an <a href="/quote/egx/TICKER/"> and the remaining <td> cells in column order
# (No., Symbol, Name, Market Cap, Price, % Change, Revenue). Parsed with a regex
# rather than a DOM library to avoid adding a dependency for one page.
_ROW = re.compile(r'<a href="/quote/egx/([A-Z0-9.]+)/">\1</a>(.*?)</tr>', re.S)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAGS = re.compile(r"<[^>]+>")

_MULT = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _num(text):
    """'463.10B' -> 4.631e11; '136.00' -> 136.0; '-' / '' -> None."""
    t = _TAGS.sub("", text or "").replace(",", "").replace("%", "").strip()
    if not t or t in {"-", "n/a", "N/A"}:
        return None
    mult = _MULT.get(t[-1].upper())
    if mult:
        t = t[:-1]
    try:
        return float(t) * (mult or 1.0)
    except ValueError:
        return None


def fetch_egx_listing(timeout: int = 30) -> pd.DataFrame:
    """Ticker, name, market cap, price and % change for every listed EGX name."""
    req = urllib.request.Request(LIST_URL, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")
    rows = []
    for ticker, rest in _ROW.findall(raw):
        cells = [_TAGS.sub("", c).strip() for c in _CELL.findall(rest)]
        # cells after the symbol column: name, market cap, price, change, revenue
        if len(cells) < 2:
            continue
        rows.append({
            "ticker": ticker.split(".")[0],
            "name_english": cells[0] or None,
            "market_cap_egp": _num(cells[1]) if len(cells) > 1 else None,
            "price": _num(cells[2]) if len(cells) > 2 else None,
            "change_pct": (_num(cells[3]) / 100.0) if len(cells) > 3 and _num(cells[3]) is not None else None,
            "revenue_egp": _num(cells[4]) if len(cells) > 4 else None,
        })
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    logger.info("fetch_egx_listing: %d rows", len(df))
    return df


def _ensure_columns(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(stocks)")}
    if "market_cap_egp" not in cols:
        conn.execute("ALTER TABLE stocks ADD COLUMN market_cap_egp REAL")
    if "reference_updated_at" not in cols:
        conn.execute("ALTER TABLE stocks ADD COLUMN reference_updated_at TEXT")


def store_reference(df: pd.DataFrame) -> dict:
    """Upsert name + market cap into `stocks`. Reference data only -- no prices."""
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    ins = upd = 0
    with get_connection() as conn:
        _ensure_columns(conn)
        have = {r["ticker"] for r in conn.execute("SELECT ticker FROM stocks")}
        for r in df.itertuples():
            if r.ticker in have:
                conn.execute("UPDATE stocks SET name_english=COALESCE(?,name_english), "
                             "market_cap_egp=?, reference_updated_at=? WHERE ticker=?",
                             (r.name_english, r.market_cap_egp, now, r.ticker)); upd += 1
            else:
                conn.execute("INSERT INTO stocks (ticker, name_english, market_cap_egp, "
                             "reference_updated_at) VALUES (?,?,?,?)",
                             (r.ticker, r.name_english, r.market_cap_egp, now)); ins += 1
    logger.info("store_reference: %d inserted, %d updated", ins, upd)
    return {"inserted": ins, "updated": upd}


def cross_check_prices(df: pd.DataFrame, tolerance: float = 0.05) -> pd.DataFrame:
    """Compare this source's price against the newest daily_ohlcv close.

    Disagreement does NOT mean Yahoo is wrong -- the two are sampled at different
    (and here, unknown) times. It means the name deserves a look before you trade
    on it, which is the whole point of having a second source.
    """
    with get_connection() as conn:
        yh = pd.read_sql(
            "SELECT ticker, adjusted_close AS yahoo, date FROM daily_ohlcv d "
            "WHERE date = (SELECT MAX(date) FROM daily_ohlcv d2 WHERE d2.ticker = d.ticker)",
            conn)
    m = df[["ticker", "price"]].merge(yh, on="ticker", how="inner").dropna()
    m = m[m["yahoo"] > 0]
    m["diff_pct"] = m["price"] / m["yahoo"] - 1.0
    m["flag"] = m["diff_pct"].abs() > tolerance
    return m.sort_values("diff_pct", key=lambda s: s.abs(), ascending=False)


def fill_missing_prices(df: pd.DataFrame = None, max_gap_sessions: int = 30) -> dict:
    """Fill price gaps for names the PRIMARY vendor has stopped publishing.

    THE FAILURE THIS FIXES. Yahoo silently drops names. CRST last printed on
    2026-08-12; by 2026-09-04 it was 19.2% of the live book with 23 days of no
    vendor price, so every risk number quietly described only 81% of the money
    under the footnote "NOT MEASURED (no price feed)". A position you cannot
    measure is not low-risk, it is unmeasured -- and the difference is invisible
    in a report that simply omits it.

    THE RULE, and why it is narrow. A filled row is written as source
    ='reference', a THIRD tier below 'vendor' (which is below 'broker'), and is
    only ever written on a date `data.calendar.valid_trading_days` has ALREADY
    accepted as a real session. It can never create a date, extend the panel, or
    become the newest row. That restriction is the whole point: a source with no
    timestamp that is allowed to mint dates is exactly the 2026-08-16 phantom
    session that reset the staleness clock and made a five-day-old signal look
    two days old. Gap-filling inside known sessions cannot do that.

    Volume is written as 0 because this source reports none, and a zero-volume
    row is already understood by MIN_TRADED_FRACTION as "not evidence of a
    session". So a filled row helps the risk model and is ignored by the
    calendar, which is the correct division of labour.

    ONLY THE LATEST SESSION IS FILLED, and that limit is load-bearing rather
    than conservative. This source publishes a single current price, not a
    history. The first version of this function back-filled every missing
    session with that one number, which for CRST wrote 14 identical closes --
    a manufactured flat line implying zero volatility for a name whose real
    problem is that it has 15 observations in 180 sessions. Valuation and risk
    measurement are different needs: one current price legitimately serves the
    first and cannot serve the second. A name too sparse to measure must keep
    reporting as unmeasurable, because that is true.
    """
    from data.calendar import valid_trading_days

    if df is None:
        df = fetch_egx_listing()
    px = df.dropna(subset=["price"]).set_index("ticker")["price"]

    sessions = valid_trading_days()
    if not len(sessions):
        return {"filled": 0, "reason": "no valid sessions"}
    latest = sessions[-1]
    # the newest accepted session only -- see the docstring on why this is not
    # widened. `max_gap_sessions` bounds how long a silence we will still call a
    # gap; beyond it the name is a delisting question, not a marking one.
    window = sessions[-1:]

    filled, names = 0, []
    with get_connection() as conn:
        from data.collectors import DataCollector
        DataCollector._ensure_source_column(conn)
        have = pd.read_sql(
            "SELECT ticker, MAX(date) AS last_date FROM daily_ohlcv "
            "WHERE source IN ('vendor','reference') GROUP BY ticker", conn)
        last = dict(zip(have["ticker"], pd.to_datetime(have["last_date"])))
        tradable = {r["ticker"] for r in conn.execute("SELECT DISTINCT ticker FROM tradable_universe")} \
            if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tradable_universe'").fetchone() \
            else set(last)
        for ticker, price in px.items():
            if ticker not in last or ticker not in tradable:
                continue
            gap = window[window > last[ticker]]
            if not len(gap):
                continue
            # bounded silence only: a name quiet for longer than this is not a
            # gap to patch, and marking it from a current quote would hide a
            # delisting behind a fresh-looking price.
            if (latest - last[ticker]).days > max_gap_sessions * 2:
                logger.warning("fill_missing_prices: %s silent since %s -- too long to "
                               "treat as a gap; left unmarked", ticker, last[ticker].date())
                continue
            for d in gap:
                conn.execute(
                    "INSERT OR IGNORE INTO daily_ohlcv "
                    "(ticker, date, open, high, low, close, volume, value_traded_egp, "
                    " adjusted_close, source) VALUES (?,?,?,?,?,?,0,0,?, 'reference')",
                    (ticker, d.strftime("%Y-%m-%d"), price, price, price, price, price))
                filled += 1
            names.append(f"{ticker}({len(gap)}d)")
    if filled:
        logger.warning("fill_missing_prices: filled %d row(s) for %d name(s) the primary "
                       "vendor stopped publishing: %s", filled, len(names), ", ".join(names[:10]))
    return {"filled": filled, "names": names, "through": str(latest.date())}


HISTORY_URL = "https://stockanalysis.com/quote/egx/{t}/history/"
_HIST_OBJ = re.compile(r'\{(a:[^{}]*?t:"\d{4}-\d{2}-\d{2}"[^{}]*)\}')
_HIST_KV = re.compile(r'(\w+):("?[-\w.:]+"?)')


def parse_history(html: str) -> pd.DataFrame:
    """Daily bars embedded in a stockanalysis /history/ page (about 6 months,
    S&P Global data, one dated row per session)."""
    out = []
    for obj in _HIST_OBJ.findall(html):
        d = dict(_HIST_KV.findall(obj))
        try:
            out.append({"date": pd.Timestamp(d["t"].strip('"')), "open": float(d["o"]),
                        "high": float(d["h"]), "low": float(d["l"]), "close": float(d["c"]),
                        "adjusted_close": float(d["a"]), "volume": float(d.get("v", 0) or 0)})
        except (KeyError, ValueError):
            continue
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).drop_duplicates("date").set_index("date").sort_index()


def backfill_history(tickers) -> dict:
    """Fill DATED history from stockanalysis for names Yahoo does not cover.

    WHY (2026-10-03). CRST and AIHC were 79% of the live book and the system could
    measure neither: Yahoo has no rows for AIHC at all and only 24 for CRST, yet
    stockanalysis carries six months of dated daily bars for both.

    WHY THIS DOES NOT BREAK THE RULE ABOVE. That rule bans UNTIMESTAMPED prices.
    These rows are dated per session by the source. The same guards as
    fill_missing_prices still apply:
      * written as source='sa_history', ranked below 'broker' and 'vendor';
      * INSERT OR IGNORE: never overwrites an existing row;
      * only on dates data.calendar.valid_trading_days has already accepted, so
        this source can never create a session or become the newest date.
    """
    from data.calendar import valid_trading_days
    from data.collectors import DataCollector
    from data.fundamentals import _fetch
    import time

    sessions = set(valid_trading_days())
    stored, per = 0, {}
    for t in tickers:
        try:
            h = parse_history(_fetch(HISTORY_URL.format(t=t)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill_history %s: %s", t, exc)
            continue
        h = h[h.index.isin(sessions)]
        rows = [(t, d.strftime("%Y-%m-%d"), r.open, r.high, r.low, r.close, r.volume,
                 r.close * r.volume, r.adjusted_close) for d, r in h.iterrows()]
        with get_connection() as conn:
            DataCollector._ensure_source_column(conn)
            before = conn.total_changes
            # A DATED bar outranks an UNDATED 'reference' fill. fill_missing_prices
            # stamps the listing page's current price on the newest session, which
            # put CRST's 29 Sep close (2.48) on 28 Sep (true close 3.10) -- a fake
            # -29% day that INSERT OR IGNORE would have preserved forever.
            dates = [r[1] for r in rows]
            if dates:
                conn.execute(f"DELETE FROM daily_ohlcv WHERE ticker = ? AND source = 'reference' "
                             f"AND date IN ({','.join('?' * len(dates))})", (t, *dates))
            conn.executemany(
                "INSERT OR IGNORE INTO daily_ohlcv (ticker, date, open, high, low, close, volume, "
                "value_traded_egp, adjusted_close, source) VALUES (?,?,?,?,?,?,?,?,?,'sa_history')", rows)
            n = conn.total_changes - before
        stored += n
        per[t] = n
        time.sleep(0.7)
    logger.info("backfill_history: %d row(s) %s", stored, per)
    return {"stored": stored, "per_ticker": per}


def refresh() -> dict:
    """Fetch, store reference data, fill vendor gaps, and cross-check."""
    df = fetch_egx_listing()
    res = store_reference(df)
    res["gap_fill"] = fill_missing_prices(df)
    chk = cross_check_prices(df)
    res.update({"rows": len(df), "compared": len(chk), "flagged": int(chk["flag"].sum())})
    return res
