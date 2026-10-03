"""
Company fundamentals -- the system's first non-price view of a business.

WHY THIS EXISTS. Until now every number in this system was derived from price:
volatility, correlation, momentum, drawdown. Price-only systems cannot
distinguish a company that has doubled because it is earning more from one that
has doubled because it is being bid up, and the EGX book has both. OIH is 21.8%
of the live book, its largest gain, and it has reported NEGATIVE net income for
four consecutive years (-533.78, -450, -184.87, against +592.81 in FY2021) --
a fact the system had no way to hold.

WHY IT IS FREE. stockanalysis.com is already the reference source for names,
market cap and cross-checked prices. Its per-stock pages carry full income
statements, balance sheets and statistics -- 5 fiscal years plus TTM,
server-rendered, no account -- INCLUDING for names Yahoo does not carry at all
(QNBE, VALU, TAQA, UBEE). A paid fundamentals feed was recommended before this
was checked; it was not necessary.

THE LOOKAHEAD PROBLEM, AND WHY available_from EXISTS.
This is the single most dangerous thing about fundamentals in a backtest. FY2025
figures describe the year ending 2025-12-31, but they were not PUBLISHED until
some months later. Joining them to prices by fiscal year lets a backtest trade
on 2025 earnings in January 2025 -- a lookahead so effective it can manufacture
any Sharpe you like, and so quiet that nothing in the output looks wrong.

Every row therefore carries `available_from = period_end + REPORTING_LAG_DAYS`,
and any point-in-time query MUST filter on it rather than on fiscal year. The lag
is deliberately conservative: being a month late with a fact costs a little
signal, while being a month early invents it.

UNITS ARE NOT NORMALISED, on purpose. The source reports in the company's own
scale (COMI revenue reads 139,166 against a true ~139bn EGP, i.e. millions).
Guessing a multiplier per row would be a silent corruption of the worst kind, so
the raw figure and the page's own unit string are stored and the caller decides.
Ratios computed inside one statement (margins, growth, leverage) are unit-free
and safe regardless.
"""

import logging
import re
import time
import urllib.error
import urllib.request

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.fundamentals")

BASE = "https://stockanalysis.com/quote/egx"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}

# EGX listing rules require annual financials within 90 days of year end and
# quarterly within 45. 120 days is past both, so a row is treated as knowable
# only once it comfortably would have been public.
REPORTING_LAG_DAYS = 120
# Q1-Q3 statements: twice the 45-day rule, because EGX companies often file late.
# A fiscal Q4 figure is the annual report seen through the quarterly page and
# keeps the annual lag.
QUARTERLY_LAG_DAYS = 90

FREQUENCIES = {"annual": "", "quarterly": "?p=quarterly"}

# Politeness. ~229 names x 3 pages is ~690 requests; this is a free public site
# already doing us a favour, and hammering it is both rude and the fastest way
# to lose the source.
REQUEST_DELAY_S = 0.7

# "financials/" became a summary page (7 income lines, mixed tables of ratios
# and segments); the full income statement moved to "income-statement/".
STATEMENTS = {
    "income": "financials/income-statement/",
    "balance": "financials/balance-sheet/",
    "cashflow": "financials/cash-flow-statement/",
}

_TAGS = re.compile(r"<[^>]+>")
_NUM = re.compile(r"^-?[\d,]+(\.\d+)?$")


def _clean(s: str) -> str:
    return _TAGS.sub("", s).replace("&nbsp;", " ").replace("&amp;", "&").strip()


def _row_label(cell: str) -> str:
    """First clean label from a header cell.

    The source nests a growth link inside the label cell, so the raw text arrives
    as 'Revenue    Revenue Growth' or 'Net Income    Net Income Growth'. Taking
    the text before the run of whitespace recovers the actual line item; without
    this every row would be keyed by a mangled string and no two tickers would
    agree on field names.
    """
    parts = re.split(r"\s{2,}", cell.strip())
    return parts[0].strip() if parts else cell.strip()


def _to_number(tok: str):
    """'139,166' -> 139166.0 ; '24.41%' -> 0.2441 ; '-' -> None."""
    t = tok.strip()
    if t in {"", "-", "--", "n/a", "N/A"}:
        return None
    pct = t.endswith("%")
    t = t.rstrip("%").replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return v / 100.0 if pct else v


def _fetch(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _parse_table(body: str) -> dict:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S)
    parsed = []
    for r in rows:
        cells = [_clean(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
        cells = [c for c in cells if c != ""]
        if cells:
            parsed.append(cells)
    if not parsed:
        return {}

    header = parsed[0]
    periods = header[1:] if header and header[0].lower().startswith("fiscal") else header
    out = {"periods": periods, "period_ends": [], "rows": {}}
    for cells in parsed[1:]:
        label = _row_label(cells[0])
        values = cells[1:]
        if label.lower().startswith("period ending"):
            # "Jun '26 Jun 30, 2026" -> 2026-06-30
            for v in values:
                m = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})", v)
                out["period_ends"].append(
                    pd.to_datetime(f"{m.group(1)} {m.group(2)} {m.group(3)}").date().isoformat()
                    if m else None)
            continue
        out["rows"][label] = [_to_number(v) for v in values]
    return out


def _parse_statement(html: str) -> dict:
    """{'periods': [...], 'period_ends': [...], 'rows': {label: [values]}}

    ALL TABLES, EACH WITH ITS OWN HEADER. The source splits one statement into
    sections -- the balance sheet is assets / liabilities / equity / debt
    metrics -- and reading only the first table is why this system held total
    assets for 212 companies and not one liability. Tables need not share
    columns, so rows are aligned on period label, not on position. A "Current"
    column is a snapshot as of the scrape date, not a reporting period, and is
    dropped. When a label appears in two tables the first one wins.
    """
    tables = [_parse_table(b) for b in re.findall(r"<table[^>]*>(.*?)</table>", html, re.S)]
    tables = [t for t in tables if t and t["rows"]]
    if not tables:
        return {}
    periods, ends = [], {}
    merged = {}
    for t in tables:
        for i, p in enumerate(t["periods"]):
            if p.lower() == "current":
                continue
            if p not in ends:
                periods.append(p)
                ends[p] = t["period_ends"][i] if i < len(t["period_ends"]) else None
        for label, values in t["rows"].items():
            if label in merged:
                continue
            merged[label] = {p: values[i] for i, p in enumerate(t["periods"])
                             if i < len(values) and p.lower() != "current"}
    return {"periods": periods, "period_ends": [ends[p] for p in periods],
            "rows": {lab: [vals.get(p) for p in periods] for lab, vals in merged.items()}}


def _lag_days(fiscal_period: str) -> int:
    """Q1-Q3 quarters use the quarterly lag; FY, TTM and fiscal Q4 the annual one."""
    p = fiscal_period.strip().upper()
    if p.startswith("Q") and not p.startswith("Q4"):
        return QUARTERLY_LAG_DAYS
    return REPORTING_LAG_DAYS


def frequency_of(fiscal_period: str) -> str:
    return "quarterly" if fiscal_period.strip().upper().startswith("Q") else "annual"


def fetch_ticker(ticker: str, statements=("income", "balance", "cashflow"),
                 frequency: str = "annual") -> dict:
    """All requested statements for one ticker. Missing pages are skipped."""
    out = {"ticker": ticker, "statements": {}}
    for name in statements:
        url = f"{BASE}/{ticker}/{STATEMENTS[name]}{FREQUENCIES[frequency]}"
        try:
            out["statements"][name] = _parse_statement(_fetch(url))
        except urllib.error.HTTPError as e:
            logger.info("fundamentals %s/%s: HTTP %s", ticker, name, e.code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fundamentals %s/%s failed: %s", ticker, name, exc)
        time.sleep(REQUEST_DELAY_S)
    return out


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker         TEXT NOT NULL,
            statement      TEXT NOT NULL,
            fiscal_period  TEXT NOT NULL,
            period_end     TEXT,
            available_from TEXT,
            line_item      TEXT NOT NULL,
            value          REAL,
            fetched_at     TEXT,
            frequency      TEXT NOT NULL DEFAULT 'annual',
            PRIMARY KEY (ticker, statement, fiscal_period, line_item)
        )""")
    # Tables created before quarterly data existed hold annual and TTM rows only.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fundamentals)")}
    if "frequency" not in cols:
        conn.execute("ALTER TABLE fundamentals ADD COLUMN frequency TEXT NOT NULL DEFAULT 'annual'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fund_avail "
                 "ON fundamentals (ticker, available_from)")


def store_ticker(payload: dict) -> int:
    """Periods label their own frequency ('Q2 2026' vs 'FY 2025'), so annual and
    quarterly rows never collide on the primary key."""
    n, now = 0, pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        _ensure_table(conn)
        for stmt, data in payload.get("statements", {}).items():
            periods = data.get("periods") or []
            ends = data.get("period_ends") or []
            for label, values in data.get("rows", {}).items():
                for i, v in enumerate(values):
                    if i >= len(periods):
                        break
                    pe = ends[i] if i < len(ends) else None
                    avail = None
                    if pe:
                        avail = (pd.Timestamp(pe)
                                 + pd.Timedelta(days=_lag_days(periods[i]))).date().isoformat()
                    conn.execute(
                        "INSERT INTO fundamentals (ticker, statement, fiscal_period, "
                        "period_end, available_from, line_item, value, fetched_at, frequency) "
                        "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(ticker, statement, "
                        "fiscal_period, line_item) DO UPDATE SET value=excluded.value, "
                        "period_end=excluded.period_end, "
                        "available_from=excluded.available_from, "
                        "fetched_at=excluded.fetched_at, frequency=excluded.frequency",
                        (payload["ticker"], stmt, periods[i], pe, avail, label, v, now,
                         frequency_of(periods[i])))
                    n += 1
    return n


def refresh(tickers=None, limit: int = None, statements=("income", "balance", "cashflow"),
            frequencies=("annual", "quarterly")) -> dict:
    """Fetch and store fundamentals. Slow by design -- see REQUEST_DELAY_S.
    Both frequencies is 6 pages per ticker, roughly 45 minutes for the universe."""
    if tickers is None:
        with get_connection() as conn:
            tickers = [r[0] for r in conn.execute(
                "SELECT DISTINCT ticker FROM stocks WHERE security_type = 'ordinary' "
                "OR security_type IS NULL ORDER BY ticker")]
    if limit:
        tickers = list(tickers)[:limit]
    stored, failed = 0, []
    for t in tickers:
        got = False
        for freq in frequencies:
            try:
                p = fetch_ticker(t, statements, freq)
                if p["statements"]:
                    stored += store_ticker(p)
                    got = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("fundamentals refresh %s/%s: %s", t, freq, exc)
        if not got:
            failed.append(t)
    return {"tickers": len(tickers), "rows_stored": stored, "failed": failed}


def as_of(ticker: str, date: str, statement: str = "income",
          frequency: str = "annual") -> pd.DataFrame:
    """Fundamentals KNOWABLE on `date` -- the only safe way to query for research.

    Filters on available_from, never on fiscal year. Querying by fiscal year is
    the lookahead this module exists to prevent. `frequency` None returns both.
    """
    with get_connection() as conn:
        _ensure_table(conn)
        sql = ("SELECT * FROM fundamentals WHERE ticker = ? AND statement = ? "
               "AND available_from IS NOT NULL AND available_from <= ?")
        params = [ticker, statement, str(date)]
        if frequency:
            sql += " AND frequency = ?"
            params.append(frequency)
        return pd.read_sql(sql + " ORDER BY period_end DESC", conn, params=params)


def latest(ticker: str, statement: str = "income", frequency: str = "annual") -> pd.DataFrame:
    """Most recent reported figures, ignoring availability. Reporting only --
    never feed this to a backtest."""
    with get_connection() as conn:
        _ensure_table(conn)
        return pd.read_sql(
            "SELECT fiscal_period, period_end, line_item, value FROM fundamentals "
            "WHERE ticker = ? AND statement = ? AND frequency = ? ORDER BY period_end DESC",
            conn, params=(ticker, statement, frequency))


def leverage_as_of(ticker: str, date: str) -> dict:
    """Debt and equity from the latest balance sheet KNOWABLE on `date`.

    Uses quarterly balance sheets when available (a balance sheet is a snapshot,
    so the newest one is the right one) and falls back to annual. Values are in
    the source's own units; the ratios are unit-free.
    """
    bal = as_of(ticker, date, "balance", frequency=None)
    bal = bal[bal["fiscal_period"] != "TTM"]
    if bal.empty:
        return {}
    pe = bal["period_end"].max()
    row = bal[bal["period_end"] == pe].drop_duplicates("line_item").set_index("line_item")["value"]
    get = lambda k: (None if k not in row or pd.isna(row[k]) else float(row[k]))
    debt, equity = get("Total Debt"), get("Shareholders' Equity")
    liab, assets = get("Total Liabilities"), get("Total Assets")
    return {
        "period_end": pe,
        "total_debt": debt, "equity": equity, "total_liabilities": liab, "total_assets": assets,
        "net_cash": get("Net Cash (Debt)"),
        "debt_to_equity": debt / equity if debt is not None and equity and equity > 0 else None,
        "liabilities_to_assets": liab / assets if liab is not None and assets else None,
        "negative_equity": equity is not None and equity <= 0,
    }


def quarterly_yoy(ticker: str, date: str, line_item: str = "Net Income") -> dict:
    """Latest knowable quarter against the same quarter a year earlier.

    Same-quarter comparison removes seasonality, which matters for EGX names
    whose Q4 books annual provisions and tax true-ups.
    """
    q = as_of(ticker, date, "income", frequency="quarterly")
    q = q[q["line_item"] == line_item].dropna(subset=["value", "period_end"])
    if q.empty:
        return {}
    q = q.assign(pe=pd.to_datetime(q["period_end"])).sort_values("pe", ascending=False)
    cur = q.iloc[0]
    prior = q[(q["pe"] - (cur["pe"] - pd.DateOffset(years=1))).abs() <= pd.Timedelta(days=10)]
    out = {"period": cur["fiscal_period"], "period_end": cur["period_end"], "value": float(cur["value"])}
    if not prior.empty:
        pv = float(prior.iloc[0]["value"])
        out["prior_value"] = pv
        out["change"] = (out["value"] - pv) / abs(pv) if pv else None
    return out
