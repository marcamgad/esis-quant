"""
Security master: the identity layer this system has never had.

THE PROBLEM IT SOLVES. Until now the only identifier here was the vendor's
ticker string, and a ticker is not a security. Three failures in September came
directly from that:

  KASABF  bought and modelled as an ordinary share. It is "Certificates Of Odin
          Egyptian Equity Investment Fund-KASAB" -- a FUND holding a basket of
          Egyptian equities, i.e. overlapping exposure to names already held
          directly, priced on NAV rather than continuously traded. Identifying it
          took a web search.
  KORA    16.8% of the live book with zero rows in daily_ohlcv. Establishing why
          (it listed on 2026-06-11, so it has ~65 sessions) took another search.
  MILS +  recommended together as "average pairwise correlation 0.227" while
  AFMC    being North Cairo Flour Mills and Alexandria Flour Mills -- the same
          business, correlated 0.644, stable across every window.

None of those is a modelling failure. All three are a missing answer to "what IS
this thing?", and that question has a canonical answer the system was not
storing.

THE THREE IDENTIFIERS, and why each is here:
  ISIN  the security's legal identity, from the exchange's own listing. Stable
        across renames and vendor spellings, which a ticker is not.
  FIGI  Bloomberg's open identifier, from Twelve Data. The practical key for
        reconciling one vendor against another -- the thing that would let a
        third price source be joined without fuzzy name matching.
  sector the exchange's own classification. No correlation matrix can tell you
        two companies do the same thing; only this can.

WHY TWELVE DATA. Its /stocks endpoint serves all 265 EGX symbols with NO API KEY
and keys them by ISIN, which joins directly to the exchange listing. Prices there
need a key and are deliberately not used: this module is about identity only, so
it has no price path to get wrong.

A NOTE ON THE 31 MISSING NAMES. 31 companies in the exchange listing are absent
from Twelve Data, and 29 are absent from Yahoo -- close to the same set. Two
independent vendors failing on the same names is evidence about those securities
(thin, suspended, or non-standard), not about the vendors, and they are recorded
as `coverage='none'` rather than quietly dropped.
"""

import json
import logging
import urllib.request

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.security_master")

TD_STOCKS_URL = "https://api.twelvedata.com/stocks?exchange=XCAI"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"}


def fetch_twelvedata_listing(timeout: int = 30) -> pd.DataFrame:
    """EGX reference rows from Twelve Data. No API key required for /stocks."""
    req = urllib.request.Request(TD_STOCKS_URL, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8", "replace"))
    rows = payload.get("data", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = [c for c in ("symbol", "name", "type", "figi_code", "cfi_code", "currency")
            if c in df.columns]
    df = df[keep].rename(columns={"symbol": "td_symbol", "name": "td_name",
                                  "type": "td_type", "figi_code": "figi",
                                  "cfi_code": "cfi"})
    logger.info("fetch_twelvedata_listing: %d rows", len(df))
    return df


def _ensure_columns(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(stocks)")}
    for name, decl in (("isin", "TEXT"), ("figi", "TEXT"), ("cfi", "TEXT"),
                       ("security_type", "TEXT"), ("coverage", "TEXT"),
                       ("master_updated_at", "TEXT")):
        if name not in cols:
            conn.execute(f"ALTER TABLE stocks ADD COLUMN {name} {decl}")


def build(store_result: bool = True) -> dict:
    """Join the exchange listing (ISIN + sector) to Twelve Data (FIGI) and store.

    The join is on ISIN, so it is exact. Every earlier attempt to reconcile
    sources here used fuzzy NAME matching and left 79-81 rows unmatched; an
    identifier join has no such failure mode.
    """
    from data.egx_sectors import load_listing, match_to_tickers, classify_security

    listing = load_listing()                       # (name, isin, sector, sec_type)
    td = fetch_twelvedata_listing()
    td_by_isin = {}
    if not td.empty:
        for r in td.itertuples():
            td_by_isin[str(r.td_symbol)] = r       # Twelve Data keys by ISIN

    tickers = match_to_tickers(listing)["matched"]  # ticker -> {name, isin, sector, ...}
    isin_to_ticker = {d["isin"]: t for t, d in tickers.items()}

    records, unmapped = [], []
    for name, isin, sector, sec_type in listing:
        t = isin_to_ticker.get(isin)
        tdrow = td_by_isin.get(isin)
        coverage = "both" if (t and tdrow) else ("vendor_only" if t else
                                                 ("td_only" if tdrow else "none"))
        rec = {"ticker": t, "isin": isin, "name": name, "sector": sector,
               "security_type": sec_type, "coverage": coverage,
               "figi": getattr(tdrow, "figi", None) if tdrow is not None else None,
               "cfi": getattr(tdrow, "cfi", None) if tdrow is not None else None,
               "td_type": getattr(tdrow, "td_type", None) if tdrow is not None else None}
        records.append(rec)
        if t is None:
            unmapped.append(rec)

    df = pd.DataFrame(records)
    out = {"listing_rows": len(listing), "twelvedata_rows": len(td),
           "with_ticker": int(df["ticker"].notna().sum()),
           "with_figi": int(df["figi"].notna().sum()),
           "coverage": df["coverage"].value_counts().to_dict(),
           "unmapped": len(unmapped)}

    if store_result:
        with get_connection() as conn:
            _ensure_columns(conn)
            now = pd.Timestamp.now().isoformat(timespec="seconds")
            n = 0
            for r in records:
                if not r["ticker"]:
                    continue
                # name_english is written from the EXCHANGE listing, and COALESCE
                # keeps any existing value rather than overwriting it. It is
                # included because the research agent's unknown_security check
                # tests exactly this column: after the first build, KASABF had a
                # sector, an ISIN, a FIGI and a security type, and the agent
                # still reported "no reference record" because the one field it
                # looks at was untouched. A master that resolves a security
                # without clearing the alarm about it is half a master.
                conn.execute(
                    "INSERT INTO stocks (ticker, isin, figi, cfi, sector, security_type, "
                    "coverage, name_english, master_updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(ticker) DO UPDATE SET isin=excluded.isin, "
                    "figi=excluded.figi, cfi=excluded.cfi, sector=excluded.sector, "
                    "security_type=excluded.security_type, coverage=excluded.coverage, "
                    "name_english=COALESCE(stocks.name_english, excluded.name_english), "
                    "master_updated_at=excluded.master_updated_at",
                    (r["ticker"], r["isin"], r["figi"], r["cfi"], r["sector"],
                     r["security_type"], r["coverage"], r["name"], now))
                n += 1
            out["stored"] = n
    out["frame"] = df
    return out


def resolve(ticker: str) -> dict:
    """Everything known about one security. The answer to 'what IS this?'."""
    with get_connection() as conn:
        _ensure_columns(conn)
        row = conn.execute(
            "SELECT ticker, name_english, sector, isin, figi, cfi, security_type, "
            "coverage, market_cap_egp FROM stocks WHERE ticker = ?", (ticker,)).fetchone()
    return dict(row) if row else {"ticker": ticker, "known": False}


def non_ordinary_holdings() -> pd.DataFrame:
    """Held positions that are not ordinary shares.

    A fund, an ETF, a rights line and a preferred share each break the risk model
    in a different way, and all four look like a ticker with a price.
    """
    from ops.portfolio_tracker import holdings
    pos, _ = holdings()
    with get_connection() as conn:
        _ensure_columns(conn)
        ref = pd.read_sql("SELECT ticker, security_type, sector, coverage FROM stocks",
                          conn).set_index("ticker")
    m = pos.merge(ref, left_on="ticker", right_index=True, how="left")
    return m[(m["security_type"].notna()) & (m["security_type"] != "ordinary")][
        ["ticker", "security_type", "sector", "value"]]
