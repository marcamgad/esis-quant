"""
EGX official sector classification and ISIN, loaded from the exchange listing.

WHY THIS MATTERS MORE THAN IT LOOKS. `stocks.sector` has been NULL for the whole
life of this system, and the cost of that was concrete: on 2026-09-04 the
optimiser was asked for the best three-name addition and returned MILS + AFMC +
LUTS at "average pairwise correlation 0.227". MILS is North Cairo Flour Mills and
AFMC is Alexandria Flour Mills -- the same business, same wheat input, same
subsidy regime -- correlated 0.644 and stable at 0.62-0.70 across every window
tested. The reported average hid one strongly-linked pair behind two near-zero
ones. A correlation matrix cannot tell you two companies do the same thing; only
a classification can.

SECURITY TYPE IS THE SECOND THING THIS FIXES. The listing contains instruments
that are not ordinary shares at all -- an index ETF, fund certificates,
subscription rights, preferred shares, two SPACs, and dollar-denominated share
classes alongside their EGP twins. KASABF was bought as though it were a share;
it is "Certificates Of Odin Egyptian Equity Investment Fund-KASAB", a fund
holding a basket of Egyptian equities, i.e. overlapping exposure to names already
held directly, priced on NAV rather than continuously traded. Nothing in the
system knew, because nothing recorded what anything WAS.

SOURCE. The exchange's own "All Listed Stocks" table, supplied 2026-09-11:
company name, ISIN, sector. Names are matched to tickers against the
stockanalysis listing; matches below the confidence bar are left UNMATCHED
rather than guessed, for the same reason as in egx_index_membership -- a silent
mis-match mislabels a security with no visible error.
"""

import csv
import difflib
import logging
import os
import re

logger = logging.getLogger("esis_quant.egx_sectors")

_HERE = os.path.dirname(os.path.abspath(__file__))
SECTOR_CSV = os.path.join(_HERE, "egx_sectors.csv")

# Instruments that are NOT ordinary shares. Detected from the listing's own
# wording rather than inferred, and kept separate because each behaves
# differently from a share in a way that breaks the risk model if ignored:
#   fund/certificate -- prices on NAV, holds a basket (overlapping exposure)
#   etf              -- same, plus it tracks an index the book may already hold
#   rights           -- expires; not a going concern
#   preferred        -- different claim, different volatility
#   spac             -- no operating business yet
#   fx_class         -- a second listing of the SAME company in another currency,
#                       so treating both as independent names double-counts it
NON_ORDINARY_PATTERNS = [
    ("etf", r"\bINDEX ETF\b|\bETF\b"),
    ("fund", r"\bFund Certificates\b|\bCertificates Of\b|\bInvestment Fund\b"),
    ("rights", r"\bSubscription Rights\b"),
    ("preferred", r"\bPreferred Shares\b"),
    ("spac", r"\bSPAC\b|\bSpecial Purpose Acquisition\b"),
    ("fx_class", r"\bIn US Dollars\b|\$\s*$"),
]


def classify_security(name: str) -> str:
    """'ordinary' unless the listing's own name says otherwise."""
    for kind, pat in NON_ORDINARY_PATTERNS:
        if re.search(pat, name, flags=re.I):
            return kind
    return "ordinary"


def _norm(s: str) -> str:
    """Aggressive normalisation for matching only -- never for display."""
    s = s.upper()
    s = re.sub(r"&AMP;", "&", s)
    s = re.sub(r"\b(S\.?A\.?E\.?|CO\.?|COMPANY|CORP|HOLDING|HOLDINGS|GROUP|"
               r"FOR|AND|THE|OF|IN|EGYPT|EGYPTIAN|LTD|PLC|INC)\b", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_listing(path: str = None) -> list:
    """[(name, isin, sector, security_type)] from the pipe-delimited listing."""
    path = path or SECTOR_CSV
    out = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="|"):
            if len(row) < 3:
                continue
            name, isin, sector = row[0].strip(), row[1].strip(), row[2].strip()
            if not name or name.lower() == "company name":
                continue
            out.append((name, isin, sector, classify_security(name)))
    return out


# Hand-verified matches the fuzzy pass cannot make, keyed by ISIN because a
# ticker can be reassigned and a name can be spelled three ways but an ISIN is
# the security. Each entry here is a deliberate assertion, not a guess:
#   KORA   -- the listing says "KORRA"; the reference source says "Korra for
#             Energy and Investment Projects". Same ISIN, same company, and the
#             normaliser cannot bridge the one-word name to the seven-word one.
#   KASABF -- absent from the reference source ENTIRELY, so no fuzzy match was
#             ever possible. It is the Odin fund; this is the only record of
#             what it is, and it is the position that prompted this module.
ISIN_OVERRIDES = {
    "EGS07911C018": "KORA",
    "EGS696Z1C017": "KASABF",
}


def match_to_tickers(listing=None, cutoff: float = 0.88) -> dict:
    """Map listing rows onto tickers using the reference source's names.

    `cutoff` is deliberately high. A loose threshold here would attach the wrong
    sector to a stock and then hide a sector concentration behind a plausible
    label, which is worse than leaving the sector blank: a missing value is
    visible and a wrong one is not.
    """
    from data.stockanalysis import fetch_egx_listing
    ref = fetch_egx_listing()
    by_norm = {}
    for t, n in zip(ref["ticker"], ref["name_english"]):
        if isinstance(n, str):
            by_norm.setdefault(_norm(n), t)
    keys = list(by_norm)

    matched, unmatched = {}, []
    for name, isin, sector, sec_type in (listing or load_listing()):
        t = ISIN_OVERRIDES.get(isin)
        if t is None:
            k = _norm(name)
            t = by_norm.get(k)
            if t is None:
                near = difflib.get_close_matches(k, keys, n=1, cutoff=cutoff)
                t = by_norm[near[0]] if near else None
        if t is None:
            unmatched.append((name, isin, sector, sec_type))
        else:
            matched[t] = {"name": name, "isin": isin, "sector": sector,
                          "security_type": sec_type}
    return {"matched": matched, "unmatched": unmatched}


def store(matched: dict = None) -> dict:
    """Write sector / ISIN / security_type into `stocks`."""
    from data.database import get_connection
    if matched is None:
        matched = match_to_tickers()["matched"]
    with get_connection() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(stocks)")}
        if "isin" not in cols:
            conn.execute("ALTER TABLE stocks ADD COLUMN isin TEXT")
        if "security_type" not in cols:
            conn.execute("ALTER TABLE stocks ADD COLUMN security_type TEXT")
        n = 0
        for t, d in matched.items():
            conn.execute(
                "INSERT INTO stocks (ticker, sector, isin, security_type) VALUES (?,?,?,?) "
                "ON CONFLICT(ticker) DO UPDATE SET sector=excluded.sector, "
                "isin=excluded.isin, security_type=excluded.security_type",
                (t, d["sector"], d["isin"], d["security_type"]))
            n += 1
    return {"stored": n}
