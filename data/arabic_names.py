"""
Arabic company names for news linking.

WHY. Until 2026-10-03 the linker knew 66 hand-entered Arabic names, and
`stocks.name_arabic` was empty. Of the first 4,991 Al Mal archive results
articles only 252 (5%) linked to a listed company: «الإسكندرية لتداول
الحاويات» (ALCN) and «أسيوط لتكرير البترول» went unmatched.

SOURCE. Mubasher (mubasher.info), free, no account:
  * /api/1/listed-companies?country=eg  -> the short name the press uses ("سي أي بي")
  * /markets/EGX/stocks/{T}/profile      -> the full name, in the page title

Each ticker gets several aliases (full name, full name without legal suffixes,
the parenthesised trade name, the short name), stored '|'-joined in
stocks.name_arabic. news_collector.build_name_index reads them; one-word aliases
there still only match inside quotation marks.

    py -3.11 -m data.arabic_names        # refresh names, then relink all news
"""

import json
import re
import time
import urllib.request
from html import unescape

from data.database import get_connection

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
LIST_URL = "https://www.mubasher.info/api/1/listed-companies?country=eg&size=400&start=0"
PROFILE_URL = "https://www.mubasher.info/markets/EGX/stocks/{}/profile"
TITLE = re.compile(r"<title>\s*بيانات شركة\s*(.*?)\s*-\s*معلومات مباشر", re.S)
_LEGAL_AR = re.compile(r"\bش\s*\.?\s*م\s*\.?\s*م\b\.?|\bش\s*\.?\s*م\s*\.?\s*[مE]\b|^شركة\s+")


def _get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def aliases(full: str, short: str = None) -> list:
    """Every form a headline is likely to use, longest first."""
    out = []
    for s in (full, short):
        if not s:
            continue
        s = unescape(s).replace('"', " ")
        inner = re.findall(r"\((.*?)\)", s)
        base = re.sub(r"\(.*?\)", " ", s)
        for v in [base, *inner]:
            v = _LEGAL_AR.sub(" ", v)
            v = re.sub(r"[-–]\s*مصر\s*$", " ", v.strip())       # "- مصر" suffix
            v = re.sub(r"\s+", " ", v).strip(" -–")
            if len(v) >= 3:
                out.append(v)
            if v.endswith(" مصر") and len(v.split()) > 2:
                out.append(v[:-4].strip())
    seen, uniq = set(), []
    for a in sorted(out, key=len, reverse=True):
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def refresh(delay_s: float = 0.7) -> dict:
    short = {r["symbol"]: r["name"] for r in json.loads(_get(LIST_URL))["rows"]}
    with get_connection() as conn:
        tickers = [r[0] for r in conn.execute("SELECT ticker FROM stocks")]
    names, missing = {}, []
    for t in tickers:
        full = None
        try:
            m = TITLE.search(_get(PROFILE_URL.format(t)))
            full = m.group(1).strip() if m else None
        except Exception:  # noqa: BLE001 -- delisted names have no profile
            pass
        a = aliases(full, short.get(t))
        if a:
            names[t] = "|".join(a)
        else:
            missing.append(t)
        time.sleep(delay_s)
    with get_connection() as conn:
        conn.executemany("UPDATE stocks SET name_arabic=? WHERE ticker=?",
                         [(v, t) for t, v in names.items()])
    return {"named": len(names), "missing": missing}


def main() -> int:
    from data.news_collector import relink_all
    print(refresh())
    print(relink_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
