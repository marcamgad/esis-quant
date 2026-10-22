"""
News and disclosure collector -- stage 1 of the news-impact pipeline.

THE PIPELINE THIS FEEDS (only stage 1 lives here):
  1. collect    headlines from Egyptian business media, company-specific queries
                and global macro, each stamped with when it was published AND
                when this system first saw it
  2. link       each item to the EGX tickers it is about
  3. assess     is this news favourable or unfavourable for THAT company, given
                its actual reported income, margins and debt?
  4. score      did the assessment turn out right? price reaction after the item
                versus a random-date baseline, kept per event type

Stage 3 is a judgement and stage 4 is what keeps it honest. Without stage 4 a
"favourable" label is an opinion with a timestamp.

WHY THE PREVIOUS COLLECTOR IS NOT ENOUGH. data/collectors.py scrapes almalnews
only. It stored 517 articles between 2026-06-11 and 2026-07-10 and then stopped,
because nothing scheduled it; only 71 (14%) were linked to a ticker, because
linking used the 47-name seed list rather than the 232 names now in `stocks`.

SOURCES, all free and account-less, probed 2026-09-13:
  Enterprise AM Egypt, Daily News Egypt, Egypt Independent -- RSS feeds
  Google News RSS -- per-company queries (English and Arabic) and macro topics.
      The most valuable of these: up to 100 items per query, reaching back
      months, which means the corpus does not start empty.
  almalnews -- existing scraper, migrated into the same table
  EGX disclosures and Mubasher -- blocked / 404; not implemented.

POINT-IN-TIME, AGAIN. Every item stores `published_at` (the publisher's time)
and `first_seen_at` (when this collector first stored it). Back-filled Google
News items are published months ago but first seen today; an event study may use
published_at to reconstruct history, and the live system can only ever have
acted on first_seen_at. Conflating the two is the news version of joining
fundamentals by fiscal year.

LINKING IS CONSERVATIVE ON PURPOSE. A wrong link attaches a stranger's news to a
position and then the impact assessment reasons about the wrong company, which
is worse than an unlinked headline. Links are recorded with their method:
  text   a full company name (English or Arabic) appears in the headline/summary
  code   the ticker appears in an exchange-style form, e.g. "(CASE: OIH)"
  query  the item came back from a query for that company but its text does not
         name it -- kept, but flagged as the weakest evidence
"""

import hashlib
import logging
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.news_collector")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      "Accept": "application/rss+xml,application/xml,*/*"}
REQUEST_DELAY_S = 1.0

RSS_FEEDS = [
    # name, url, region, kind, lang
    ("enterprise_am", "https://enterpriseam.com/egypt/feed/", "egypt", "business", "en"),
    ("daily_news_egypt", "https://www.dailynewsegypt.com/feed/", "egypt", "business", "en"),
    ("egypt_independent", "https://www.egyptindependent.com/feed/", "egypt", "general", "en"),
]

# Macro topics that move the EGX as a whole. Egyptian first, then global drivers
# the EGX is exposed to: devaluation risk, energy, rates, Gulf and IMF flows.
MACRO_QUERIES = [
    ("cbe_rates", "Central Bank of Egypt interest rate", "egypt"),
    ("egp_fx", "Egyptian pound exchange rate", "egypt"),
    ("egypt_inflation", "Egypt inflation", "egypt"),
    ("egypt_imf", "Egypt IMF", "egypt"),
    ("suez_revenue", "Suez Canal revenue", "egypt"),
    ("egx_market", "Egyptian Exchange EGX30", "egypt"),
    ("brent_oil", "Brent crude oil price", "global"),
    ("fed_rates", "Federal Reserve interest rates", "global"),
    ("gold_price", "gold price", "global"),
    ("gulf_investment_egypt", "Gulf investment Egypt", "global"),
]

# Suffixes stripped when building search/link names. Matching "Holding" alone
# would link half the exchange to every holding-company headline.
_GENERIC = r"\b(S\.?A\.?E\.?|Co\.?|Company|Corp\.?|Corporation|Holding|Holdings|Group|For|And|The|Of|Egypt|Egyptian|PLC|Ltd\.?|Inc\.?)\b"


def _norm(s: str) -> str:
    s = re.sub(r"&amp;", "&", s or "")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(_GENERIC, " ", s, flags=re.I)
    s = re.sub(r"[^\w\s&-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _hash(url: str) -> str:
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()


def _fetch(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _gnews_url(q: str, lang: str = "en") -> str:
    if lang == "ar":
        return ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
                + "&hl=ar&gl=EG&ceid=EG:ar")
    return ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
            + "&hl=en-US&gl=EG&ceid=EG:en")


def parse_rss(xml_text: str) -> list:
    """[{headline, url, summary, published_at, publisher}] from an RSS document."""
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("RSS parse failed: %s", exc)
        return out
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        desc = re.sub(r"<[^>]+>", " ", item.findtext("description") or "")
        desc = re.sub(r"\s+", " ", desc).strip()
        pub = item.findtext("pubDate")
        published = None
        if pub:
            try:
                published = parsedate_to_datetime(pub).astimezone(tz=None).replace(tzinfo=None)
            except (TypeError, ValueError):
                published = None
        src_el = item.find("source")
        publisher = (src_el.text or "").strip() if src_el is not None and src_el.text else None
        # Google News appends " - Publisher" to titles; keep the headline clean.
        if publisher and title.endswith(" - " + publisher):
            title = title[: -len(" - " + publisher)].strip()
        out.append({"headline": title, "url": link, "summary": desc[:1000],
                    "published_at": published, "publisher": publisher})
    return out


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_items (
            id            INTEGER PRIMARY KEY,
            url_hash      TEXT NOT NULL UNIQUE,
            url           TEXT NOT NULL,
            feed          TEXT NOT NULL,
            publisher     TEXT,
            region        TEXT,
            kind          TEXT,
            lang          TEXT,
            query         TEXT,
            headline      TEXT NOT NULL,
            summary       TEXT,
            published_at  TEXT,
            first_seen_at TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_links (
            news_id  INTEGER NOT NULL,
            ticker   TEXT NOT NULL,
            method   TEXT NOT NULL,
            PRIMARY KEY (news_id, ticker)
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_pub ON news_items (published_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_links_ticker ON news_links (ticker)")


def store_items(items: list, feed: str, region: str, kind: str, lang: str,
                query: str = None, query_ticker: str = None) -> int:
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    new = 0
    with get_connection() as conn:
        _ensure_tables(conn)
        for it in items:
            cur = conn.execute(
                "INSERT OR IGNORE INTO news_items (url_hash, url, feed, publisher, region, kind, "
                "lang, query, headline, summary, published_at, first_seen_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (_hash(it["url"]), it["url"], feed, it.get("publisher"), region, kind, lang,
                 query, it["headline"], it.get("summary"),
                 it["published_at"].isoformat(timespec="seconds") if it.get("published_at") else None,
                 now))
            if cur.rowcount:
                new += 1
            if query_ticker:
                nid = conn.execute("SELECT id FROM news_items WHERE url_hash = ?",
                                   (_hash(it["url"]),)).fetchone()[0]
                conn.execute("INSERT OR IGNORE INTO news_links (news_id, ticker, method) "
                             "VALUES (?,?,'query')", (nid, query_ticker))
    return new


# --------------------------------------------------------------------------
# Linking
# --------------------------------------------------------------------------

# Only LEGAL suffixes are stripped when building a link name. The first version
# also stripped "Holding", "Group" and "Egypt", which turned "Valmore Holding"
# into "Valmore" (it then matched an obituary and a man named Valmore Vachon) and
# "Telecom Egypt Company" into "Telecom" (it matched an unrelated US telecom IPO).
# Those words are part of how a company is actually named in a headline.
_LEGAL = r"\b(S\.?A\.?E\.?|S\.?A\.?|Co\.?|Company|Corp\.?|Corporation|PLC|Ltd\.?|Inc\.?|Free Zone)\b"

# An item that links more than this many tickers is a market round-up or a price
# table ("EGX closes: COMI up, SWDY down, ...") and says nothing specific about any
# one of them. Its links are kept but marked 'roundup' so impact assessment can
# exclude them.
ROUNDUP_TICKERS = 6


def _link_name(s: str) -> str:
    s = re.sub(r"&amp;", "&", s or "")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(_LEGAL, " ", s, flags=re.I)
    s = re.sub(r"\b(for|and|of|the)\s*$", " ", s.strip(), flags=re.I)
    s = re.sub(r"[^\w\s&'-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_name_index() -> list:
    """[(pattern, ticker, kind)] from every name the system knows.

    English names come from `stocks` (reference source + exchange listing);
    Arabic names from config.ARABIC_COMPANY_NAMES. An English link name must be
    at least TWO words: every single-word name tried here ("Valmore", "Telecom",
    "Delta") matched something that was not the company.
    """
    idx = []
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT ticker, name_english FROM stocks WHERE name_english IS NOT NULL").fetchall()
    seen = set()
    for t, name in rows:
        n = _link_name(name)
        if len(n.split()) < 2:
            continue
        key = (n.lower(), t)
        if key in seen:
            continue
        seen.add(key)
        idx.append((re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)", re.I), t, "text"))
    arabic_names = dict(getattr(__import__("config"), "ARABIC_COMPANY_NAMES", {}) or {})
    for t, aliases in QUERY_ALIASES.items():
        for a in aliases:
            arabic_names.setdefault(a, t)
    pairs = list(arabic_names.items())
    # Full and short names from Mubasher (data/arabic_names.py), '|'-joined.
    with get_connection() as conn:
        for t, joined in conn.execute(
                "SELECT ticker, name_arabic FROM stocks WHERE name_arabic IS NOT NULL AND name_arabic <> ''"):
            pairs += [(a, t) for a in joined.split("|") if a]
    seen = set()
    for a, t in pairs:
        if (a, t) in seen:
            continue
        seen.add((a, t))
        if not re.search(r"[؀-ۿ]", a):
            if len(a.split()) >= 2:
                idx.append((re.compile(r"(?<!\w)" + re.escape(a) + r"(?!\w)", re.I), t, "text"))
            continue
        if len(a) < 3:
            continue
        if len(a.split()) >= 2 or a in DISTINCT_ARABIC:
            idx.append((re.compile(r"(?<![؀-ۿ])" + _arabic_pattern(a) + r"(?![؀-ۿ])"), t, "text"))
        else:
            # A one-word Arabic name is often an ordinary word too: "فوري"
            # (Fawry) means "immediate", and linked a Luxor governor's order about
            # "immediate handling of violations" to FWRY. Arabic business press
            # sets company names in quotation marks, so a single word only counts
            # when quoted.
            idx.append((re.compile(r"[«\"“']\s*" + _arabic_pattern(a) + r"\s*[»\"”']"), t, "text"))
    return idx


# Spellings Arabic press varies freely: hamza on alef, taa marbuta vs haa, final
# yaa vs alef maqsura, "و" attached or spaced ("و التنمية" / "والتنمية").
_AR_VARIANTS = {"ا": "[اأإآ]", "أ": "[اأإآ]", "إ": "[اأإآ]", "آ": "[اأإآ]",
                "ة": "[ةه]", "ه": "[ةه]", "ى": "[ىي]", "ي": "[ىي]"}


def _arabic_pattern(name: str) -> str:
    pat = ""
    words = name.split()
    for i, w in enumerate(words):
        pat += "".join(_AR_VARIANTS.get(ch, re.escape(ch)) for ch in w)
        if i < len(words) - 1:
            pat += r"\s*" if w == "و" else r"\s+"       # "و التنمية" == "والتنمية"
    return pat


# One-word Arabic names distinctive enough to match unquoted.
DISTINCT_ARABIC = {"أموك"}

# Text that contains a company's name but is about a different entity.
LINK_EXCLUDE = {
    # "Catalyst Partners" (CPME) vs Abu Dhabi Catalyst Partners, an unrelated firm
    "CPME": [re.compile(r"Abu Dhabi Catalyst", re.I)],
}


_CODE = re.compile(r"(?:CASE|EGX)\s*[:\-]\s*([A-Z]{3,6})\b|\(([A-Z]{3,6})\.CA\)")


def link_text(text: str, index: list, valid_tickers: set) -> dict:
    """{ticker: method} for a piece of text."""
    found = {}
    for pat, t, kind in index:
        if pat.search(text) and not any(x.search(text) for x in LINK_EXCLUDE.get(t, ())):
            found.setdefault(t, kind)
    for m in _CODE.finditer(text):
        t = m.group(1) or m.group(2)
        if t in valid_tickers:
            found.setdefault(t, "code")
    return found


def relink_all() -> dict:
    """Re-run text/code linking over every stored item. Safe to repeat."""
    index = build_name_index()
    with get_connection() as conn:
        _ensure_tables(conn)
        # Derived links are rebuilt from scratch so a tightened matcher also
        # REMOVES the links the old one got wrong. Query links are facts about
        # how an item was fetched and are kept.
        conn.execute("DELETE FROM news_links WHERE method IN ('text', 'code', 'roundup')")
        valid = {r[0] for r in conn.execute("SELECT ticker FROM stocks")}
        items = conn.execute("SELECT id, headline, summary FROM news_items").fetchall()
        n = roundups = 0
        for nid, h, s in items:
            found = link_text(f"{h} {s or ''}", index, valid)
            if len(found) > ROUNDUP_TICKERS:
                found = {t: "roundup" for t in found}
                roundups += 1
            for t, method in found.items():
                cur = conn.execute(
                    "INSERT INTO news_links (news_id, ticker, method) VALUES (?,?,?) "
                    "ON CONFLICT(news_id, ticker) DO UPDATE SET method = excluded.method",
                    (nid, t, method))
                n += cur.rowcount
    return {"items": len(items), "links_written": n, "roundup_items": roundups}


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def collect_feeds() -> dict:
    out = {}
    for name, url, region, kind, lang in RSS_FEEDS:
        try:
            out[name] = store_items(parse_rss(_fetch(url)), name, region, kind, lang)
        except Exception as exc:  # noqa: BLE001
            logger.warning("feed %s failed: %s", name, exc)
            out[name] = f"error: {exc}"
        time.sleep(REQUEST_DELAY_S)
    return out


def collect_macro() -> dict:
    out = {}
    for key, q, region in MACRO_QUERIES:
        try:
            out[key] = store_items(parse_rss(_fetch(_gnews_url(q))), "gnews_macro",
                                   region, "macro", "en", query=key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("macro query %s failed: %s", key, exc)
            out[key] = f"error: {exc}"
        time.sleep(REQUEST_DELAY_S)
    return out


# Names the press actually uses, where the legal name returns nothing as an exact
# phrase. First run: "Korra for Energy and Investment Projects", "Creast Mark For
# Contracting And Real Estate Development" and the Odin fund certificate name each
# returned ZERO items, while the positions were 14-21% of the live book.
QUERY_ALIASES = {
    "KORA": ["Korra Energi", "كورا للطاقة"],
    "CRST": ["Creast Mark", "كريست مارك"],
    "KASABF": ["Odin Investments fund", "صندوق أودن"],
    "OIH": ["Orascom Investment"],
    "AMOC": ["Alexandria Mineral Oils", "أموك"],
    "TALM": ["Taaleem Management"],
    "MOED": ["Egyptian Modern Education"],
    "GTWL": ["Golden Textiles"],
    "GSSC": ["General Silos"],
}


def company_targets(extra: int = 25) -> list:
    """Held names always, plus a rotating slice of the rest by market cap.

    Rotation keeps each run to a polite number of queries while every listed
    company is still visited every few days.
    """
    with get_connection() as conn:
        held = [r[0] for r in conn.execute("SELECT ticker FROM user_positions")]
        rest = [r[0] for r in conn.execute(
            "SELECT ticker FROM stocks WHERE name_english IS NOT NULL "
            "AND (security_type IS NULL OR security_type = 'ordinary') "
            "ORDER BY COALESCE(market_cap_egp, 0) DESC")]
    rest = [t for t in rest if t not in held]
    if not rest:
        return held
    start = (pd.Timestamp.now().dayofyear * extra) % len(rest)
    return held + (rest[start:] + rest[:start])[:extra]


def collect_companies(tickers=None, arabic: bool = True) -> dict:
    tickers = tickers or company_targets()
    import config
    ar_by_ticker = {}
    for ar, t in getattr(config, "ARABIC_COMPANY_NAMES", {}).items():
        if len(ar) >= 4 and re.search(r"[؀-ۿ]", ar):
            ar_by_ticker.setdefault(t, ar)
    with get_connection() as conn:
        names = dict(conn.execute(
            "SELECT ticker, name_english FROM stocks WHERE name_english IS NOT NULL").fetchall())
    out = {}
    for t in tickers:
        queries = []
        for alias in QUERY_ALIASES.get(t, []):
            queries.append((alias, "ar" if re.search(r"[؀-ۿ]", alias) else "en"))
        name = _norm(names.get(t, ""))
        if name and not any(lang == "en" for _, lang in queries):
            queries.append((name, "en"))
        got = 0
        for q, lang in queries:
            try:
                got += store_items(parse_rss(_fetch(_gnews_url(f'"{q}"', lang))), "gnews_company",
                                   "egypt", "company", lang, query=t, query_ticker=t)
            except Exception as exc:  # noqa: BLE001
                logger.warning("company query %s (%s) failed: %s", t, q, exc)
            time.sleep(REQUEST_DELAY_S)
        if arabic and t in ar_by_ticker and not any(lang == "ar" for _, lang in queries):
            try:
                got += store_items(parse_rss(_fetch(_gnews_url(ar_by_ticker[t], "ar"))),
                                   "gnews_company", "egypt", "company", "ar",
                                   query=t, query_ticker=t)
            except Exception as exc:  # noqa: BLE001
                logger.warning("arabic company query %s failed: %s", t, exc)
            time.sleep(REQUEST_DELAY_S)
        out[t] = got
    return out


def migrate_almalnews() -> int:
    """Copy the legacy news_articles corpus into news_items (idempotent)."""
    with get_connection() as conn:
        _ensure_tables(conn)
        try:
            rows = conn.execute("SELECT url, headline_arabic, body_arabic, published_at "
                                "FROM news_articles WHERE url IS NOT NULL").fetchall()
        except Exception:  # noqa: BLE001
            return 0
    items = []
    for url, h, body, pub in rows:
        if not h:
            continue
        items.append({"url": url, "headline": h, "summary": (body or "")[:1000],
                      "published_at": pd.Timestamp(pub).to_pydatetime() if pub else None,
                      "publisher": "almalnews"})
    return store_items(items, "almalnews", "egypt", "business", "ar")


def run(companies: bool = True) -> dict:
    res = {"feeds": collect_feeds(), "macro": collect_macro()}
    if companies:
        res["companies"] = collect_companies()
    res["almalnews_migrated"] = migrate_almalnews()
    res["relink"] = relink_all()
    with get_connection() as conn:
        res["totals"] = {
            "items": conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0],
            "linked_items": conn.execute("SELECT COUNT(DISTINCT news_id) FROM news_links").fetchone()[0],
            "tickers_with_news": conn.execute("SELECT COUNT(DISTINCT ticker) FROM news_links").fetchone()[0],
        }
    return res


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    r = run()
    t = r["totals"]
    print(f"news items {t['items']} | linked {t['linked_items']} | tickers with news {t['tickers_with_news']}")
    print("feeds:", r["feeds"])
    print("macro:", r["macro"])
    print("companies:", {k: v for k, v in r.get("companies", {}).items()})
    print("almalnews migrated:", r["almalnews_migrated"], "| relink:", r["relink"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
