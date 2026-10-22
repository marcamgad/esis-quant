"""
Arabic financial text corpus for pretraining your own encoder (Agent 4, data side).

WHY. The audit plan's from-scratch encoder needs 20-50M tokens of Arabic
financial text before masked-language pretraining is worth running. On
2026-10-03 the system held ~0.3M words. Training on that would memorise it.

SOURCE. Al Mal (almalnews.com), the Arabic business daily already in the news
feed. Its robots.txt allows all crawlers and its sitemap index lists ~760,000
article URLs (76 x 10,000). At ~250 words an article, ~100,000 articles reach
the target.

POLITE BY CONSTRUCTION. One request per REQUEST_DELAY_S, a per-run cap, newest
first, and the run stops at the first 403/429 instead of retrying. Run nightly;
it resumes where it stopped (already-fetched URLs are skipped).

The corpus is for PRETRAINING ONLY. Article text never feeds a signal, and
evaluation stays forward-only on items first seen after the model's cutoff.

    py -3.11 -m data.corpus_collector              # one run (default 1,500 articles)
    py -3.11 -m data.corpus_collector status
"""

import logging
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.corpus")

SITEMAP_INDEX = "https://almalnews.com/sitemaps.xml"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
REQUEST_DELAY_S = 1.0
TARGET_WORDS = 20_000_000


def _get(url: str, timeout: int = 25) -> str:
    parts = urllib.parse.urlsplit(url)
    safe = urllib.parse.urlunsplit((parts.scheme, parts.netloc, urllib.parse.quote(parts.path),
                                    parts.query, parts.fragment))
    req = urllib.request.Request(safe, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_article(html: str) -> dict:
    title = re.search(r'<meta property="og:title" content="([^"]*)"', html)
    pub = re.search(r'<meta property="article:published_time" content="([^"]*)"', html)
    paras = [re.sub(r"<[^>]+>", " ", p) for p in re.findall(r"<p[^>]*>(.*?)</p>", html, re.S)]
    body = re.sub(r"\s+", " ", unescape(" ".join(paras))).strip()
    return {"title": unescape(title.group(1)) if title else None,
            "published_at": pub.group(1) if pub else None, "body": body}


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS corpus_articles (
        url TEXT PRIMARY KEY, source TEXT, title TEXT, body TEXT, words INTEGER,
        published_at TEXT, fetched_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS corpus_queue (
        url TEXT PRIMARY KEY, kind TEXT, sitemap INTEGER, status TEXT DEFAULT 'todo', queued_at TEXT)""")


# Results articles are found by their headline slug, which Al Mal puts in the URL.
EARNINGS_SLUG = re.compile(r"أرباح|ارباح|خسائر|خسارة|نتائج-أعمال|نتائج-الأعمال|صافي-ربح|صافي-الربح|"
                           r"إيرادات|ايرادات|القوائم-المالية")


def _get_retry(url: str, tries: int = 3, timeout: int = 90) -> str:
    for k in range(tries):
        try:
            return _get(url, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except Exception:  # noqa: BLE001 -- timeouts on the large sitemap files
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def queue_earnings(max_sitemap: int = 45) -> dict:
    """Queue every results article (by headline slug) in sitemaps 1..max_sitemap
    (sitemap 1 = newest; ~45 reaches back to 2016, where price data starts)."""
    subs = re.findall(r"<loc>(.*?)</loc>", _get_retry(SITEMAP_INDEX))[:max_sitemap]
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    n = 0
    for i, sm in enumerate(subs, 1):
        urls = [u for u in re.findall(r"<loc>(.*?)</loc>", _get_retry(sm))
                if EARNINGS_SLUG.search(urllib.parse.unquote(u))]
        with get_connection() as conn:
            _ensure(conn)
            conn.executemany("INSERT OR IGNORE INTO corpus_queue (url, kind, sitemap, queued_at) "
                             "VALUES (?, 'earnings', ?, ?)", [(u, i, now) for u in urls])
        n += len(urls)
        time.sleep(REQUEST_DELAY_S)
    return {"sitemaps": len(subs), "queued": n}


def _ingest_as_news(url: str, a: dict, index, valid) -> int:
    """Store a fetched results article as a news item with company links, so the
    existing extraction and event-study code sees it with its TRUE publication
    time. first_seen_at stays the fetch time: the system did not see it then."""
    from data.news_collector import store_items, link_text, _hash, ROUNDUP_TICKERS
    pub = pd.to_datetime(a["published_at"], errors="coerce", utc=True)
    item = {"url": url, "headline": a["title"] or "", "summary": a["body"][:400],
            "publisher": "almalnews (archive)",
            "published_at": None if pd.isna(pub) else pub.tz_convert("Africa/Cairo").tz_localize(None)}
    store_items([item], feed="almalnews_archive", region="egypt", kind="company", lang="ar")
    found = link_text(item["headline"], index, valid)          # headline only: bodies name many firms
    if not found or len(found) > ROUNDUP_TICKERS:
        return 0
    with get_connection() as conn:
        nid = conn.execute("SELECT id FROM news_items WHERE url_hash = ?", (_hash(url),)).fetchone()[0]
        for t, method in found.items():
            conn.execute("INSERT OR IGNORE INTO news_links (news_id, ticker, method) VALUES (?,?,?)",
                         (nid, t, method))
    return len(found)


def run_earnings(max_articles: int = 2000, max_minutes: float = 120.0) -> dict:
    """Fetch queued results articles, newest first; store text AND a linked news item."""
    from data.news_collector import build_name_index
    t0 = time.time()
    index = build_name_index()
    with get_connection() as conn:
        _ensure(conn)
        valid = {r[0] for r in conn.execute("SELECT ticker FROM stocks")}
        todo = [r[0] for r in conn.execute("SELECT url FROM corpus_queue WHERE status='todo' "
                                           "ORDER BY sitemap, url DESC LIMIT ?", (max_articles,))]
    fetched = linked = 0
    stopped = None
    for u in todo:
        if (time.time() - t0) / 60 > max_minutes:
            break
        status = "done"
        try:
            a = parse_article(_get(u))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                stopped = f"HTTP {e.code} -- stopping politely"
                break
            a, status = None, f"http {e.code}"
        except Exception:  # noqa: BLE001
            a, status = None, "error"
        try:
            if a and a["body"]:
                with get_connection() as conn:
                    conn.execute("INSERT OR IGNORE INTO corpus_articles VALUES (?,?,?,?,?,?,?)",
                                 (u, "almalnews", a["title"], a["body"], len(a["body"].split()),
                                  a["published_at"], pd.Timestamp.now().isoformat(timespec="seconds")))
                linked += bool(_ingest_as_news(u, a, index, valid))
                fetched += 1
            with get_connection() as conn:
                conn.execute("UPDATE corpus_queue SET status=? WHERE url=?", (status, u))
        except sqlite3.OperationalError as e:
            # another job held the write lock past the 30 s busy timeout (the daily
            # run, a labeller session). The URL stays 'todo' and is retried next run.
            if "locked" not in str(e):
                raise
            logger.warning("database locked; leaving %s queued", u)
            time.sleep(30)
        time.sleep(REQUEST_DELAY_S)
    with get_connection() as conn:
        left = conn.execute("SELECT COUNT(*) FROM corpus_queue WHERE status='todo'").fetchone()[0]
    return {"fetched": fetched, "linked_to_a_company": linked, "queue_left": left,
            "minutes": round((time.time() - t0) / 60, 1), "stopped": stopped}


def status() -> dict:
    with get_connection() as conn:
        _ensure(conn)
        n, w = conn.execute("SELECT COUNT(*), COALESCE(SUM(words),0) FROM corpus_articles").fetchone()
    return {"articles": n, "words": w, "progress": w / TARGET_WORDS}


def run(max_articles: int = 1500, max_minutes: float = 45.0) -> dict:
    t0 = time.time()
    with get_connection() as conn:
        _ensure(conn)
        have = {r[0] for r in conn.execute("SELECT url FROM corpus_articles")}
    subs = re.findall(r"<loc>(.*?)</loc>", _get(SITEMAP_INDEX))
    fetched, stopped = 0, None
    for sm in subs:                                           # sitemap1 = newest
        if fetched >= max_articles or (time.time() - t0) / 60 > max_minutes or stopped:
            break
        urls = [u for u in re.findall(r"<loc>(.*?)</loc>", _get(sm)) if u not in have]
        time.sleep(REQUEST_DELAY_S)
        for u in urls:
            if fetched >= max_articles or (time.time() - t0) / 60 > max_minutes:
                break
            try:
                a = parse_article(_get(u))
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    stopped = f"HTTP {e.code} at {u} -- stopping politely"
                    break
                continue
            except Exception:  # noqa: BLE001
                continue
            finally:
                time.sleep(REQUEST_DELAY_S)
            if len(a["body"].split()) < 40:
                continue                                      # galleries, stubs
            with get_connection() as conn:
                conn.execute("INSERT OR IGNORE INTO corpus_articles VALUES (?,?,?,?,?,?,?)",
                             (u, "almalnews", a["title"], a["body"], len(a["body"].split()),
                              a["published_at"], pd.Timestamp.now().isoformat(timespec="seconds")))
            fetched += 1
    return {"fetched": fetched, "minutes": round((time.time() - t0) / 60, 1), "stopped": stopped, **status()}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    if argv[:1] == ["nightly"]:
        with get_connection() as conn:
            _ensure(conn)
            left = conn.execute("SELECT COUNT(*) FROM corpus_queue WHERE status='todo'").fetchone()[0]
        if left:
            print(run_earnings(max_articles=1500, max_minutes=45))
            return 0
        argv = ["1500"]
    if argv[:1] == ["queue-earnings"]:
        print(queue_earnings())
        return 0
    if argv[:1] == ["earnings"]:
        n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 2000
        print(run_earnings(max_articles=n))
        return 0
    if argv[:1] == ["status"]:
        s = status()
    else:
        n = int(argv[0]) if argv and argv[0].isdigit() else 1500
        s = run(max_articles=n)
    print(f"corpus: {s['articles']:,} articles, {s['words']:,} words "
          f"({s['progress']:.1%} of the {TARGET_WORDS/1e6:.0f}M-word pretraining target)"
          + (f" | this run {s['fetched']} in {s['minutes']} min" if "fetched" in s else "")
          + (f" | {s['stopped']}" if s.get("stopped") else ""))
    return 0


if __name__ == "__main__":
    from ops.keep_awake import keep_awake
    with keep_awake():                 # the 2026-10-03 fetch died when the laptop slept
        raise SystemExit(main())
