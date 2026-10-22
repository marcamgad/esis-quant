"""
Layer 1 data collectors: EGX prices, macro daily series, Arabic news.

Every fetch is wrapped so one bad ticker/request can't abort a batch job --
failures are logged and skipped, never raised out of the batch methods.
"""

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.collectors")


class FetchError(RuntimeError):
    """A data fetch genuinely FAILED (network/API), as opposed to succeeding
    and returning no rows. Kept distinct so callers can alert on failures
    instead of mistaking them for 'this ticker has no data' (audit fix
    FETCH-01)."""

_TBILL_DISCOUNT_RATE_PATTERN = re.compile(r"معدل\s*خصم\s*(\d+(?:\.\d+)?)\s*%")


def extract_tbill_discount_rate(text: str) -> list:
    """Extract Egyptian Treasury-bill discount rate mentions (e.g. the
    phrase "بمعدل خصم 24.7%") from Arabic financial text. Confirmed
    working against a real almalnews.com headline (2026-07-07): correctly
    matched "... بمعدل خصم 24.7% ..." and did NOT false-positive on other
    articles that merely contain the word "خصم" (discount/deduction) in
    an unrelated context, since the full "معدل خصم <number>%" phrase must
    appear together.

    This is NOT a substitute for the yield-spread HMM feature dropped in
    models/regime_features (see that module's docstring) -- it only fires
    when a scraped article happens to mention a T-bill auction result, so
    coverage is sparse/event-driven, not a continuous daily series. Real
    data, useful as an occasional reference point; not training data.
    """
    return [float(m) for m in _TBILL_DISCOUNT_RATE_PATTERN.findall(text)]


class DataCollector:
    def __init__(self):
        self.session_headers = {"User-Agent": config.SCRAPE_USER_AGENT}

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------
    def fetch_egx_prices(self, ticker: str, start_date: str, end_date: str = None,
                         retries: int = None, backoff: float = None) -> pd.DataFrame:
        """Fetch daily OHLCV for one EGX ticker from Yahoo Finance.

        ticker: bare EGX ticker (e.g. "COMI"), ".CA" suffix is added here.
        Returns a DataFrame indexed by date with columns:
          open, high, low, close, volume, value_traded_egp, adjusted_close

        AUDIT FIX (FETCH-01): this used to swallow every exception and return
        an empty DataFrame, making a NETWORK/API FAILURE indistinguishable from
        "this ticker legitimately has no data" -- so a broken yfinance session
        silently shrank the universe instead of alerting anyone. Now:
          * transient failures are RETRIED with exponential backoff, and
          * an exhausted retry raises FetchError (a real failure), while
          * a successful call that genuinely returns no rows returns an empty
            frame (a real, different fact).
        Batch callers (fetch_all_historical) catch FetchError per ticker and
        report it as a FAILURE, not as a skip.
        """
        retries = config.FETCH_RETRIES if retries is None else retries
        backoff = config.FETCH_BACKOFF_SECONDS if backoff is None else backoff
        symbol = f"{ticker}{config.YFINANCE_SUFFIX}"

        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                hist = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=False)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    wait = backoff * (2 ** (attempt - 1))
                    logger.warning("fetch_egx_prices(%s) attempt %d/%d failed: %s -- retrying in %.1fs",
                                   symbol, attempt, retries, exc, wait)
                    time.sleep(wait)
        else:
            raise FetchError(f"fetch_egx_prices({symbol}) failed after {retries} attempts: {last_exc}")

        if hist.empty:
            # NOT an error: the call succeeded and the ticker has no rows here.
            logger.warning("fetch_egx_prices(%s) returned no rows (call succeeded)", symbol)
            return pd.DataFrame()

        hist = hist.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )
        hist.index = hist.index.tz_localize(None).normalize()
        hist["value_traded_egp"] = hist["close"] * hist["volume"]
        return hist[["open", "high", "low", "close", "volume", "value_traded_egp", "adjusted_close"]]

    def fetch_all_historical(self, tickers=None, start_date=None, end_date=None):
        """Fetch and store history for every ticker in `tickers` (defaults
        to config.EGX_SEED_TICKERS). NOTHING is forward-filled -- only
        dates Yahoo Finance actually returned a row for are stored; every
        other date is simply absent from daily_ohlcv for that ticker.

        This used to forward-fill short (<= MAX_FORWARD_FILL_DAYS) gaps.
        Two real bugs found that behavior, not just one:
          1. Reindexing onto a naive Mon-Fri calendar fabricated a phantom
             Friday price for EGX (which trades Sun-Thu) -- fixed earlier
             by switching to an EGX-specific trading calendar.
          2. Even with the right calendar, forward-filling still produces
             a mechanically-EXACT copy of the prior day's row. Confirmed
             on real data (2026-07-07 fetch): Yahoo's raw feed had a
             single-day hole in the MIDDLE of otherwise-contiguous real
             data for ABUK.CA (nothing between 2026-07-01 and 07-05,
             despite EGX being open on Thursday 07-02) -- a reporting lag,
             not evidence of no trading. Forward-filling manufactured a
             specific close price AND volume for that day, which in turn
             showed up as an exact 0.000000 entry in every downstream
             return-based calculation (market proxy, momentum, Amihud) --
             indistinguishable from a real, confirmed zero-return day.
        Rather than patch this a third time (e.g. filling price but not
        volume, tried and still wrong -- a flat price still fakes a zero
        return), the fix is to stop forward-filling at all. Every
        consumer of this data (models/regime_features.py,
        models/factor_returns.py) already treats missing dates correctly
        via coverage thresholds/dropna/skipna -- a ticker simply absent on
        a date is handled the same way whether that's a genuine holiday,
        a halt, or a Yahoo reporting gap, without ever claiming to know a
        price or volume we don't actually have.

        Gap lengths are still logged (informational -- worth knowing if a
        ticker looks suspended), just no longer acted on.

        NOTE: this does not yet reconcile corporate_actions against Yahoo's
        own adjustment -- for the seed tickers tested, Yahoo's adjusted
        close already accounts for splits/dividends, so double-adjustment
        would be wrong. Layer 1 stores Yahoo's adjustment as-is; a future
        pass should verify it against manually-sourced corporate_actions
        rows before this is relied on for backtesting.
        """
        tickers = tickers if tickers is not None else [t["ticker"] for t in config.EGX_SEED_TICKERS]
        start_date = start_date or config.HISTORY_START_DATE
        fetched, skipped, failed = 0, [], []

        for ticker in tickers:
            # AUDIT FIX (FETCH-01): distinguish a genuine FAILURE (raises, after
            # retries) from an empty-but-successful fetch. Failures are collected
            # and surfaced as an error at the end -- previously both looked like
            # a silent "skip" and the universe just quietly shrank.
            try:
                df = self.fetch_egx_prices(ticker, start_date, end_date)
            except FetchError as exc:
                logger.error("fetch_all_historical: %s FAILED: %s", ticker, exc)
                failed.append(ticker)
                continue
            if df.empty:
                skipped.append(ticker)
                continue

            full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
            # EGX trades Sunday-Thursday, closed Friday/Saturday (Python
            # weekday: Mon=0 ... Sun=6).
            egx_trading_days = full_index[~full_index.weekday.isin([4, 5])]
            missing = egx_trading_days.difference(df.index)
            if len(missing):
                logger.info("%s: %d/%d expected EGX trading days have no Yahoo Finance data "
                            "(reporting gaps or halts, not filled)",
                            ticker, len(missing), len(egx_trading_days))

            self._store_ohlcv(ticker, df)
            fetched += 1

        if failed:
            # Loud, not silent: a fetch failure means the universe we are about
            # to trade is NOT the universe we think it is.
            logger.error("fetch_all_historical: %d/%d tickers FAILED to fetch (%s) -- "
                         "downstream universe is incomplete; do NOT trade on this run "
                         "without re-fetching", len(failed), len(tickers), failed)
        logger.info("fetch_all_historical: %d/%d tickers stored, skipped(no data)=%s, failed=%s",
                    fetched, len(tickers), skipped, failed)
        return {"fetched": fetched, "skipped": skipped, "failed": failed,
                "ok": len(failed) == 0}

    @staticmethod
    def _ensure_source_column(conn):
        """Add daily_ohlcv.source in place for databases created before marks
        were attributed. Existing rows default to 'vendor', which is the honest
        value: they came from the price feed and nothing else is known."""
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(daily_ohlcv)")}
        if "source" not in cols:
            conn.execute("ALTER TABLE daily_ohlcv ADD COLUMN source TEXT DEFAULT 'vendor'")

    def _store_ohlcv(self, ticker: str, df: pd.DataFrame):
        # A bar with no close is not a price (Stage 0). One such row for MFSC on
        # 2026-10-01 became the newest date in the table, the universe was read
        # on a day only that name "traded", and it collapsed to 1 name.
        if len(df) and {"close", "adjusted_close"} <= set(df.columns):
            bad = df["close"].isna() | df["adjusted_close"].isna()
            if bad.any():
                logger.warning("_store_ohlcv: %s dropped %d bar(s) with no close: %s", ticker,
                               int(bad.sum()), [d.strftime("%Y-%m-%d") for d in df.index[bad]])
                df = df[~bad]
        with get_connection() as conn:
            self._ensure_source_column(conn)
            broker_dates = set()
            if len(df):
                lo = df.index.min().strftime("%Y-%m-%d")
                hi = df.index.max().strftime("%Y-%m-%d")
                # BROKER MARKS ARE NOT THE VENDOR'S TO OVERWRITE (fix 2026-08-18).
                # These are prices the operator read off their actual broker
                # account -- the only VERIFIED prices in the database. On
                # 2026-08-16 the vendor returned a carried-forward zero-volume
                # bar for every ticker and the delete-then-replace below wiped
                # all nine broker marks, silently restoring stale Aug-13 prices
                # over real ones. Verified data outranks vendor data, always.
                broker_dates = {r["date"] for r in conn.execute(
                    "SELECT date FROM daily_ohlcv WHERE ticker = ? AND date BETWEEN ? AND ? "
                    "AND source = 'broker'", (ticker, lo, hi))}
                # INSERT OR REPLACE only overwrites rows for dates present
                # in `df` -- it can never remove a row for a date that USED
                # to be there (e.g. a stale forward-filled row from before
                # forward-filling was removed from fetch_all_historical, or
                # any date Yahoo has since retracted). Delete the covered
                # range first so the table always reflects exactly what the
                # latest fetch returned, not an accumulation of past runs.
                # sa_history rows (dated stockanalysis backfill) are also kept: a
                # vendor row on the same date still replaces them below, but a date
                # the vendor simply lacks keeps its backfilled price.
                conn.execute(
                    "DELETE FROM daily_ohlcv WHERE ticker = ? AND date BETWEEN ? AND ? "
                    "AND (source IS NULL OR source NOT IN ('broker', 'sa_history'))",
                    (ticker, lo, hi),
                )
                if broker_dates:
                    df = df[~df.index.strftime("%Y-%m-%d").isin(broker_dates)]
                    logger.info("_store_ohlcv: %s kept %d broker mark(s), vendor rows skipped for %s",
                                ticker, len(broker_dates), sorted(broker_dates))
            rows = [
                (
                    ticker,
                    idx.strftime("%Y-%m-%d"),
                    float(r["open"]) if pd.notna(r["open"]) else None,
                    float(r["high"]) if pd.notna(r["high"]) else None,
                    float(r["low"]) if pd.notna(r["low"]) else None,
                    float(r["close"]) if pd.notna(r["close"]) else None,
                    float(r["volume"]) if pd.notna(r["volume"]) else None,
                    float(r["value_traded_egp"]) if pd.notna(r["value_traded_egp"]) else None,
                    float(r["adjusted_close"]) if pd.notna(r["adjusted_close"]) else None,
                )
                for idx, r in df.iterrows()
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO daily_ohlcv
                   (ticker, date, open, high, low, close, volume, value_traded_egp,
                    adjusted_close, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'vendor')""",
                rows,
            )
            avg_vol = df["value_traded_egp"].tail(30).mean()
            if pd.notna(avg_vol):
                conn.execute(
                    "UPDATE stocks SET avg_30d_volume_egp = ? WHERE ticker = ?",
                    (float(avg_vol), ticker),
                )

    # ------------------------------------------------------------------
    # Macro data
    # ------------------------------------------------------------------
    def fetch_macro_daily(self, start_date: str = None, end_date: str = None):
        """Fetch USD/EGP, Brent, gold, natgas, US 10Y yield, and an EGX30
        proxy, then upsert into macro_daily keyed by date.

        egx30_level/egx30_volume are best-effort: Yahoo has no working
        historical series for the EGX30 index under any symbol tried (see
        config.EGX30_PROXY_TICKER docstring note) -- ^CASE30 only returns
        a same-day snapshot, so historical backfill for those two columns
        will mostly be NULL until a better source is wired in.

        cbe_overnight_rate, egypt_10y_yield, egypt_cds_5y, global_risk_index
        have no source implemented in Layer 1 and are left NULL.
        """
        start_date = start_date or config.HISTORY_START_DATE
        series = {}
        for col, symbol in config.MACRO_YFINANCE_TICKERS.items():
            try:
                h = yf.Ticker(symbol).history(start=start_date, end=end_date)
                if not h.empty:
                    h.index = h.index.tz_localize(None).normalize()
                    series[col] = h["Close"]
                else:
                    logger.warning("fetch_macro_daily: %s (%s) returned no rows", col, symbol)
            except Exception as exc:
                logger.warning("fetch_macro_daily: %s (%s) failed: %s", col, symbol, exc)

        try:
            h = yf.Ticker(config.EGX30_PROXY_TICKER).history(start=start_date, end=end_date)
            if not h.empty:
                h.index = h.index.tz_localize(None).normalize()
                series["egx30_level"] = h["Close"]
                series["egx30_volume"] = h["Volume"]
        except Exception as exc:
            logger.warning("fetch_macro_daily: EGX30 proxy failed: %s", exc)

        if not series:
            logger.error("fetch_macro_daily: no series fetched, aborting")
            return {"stored": 0}

        merged = pd.DataFrame(series)
        self._cross_check_usd_egp_latest(merged)
        self._store_macro(merged)
        egx = self.fetch_egx30_history()
        return {"stored": len(merged), "egx30": egx}

    def _cross_check_usd_egp_latest(self, merged: pd.DataFrame):
        """Best-effort sanity check of the latest USD/EGP against a second
        free source. Logs a warning on large divergence; never raises.
        """
        if "usd_egp" not in merged.columns or merged["usd_egp"].dropna().empty:
            return
        try:
            r = requests.get(config.EXCHANGERATE_API_LATEST_URL, timeout=config.SCRAPE_TIMEOUT_SECONDS)
            r.raise_for_status()
            rate = r.json().get("rates", {}).get("EGP")
            if rate:
                yf_latest = merged["usd_egp"].dropna().iloc[-1]
                pct_diff = abs(rate - yf_latest) / yf_latest
                if pct_diff > 0.02:
                    logger.warning(
                        "USD/EGP cross-check divergence: yfinance=%.4f open.er-api=%.4f (%.1f%%)",
                        yf_latest, rate, pct_diff * 100,
                    )
        except Exception as exc:
            logger.info("USD/EGP cross-check skipped: %s", exc)

    def _store_macro(self, df: pd.DataFrame):
        # COALESCE: a column this fetch did not get must NOT overwrite the value
        # already stored. Before 2026-10-04 it did, so every Yahoo run (which has
        # no EGX30 history) blanked egx30_level for every date it touched -- only
        # 3 of 3,067 rows survived.
        with get_connection() as conn:
            for idx, r in df.iterrows():
                cols = ["usd_egp", "brent_oil_usd", "gold_usd", "natural_gas_usd",
                        "us_10y_yield", "egx30_level", "egx30_volume"]
                values = [float(r[c]) if c in df.columns and pd.notna(r[c]) else None for c in cols]
                conn.execute(
                    f"""INSERT INTO macro_daily (date, {', '.join(cols)})
                        VALUES (?, {', '.join(['?'] * len(cols))})
                        ON CONFLICT(date) DO UPDATE SET
                        {', '.join(f'{c}=COALESCE(excluded.{c}, macro_daily.{c})' for c in cols)}""",
                    [idx.strftime("%Y-%m-%d")] + values,
                )

    # Mubasher publishes the full daily EGX30 history (1998-) as a CSV linked from
    # its index page (free, no account). Found 2026-10-04; Yahoo has no history.
    MUBASHER_EGX30_PAGE = "https://www.mubasher.info/markets/EGX/indices/EGX30"
    MUBASHER_EGX30_CSV = ("https://static.mubasher.info/File.MubasherCharts/"
                          "File.Historical_Index_Charts_Dir/39d67d8ab4f4c10bf22aa353e27879133c.csv")

    def fetch_egx30_history(self) -> dict:
        """EGX30 daily close + traded value from Mubasher into macro_daily.
        The CSV link is re-read from the index page each time (the file name is a
        hash and may change); the known link is the fallback."""
        import io
        import re
        import urllib.request
        ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}
        url = self.MUBASHER_EGX30_CSV
        try:
            page = urllib.request.urlopen(urllib.request.Request(self.MUBASHER_EGX30_PAGE, headers=ua),
                                          timeout=30).read().decode("utf-8", "replace")
            m = re.search(r'historical-data-url="([^"]+\.csv)"', page)
            if m:
                url = m.group(1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_egx30_history: index page failed (%s); using the known CSV link", exc)
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=60).read()
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_egx30_history: CSV failed: %s", exc)
            return {"stored": 0, "error": str(exc)}
        df = pd.read_csv(io.BytesIO(raw), header=None,
                         names=["ts", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["ts"].str.slice(0, 10), errors="coerce")
        df = df.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
        df = df[df["date"] >= pd.Timestamp(config.HISTORY_START_DATE)].set_index("date")
        out = pd.DataFrame({"egx30_level": df["close"].astype(float),
                            "egx30_volume": pd.to_numeric(df["volume"], errors="coerce")})
        self._store_macro(out)
        return {"stored": len(out), "first": str(out.index.min().date()), "last": str(out.index.max().date())}

    # ------------------------------------------------------------------
    # CBE monetary policy decisions -- NOT IMPLEMENTED
    # ------------------------------------------------------------------
    def scrape_cbe_decisions(self):
        """cbe.org.eg rejects plain HTTP requests with a WAF challenge page
        ("Request Rejected" / support-ID, confirmed 2026-07-07) rather than
        serving real content -- this is not a parsing problem, it's a bot
        block. Implementing this reliably needs a real browser engine
        (Playwright/Selenium) and may still be blocked or require rotating
        IPs/user agents.

        Degrades GRACEFULLY (OPS-BLOCKED-01): logs a warning and returns a
        result dict with status="unavailable" rather than raising, so an
        automated pipeline run doesn't crash on an unimplemented optional
        source. Callers should treat CBE decisions as best-effort/missing.
        """
        logger.warning(
            "scrape_cbe_decisions: cbe.org.eg is behind a WAF (plain HTTP is rejected); "
            "no CBE rate-decision events collected. Needs a headless-browser scraper "
            "(see config.CBE_MPC_DECISIONS_URL). Returning empty result, not raising."
        )
        return {"status": "unavailable", "reason": "waf_block", "events": 0}

    # ------------------------------------------------------------------
    # Arabic news (almalnews.com)
    # ------------------------------------------------------------------
    def _fetch_almal_listing_links(self, page: int) -> list:
        """Return the article URLs found on one almalnews stocks-category
        listing page. Page 1 is the newest; higher page numbers page BACK
        in time, which is how we accumulate articles spread across dates
        (essential for building a real multi-day sentiment series rather
        than a one-day snapshot). Returns [] on any fetch/parse failure.
        """
        listing_url = f"{config.ALMAL_BASE_URL}{config.ALMAL_CATEGORY_PATH}{page}/"
        try:
            r = requests.get(listing_url, headers=self.session_headers, timeout=config.SCRAPE_TIMEOUT_SECONDS)
            r.raise_for_status()
        except Exception as exc:
            logger.warning("scrape_arabic_news: listing page %d fetch failed: %s", page, exc)
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for tag in soup.find_all(["h1", "h2", "h3"]):
            a = tag.find("a", href=True)
            if a and re.match(r"^/\d+/", a["href"]):
                links.append(config.ALMAL_BASE_URL + a["href"])
        return list(dict.fromkeys(links))  # de-dupe, preserve order

    def scrape_arabic_news(self, max_articles: int = 20, pages: int = 1):
        """Scrape stock-market headlines from almalnews.com, across `pages`
        listing pages (page 1 = newest, higher pages = older), up to
        `max_articles` NEW articles total.

        Confirmed working selectors (2026-07-07):
          - listing pages: headline links are <a href> inside h1/h2/h3
          - article body: CSS class `.article-content`
          - published time: <meta property="article:published_time">
          - headline text: <meta property="og:title">

        Only almalnews.com is implemented. mubasher.info and
        egx.com.eg/disclosures were not inspected/tested and are not
        implemented -- listing them as done would be dishonest given
        neither has a verified selector set here.

        URLs already in news_articles are skipped WITHOUT re-fetching the
        article body (politeness + speed), so this is cheap to re-run on a
        schedule -- the intended way to accumulate a real historical corpus
        over time. Ticker mentions are tagged via _match_tickers (Arabic +
        English name dictionaries).
        """
        # collect candidate links across pages
        candidates = []
        for page in range(1, pages + 1):
            candidates.extend(self._fetch_almal_listing_links(page))
            if page < pages:
                time.sleep(config.SCRAPE_REQUEST_DELAY_SECONDS)
        candidates = list(dict.fromkeys(candidates))  # global de-dupe

        # skip URLs we already have (no re-fetch)
        with get_connection() as conn:
            existing = {row[0] for row in conn.execute("SELECT url FROM news_articles").fetchall()}
        fresh = [u for u in candidates if u not in existing][:max_articles]

        name_dict = {
            t["ticker"]: t["name_english"]
            for t in config.EGX_SEED_TICKERS
            if t["name_english"]
        }

        stored = 0
        for url in fresh:
            article = self._fetch_almal_article(url)
            if article is None:
                continue
            mentioned = self._match_tickers(article["body"] + " " + article["headline"], name_dict)
            self._store_article(article, mentioned)
            stored += 1
            time.sleep(config.SCRAPE_REQUEST_DELAY_SECONDS)

        logger.info("scrape_arabic_news: stored %d new articles (%d candidates across %d page(s), %d already known)",
                    stored, len(candidates), pages, len(candidates) - len(fresh))
        return {"stored": stored, "candidates": len(candidates), "pages": pages}

    def _fetch_almal_article(self, url: str):
        try:
            r = requests.get(url, headers=self.session_headers, timeout=config.SCRAPE_TIMEOUT_SECONDS)
            r.raise_for_status()
        except Exception as exc:
            logger.warning("almal article fetch failed (%s): %s", url, exc)
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        body_el = soup.select_one(".article-content")
        title_meta = soup.find("meta", property="og:title")
        time_meta = soup.find("meta", property="article:published_time")

        if body_el is None or title_meta is None:
            logger.warning("almal article missing expected structure: %s", url)
            return None

        published_at = None
        if time_meta and time_meta.get("content"):
            try:
                published_at = datetime.fromisoformat(time_meta["content"])
            except ValueError:
                pass

        return {
            "headline": title_meta.get("content", "").split(" -")[0].strip(),
            "body": body_el.get_text(strip=True),
            "url": url,
            "published_at": published_at,
        }

    @staticmethod
    def _match_tickers(text: str, name_dict: dict) -> list:
        """Match EGX tickers mentioned in text using both English company
        names (name_dict, for any English text) and Arabic company names
        (config.ARABIC_COMPANY_NAMES, for Arabic text). Returns a
        deduplicated list of matched tickers preserving first-match order.
        Case-insensitive for English names; exact substring for Arabic
        (Arabic text is case-invariant by nature but whitespace-sensitive).
        """
        found = []
        seen = set()

        # English name matching (existing behavior, kept for robustness)
        for ticker, name in name_dict.items():
            if name and name.lower() in text.lower():
                if ticker not in seen:
                    found.append(ticker)
                    seen.add(ticker)

        # Arabic name matching (new -- this is what actually fires on Arabic text)
        for arabic_name, ticker in config.ARABIC_COMPANY_NAMES.items():
            if arabic_name in text:
                if ticker not in seen:
                    found.append(ticker)
                    seen.add(ticker)

        return found

    def retag_existing_articles(self):
        """Re-run _match_tickers against every already-stored article's
        text and update tickers_mentioned. Needed because ticker matching
        happens once, at scrape time -- if config.ARABIC_COMPANY_NAMES
        gains entries later (or a bug in matching gets fixed), articles
        scraped before that change are stuck with their stale (often
        empty) tickers_mentioned forever unless something goes back and
        reprocesses them. Safe to re-run any time; purely a re-derivation
        of tickers_mentioned from already-stored text, no new HTTP calls.
        """
        name_dict = {
            t["ticker"]: t["name_english"]
            for t in config.EGX_SEED_TICKERS
            if t["name_english"]
        }
        updated = 0
        with get_connection() as conn:
            rows = conn.execute("SELECT id, headline_arabic, body_arabic, tickers_mentioned FROM news_articles").fetchall()
            for row in rows:
                text = (row["body_arabic"] or "") + " " + (row["headline_arabic"] or "")
                mentioned = self._match_tickers(text, name_dict)
                new_value = ",".join(mentioned)
                if new_value != (row["tickers_mentioned"] or ""):
                    conn.execute(
                        "UPDATE news_articles SET tickers_mentioned = ? WHERE id = ?",
                        (new_value, row["id"]),
                    )
                    updated += 1
        logger.info("retag_existing_articles: updated %d/%d articles", updated, len(rows))
        return {"updated": updated, "total": len(rows)}

    def _store_article(self, article: dict, mentioned_tickers: list):
        with get_connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO news_articles
                   (source, headline_arabic, body_arabic, url, published_at, tickers_mentioned)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    "almalnews",
                    article["headline"],
                    article["body"],
                    article["url"],
                    article["published_at"].isoformat() if article["published_at"] else None,
                    ",".join(mentioned_tickers),
                ),
            )

    def scan_news_for_tbill_rates(self):
        """Scan already-scraped news_articles for T-bill discount rate
        mentions (see extract_tbill_discount_rate) and store any found
        into macro_events (event_type='EGY_TBILL_DISCOUNT_RATE'). Sparse/
        event-driven by nature -- see extract_tbill_discount_rate
        docstring for why this can't backfill deep history.
        """
        stored = 0
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT headline_arabic, body_arabic, published_at, url FROM news_articles"
            ).fetchall()
            for row in rows:
                if not row["published_at"]:
                    continue
                text = (row["headline_arabic"] or "") + " " + (row["body_arabic"] or "")
                rates = extract_tbill_discount_rate(text)
                if not rates:
                    continue
                event_date = row["published_at"][:10]
                rate = rates[0]
                existing = conn.execute(
                    "SELECT 1 FROM macro_events WHERE event_type = 'EGY_TBILL_DISCOUNT_RATE' "
                    "AND date = ? AND actual_value = ?",
                    (event_date, rate),
                ).fetchone()
                if existing:
                    continue
                conn.execute(
                    """INSERT INTO macro_events (date, event_type, description, actual_value)
                       VALUES (?, 'EGY_TBILL_DISCOUNT_RATE', ?, ?)""",
                    (event_date, f"T-bill discount rate mention: {row['url']}", rate),
                )
                stored += 1
        logger.info("scan_news_for_tbill_rates: stored %d new T-bill discount rate mention(s)", stored)
        return {"stored": stored}
