"""
Research-operations agent: watches the DATA, never the market.

WHY THIS EXISTS. Between 2026-08-13 and 2026-09-11 every material failure in
this system was a data-meaning failure, and not one of the eleven modules in
analysis/ could see any of them:

    2026-08-16  a fabricated session reset the staleness clock; a five-day-old
                signal reported itself as two days old
    2026-08-18  a fetch overwrote nine broker marks with stale vendor prices
    2026-08-20  stocks.egx70_member had never been populated, so index questions
                were unanswerable and the tempting proxy was INVERTED
    2026-09-01  the index membership list silently passed its own expiry date
    2026-09-04  CRST reached 21% of the live book with 5 usable returns in 180
                sessions, reported as a two-line footnote
    2026-09-11  KORA (16.8%) and KASABF (10.2%) joined it; 40% of the book
                became unmeasurable, and KASABF turned out not to be a share at
                all but a closed-end fund

deflated_sharpe, significance, robustness_suite, regime_validation, alpha_decay
and the rest are excellent at asking "is this number real?". None of them asks
"is this number ABOUT what I think it is about?". That second question is the
entire job of this module.

WHAT IT DELIBERATELY DOES NOT DO. It does not select securities, size positions,
time entries, or interpret the regime. Five independent selection methods have
been tested on this data -- selection alpha (t = -0.61), Fama-MacBeth (max |t|
1.93), rank persistence (t = 1.01), a contrarian composite (p = 0.985) and a
multi-timeframe momentum composite which came back SIGNIFICANTLY NEGATIVE
(-20.9%/yr, permutation p = 0.010). ops/news_brief.py is the standing warning:
an NLP component pointed at this system scored F1 0.33 and had to be quarantined
as "CONTEXT ONLY". An agent that opined on what to buy would be that failure
again with better prose, and prose is worse than a t-statistic precisely because
it reads as informative when it is not.

ARCHITECTURE, AND WHY IT IS NOT "AN LLM WITH TOOLS". Of the failures listed
above, six are detectable by a SQL query and two require reading unstructured
prose. So the core is a DETERMINISTIC scanner -- cheap, testable, runs unattended
-- and language work is an explicit, optional escalation on the few findings that
need it. Building it the other way round (an LLM that decides what to check)
would make the reliable 75% depend on the unreliable 25%.

THE WRITE BOUNDARY IS ENFORCED IN CODE. `_GuardedConnection` raises on any write
outside the allow-list. This is deliberately not a docstring promise: the
2026-08-18 incident was a function that was *supposed* to leave broker marks
alone and did not, and the fix was a mechanism, not a comment. The agent may
write reference facts and its own findings. It may never write a price, a
weight, a mark, a position or an order.
"""

import datetime
import hashlib
import logging
import sqlite3

import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.research_agent")

# Tables the agent is permitted to modify. Everything else is read-only to it.
# `stocks` holds reference facts (name, sector, market cap, index membership);
# `research_findings` is the agent's own log. Prices, positions, marks, weights,
# orders and the trade ledger are all absent from this list on purpose.
WRITABLE = {"stocks", "research_findings"}

# Severity is about what the finding does to a DECISION, not how unusual it is.
#   critical -- a number you are already relying on is wrong or unknowable
#   high     -- a decision input is stale/missing and will mislead soon
#   medium   -- a real defect with no immediate decision consequence
#   info     -- worth knowing, nothing to do
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "info": 3}

# Share of book value that may be unmeasurable before it is a critical finding.
# Matches analysis.portfolio_diagnostic.UNMEASURED_ALARM so the two cannot drift
# apart and disagree about the same book.
UNMEASURED_CRITICAL = 0.10

# A vendor silence longer than this is a dead feed, not a gap.
STALE_FEED_SESSIONS = 10

# Minimum usable returns (in the last 180 sessions) for a name to carry a
# volatility or a correlation. Mirrors portfolio_diagnostic's `priced` test.
MIN_USABLE_RETURNS = 90
LOOKBACK = 180


class WriteBoundaryViolation(RuntimeError):
    """The agent attempted to write outside its allow-list."""


# The engine-level boundary itself lives in ops.agent_guard, shared by every agent.
from ops.agent_guard import make_authorizer as _make_authorizer  # noqa: E402

_authorize = _make_authorizer(WRITABLE)


class _GuardedConnection:
    """A sqlite connection that refuses writes outside WRITABLE.

    Wraps rather than subclasses because sqlite3.Connection is a C type whose
    execute() cannot be reliably overridden by inheritance. Every statement
    passes through _check() before reaching the real connection, so there is no
    path from agent code to a price table that does not go through this gate.
    """

    _WRITE_VERBS = ("insert", "update", "delete", "drop", "alter", "replace", "create")

    def __init__(self, conn):
        self._conn = conn
        # THE REAL BOUNDARY (Stage 0, 2026-10-03). _check() below parses SQL
        # text, and text parsing has holes: `WITH t AS (SELECT 1) DELETE FROM
        # daily_ohlcv` starts with WITH, which is not a write verb, and it
        # emptied the table in the audit's reproduction. SQLite's authorizer is
        # called by the engine for every table a compiled statement touches,
        # whatever the SQL looks like -- CTEs, triggers, subqueries, REPLACE,
        # UPSERT. _check() stays as a first layer for its clearer message.
        conn.set_authorizer(_authorize)

    def _run(self, fn, sql, *a, **kw):
        try:
            return fn(sql, *a, **kw)
        except sqlite3.DatabaseError as exc:
            if "not authorized" in str(exc):
                raise WriteBoundaryViolation(
                    f"research_agent write refused by the database engine: {sql[:120]!r}. "
                    f"Writable: {sorted(WRITABLE)}.") from exc
            raise

    @staticmethod
    def _target_table(sql: str) -> str:
        """Table a write statement targets, or '?' when it cannot be determined.

        Returning '?' for an unparsed statement is deliberate: '?' is not in
        WRITABLE, so anything this function fails to understand is REFUSED
        rather than allowed. A guard that fails open is not a guard.

        The noise words matter. `CREATE TABLE IF NOT EXISTS x` and
        `DROP TABLE IF EXISTS x` put keywords between the verb and the name, and
        the first version of this parser read the `IF` as the table -- which
        blocked the agent's own findings table on the very first run. That was
        the right failure (refuse rather than permit), but it has to be handled
        rather than tolerated.

        `CREATE INDEX ... ON t` is resolved to `t`, not to the index name: what
        matters for safety is which table is being touched.
        """
        toks = [t.strip('"`[];') for t in sql.replace("(", " ( ").split()]
        low = [t.lower() for t in toks]
        NOISE = {"if", "not", "exists", "temp", "temporary", "unique",
                 "or", "ignore", "replace", "rollback", "abort", "fail"}

        def name_after(i):
            """First non-noise token after position i."""
            for j in range(i + 1, len(toks)):
                if low[j] in NOISE:
                    continue
                return toks[j].split("(")[0]
            return "?"

        verb = low[0] if low else ""
        # CREATE INDEX ... ON <table>  -- the ON target is the real subject
        if verb == "create" and "index" in low:
            if "on" in low:
                return name_after(low.index("on"))
            return "?"
        for i, t in enumerate(low):
            if t in ("into", "update", "table"):
                return name_after(i)
            if t == "from" and verb == "delete":
                return name_after(i)
        return "?"

    def _check(self, sql: str):
        verb = sql.strip().split(None, 1)[0].lower() if sql.strip() else ""
        if verb not in self._WRITE_VERBS:
            return
        table = self._target_table(sql)
        if table not in WRITABLE:
            raise WriteBoundaryViolation(
                f"research_agent may not {verb.upper()} {table!r}. "
                f"Writable: {sorted(WRITABLE)}. The agent reports findings; it "
                f"does not change prices, marks, positions or orders.")

    def execute(self, sql, *a, **kw):
        self._check(sql)
        return self._run(self._conn.execute, sql, *a, **kw)

    def executemany(self, sql, *a, **kw):
        self._check(sql)
        return self._run(self._conn.executemany, sql, *a, **kw)

    def executescript(self, sql, *a, **kw):
        # A script is many statements; check each one rather than the blob.
        for stmt in sql.split(";"):
            if stmt.strip():
                self._check(stmt)
        return self._run(self._conn.executescript, sql, *a, **kw)

    def cursor(self, *a, **kw):
        """Return a GUARDED cursor.

        THIS IS THE HOLE THE FIRST VERSION HAD. __getattr__ forwarded every
        unknown attribute to the real connection, and `cursor` is an unknown
        attribute -- so `conn.cursor().execute("DELETE FROM daily_ohlcv")` went
        straight past the boundary. pandas.read_sql takes exactly that path,
        which means the guard was already being bypassed on every check in this
        module. It happened to be harmless (pandas only SELECTs) but the guard
        was decorative, not load-bearing.

        A boundary with one unguarded door is not a boundary, and the door you
        did not think of is the one that gets used.
        """
        return _GuardedCursor(self._conn.cursor(*a, **kw), self._check, self._run)

    def __getattr__(self, name):
        # Deliberately narrow. Anything that can execute SQL must be wrapped
        # above; this forwards only the inert remainder (commit, rollback,
        # row_factory, close, context-manager protocol...).
        if name in ("execute", "executemany", "executescript", "cursor"):
            raise AttributeError(name)   # unreachable, but fails closed if moved
        return getattr(self._conn, name)


class _GuardedCursor:
    """Cursor wrapper applying the same check as the connection."""

    def __init__(self, cur, check, run):
        self._cur, self._check, self._run = cur, check, run

    def execute(self, sql, *a, **kw):
        self._check(sql)
        return self._run(self._cur.execute, sql, *a, **kw)

    def executemany(self, sql, *a, **kw):
        self._check(sql)
        return self._run(self._cur.executemany, sql, *a, **kw)

    def executescript(self, sql, *a, **kw):
        for stmt in sql.split(";"):
            if stmt.strip():
                self._check(stmt)
        return self._run(self._cur.executescript, sql, *a, **kw)

    def __iter__(self):
        return iter(self._cur)

    def __getattr__(self, name):
        if name in ("execute", "executemany", "executescript"):
            raise AttributeError(name)
        return getattr(self._cur, name)


def _read_sql(sql, conn, params=None):
    """pandas.read_sql through the guarded connection, without the noise.

    pandas warns because _GuardedConnection is not literally a
    sqlite3.Connection. That warning is about TYPE CHECKING, not safety: reads
    still pass through _GuardedCursor (see the pandas test), so a write smuggled
    into a read call is still refused. The filter is scoped to this one call so
    a genuine pandas warning elsewhere is not hidden -- a log full of warnings
    nobody can act on is how the real ones get ignored.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*only supports SQLAlchemy.*",
                                category=UserWarning)
        if params is not None:
            return pd.read_sql(sql, conn, params=params)
        return pd.read_sql(sql, conn)


def _ensure_findings_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research_findings (
            id           INTEGER PRIMARY KEY,
            first_seen   TEXT NOT NULL,
            last_seen    TEXT NOT NULL,
            fingerprint  TEXT NOT NULL UNIQUE,
            severity     TEXT NOT NULL,
            kind         TEXT NOT NULL,
            subject      TEXT,
            summary      TEXT NOT NULL,
            detail       TEXT,
            needs_human  INTEGER DEFAULT 0,
            resolved_at  TEXT
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_open "
                 "ON research_findings (resolved_at, severity)")


def _fingerprint(kind: str, subject: str) -> str:
    """Stable identity for a finding, so a problem that persists for six days is
    ONE finding seen six times rather than six findings. Without this the log
    becomes a scroll of duplicates and stops being read -- which is how the
    'NOT MEASURABLE: CRST' footnote survived three weeks of daily reports."""
    return hashlib.sha1(f"{kind}|{subject}".encode()).hexdigest()[:16]


class Finding:
    __slots__ = ("severity", "kind", "subject", "summary", "detail", "needs_human")

    def __init__(self, severity, kind, subject, summary, detail="", needs_human=False):
        self.severity, self.kind, self.subject = severity, kind, subject
        self.summary, self.detail, self.needs_human = summary, detail, needs_human

    def __repr__(self):
        return f"<{self.severity}:{self.kind}:{self.subject}>"


# --------------------------------------------------------------------------
# CHECKS. Each returns a list[Finding] and takes a read-only connection.
# Every one of these corresponds to a failure that actually happened.
# --------------------------------------------------------------------------

def check_unmeasurable_holdings(conn) -> list:
    """Positions with too little price history to carry a risk number.

    THE FAILURE: on 2026-09-11 CRST + KORA + KASABF were 39.9% of book value
    with 5, 0 and 0 usable returns respectively. The portfolio diagnostic
    excluded them and reported a volatility for the other 60%, which is not a
    volatility for the portfolio. An unmeasured position reads as a safe one,
    because absence of a risk number looks exactly like absence of risk.
    """
    from backtest.engine import load_price_returns
    pos = _read_sql("SELECT ticker, shares FROM user_positions", conn)
    if pos.empty:
        return []
    px = _read_sql(
        "SELECT ticker, adjusted_close FROM daily_ohlcv d WHERE date = "
        "(SELECT MAX(date) FROM daily_ohlcv d2 WHERE d2.ticker = d.ticker)", conn
    ).set_index("ticker")["adjusted_close"]
    rets = load_price_returns()

    out, unmeasured_value, total_value, names = [], 0.0, 0.0, []
    for t, sh in zip(pos["ticker"], pos["shares"]):
        val = float(sh) * float(px.get(t, 0.0) or 0.0)
        total_value += val
        n = int(rets[t].tail(LOOKBACK).notna().sum()) if t in rets.columns else 0
        if n < MIN_USABLE_RETURNS:
            unmeasured_value += val
            names.append(f"{t}({n} returns)")
    if not names or total_value <= 0:
        return out
    share = unmeasured_value / total_value
    sev = "critical" if share >= UNMEASURED_CRITICAL else "medium"
    out.append(Finding(
        sev, "unmeasurable_book", "portfolio",
        f"{share:.0%} of book value cannot carry a volatility or correlation",
        f"{unmeasured_value:,.0f} of {total_value:,.0f} EGP: {', '.join(names)}. "
        f"Every portfolio-level risk figure describes only the remaining "
        f"{1-share:.0%}."))
    return out


def check_positions_outside_universe(conn) -> list:
    """Holdings the system has no price history for at all.

    THE FAILURE: KORA and KASABF were bought on 2026-09-11 and had zero rows in
    daily_ohlcv. Nothing anywhere said so -- they simply did not appear in any
    model, and a position that is absent from a risk report is indistinguishable
    from one that is fine.
    """
    pos = _read_sql("SELECT ticker FROM user_positions", conn)["ticker"].tolist()
    if not pos:
        return []
    have = {r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM daily_ohlcv WHERE ticker IN (%s)"
        % ",".join("?" * len(pos)), pos)}
    missing = sorted(set(pos) - have)
    return [Finding("critical", "no_price_history", t,
                    f"{t} is held but has NO rows in daily_ohlcv",
                    "It cannot appear in any risk model, optimiser or diagnostic. "
                    "Either load its history or record explicitly that it is "
                    "held outside the measured book.", needs_human=True)
            for t in missing]


def check_dead_vendor_feeds(conn) -> list:
    """Names the primary vendor has stopped publishing.

    THE FAILURE: Yahoo's last CRST print was 2026-08-12. By 2026-09-04 the
    position was 21% of the book and every mark for three weeks was a value the
    operator had typed. A feed does not announce that it has stopped.
    """
    from data.calendar import valid_trading_days
    sessions = valid_trading_days()
    if len(sessions) < STALE_FEED_SESSIONS + 1:
        return []
    cutoff = sessions[-STALE_FEED_SESSIONS]
    held = _read_sql("SELECT ticker FROM user_positions", conn)["ticker"].tolist()
    if not held:
        return []
    last = _read_sql(
        "SELECT ticker, MAX(date) AS d FROM daily_ohlcv WHERE source = 'vendor' "
        "AND ticker IN (%s) GROUP BY ticker" % ",".join("?" * len(held)),
        conn, params=held)
    out = []
    for t, d in zip(last["ticker"], pd.to_datetime(last["d"])):
        if d < cutoff:
            gap = int((sessions > d).sum())
            out.append(Finding(
                "high", "dead_vendor_feed", t,
                f"{t}: no vendor price for {gap} sessions (last {d.date()})",
                "Marks since then come from the broker or the reference source, "
                "so there is no independent price to cross-check against."))
    return out


def check_index_membership_freshness(conn) -> list:
    """Has the stored EGX index membership passed its own review date?

    THE FAILURE: the first version of data/egx_index_membership.py recorded its
    expiry in a docstring. It then sat a week past that date being used to answer
    index questions, and nothing said so. This check exists because a date that
    matters must be checked, not written down.
    """
    try:
        from data import egx_index_membership as m
    except Exception as exc:  # noqa: BLE001
        return [Finding("medium", "index_membership", "egx",
                        "index membership module could not be imported", str(exc))]
    today = datetime.date.today().isoformat()
    nxt = getattr(m, "NEXT_REVIEW", None)
    if nxt and today >= nxt:
        return [Finding("high", "index_membership", "egx",
                        f"EGX index membership expired on {nxt}",
                        f"The stored list reflects the "
                        f"{getattr(m,'REVIEW_APPLIED','?')} review. EGX rebalances "
                        f"semi-annually; re-scrape constituents before using "
                        f"egx70_tickers().", needs_human=True)]
    unmatched = getattr(m, "UNMATCHED", [])
    if unmatched:
        return [Finding("info", "index_membership", "egx",
                        f"{len(unmatched)} index constituents unmapped to tickers",
                        "Deliberately not guessed: " + "; ".join(unmatched))]
    return []


def check_assumed_prices(conn) -> list:
    """Trades booked at an estimate rather than a real fill.

    THE FAILURE: four exits on 2026-09-04 were recorded at my last marks because
    the execution prices were not supplied, carrying -179.59 of ESTIMATED
    realised P&L that was then reported beside genuine figures. Realised P&L is
    the one number a backtest cannot flatter; it is the last place an unlabelled
    guess belongs.
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_trade_ledger)")}
    if "price_source" not in cols:
        return []
    df = _read_sql(
        "SELECT ticker, action, shares, price, traded_at FROM user_trade_ledger "
        "WHERE price_source = 'assumed' ORDER BY traded_at", conn)
    if df.empty:
        return []
    return [Finding(
        "medium", "assumed_fill_prices", "ledger",
        f"{len(df)} trade(s) booked at estimated prices, not real fills",
        "; ".join(f"{r.action} {r.shares:g} {r.ticker} @ {r.price:.4f} "
                  f"({str(r.traded_at)[:10]})" for r in df.itertuples())
        + ". Realised P&L computed from these is an estimate.", needs_human=True)]


def check_killed_runs(conn) -> list:
    """Jobs that started and never finished.

    THE FAILURE: the scheduler returned 0xC000013A on 2026-09-01, 09-02 and
    09-03 while job_runs stayed empty for that window, because rows were only
    written on completion. From the database alone, three kills were
    indistinguishable from three days of the laptop being switched off.
    """
    try:
        from ops.health import killed_runs
        df = killed_runs()
    except Exception:  # noqa: BLE001
        return []
    if df is None or df.empty:
        return []
    return [Finding("high", "killed_job_runs", "scheduler",
                    f"{len(df)} job run(s) started but never finished",
                    "; ".join(f"{r.job} @ {str(r.started_at)[:16]}"
                              for r in df.itertuples()))]


def check_price_source_disagreement(conn, tolerance: float = 0.05) -> list:
    """Names where the two independent price sources disagree materially.

    Disagreement is not proof either source is wrong -- they sample at different
    and here unknown times. It means the name should not be traded on today's
    number without a look. Held names are escalated because a bad mark on a
    position you own misprices your book, not just a candidate.
    """
    try:
        from data.stockanalysis import fetch_egx_listing, cross_check_prices
        chk = cross_check_prices(fetch_egx_listing(), tolerance=tolerance)
    except Exception as exc:  # noqa: BLE001
        return [Finding("info", "price_cross_check", "reference",
                        "cross-check unavailable", str(exc))]
    flagged = chk[chk["flag"]] if "flag" in chk.columns else chk.iloc[0:0]
    if flagged.empty:
        return []
    held = set(_read_sql("SELECT ticker FROM user_positions", conn)["ticker"])
    out = []
    hits = flagged[flagged["ticker"].isin(held)]
    for r in hits.itertuples():
        out.append(Finding("critical", "price_disagreement_held", r.ticker,
                           f"{r.ticker}: sources differ {r.diff_pct:+.1%}",
                           f"reference {r.price:.4f} vs vendor {r.yahoo:.4f}. "
                           f"This is a position you HOLD, so one of these marks "
                           f"is mispricing your book.", needs_human=True))
    others = flagged[~flagged["ticker"].isin(held)]
    if len(others):
        worst = "; ".join(f"{r.ticker} {r.diff_pct:+.0%}"
                          for r in others.head(8).itertuples())
        out.append(Finding("medium", "price_disagreement", "universe",
                           f"{len(others)} universe name(s) disagree >{tolerance:.0%}",
                           worst + ("; ..." if len(others) > 8 else "")))
    return out


def check_unknown_securities(conn) -> list:
    """Held tickers absent from the reference table -- identity unknown.

    THE FAILURE: KASABF was bought as if it were a share. It is Certificates of
    Odin Egyptian Equity Investment Fund -- a closed-end FUND holding a basket of
    Egyptian equities, i.e. overlapping exposure to names already held directly,
    priced on NAV rather than continuous trading. Nothing in the system knew
    that, because nothing checked what it was. This is the one finding class
    that genuinely needs a language model to resolve.
    """
    pos = _read_sql("SELECT ticker FROM user_positions", conn)["ticker"].tolist()
    if not pos:
        return []
    known = {r[0] for r in conn.execute(
        "SELECT ticker FROM stocks WHERE ticker IN (%s) AND name_english IS NOT NULL"
        % ",".join("?" * len(pos)), pos)}
    return [Finding("high", "unknown_security", t,
                    f"{t} is held but has no reference record",
                    "Security type (ordinary share / fund / certificate), sector, "
                    "listing date and ISIN are all unknown. A fund or a recent "
                    "IPO behaves nothing like an ordinary share and cannot be "
                    "modelled as one.", needs_human=True)
            for t in sorted(set(pos) - known)]


def check_sector_concentration(conn, alarm: float = 0.25) -> list:
    """Book weight in a single exchange sector.

    THE FAILURE: on 2026-09-04 the optimiser returned MILS + AFMC + LUTS as the
    best three-name addition, reporting "average pairwise correlation 0.227".
    MILS is North Cairo Flour Mills and AFMC is Alexandria Flour Mills --
    correlated 0.644, stable 0.62-0.70 across every window -- and the average hid
    that one strong pair behind two near-zero ones. A correlation matrix cannot
    know two companies are in the same business.

    By 2026-09-11 the live book held 31.3% in Contracting & Construction
    Engineering (CRST + KORA), and BOTH of those positions are unmeasurable. The
    largest sector bet in the book was also its largest blind spot, and nothing
    anywhere said so, because `stocks.sector` had been NULL since the system was
    written.
    """
    import pandas as _pd
    pos = _read_sql("SELECT ticker, shares FROM user_positions", conn)
    if pos.empty:
        return []
    px = _read_sql(
        "SELECT ticker, adjusted_close FROM daily_ohlcv d WHERE date = "
        "(SELECT MAX(date) FROM daily_ohlcv d2 WHERE d2.ticker = d.ticker)", conn
    ).set_index("ticker")["adjusted_close"]
    ref = _read_sql("SELECT ticker, sector FROM stocks", conn).set_index("ticker")["sector"]
    pos["value"] = [float(s) * float(px.get(t, 0.0) or 0.0)
                    for t, s in zip(pos["ticker"], pos["shares"])]
    pos["sector"] = pos["ticker"].map(ref)
    total = pos["value"].sum()
    if total <= 0:
        return []

    out = []
    unknown = pos[pos["sector"].isna()]
    if len(unknown):
        w = unknown["value"].sum() / total
        out.append(Finding(
            "medium", "sector_unknown", "portfolio",
            f"{w:.0%} of the book has no sector classification",
            "Unclassified: " + ", ".join(unknown["ticker"]) +
            ". Sector concentration cannot be measured for these."))

    grp = pos.dropna(subset=["sector"]).groupby("sector")["value"].agg(["sum", "count"])
    grp["w"] = grp["sum"] / total
    for sector, r in grp.sort_values("w", ascending=False).iterrows():
        if r["w"] >= alarm:
            names = ", ".join(sorted(pos.loc[pos["sector"] == sector, "ticker"]))
            out.append(Finding(
                "high", "sector_concentration", str(sector)[:40],
                f"{r['w']:.0%} of the book is in one sector: {sector}",
                f"{int(r['count'])} position(s): {names}. Names in one sector share "
                f"inputs, regulation and demand shocks that a correlation estimate "
                f"measured on quiet periods will understate."))
    return out


def check_non_ordinary_holdings(conn) -> list:
    """Held instruments that are not ordinary shares.

    KASABF is a closed-end fund holding a basket of Egyptian equities: some of
    its "diversification" is re-exposure to names already held directly, and it
    prices on NAV rather than continuous trading. An ETF, a rights line, a
    preferred share and a SPAC each break the risk model differently, and all of
    them look like a ticker with a price.
    """
    df = _read_sql(
        "SELECT p.ticker, s.security_type, s.sector FROM user_positions p "
        "JOIN stocks s ON s.ticker = p.ticker "
        "WHERE s.security_type IS NOT NULL AND s.security_type != 'ordinary'", conn)
    if df.empty:
        return []
    return [Finding(
        "high", "non_ordinary_instrument", r.ticker,
        f"{r.ticker} is a {r.security_type}, not an ordinary share",
        "It cannot be modelled as a single-name equity: a fund or ETF holds a "
        "basket (overlapping exposure to names you may already own) and prices "
        "on NAV; rights expire; preferred shares carry a different claim.",
        needs_human=True) for r in df.itertuples()]


def check_fundamentals_of_holdings(conn, loss_years: int = 3,
                                   stale_months: int = 15) -> list:
    """Reported financial facts about positions you hold.

    THE LINE THIS CHECK MUST NOT CROSS. It reports what a company HAS REPORTED.
    It does not rank, score, or imply what to buy or sell. That distinction is
    load-bearing: a screen built on these same fields would be a selection model,
    and selection on this data has now been tested six ways -- four null, one
    significantly negative (-20.9%/yr, permutation p = 0.010). A loss-making
    company can rise for years, and OIH is 21.8% of this book, up 30.9%, with
    four consecutive annual losses. Both facts are true at once, and the job here
    is to make sure the second one is not invisible.

    Why it earns a place at all: before 2026-09-11 every number in this system
    came from price. A position could halve its revenue and nothing anywhere
    would change.
    """
    import pandas as _pd
    held = _read_sql("SELECT ticker FROM user_positions", conn)["ticker"].tolist()
    if not held:
        return []
    try:
        f = _read_sql(
            "SELECT ticker, statement, fiscal_period, period_end, line_item, value "
            "FROM fundamentals WHERE ticker IN (%s)" % ",".join("?" * len(held)),
            conn, tuple(held))
    except Exception:
        return []          # table not created yet -- nothing to report
    if f.empty:
        return []

    out = []
    inc = f[f["statement"] == "income"]
    for t in held:
        d = inc[inc["ticker"] == t]
        if d.empty:
            continue

        # Consecutive reported annual losses, newest first.
        ni = (d[d["line_item"] == "Net Income"]
              .dropna(subset=["period_end", "value"])
              .sort_values("period_end", ascending=False))
        # Fiscal years only: quarterly rows share this table since 2026-09-14.
        ni = ni[ni["fiscal_period"].str.startswith("FY", na=False)]
        streak = 0
        for v in ni["value"]:
            if v < 0:
                streak += 1
            else:
                break
        if streak >= loss_years:
            out.append(Finding(
                "medium", "reported_losses", t,
                f"{t} has reported a net loss in {streak} consecutive years",
                "A reported fact, not a recommendation: loss-making companies "
                "can and do appreciate. It is recorded because until now nothing "
                "in this system could see a company's earnings at all."))

        # Fundamentals that have stopped arriving.
        last = _pd.to_datetime(d["period_end"], errors="coerce").max()
        if _pd.notna(last):
            months = (_pd.Timestamp.now() - last).days / 30.44
            if months > stale_months:
                out.append(Finding(
                    "medium", "stale_fundamentals", t,
                    f"{t}: no reported financials for {months:.0f} months "
                    f"(last period {last.date()})",
                    "Either the company has stopped reporting or the source has "
                    "stopped carrying it; both are worth knowing about a position "
                    "you hold."))
    return out


ALL_CHECKS = (
    check_unmeasurable_holdings,
    check_sector_concentration,
    check_non_ordinary_holdings,
    check_fundamentals_of_holdings,
    check_positions_outside_universe,
    check_dead_vendor_feeds,
    check_unknown_securities,
    check_index_membership_freshness,
    check_price_source_disagreement,
    check_assumed_prices,
    check_killed_runs,
)


def scan(persist: bool = True, checks=ALL_CHECKS) -> pd.DataFrame:
    """Run every integrity check. Returns findings, newest severity first.

    A failing CHECK must never take down the scan: the point of this module is
    to report problems, and a reporter that dies on the first one it cannot
    evaluate is worse than no reporter. Each check is isolated.
    """
    findings = []
    with get_connection() as raw:
        conn = _GuardedConnection(raw)
        _ensure_findings_table(conn)
        for fn in checks:
            try:
                findings.extend(fn(conn) or [])
            except Exception as exc:  # noqa: BLE001
                logger.warning("check %s failed (non-fatal): %s", fn.__name__, exc)
                findings.append(Finding("info", "check_failed", fn.__name__,
                                        f"check {fn.__name__} could not run", str(exc)))
        findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
        if persist:
            now = pd.Timestamp.now().isoformat(timespec="seconds")
            # CLOSURE (Stage 4). A finding this scan no longer produces is no
            # longer true -- KORA/KASABF findings sat "open" and "critical" for
            # weeks after both were sold. Close what was not seen; checks that
            # failed to run this time keep their findings open (absence of a
            # check is not evidence the problem is gone).
            failed = {f.subject for f in findings if f.kind == "check_failed"}
            seen = {_fingerprint(f.kind, f.subject or "") for f in findings}
            if not failed:
                for (fp,) in conn.execute("SELECT fingerprint FROM research_findings "
                                          "WHERE resolved_at IS NULL").fetchall():
                    if fp not in seen:
                        conn.execute("UPDATE research_findings SET resolved_at = ?, "
                                     "detail = COALESCE(detail,'') || ' | auto-closed: no longer observed' "
                                     "WHERE fingerprint = ?", (now, fp))
            for f in findings:
                fp = _fingerprint(f.kind, f.subject or "")
                conn.execute(
                    "INSERT INTO research_findings (first_seen, last_seen, fingerprint, "
                    "severity, kind, subject, summary, detail, needs_human) "
                    "VALUES (?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(fingerprint) DO UPDATE SET last_seen = excluded.last_seen, "
                    "severity = excluded.severity, summary = excluded.summary, "
                    "detail = excluded.detail, resolved_at = NULL",
                    (now, now, fp, f.severity, f.kind, f.subject, f.summary,
                     f.detail, 1 if f.needs_human else 0))
    return pd.DataFrame([{"severity": f.severity, "kind": f.kind, "subject": f.subject,
                          "summary": f.summary, "detail": f.detail,
                          "needs_human": f.needs_human} for f in findings])


def format_scan(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "RESEARCH AGENT — no integrity findings."
    L = ["=" * 70, "RESEARCH AGENT — DATA INTEGRITY SCAN", "=" * 70]
    counts = df["severity"].value_counts().to_dict()
    L.append("  " + "  ".join(f"{k}:{counts[k]}" for k in
                              ("critical", "high", "medium", "info") if k in counts))
    L.append("")
    for sev in ("critical", "high", "medium", "info"):
        rows = df[df["severity"] == sev]
        if rows.empty:
            continue
        L.append(f"{sev.upper()}")
        for r in rows.itertuples():
            flag = "  [NEEDS HUMAN]" if r.needs_human else ""
            L.append(f"  * {r.summary}{flag}")
            if r.detail:
                for line in _wrap(str(r.detail), 64):
                    L.append(f"      {line}")
        L.append("")
    L.append("  This agent reports on DATA, never on what to buy. Selection has")
    L.append("  been tested five ways on this data and is null or negative.")
    return "\n".join(L)


def _wrap(text: str, width: int) -> list:
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def open_findings(severity: str = None) -> pd.DataFrame:
    """Unresolved findings from the log."""
    q = "SELECT * FROM research_findings WHERE resolved_at IS NULL"
    params = ()
    if severity:
        q += " AND severity = ?"
        params = (severity,)
    with get_connection() as conn:
        _ensure_findings_table(_GuardedConnection(conn))
        return _read_sql(q + " ORDER BY last_seen DESC", conn, params=params)


def resolve(fingerprint: str, note: str = "") -> bool:
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as raw:
        conn = _GuardedConnection(raw)
        cur = conn.execute(
            "UPDATE research_findings SET resolved_at = ?, "
            "detail = COALESCE(detail,'') || ' | resolved: ' || ? "
            "WHERE fingerprint = ?", (now, note, fingerprint))
        return cur.rowcount > 0


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(format_scan(scan()))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
