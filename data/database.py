"""
SQLite schema for ESIS-QUANT.

Creates every table from the full 8-layer design (Layer 1 through Layer 8)
up front, since they're cheap DDL and later layers will need them -- but
only the Layer 1 tables (stocks, daily_ohlcv, intraday_stats,
corporate_actions, macro_daily, macro_events, news_articles) are actually
populated by anything implemented so far.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager

import config

logger = logging.getLogger("esis_quant.database")

SCHEMA = """
CREATE TABLE IF NOT EXISTS stocks (
    ticker TEXT PRIMARY KEY,
    name_arabic TEXT,
    name_english TEXT,
    sector TEXT,
    sub_sector TEXT,
    egx30_member INTEGER,
    egx70_member INTEGER,
    is_soe INTEGER,
    float_pct REAL,
    listing_date DATE,
    avg_30d_volume_egp REAL
);

CREATE TABLE IF NOT EXISTS daily_ohlcv (
    ticker TEXT,
    date DATE,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    value_traded_egp REAL,
    adjusted_close REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS intraday_stats (
    ticker TEXT,
    date DATE,
    opening_auction_volume REAL,
    closing_auction_volume REAL,
    intraday_high_time TEXT,
    intraday_low_time TEXT,
    bid_ask_spread_avg REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    ticker TEXT,
    date DATE,
    action_type TEXT,
    ratio REAL,
    amount_egp REAL,
    PRIMARY KEY (ticker, date, action_type)
);

CREATE TABLE IF NOT EXISTS macro_daily (
    date DATE PRIMARY KEY,
    usd_egp REAL,
    eur_egp REAL,
    gbp_egp REAL,
    cbe_overnight_rate REAL,
    egypt_10y_yield REAL,
    us_10y_yield REAL,
    brent_oil_usd REAL,
    natural_gas_usd REAL,
    gold_usd REAL,
    egx30_level REAL,
    egx30_volume REAL,
    egypt_cds_5y REAL,
    global_risk_index REAL
);

CREATE TABLE IF NOT EXISTS macro_events (
    id INTEGER PRIMARY KEY,
    date DATE,
    event_type TEXT,
    description TEXT,
    previous_value REAL,
    actual_value REAL,
    consensus_value REAL,
    surprise_zscore REAL
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY,
    source TEXT,
    headline_arabic TEXT,
    body_arabic TEXT,
    url TEXT UNIQUE,
    published_at DATETIME,
    tickers_mentioned TEXT,
    sentiment_score REAL,
    sentiment_magnitude REAL,
    sentiment_confidence REAL
);

CREATE TABLE IF NOT EXISTS factor_exposures (
    ticker TEXT,
    date DATE,
    beta_market REAL,
    beta_size REAL,
    beta_value REAL,
    beta_momentum REAL,
    beta_quality REAL,
    beta_liquidity REAL,
    beta_egp_sensitivity REAL,
    beta_oil_sensitivity REAL,
    idiosyncratic_vol REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS signals (
    ticker TEXT,
    date DATE,
    regime TEXT,
    alpha_score REAL,
    sentiment_score REAL,
    causal_signal REAL,
    tfm_probability REAL,
    composite_score REAL,
    signal_direction TEXT,
    confidence REAL,
    suggested_weight REAL,
    stop_loss REAL,
    take_profit REAL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS portfolio_history (
    date DATE,
    ticker TEXT,
    weight REAL,
    entry_price REAL,
    current_price REAL,
    unrealized_pnl REAL,
    realized_pnl REAL,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS performance_attribution (
    date DATE PRIMARY KEY,
    total_return REAL,
    alpha REAL,
    beta_contribution REAL,
    factor_contribution REAL,
    sentiment_contribution REAL,
    selection_contribution REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    max_drawdown REAL,
    var_95 REAL,
    cvar_95 REAL
);

-- Layer 2 (regime detection). Not in the original 8-table Layer 1 list --
-- added when Layer 2 was built. `regime` here is a market-wide daily
-- state, not per-ticker, hence a separate table rather than overloading
-- the per-(ticker,date) `signals.regime` column.
CREATE TABLE IF NOT EXISTS regime_history (
    date DATE PRIMARY KEY,
    regime TEXT,
    prob_bull REAL,
    prob_bear REAL,
    prob_high_vol REAL,
    prob_recovery REAL,
    param_vintage_date DATE
);

CREATE TABLE IF NOT EXISTS regime_statistics (
    regime TEXT PRIMARY KEY,
    mean_return REAL,
    volatility REAL,
    avg_cross_sectional_correlation REAL,
    avg_duration_days REAL,
    transition_to_bull REAL,
    transition_to_bear REAL,
    transition_to_high_vol REAL,
    transition_to_recovery REAL
);

-- Layer 4 (Arabic NLP sentiment). Daily aggregated sentiment per ticker,
-- distinct from news_articles' per-article sentiment_score -- this is the
-- cross-article weighted daily rollup models/sentiment/pipeline.py produces.
CREATE TABLE IF NOT EXISTS daily_sentiment (
    ticker TEXT,
    date DATE,
    sentiment_score REAL,
    sentiment_5d_avg REAL,
    sentiment_momentum REAL,
    sentiment_volume INTEGER,
    sentiment_surprise REAL,
    PRIMARY KEY (ticker, date)
);

-- The user's ACTUAL portfolio (the interactive workflow): what you really hold,
-- so the system can compare it to its own target book and tell you what to
-- buy/trim/hold/sell to align. Separate from every backtest/paper table.
CREATE TABLE IF NOT EXISTS user_positions (
    ticker TEXT PRIMARY KEY,
    shares REAL,
    avg_cost REAL,
    updated_at DATETIME
);
CREATE TABLE IF NOT EXISTS user_account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL,
    updated_at DATETIME
);
-- Log of your DECISION on each suggestion: did you take the system's advice,
-- decline it, or act on your own view? Lets us later compare "followed the
-- system" vs "went my own way".
CREATE TABLE IF NOT EXISTS user_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at DATETIME,
    run_id TEXT,
    as_of_date DATE,
    decision TEXT,   -- ACCEPTED / DECLINED / OWN
    note TEXT
);
-- Ledger of the trades YOU tell the system you made (audit of your own actions).
CREATE TABLE IF NOT EXISTS user_trade_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    traded_at DATETIME,
    action TEXT,
    ticker TEXT,
    shares REAL,
    price REAL
);

-- Production audit trail (Part 2.3): every generated trade list is logged with
-- the regime/exposure it was based on, so what the system SAID can later be
-- compared against what the market actually did. One row per (run, ticker).
CREATE TABLE IF NOT EXISTS trade_list_log (
    run_id TEXT,
    generated_at DATETIME,
    as_of_date DATE,
    regime TEXT,
    exposure REAL,
    ticker TEXT,
    weight REAL,
    PRIMARY KEY (run_id, ticker)
);

-- Independent FORWARD paper-trading record (Part 2.4). Deliberately separate
-- from any backtest table: this is the out-of-sample track record, marked to
-- real EGX closes, and must never be blended into backtest statistics.
CREATE TABLE IF NOT EXISTS paper_trading_nav (
    date DATE PRIMARY KEY,
    nav REAL,
    daily_return REAL,
    exposure REAL,
    regime TEXT,
    run_id TEXT,
    n_positions INTEGER
);

-- T+2 SETTLEMENT. EGX settles equity trades on T+2, but the tracker credited
-- sale proceeds to cash the instant a sell was recorded. That let the allocator
-- size a buy against money the account did not yet have -- the kind of error
-- that shows up as a rejected order or an unintended margin position, not as a
-- crash. One row per cash movement that is not yet settled.
--
-- NOTE ON BROKER POLICY: many brokers permit REINVESTING unsettled proceeds and
-- restrict only withdrawal. Thndr's exact rule is not verified here, so the
-- default is the CONSERVATIVE one (unsettled proceeds are not spendable) and it
-- is switchable via config.ALLOW_UNSETTLED_REINVESTMENT. Assuming the
-- permissive rule without checking is how accounts end up in a settlement
-- violation.
CREATE TABLE IF NOT EXISTS pending_settlements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date DATE,
    settles_on DATE,       -- trade_date + 2 TRADING sessions, not calendar days
    ticker TEXT,
    side TEXT,
    amount REAL,           -- signed: + for sale proceeds incoming
    settled INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pending_settles_on ON pending_settlements (settles_on);

-- EXECUTION QUALITY (Phase 0.2). The ticket workflow is human-in-the-loop:
-- the system prints a target at the close of t, a person types it into the
-- broker app some time later, and the fill lands at whatever the market gave.
-- Until now only shares+price were kept, so the gap between the SIGNAL price
-- (the close the optimizer sized on) and the actual FILL was never measured --
-- meaning backtest/costs.py's slippage assumption could never be checked
-- against reality. One row per filled line item.
CREATE TABLE IF NOT EXISTS execution_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_date DATE,          -- the day the ticket was generated
    as_of_date DATE,           -- the signal's as-of session
    ticket_generated_at DATETIME,
    filled_at DATETIME,        -- when the human actually executed (entered)
    side TEXT,                 -- buy / sell
    ticker TEXT,
    shares REAL,
    signal_price REAL,         -- close of t: what the optimizer sized on
    fill_price REAL,           -- what the account actually got
    shortfall REAL,            -- signed implementation shortfall, fraction
    notional REAL,
    delay_minutes REAL         -- fill time minus ticket time, if both known
);
CREATE INDEX IF NOT EXISTS idx_exec_quality_date ON execution_quality (ticket_date);

-- BROKER RECONCILIATION (Part 3). The internal book is our own arithmetic; the
-- broker statement is ground truth. Divergence between them is how a trade
-- record quietly stops describing reality. One row per reconciliation run.
CREATE TABLE IF NOT EXISTS broker_reconciliation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recon_date DATE,
    checked_at DATETIME,
    ok INTEGER,                -- 1 if within tolerance
    n_position_breaks INTEGER,
    cash_internal REAL,
    cash_broker REAL,
    cash_diff REAL,
    detail TEXT                -- human-readable break list
);

-- ENSEMBLE regime signal (the live regime source as of 2026-08).
-- A single HMM fit is too noisy to trade: refitting on identical data with a
-- different EM seed swung BEAR-day counts 478-790 and the overlay's
-- contribution by ~10pp of CAGR (analysis/regime_stability.py). Averaging
-- P(BEAR) over independent refits cancels that estimation noise -- two disjoint
-- ensembles agree where single fits did not -- and yields a CONTINUOUS
-- confidence, so days the fits disagree get partial de-risking instead of an
-- arbitrary on/off. Live exposure = 1 - p_bear.
CREATE TABLE IF NOT EXISTS regime_ensemble (
    date DATE PRIMARY KEY,
    p_bear REAL,          -- fraction of independent refits labelling the day BEAR
    n_fits INTEGER,       -- how many fits voted (more = less estimation noise)
    computed_at DATETIME
);

-- Outcome of every scheduled/manual job run. The audit's worst real-world
-- failure was SILENT: the daily job died on a holiday data glitch and nobody
-- noticed for 3 days -- during which a BEAR de-risk signal could never have
-- reached the user. A job that fails must leave a record that something is
-- watching, so ops/health.py can shout about it.
CREATE TABLE IF NOT EXISTS job_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job TEXT,             -- 'daily_run', etc.
    started_at DATETIME,
    finished_at DATETIME,
    ok INTEGER,           -- 1 success, 0 failure
    as_of_date DATE,      -- the session the run produced a book for (if any)
    message TEXT          -- error text on failure, summary on success
);
CREATE INDEX IF NOT EXISTS idx_job_runs_finished ON job_runs (finished_at);

-- Daily snapshot of YOUR book (positions + cash) next to the system's book,
-- written once per day by the daily report. This is what lets the report get
-- "smart": with a value for your book AND the system's book on each date, it
-- can measure -- not assert -- whether your accept/decline/own choices helped.
-- One row per date; INSERT OR REPLACE keeps it idempotent within a day.
CREATE TABLE IF NOT EXISTS user_daily_snapshot (
    date DATE PRIMARY KEY,
    user_value REAL,      -- your positions marked to latest close + cash
    cash REAL,
    n_positions INTEGER,
    decision TEXT,        -- your decision logged for that day's suggestion (or NULL)
    regime TEXT,
    exposure REAL,
    system_nav REAL       -- the system paper book's NAV that day (for comparison)
);

-- Secondary indexes for the date/time columns filtered or ordered on most
-- (the composite primary keys above only accelerate ticker-leading lookups;
-- date-range scans like "WHERE date >= ?" or "WHERE published_at >= ?" need
-- their own index to avoid full-table scans as history accumulates).
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON daily_ohlcv (date);
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles (published_at);
CREATE INDEX IF NOT EXISTS idx_factor_exposures_date ON factor_exposures (date);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals (date);
CREATE INDEX IF NOT EXISTS idx_daily_sentiment_date ON daily_sentiment (date);
CREATE INDEX IF NOT EXISTS idx_macro_events_date ON macro_events (date);
"""


@contextmanager
def get_connection():
    """Yield a sqlite3 connection with row factory + concurrency PRAGMAs.

    Uses a context manager so callers get transactional commit/rollback
    without every call site having to remember to close the connection.

    WAL (write-ahead logging) lets readers and a writer proceed
    concurrently instead of blocking on the default rollback journal, and
    a 30s busy timeout makes a transient lock wait-and-retry rather than
    immediately raising "database is locked" -- both matter once data
    collection, scraping and modeling can run in parallel.
    """
    # ESIS_DB_READONLY=1 (set by the MCP server, esis_mcp): every connection opened
    # here is read-only at the ENGINE level, so no code path reached through a
    # tool can write. The watchlist's own writer opens a separate connection
    # guarded to its four tables (ops/watchlist._writer).
    if os.environ.get("ESIS_DB_READONLY") == "1":
        conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON;")
    else:
        conn = sqlite3.connect(config.DB_PATH, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate_add_missing_primary_keys(conn):
    """Backfill primary keys onto corporate_actions / portfolio_history for
    databases created before those PKs were added to the schema. CREATE
    TABLE IF NOT EXISTS never alters an existing table, so we detect a
    missing PK and rebuild -- but ONLY when the table is empty, so this can
    never destroy data (both tables are unpopulated until Layers 1/7 run).
    """
    for table in ("corporate_actions", "portfolio_history"):
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        has_pk = any(col["pk"] for col in info)
        if info and not has_pk:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if n == 0:
                conn.execute(f"DROP TABLE {table}")
                logger.info("Migration: dropped empty %s to recreate it with a primary key", table)
            else:
                logger.warning("Migration: %s lacks a primary key but has %d rows -- skipping "
                               "rebuild to avoid data loss; deduplicate and migrate manually", table, n)


def _migrate_add_regime_vintage_column(conn):
    """Add regime_history.param_vintage_date to pre-existing databases (records
    which walk-forward HMM parameter fit produced each regime row). ALTER TABLE
    ADD COLUMN is safe/idempotent-guarded here by an existence check."""
    cols = {c["name"] for c in conn.execute("PRAGMA table_info(regime_history)")}
    if cols and "param_vintage_date" not in cols:
        conn.execute("ALTER TABLE regime_history ADD COLUMN param_vintage_date DATE")
        logger.info("Migration: added regime_history.param_vintage_date")


def init_db():
    """Create every table/index in SCHEMA if absent, then run lightweight
    migrations for pre-existing databases. Idempotent."""
    with get_connection() as conn:
        _migrate_add_missing_primary_keys(conn)
        _migrate_add_regime_vintage_column(conn)
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", config.DB_PATH)


def list_tables():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [r["name"] for r in rows]


def seed_stocks(seed_list=None):
    """Insert/refresh the seed ticker list into `stocks`. Does not overwrite
    fields a later, more authoritative source may have already populated
    (uses INSERT OR IGNORE, then a targeted UPDATE only for NULL columns).
    """
    seed_list = seed_list if seed_list is not None else config.EGX_SEED_TICKERS
    with get_connection() as conn:
        for row in seed_list:
            conn.execute(
                "INSERT OR IGNORE INTO stocks (ticker, name_english) VALUES (?, ?)",
                (row["ticker"], row["name_english"]),
            )
            if row["name_english"] is not None:
                conn.execute(
                    "UPDATE stocks SET name_english = ? "
                    "WHERE ticker = ? AND name_english IS NULL",
                    (row["name_english"], row["ticker"]),
                )
    logger.info("Seeded %d tickers into stocks table", len(seed_list))
