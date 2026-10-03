"""
Single, DB-level definition of the tradable universe (Phase A3).

Before this, liquidity/staleness filtering lived only inside TFT windowing
(config.TFT_MAX_STALE_FRAC), so every other layer either traded dead names
(the ORAS problem: ~99% zero-return days in its test span) or invented its
own ad-hoc threshold. This module gives ONE `is_tradable(ticker, date)`
rule, materializes it into a `tradable_universe` table, and exposes helpers
every consumer (factor model, causal signals, HMM features, backtest) can
share so the investable set is identical across layers on any given date.

Tradable on date T = over the trailing window:
  * median EGP value traded >= TRADABLE_MIN_MEDIAN_VALUE (real liquidity), AND
  * fraction of zero-return days <= config.TFT_MAX_STALE_FRAC (not halted/stale).
The thresholds mirror the existing TFT staleness filter so the two never
diverge; TRADABLE_MIN_MEDIAN_VALUE is stated here as the one liquidity floor.
"""

import logging

import numpy as np
import pandas as pd

import config
from data.database import get_connection

logger = logging.getLogger("esis_quant.universe")

TRADABLE_WINDOW_DAYS = 60
TRADABLE_MIN_MEDIAN_VALUE = 50_000.0   # EGP median daily value traded over the window

# Sector classification for the seed universe. These are stable, public facts
# about large, well-known EGX-listed companies (unlike corporate-action dates,
# a company's sector is not something that needs a live feed). Tickers whose
# identity config.ARABIC_COMPANY_NAMES explicitly flags as UNCONFIRMED are left
# as None rather than guessed -- honesty over coverage (house rule 7).
SECTOR_MAP = {
    "COMI": ("Financials", "Banks"), "ADIB": ("Financials", "Banks"),
    "HRHO": ("Financials", "Investment Services"), "EFIH": ("Financials", "Investment Services"),
    "EKHO": ("Financials", "Holding"), "FWRY": ("Technology", "Fintech"),
    "SWDY": ("Industrials", "Electrical Equipment"), "ORAS": ("Industrials", "Construction"),
    "TMGH": ("Real Estate", "Development"), "ORHD": ("Real Estate", "Development"),
    "PHDC": ("Real Estate", "Development"),
    "ABUK": ("Materials", "Fertilizers"), "MFPC": ("Materials", "Fertilizers"),
    "SKPC": ("Materials", "Petrochemicals"), "AMOC": ("Energy", "Refining"),
    "ETEL": ("Communication Services", "Telecom"),
    "EAST": ("Consumer Staples", "Tobacco"), "JUFO": ("Consumer Staples", "Food & Beverage"),
    "EFID": ("Consumer Staples", "Food & Beverage"), "SUGR": ("Consumer Staples", "Food & Beverage"),
    "OLFI": ("Consumer Staples", "Food & Beverage"),
    "ORWE": ("Consumer Discretionary", "Textiles"), "CIRA": ("Consumer Discretionary", "Education"),
    "ISPH": ("Health Care", "Diagnostics"), "CLHO": ("Health Care", "Hospitals"),
    "EGAL": ("Materials", "Aluminium"),
    # ELSH: identity UNCONFIRMED in config -> intentionally omitted (stays NULL)
}


def populate_sectors() -> int:
    """Write SECTOR_MAP into stocks.sector / stocks.sub_sector. Returns the
    number of tickers updated. Leaves unlisted tickers untouched (NULL).
    """
    with get_connection() as conn:
        n = 0
        for ticker, (sector, sub) in SECTOR_MAP.items():
            cur = conn.execute("UPDATE stocks SET sector=?, sub_sector=? WHERE ticker=?",
                               (sector, sub, ticker))
            n += cur.rowcount
    logger.info("populate_sectors: set sector/sub_sector for %d tickers", n)
    return n


def live_universe_tickers() -> set:
    """The ticker set the LIVE book is allowed to trade.

    WHY THIS EXISTS. On 2026-08-06 the research panel was widened from 51 to 198
    tickers so a point-in-time universe could be measured (data/pit_universe.py).
    `build_tradable_universe` reads everything in daily_ohlcv, so it silently
    made all 198 tradable -- the live book went from 27 to 73 positions without
    anyone deciding that. It was corrected by hand, and then a later rebuild
    undid the correction, because a manual DELETE is not a policy.

    The live universe is now DECLARED here rather than repaired afterwards.
    Widening it is a deliberate edit to this function, not a side effect of
    ingesting research data.
    """
    # WIDENED 2026-08-10 AT THE OPERATOR'S EXPLICIT REQUEST, from the 47-name
    # hand-picked seed to the FULL EGX listing that has price history.
    #
    # This is the change this function exists to make deliberate. Two things
    # follow from it and neither is a side effect:
    #
    #   * IT REMOVES SELECTION BIAS from the live book. The seed list was 20
    #     names chosen on 2026 liquidity and applied back to 2015; the
    #     point-in-time study (data/pit_universe.py) measured that bias at about
    #     10pp of CAGR and 0.7 of Sharpe.
    #   * IT WILL THEREFORE LOWER REPORTED PERFORMANCE. The narrow universe's
    #     numbers were inflated BY the bias. A broader universe is more honest,
    #     not better. Expect the backtest figures to fall, and understand that
    #     the fall is the correction, not a degradation.
    #
    # Membership is still date-gated by the liquidity/staleness filter below, so
    # thin names are excluded on the days they are thin, and delisted names drop
    # out automatically after their final print.
    if getattr(config, "LIVE_UNIVERSE_FULL_EGX", False):
        with get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT ticker FROM daily_ohlcv").fetchall()
        return {r[0] for r in rows}

    live = {t["ticker"] for t in config.EGX_SEED_TICKERS}
    # delisted names recovered 2026-08-05 for the survivorship correction; they
    # are date-limited by the liquidity filter and cannot trade after delisting
    live |= {"ESRS", "UASG", "SUCE", "OCIC"}
    return live


def build_tradable_universe(window: int = TRADABLE_WINDOW_DAYS,
                            tickers: set = None,
                            all_tickers: bool = False) -> pd.DataFrame:
    """Compute is_tradable for every (ticker, date) and materialize it into
    the tradable_universe table. Returns the long DataFrame.

    Restricted to `live_universe_tickers()` by DEFAULT. Pass `all_tickers=True`
    for research over the full panel -- that path must never write the live
    table, so callers doing research should use data/pit_universe.py instead.
    """
    with get_connection() as conn:
        px = pd.read_sql("SELECT ticker, date, adjusted_close, value_traded_egp FROM daily_ohlcv "
                         "ORDER BY ticker, date", conn, parse_dates=["date"])
    if not all_tickers:
        keep = set(tickers) if tickers is not None else live_universe_tickers()
        before = px["ticker"].nunique()
        px = px[px["ticker"].isin(keep)]
        logger.info("build_tradable_universe: restricted %d -> %d tickers (live universe)",
                    before, px["ticker"].nunique())
    if px.empty:
        # An empty panel means the restriction (or the DB) left nothing to
        # evaluate. pd.concat([]) raises, so without this the caller gets an
        # opaque ValueError instead of an empty universe it can check for.
        logger.warning("build_tradable_universe: no price rows after filtering — "
                       "returning an empty universe (nothing will be tradable)")
        return pd.DataFrame(columns=["ticker", "date", "is_tradable"])
    parts = []
    for ticker, g in px.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        ret = g["adjusted_close"].pct_change()
        stale = (ret.abs() < 1e-9).astype(float)
        med_val = g["value_traded_egp"].rolling(window, min_periods=window // 2).median()
        stale_frac = stale.rolling(window, min_periods=window // 2).mean()
        # Staleness gate: TRADABLE_MAX_STALE_FRAC, not TFT_MAX_STALE_FRAC.
        # The TFT constant (0.50) was set for a hand-picked large-cap list and
        # admits names that do not move on half their sessions -- exactly what
        # fools the covariance estimator once the universe is widened.
        max_stale = getattr(config, 'TRADABLE_MAX_STALE_FRAC', config.TFT_MAX_STALE_FRAC)
        tradable = ((med_val >= TRADABLE_MIN_MEDIAN_VALUE) & (stale_frac <= max_stale))
        parts.append(pd.DataFrame({"ticker": ticker, "date": g["date"],
                                   "is_tradable": tradable.fillna(False).astype(int)}))
    uni = pd.concat(parts, ignore_index=True)

    with get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS tradable_universe "
                     "(ticker TEXT, date DATE, is_tradable INTEGER, PRIMARY KEY (ticker, date))")
        conn.execute("DELETE FROM tradable_universe")
        conn.executemany("INSERT INTO tradable_universe (ticker, date, is_tradable) VALUES (?, ?, ?)",
                         [(r.ticker, r.date.strftime("%Y-%m-%d"), int(r.is_tradable)) for r in uni.itertuples()])
    logger.info("build_tradable_universe: %d (ticker,date) rows, %.1f%% tradable",
                len(uni), 100 * uni["is_tradable"].mean())
    return uni


def load_tradable() -> pd.DataFrame:
    """Long (ticker, date, is_tradable) frame; empty if not built yet."""
    with get_connection() as conn:
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tradable_universe'").fetchone()
        if not exists:
            return pd.DataFrame(columns=["ticker", "date", "is_tradable"])
        return pd.read_sql("SELECT ticker, date, is_tradable FROM tradable_universe", conn, parse_dates=["date"])


def tradable_mask(scores_or_panel: pd.DataFrame) -> pd.DataFrame:
    """Inner-join a (ticker, date, ...) frame against the tradable universe,
    keeping only rows that are tradable on that date. If the universe table
    isn't built, returns the input unchanged (with a warning) so callers
    degrade gracefully rather than silently dropping everything.
    """
    uni = load_tradable()
    if uni.empty:
        logger.warning("tradable_mask: tradable_universe not built -- returning input unfiltered")
        return scores_or_panel
    merged = scores_or_panel.merge(uni[uni["is_tradable"] == 1][["ticker", "date"]],
                                   on=["ticker", "date"], how="inner")
    return merged


# --- mandatory bias disclosure  (audit P2.5) --------------------------------

def survivorship_disclosure(short: bool = False) -> str:
    """The universe-bias statement that MUST head every generated report.

    WHY IT IS A FUNCTION AND NOT A DOCSTRING. The 2026-08-05 audit noted the
    bias was documented in data/universe_discovery.py -- where nobody reading a
    performance number would ever see it. A disclosure that lives next to the
    code instead of next to the number is not a disclosure. This is emitted into
    report headers so it cannot be read past.

    IT IS DISCLOSURE, NOT REMEDIATION. Four delisted names (ESRS, UASG, SUCE,
    OCIC) were recovered on 2026-08-05 and are now in the panel, which lowered
    core CAGR by 3.18pp. That is a LOWER BOUND on the bias: those four were
    found by guessing ticker symbols, so an unknown number remain missing.
    """
    if short:
        return ("UNIVERSE IS SURVIVORSHIP- AND SELECTION-BIASED — 20 of 47 names were "
                "chosen on 2026 liquidity and applied back to 2015; only 4 delisted "
                "names have been recovered. Treat all figures as upper bounds.")
    return (
        "!! UNIVERSE BIAS — READ BEFORE USING ANY NUMBER BELOW !!\n"
        "  SELECTION BIAS: 20 of the 47 seed names (config.EGX_SEED_TICKERS) were added\n"
        "    on 2026-07-28, ranked by liquidity AS OF 2026, then backtested from 2015.\n"
        "    Names that were liquid in 2015 but faded before 2026 were never considered.\n"
        "  SURVIVORSHIP BIAS: partially corrected. ESRS, UASG, SUCE and OCIC were\n"
        "    recovered 2026-08-05 and are in the panel; including them cut core CAGR\n"
        "    from 25.84% to 22.66% and turned system alpha negative. They were found by\n"
        "    GUESSING ticker symbols, so an unknown number of delisted names are still\n"
        "    missing and the true bias is LARGER than the 3.18pp already measured.\n"
        "  CONSEQUENCE: every CAGR, Sharpe, alpha and cost figure produced by this\n"
        "    system is an UPPER BOUND on what was actually achievable."
    )
