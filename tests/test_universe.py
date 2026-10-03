"""
Tests for the Phase-A3 tradable-universe definition. Known-answer: a liquid,
actively-trading name is tradable; a stale (all-zero-return) or illiquid name
is not; and populate_sectors writes the expected classification.
"""

import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from data import universe


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "u.db"))
    database.init_db()
    database.seed_stocks()
    return config.DB_PATH


def _insert_prices(ticker, dates, close, value):
    with get_connection() as conn:
        conn.executemany(
            "INSERT INTO daily_ohlcv (ticker, date, close, adjusted_close, volume, value_traded_egp) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            [(ticker, d.strftime("%Y-%m-%d"), c, c, v) for d, c, v in zip(dates, close, value)],
        )


def test_tradable_flags_liquid_vs_stale(temp_db):
    dates = pd.bdate_range("2020-01-01", periods=120)
    rng = np.random.default_rng(0)
    # LIQUID: moving price, high value traded
    liquid_close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, len(dates)))
    _insert_prices("LIQD", dates, liquid_close, value=np.full(len(dates), 5_000_000.0))
    # STALE: flat price (zero returns), decent value -> fails the staleness test
    _insert_prices("STAL", dates, np.full(len(dates), 50.0), value=np.full(len(dates), 5_000_000.0))
    # ILLIQUID: moving price but tiny value traded -> fails the liquidity floor
    _insert_prices("ILLQ", dates, liquid_close, value=np.full(len(dates), 100.0))

    # LIQUID/STALE are synthetic tickers outside the live universe, so this
    # test must ask for the full panel explicitly (see live_universe_tickers)
    uni = universe.build_tradable_universe(window=60, all_tickers=True)
    last = uni[uni["date"] == uni["date"].max()].set_index("ticker")["is_tradable"]
    assert last["LIQD"] == 1
    assert last["STAL"] == 0
    assert last["ILLQ"] == 0


def test_populate_sectors_writes_known_classifications(temp_db):
    n = universe.populate_sectors()
    assert n > 0
    with get_connection() as conn:
        row = conn.execute("SELECT sector, sub_sector FROM stocks WHERE ticker='COMI'").fetchone()
    assert row["sector"] == "Financials" and row["sub_sector"] == "Banks"


def test_tradable_mask_degrades_gracefully_without_table(temp_db):
    # no tradable_universe table built -> mask returns input unchanged (warned)
    df = pd.DataFrame({"ticker": ["X"], "date": [pd.Timestamp("2020-01-01")], "score": [1.0]})
    out = universe.tradable_mask(df)
    assert len(out) == 1


# --- the live universe must not widen as a side effect of research ingest ----

def test_build_tradable_universe_restricts_to_the_live_set_by_default(tmp_path, monkeypatch):
    """On 2026-08-06 the research panel grew 51 -> 198 tickers so a point-in-time
    universe could be measured. build_tradable_universe reads everything in
    daily_ohlcv, so it silently made all 198 tradable and the live book jumped
    from 27 to 73 positions with nobody deciding that. Restriction is now policy,
    not a manual DELETE anyone has to remember."""
    import config
    from data import database
    from data.database import get_connection
    from data import universe as du

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "u.db"))
    database.init_db()
    dates = pd.bdate_range("2024-01-01", periods=260)
    rows = []
    for t in ("COMI", "RESEARCH_ONLY"):
        for i, d in enumerate(dates):
            rows.append((t, d.strftime("%Y-%m-%d"), 100.0 + i, 100.0 + i, 0, 5e6))
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, "
                         "adjusted_close, volume, value_traded_egp) VALUES (?,?,?,?,?,?)", rows)

    monkeypatch.setattr(du, "live_universe_tickers", lambda: {"COMI"})
    uni = du.build_tradable_universe()
    assert set(uni["ticker"]) == {"COMI"}
    assert "RESEARCH_ONLY" not in set(du.load_tradable()["ticker"])


def test_all_tickers_escape_hatch_still_works(tmp_path, monkeypatch):
    """Research over the full panel must remain possible -- explicitly."""
    import config
    from data import database
    from data.database import get_connection
    from data import universe as du

    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "u2.db"))
    database.init_db()
    dates = pd.bdate_range("2024-01-01", periods=260)
    rows = []
    for t in ("COMI", "RESEARCH_ONLY"):
        for i, d in enumerate(dates):
            rows.append((t, d.strftime("%Y-%m-%d"), 100.0 + i, 100.0 + i, 0, 5e6))
    with get_connection() as conn:
        conn.executemany("INSERT OR REPLACE INTO daily_ohlcv (ticker, date, close, "
                         "adjusted_close, volume, value_traded_egp) VALUES (?,?,?,?,?,?)", rows)

    monkeypatch.setattr(du, "live_universe_tickers", lambda: {"COMI"})
    uni = du.build_tradable_universe(all_tickers=True)
    assert set(uni["ticker"]) == {"COMI", "RESEARCH_ONLY"}


def test_live_universe_includes_the_recovered_delisted_names():
    """The four survivorship names must stay in the live definition -- the
    liquidity filter date-limits them, so they cannot trade after delisting."""
    from data.universe import live_universe_tickers
    live = live_universe_tickers()
    assert {"ESRS", "UASG", "SUCE", "OCIC"} <= live
    assert len(live) >= 47
