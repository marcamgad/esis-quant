import pytest

import config
from data import database


EXPECTED_TABLES = {
    "stocks", "daily_ohlcv", "intraday_stats", "corporate_actions",
    "macro_daily", "macro_events", "news_articles", "factor_exposures",
    "signals", "portfolio_history", "performance_attribution",
    "regime_history", "regime_statistics", "daily_sentiment",
}


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_esis.db"))
    database.init_db()
    return config.DB_PATH


def test_init_db_creates_all_expected_tables(temp_db):
    assert EXPECTED_TABLES.issubset(set(database.list_tables()))


def test_init_db_is_idempotent(temp_db):
    database.init_db()
    database.init_db()
    assert EXPECTED_TABLES.issubset(set(database.list_tables()))


def test_seed_stocks_inserts_rows(temp_db):
    seed = [
        {"ticker": "AAA", "name_english": "Alpha Co"},
        {"ticker": "BBB", "name_english": None},
    ]
    database.seed_stocks(seed)

    with database.get_connection() as conn:
        rows = conn.execute("SELECT ticker, name_english FROM stocks ORDER BY ticker").fetchall()

    assert [dict(r) for r in rows] == [
        {"ticker": "AAA", "name_english": "Alpha Co"},
        {"ticker": "BBB", "name_english": None},
    ]


def test_seed_stocks_does_not_clobber_existing_name(temp_db):
    database.seed_stocks([{"ticker": "AAA", "name_english": None}])
    with database.get_connection() as conn:
        conn.execute("UPDATE stocks SET name_english = 'Manually Verified Name' WHERE ticker = 'AAA'")

    # Re-seeding with a None name should not wipe out the manually-set name.
    database.seed_stocks([{"ticker": "AAA", "name_english": None}])

    with database.get_connection() as conn:
        row = conn.execute("SELECT name_english FROM stocks WHERE ticker = 'AAA'").fetchone()
    assert row["name_english"] == "Manually Verified Name"
