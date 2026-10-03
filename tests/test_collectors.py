import pytest

import config
from data import database
from data.collectors import DataCollector, extract_tbill_discount_rate


def test_extract_tbill_discount_rate_matches_real_headline():
    # Real headline scraped from almalnews.com on 2026-07-07.
    text = "البورصة تتيح تداول أذون خزانة مصرية بقيمة 25.19 مليار جنيه بمعدل خصم 24.7% بدءًا من اليوم"
    assert extract_tbill_discount_rate(text) == [24.7]


def test_extract_tbill_discount_rate_no_false_positive_on_unrelated_mention():
    # "خصم" (discount/deduction) appears without the full "معدل خصم N%"
    # phrase -- must not match.
    text = '"أرابيا للاستثمار" تسدل الستار على قضية خصم مستحقات بقيمة 6.56 مليون جنيه'
    assert extract_tbill_discount_rate(text) == []


def test_extract_tbill_discount_rate_handles_no_match_text():
    assert extract_tbill_discount_rate("مخاوف الذكاء الاصطناعي تضغط على العقود الآجلة") == []


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_esis.db"))
    database.init_db()
    return config.DB_PATH


def _insert_article(headline, published_at, url):
    with database.get_connection() as conn:
        conn.execute(
            """INSERT INTO news_articles (source, headline_arabic, body_arabic, url, published_at)
               VALUES ('almalnews', ?, '', ?, ?)""",
            (headline, url, published_at),
        )


def test_scan_news_for_tbill_rates_stores_match(temp_db):
    _insert_article(
        "البورصة تتيح تداول أذون خزانة مصرية بقيمة 25.19 مليار جنيه بمعدل خصم 24.7% بدءًا من اليوم",
        "2026-07-07T13:22:02", "https://example.com/1",
    )
    _insert_article("مخاوف الذكاء الاصطناعي تضغط على العقود الآجلة", "2026-07-07T13:23:17", "https://example.com/2")

    result = DataCollector().scan_news_for_tbill_rates()
    assert result["stored"] == 1

    with database.get_connection() as conn:
        row = conn.execute(
            "SELECT date, actual_value FROM macro_events WHERE event_type = 'EGY_TBILL_DISCOUNT_RATE'"
        ).fetchone()
    assert row["date"] == "2026-07-07"
    assert row["actual_value"] == 24.7


def test_scan_news_for_tbill_rates_is_idempotent(temp_db):
    _insert_article(
        "البورصة تتيح تداول أذون خزانة مصرية بقيمة 25.19 مليار جنيه بمعدل خصم 24.7% بدءًا من اليوم",
        "2026-07-07T13:22:02", "https://example.com/1",
    )
    collector = DataCollector()
    collector.scan_news_for_tbill_rates()
    second_result = collector.scan_news_for_tbill_rates()  # re-run should not duplicate
    assert second_result["stored"] == 0

    with database.get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM macro_events WHERE event_type = 'EGY_TBILL_DISCOUNT_RATE'"
        ).fetchone()[0]
    assert count == 1


def test_arabic_ticker_matching():
    from data.collectors import DataCollector
    import config

    # _match_tickers is a static method, call directly
    # Ensure config.ARABIC_COMPANY_NAMES is populated (Fix 2 must be done first)
    assert hasattr(config, "ARABIC_COMPANY_NAMES"), \
        "ARABIC_COMPANY_NAMES missing from config -- Fix 2 not applied"

    name_dict = {}  # empty English dict -- testing Arabic path only

    result = DataCollector._match_tickers("البنك التجاري الدولي يعلن نتائجه", name_dict)
    assert "COMI" in result

    result = DataCollector._match_tickers("هيرميس تستحوذ على شركة جديدة", name_dict)
    assert "HRHO" in result

    result = DataCollector._match_tickers("لا يوجد اسم شركة هنا", name_dict)
    assert result == []

    result = DataCollector._match_tickers(
        "البنك التجاري الدولي وهيرميس يتعاونان", name_dict
    )
    assert "COMI" in result
    assert "HRHO" in result
    assert len(result) == len(set(result))  # no duplicates


def test_retag_existing_articles_updates_stale_empty_mentions(temp_db):
    # Simulates an article scraped BEFORE the Arabic matcher existed --
    # stored with an empty tickers_mentioned that should get backfilled.
    _insert_article("البنك التجاري الدولي يعلن نتائجه", "2026-07-07T10:00:00", "https://example.com/1")

    result = DataCollector().retag_existing_articles()
    assert result["updated"] == 1

    with database.get_connection() as conn:
        row = conn.execute("SELECT tickers_mentioned FROM news_articles WHERE url = 'https://example.com/1'").fetchone()
    assert row["tickers_mentioned"] == "COMI"


def test_retag_existing_articles_is_idempotent(temp_db):
    _insert_article("البنك التجاري الدولي يعلن نتائجه", "2026-07-07T10:00:00", "https://example.com/1")
    collector = DataCollector()
    collector.retag_existing_articles()
    second_result = collector.retag_existing_articles()
    assert second_result["updated"] == 0  # already correctly tagged, nothing to change
