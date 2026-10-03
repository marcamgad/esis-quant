"""
News collector tests. Every false-positive case here is a real headline the
first versions of the linker attached to the wrong company.

A wrong link is worse than no link: the impact stage would reason about the
wrong company's income statement and present the result with confidence.
"""

import pytest

from data import news_collector as nc

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Orascom Investment appoints Naguib Sawiris as Chairman - Reuters</title>
<link>https://example.com/a</link><pubDate>Thu, 20 Aug 2026 10:00:00 GMT</pubDate>
<source url="https://reuters.com">Reuters</source><description>&lt;b&gt;Board&lt;/b&gt; change</description></item>
<item><title></title><link>https://example.com/empty</link></item>
</channel></rss>"""


def test_parse_rss_strips_publisher_suffix_and_html():
    items = nc.parse_rss(RSS)
    assert len(items) == 1                      # the empty-title item is dropped
    it = items[0]
    assert it["headline"] == "Orascom Investment appoints Naguib Sawiris as Chairman"
    assert it["publisher"] == "Reuters"
    assert it["summary"] == "Board change"
    assert it["published_at"] is not None


def test_parse_rss_survives_malformed_xml():
    assert nc.parse_rss("<rss><item><title>broken") == []


def test_link_name_keeps_holding_group_and_egypt():
    """Stripping these produced 'Valmore' and 'Telecom', which matched strangers."""
    assert nc._link_name("Valmore Holding S.A.E.") == "Valmore Holding"
    assert nc._link_name("Telecom Egypt Company") == "Telecom Egypt"
    assert nc._link_name("Korra for Energy and Investment Projects (S.A.E.)") == \
        "Korra for Energy and Investment Projects"


@pytest.fixture(scope="module")
def index():
    return nc.build_name_index()


@pytest.fixture(scope="module")
def valid():
    from data.database import get_connection
    with get_connection() as conn:
        return {r[0] for r in conn.execute("SELECT ticker FROM stocks")}


@pytest.mark.parametrize("headline,wrong", [
    ("Valmore L. Girard Obituary Jan 24, 2012", "VLMR"),
    ("Valmore Vachon", "VLMRA"),
    ("Telecom infrastructure firm GTS to go public via SPAC in $400M deal", "ETEL"),
    ("محافظ الأقصر يوجه باستمرار الحملات الرقابية والتعامل الفوري مع أي مخالفات تموينية", "FWRY"),
    ("Abu Dhabi Catalyst Partners takes stake in ADGM-based private equity firm Olive Rock", "CPME"),
])
def test_known_false_positives_do_not_link(index, valid, headline, wrong):
    assert wrong not in nc.link_text(headline, index, valid)


@pytest.mark.parametrize("headline,right", [
    ("Orascom Investment Holding swings deeper into the red despite a 4x revenue surge", "OIH"),
    ("Valmore Holding Revenues Grow 24% YoY in 2025", "VLMR"),
    ("Korra Energi’s 11% IPO ends EGX’s wait-and-see", "KORA"),
    ("\"نعيم\" يحدد القيمة العادلة لسهم \"أموك\" عند 14 جنيهاً", "AMOC"),
    ("Orascom Investment Holding S.A.E.(CASE: OIH) dropped from EGX 30 Index", "OIH"),
])
def test_genuine_mentions_do_link(index, valid, headline, right):
    assert right in nc.link_text(headline, index, valid)


def test_exchange_code_form_links(valid):
    got = nc.link_text("Shares of (CASE: AMOC) rose", [], valid)
    assert got.get("AMOC") == "code"


def test_query_aliases_cover_every_position_that_returned_nothing():
    for t in ("KORA", "CRST", "KASABF"):
        assert nc.QUERY_ALIASES.get(t), f"{t} returned zero items without an alias"


def test_hash_is_stable():
    assert nc._hash("https://a.b/c ") == nc._hash("https://a.b/c")
