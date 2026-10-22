"""Agent 4 (disclosure reader, rules stage): extraction cases from real headlines."""

import pytest

from data import disclosure_reader as dr
from data.corpus_collector import parse_article


@pytest.mark.parametrize("text,ticker,expect", [
    ("أرباح السويدي إليكتريك ترتفع إلى 10.6 مليار جنيه خلال النصف الأول من 2026", "SWDY",
     {"kind": "result", "metric": "net_profit", "value_m": 10600.0, "period": "H1", "fiscal_year": 2026}),
    ("Juhayna’s profits hit LE71.6M during Q1 2019", "JUFO",
     {"metric": "net_profit", "value_m": 71.6, "period": "Q1", "fiscal_year": 2019, "currency": "EGP"}),
    ("SIDPEC Net profits Surge to EGP 2.5B in 2024 Despite Global Challenges", "SKPC",
     {"metric": "net_profit", "value_m": 2500.0, "period": "FY", "fiscal_year": 2024}),
    ("أرباح الدلتا للسكر تتراجع 85% إلى 183.9 مليون جنيه بالعام المالي 2025/2026", "SUGR",
     {"value_m": 183.9, "pct_change": -85.0, "period": "FY", "period_end": "2026-06-30"}),
    ("هيبكو للاستثمارات تحقق صافي ربح 41.259 مليون جنيه مقابل 19.337 مليون جنيه", "HBCO",
     {"value_m": 41.259, "prior_m": 19.337}),
    ("المصرية للاتصالات تسجل ٣٫٦ مليار جنيه صافي أرباح في الربع الأول", "ETEL",
     {"value_m": 3600.0, "period": "Q1"}),                                  # Arabic-Indic digits
])
def test_extracts_results(text, ticker, expect):
    got = dr.extract_figures(text, ticker, seen="2026-08-01")[0]
    for k, v in expect.items():
        assert got[k] == pytest.approx(v) if isinstance(v, float) else got[k] == v, k


def test_refuses_to_call_targets_results():
    got = dr.extract_figures('"أموك" تستهدف ربح 1.1 مليار جنيه بموازنة 2025-2026', "AMOC")[0]
    assert got["kind"] == "guidance" and got["confidence"] < 0.6


def test_developer_sales_are_presales():
    got = dr.extract_figures('"طلعت مصطفى" المصرية تحقق مبيعات بـ 219.1 مليار جنيه في 6 أشهر', "TMGH")[0]
    assert got["metric"] == "presales" and got["period"] == "H1"


def test_foreign_currency_flags_a_probable_wrong_link():
    got = dr.extract_figures("Arabian Cement sees profits fall to SAR44.1m in 1H25", "ARCC")[0]
    assert got["currency"] == "SAR" and got["confidence"] <= 0.3


def test_losses_are_negative():
    got = dr.extract_figures("خسائر الشركة تبلغ 526 مليون جنيه خلال 2025", "X")[0]
    assert got["metric"] == "net_loss" and got["value_m"] == -526.0


def test_interface_logs_every_call():
    dr._CALL_BUFFER.clear()
    dr.extract("figures", "أرباح 5 مليون جنيه في الربع الأول", ticker="X")
    assert len(dr._CALL_BUFFER) == 1 and dr._CALL_BUFFER[0][1] == "figures"


def test_analyst_forecasts_are_not_results():
    got = dr.extract_figures("بنوك استثمار: زيادة الأسعار تقفز بأرباح «النساجون الشرقيون» 7.8% خلال الربع الأول",
                             "ORWE")[0]
    assert got["kind"] == "guidance"


def test_earnings_slug_selects_results_articles():
    import urllib.parse
    from data.corpus_collector import EARNINGS_SLUG
    yes = "https://almalnews.com/2113972/" + urllib.parse.quote("«هيبكو»-تحقق-نموًا-41-6-في-أرباح") + "/"
    no = "https://almalnews.com/2139038/" + urllib.parse.quote("برلمانية-منتدى-العلمين-أفريقيا") + "/"
    assert EARNINGS_SLUG.search(urllib.parse.unquote(yes))
    assert not EARNINGS_SLUG.search(urllib.parse.unquote(no))


def test_corpus_article_parse():
    html = ('<meta property="og:title" content="عنوان">'
            '<meta property="article:published_time" content="2026-10-01T10:00:00+03:00">'
            '<p>الفقرة <b>الأولى</b></p><p>الفقرة الثانية &amp; المزيد</p>')
    a = parse_article(html)
    assert a["title"] == "عنوان" and a["published_at"].startswith("2026-10-01")
    assert a["body"] == "الفقرة الأولى الفقرة الثانية & المزيد"
