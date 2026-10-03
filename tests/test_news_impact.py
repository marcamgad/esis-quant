"""
Stage 3 tests. The misclassification cases are real headlines the first rule set
got wrong on positions the operator holds; the fundamentals cases pin the
requirement that the same event can be good for one company and bad for another.
"""

import pandas as pd
import pytest

from analysis import news_impact as ni

PROFITABLE = {"net_income": 1488.0, "revenue": 37623.0, "rev_growth": 0.11, "loss_streak": 0,
              "op_cash_flow": 900.0, "period_end": "2025-06-30"}
LOSS_MAKING = {"net_income": -534.0, "revenue": 1394.0, "rev_growth": 1.90, "loss_streak": 3,
               "op_cash_flow": -120.0, "period_end": "2025-12-31"}


def call(text, fund=PROFITABLE, price=None, link="text"):
    return ni.assess(text, "TEST", "2026-08-01", fund, price, link)


# ---- real misclassifications -------------------------------------------------

@pytest.mark.parametrize("headline,not_type", [
    ("مساهم رئيسي يرفع حصته في أسهم كرستمارك للمقاولات والتطوير العقاري إلى 15.3%", "debt_financing"),
    ("إيقاف التداول على 11 سهمًا بالبورصة لتجاوز الحدود السعرية", "legal_regulatory"),
    ("Dr. Mahendra Kumar Mishra awarded Padma Shri 2026 for Literature and Education", "contract_win"),
])
def test_first_version_mistakes_do_not_recur(headline, not_type):
    assert call(headline)["event_type"] != not_type


def test_stake_increase_is_recognised():
    r = call("مساهم رئيسي يرفع حصته في أسهم كرستمارك للمقاولات والتطوير العقاري إلى 15.3%")
    assert r["event_type"] == "stake_increase" and r["direction"] == 1


def test_rationale_never_contradicts_direction():
    """A favourable call explained as 'no clear direction' is a bug in the words."""
    for h in ("Major shareholder raises stake to 15%", "Company launches share buyback",
              "Fitch upgrades company rating", "Major shareholder cuts stake to 5%"):
        r = call(h)
        if r["direction"] != 0:
            assert "no clear direction" not in r["rationale"], h


def test_price_limit_halt_is_neutral():
    r = call("إيقاف التداول على 11 سهمًا بالبورصة لتجاوز الحدود السعرية")
    assert r["event_type"] == "price_limit_halt" and r["direction"] == 0


def test_search_only_links_are_never_more_than_low_confidence():
    r = call("AMOC profit jumps 109% in first half", link="query")
    assert r["confidence"] == "low"


# ---- same event, different company ------------------------------------------

def test_capital_increase_depends_on_profitability():
    h = "Company announces capital increase via rights issue"
    assert call(h, PROFITABLE)["direction"] == 1
    assert call(h, LOSS_MAKING)["direction"] == -1


def test_debt_financing_depends_on_profitability():
    h = "Company signs EGP 2 billion financing facility with banks"
    assert call(h, LOSS_MAKING)["direction"] == -1
    assert call(h, PROFITABLE)["direction"] == 0


def test_dividend_from_a_loss_maker_is_not_favourable():
    h = "Board approves cash dividend of EGP 0.5 per share"
    assert call(h, PROFITABLE)["direction"] == 1
    assert call(h, LOSS_MAKING)["direction"] == 0


def test_revenue_growth_at_a_loss_maker_is_not_favourable():
    assert call("Revenue surges 190% year on year", LOSS_MAKING)["direction"] == 0


def test_earnings_outrank_revenue_in_the_same_headline():
    r = call("Orascom Investment Holding swings deeper into the red despite a 4x revenue surge", LOSS_MAKING)
    assert r["event_type"] == "earnings_down" and r["direction"] == -1


def test_profit_after_losses_is_flagged_as_turnaround():
    r = call("Company swings to profit in first half", LOSS_MAKING)
    assert r["event_type"] == "earnings_up" and r["strength"] == 3 and "turnaround" in r["rationale"]


def test_rationale_cites_the_numbers_used():
    r = call("Company announces capital increase", LOSS_MAKING)
    assert "-534" in r["rationale"] and "3 loss year" in r["rationale"]


def test_missing_fundamentals_lowers_confidence_for_context_dependent_events():
    assert call("Company announces capital increase", {})["confidence"] != "high"


# ---- fair value & sizing ------------------------------------------------------

def test_fair_value_compared_with_price_on_the_day():
    up = call('"نعيم" يحدد القيمة العادلة لسهم "أموك" عند 14 جنيهاً', price=11.0)
    flat = call('"نعيم" يحدد القيمة العادلة لسهم "أموك" عند 14 جنيهاً', price=13.8)
    assert up["direction"] == 1 and flat["direction"] == 0


def test_contract_sized_against_revenue():
    r = call("El Sewedy wins EGP 1.5 billion cable contract",
             {**PROFITABLE, "revenue": 3000.0})
    assert "50% of annual revenue" in r["rationale"] and r["strength"] == 3


def test_mixed_earnings_wording_is_not_called():
    r = call("Profit rises in Q2 but net loss for the year widens")
    assert r["direction"] == 0


def test_unrecognised_headline_is_other_and_low():
    r = call("Company hosts annual charity football match")
    assert r["event_type"] == "other" and r["confidence"] == "low"


# ---- duplicate coverage -------------------------------------------------------

def test_duplicate_coverage_counts_as_one_event():
    df = pd.DataFrame([
        {"ticker": "AMOC", "event_type": "earnings_up", "date": "2026-07-30", "direction": 1,
         "strength": 3, "confidence": "high"},
        {"ticker": "AMOC", "event_type": "earnings_up", "date": "2026-07-30", "direction": 1,
         "strength": 2, "confidence": "high"},
        {"ticker": "AMOC", "event_type": "earnings_up", "date": "2026-07-31", "direction": 1,
         "strength": 3, "confidence": "medium"},
        {"ticker": "AMOC", "event_type": "earnings_up", "date": "2026-10-30", "direction": 1,
         "strength": 2, "confidence": "high"},
    ])
    ev = ni.dedupe_events(df)
    assert len(ev) == 2
