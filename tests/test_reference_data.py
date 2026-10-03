"""
Tests for the modules added 2026-09-11: sector limits, fundamentals parsing,
security classification.

All offline. The fundamentals and security-master modules fetch from public
sites in production, but a test suite that needs the network fails for reasons
unrelated to the code, and a flaky suite trains people to ignore it. Parsing is
tested on HTML shaped like the real pages; storage is tested against the real
database with a sentinel ticker that is removed afterwards.
"""

import pandas as pd
import pytest

from data import egx_sectors as es
from data import fundamentals as fd
from portfolio import sector_cap as sc


# --------------------------------------------------------------------------
# Sector limits
# --------------------------------------------------------------------------

FLOUR = {"MILS": "Food", "AFMC": "Food", "LUTS": "Agriculture",
         "OIH": "IT", "AMOC": "Energy", "CRST": "Contracting", "KORA": "Contracting"}


def test_select_diverse_blocks_the_flour_mill_pair():
    """The exact 2026-09-04 failure: top-3 by weight was two flour mills."""
    target = pd.Series({"MILS": 0.30, "AFMC": 0.28, "LUTS": 0.22, "OIH": 0.20})
    chosen = sc.select_diverse(target, FLOUR, max_names=3, max_per_sector=1)
    assert not {"MILS", "AFMC"} <= chosen
    assert chosen == {"MILS", "LUTS", "OIH"}


def test_select_diverse_without_sector_data_is_plain_top_n():
    target = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    assert sc.select_diverse(target, {}, max_names=2) == {"A", "B"}


def test_select_diverse_keeps_held_names_and_counts_them():
    target = pd.Series({"MILS": 0.30, "AFMC": 0.28, "LUTS": 0.22, "OIH": 0.20})
    chosen = sc.select_diverse(target, FLOUR, max_names=3, held={"AFMC"},
                               max_per_sector=1)
    assert "AFMC" in chosen          # held is never evicted
    assert "MILS" not in chosen      # AFMC already used the Food quota


def test_select_diverse_fills_remaining_slots_when_quota_starves_it():
    """A ticket shorter than it can afford concentrates cost -- fill it."""
    target = pd.Series({"MILS": 0.5, "AFMC": 0.5})
    assert sc.select_diverse(target, FLOUR, max_names=2, max_per_sector=1) == {"MILS", "AFMC"}


def test_apply_sector_cap_enforces_the_limit():
    w = pd.Series({"CRST": 0.30, "KORA": 0.25, "OIH": 0.20, "AMOC": 0.15, "LUTS": 0.10})
    out, info = sc.apply_sector_cap(w, FLOUR, cap=0.30)
    by_sector = out.groupby(pd.Series({t: FLOUR[t] for t in out.index})).sum()
    assert by_sector.max() <= 0.30 + 1e-6
    assert abs(out.sum() - 1.0) < 1e-9
    assert info["capped_sectors"] == ["Contracting"]


def test_apply_sector_cap_respects_max_weight():
    w = pd.Series({"CRST": 0.35, "KORA": 0.35, "OIH": 0.15, "AMOC": 0.15})
    out, _ = sc.apply_sector_cap(w, FLOUR, cap=0.40, max_weight=0.35)
    assert out.max() <= 0.35 + 1e-6


def test_apply_sector_cap_relaxes_when_infeasible_instead_of_losing_money():
    """Two sectors can never both sit under a 30% cap; say so, don't force it."""
    w = pd.Series({"CRST": 0.5, "OIH": 0.5})
    out, info = sc.apply_sector_cap(w, FLOUR, cap=0.30)
    assert info["applied"] is False and "infeasible" in info["reason"]
    assert abs(out.sum() - 1.0) < 1e-9


def test_unknown_sectors_are_not_pooled():
    """Names lacking a classification must not be capped together."""
    w = pd.Series({"X1": 0.34, "X2": 0.33, "X3": 0.33})
    out, info = sc.apply_sector_cap(w, {"OTHER": "IT"}, cap=0.40)
    assert info["capped_sectors"] == []
    assert out.round(6).equals(w.round(6))


# --------------------------------------------------------------------------
# Fundamentals parsing
# --------------------------------------------------------------------------

PAGE = """
<table><thead>
<tr><th>Fiscal Year</th><th>TTM</th><th>FY 2025</th><th>FY 2024</th></tr>
</thead><tbody>
<tr><td>Period Ending</td><td>Jun '26 Jun 30, 2026</td><td>Dec '25 Dec 31, 2025</td><td>Dec '24 Dec 31, 2024</td></tr>
<tr><td>Revenue    <a>Revenue Growth</a></td><td>139,166</td><td>128,542</td><td>114,754</td></tr>
<tr><td>Revenue Growth</td><td>24.41%</td><td>12.02%</td><td>-</td></tr>
<tr><td>Net Income    <a>Net Income Growth</a></td><td>-533.78</td><td>-450</td><td>592.81</td></tr>
</tbody></table>
"""


def test_row_label_strips_the_nested_growth_link():
    assert fd._row_label("Revenue    Revenue Growth") == "Revenue"
    assert fd._row_label("Net Income    Net Income Growth") == "Net Income"
    assert fd._row_label("Total Debt") == "Total Debt"


@pytest.mark.parametrize("tok,expected", [
    ("139,166", 139166.0), ("-533.78", -533.78), ("24.41%", 0.2441),
    ("-", None), ("", None), ("n/a", None), ("abc", None),
])
def test_to_number(tok, expected):
    got = fd._to_number(tok)
    assert got == pytest.approx(expected) if expected is not None else got is None


def test_parse_statement_reads_periods_ends_and_rows():
    p = fd._parse_statement(PAGE)
    assert p["periods"] == ["TTM", "FY 2025", "FY 2024"]
    assert p["period_ends"] == ["2026-06-30", "2025-12-31", "2024-12-31"]
    assert p["rows"]["Revenue"] == [139166.0, 128542.0, 114754.0]
    assert p["rows"]["Net Income"] == [-533.78, -450.0, 592.81]
    assert p["rows"]["Revenue Growth"][2] is None


SPLIT_PAGE = """
<table><tr><th>Fiscal Quarter</th><th>Q2 2026</th><th>Q1 2026</th></tr>
<tr><td>Period Ending</td><td>Jun '26 Jun 30, 2026</td><td>Mar '26 Mar 31, 2026</td></tr>
<tr><td>Total Assets</td><td>1,000</td><td>900</td></tr></table>
<p>ad</p>
<table><tr><th>Fiscal Quarter</th><th>Q2 2026</th><th>Q1 2026</th></tr>
<tr><td>Period Ending</td><td>Jun '26 Jun 30, 2026</td><td>Mar '26 Mar 31, 2026</td></tr>
<tr><td>Total Debt</td><td>400</td><td>350</td></tr>
<tr><td>Shareholders' Equity</td><td>500</td><td>480</td></tr></table>
<table><tr><th>Fiscal Quarter</th><th>Current</th><th>Q1 2026</th></tr>
<tr><td>Period Ending</td><td>Sep '26 Sep 13, 2026</td><td>Mar '26 Mar 31, 2026</td></tr>
<tr><td>PE Ratio</td><td>12</td><td>10</td></tr></table>
"""


def test_parse_statement_reads_every_section_table():
    """The site splits the balance sheet into sections; liabilities and equity
    were lost for months because only the first table was read."""
    p = fd._parse_statement(SPLIT_PAGE)
    assert p["periods"] == ["Q2 2026", "Q1 2026"]            # "Current" snapshot dropped
    assert p["rows"]["Total Assets"] == [1000.0, 900.0]
    assert p["rows"]["Total Debt"] == [400.0, 350.0]
    assert p["rows"]["Shareholders' Equity"] == [500.0, 480.0]
    assert p["rows"]["PE Ratio"] == [None, 10.0]              # aligned by label, not position


def test_quarterly_lag_and_frequency():
    assert fd._lag_days("Q2 2026") == fd.QUARTERLY_LAG_DAYS
    assert fd._lag_days("Q4 2025") == fd.REPORTING_LAG_DAYS   # Q4 is the annual report
    assert fd._lag_days("FY 2025") == fd.REPORTING_LAG_DAYS
    assert fd.frequency_of("Q1 2026") == "quarterly"
    assert fd.frequency_of("TTM") == "annual"


def test_quarterly_rows_are_point_in_time_and_give_leverage(sentinel_cleanup):
    payload = {"ticker": SENTINEL, "statements": {"balance": fd._parse_statement(SPLIT_PAGE)}}
    fd.store_ticker(payload)
    assert fd.leverage_as_of(SENTINEL, "2026-06-15") == {}                 # Q1 not public until 2026-06-29
    lev = fd.leverage_as_of(SENTINEL, "2026-07-01")
    assert lev["period_end"] == "2026-03-31"
    assert lev["debt_to_equity"] == pytest.approx(350 / 480)
    assert fd.leverage_as_of(SENTINEL, "2026-10-01")["total_debt"] == 400.0
    assert fd.as_of(SENTINEL, "2026-10-01", "balance").empty              # default is annual only


def test_parse_statement_without_a_table_returns_empty():
    assert fd._parse_statement("<html>no table</html>") == {}


SENTINEL = "__TEST_FUND__"


@pytest.fixture
def sentinel_cleanup():
    yield
    from data.database import get_connection
    with get_connection() as conn:
        conn.execute("DELETE FROM fundamentals WHERE ticker = ?", (SENTINEL,))


def test_available_from_prevents_lookahead(sentinel_cleanup):
    """THE property the module exists for: a fiscal year is not knowable on the
    day it closes. FY2025 ends 2025-12-31 and must be invisible on 2026-01-15."""
    payload = {"ticker": SENTINEL, "statements": {"income": fd._parse_statement(PAGE)}}
    assert fd.store_ticker(payload) > 0

    jan = fd.as_of(SENTINEL, "2026-01-15")
    assert set(jan["period_end"]) == {"2024-12-31"}

    later = fd.as_of(SENTINEL, "2026-05-15")        # 2025-12-31 + 120d = 2026-04-30
    assert "2025-12-31" in set(later["period_end"])
    assert "2026-06-30" not in set(later["period_end"])

    row = later[later["period_end"] == "2025-12-31"].iloc[0]
    lag = (pd.Timestamp(row["available_from"]) - pd.Timestamp(row["period_end"])).days
    assert lag == fd.REPORTING_LAG_DAYS


# --------------------------------------------------------------------------
# Security classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,kind", [
    ("Certificates Of Odin Egyptian Equity Investment Fund-KASAB", "fund"),
    ("Egyptians Real Estate Fund Certificates", "fund"),
    ("EGX 30 INDEX ETF", "etf"),
    ("Subscription Rights Of Lotus For Agricultural Invest&Dev-1", "rights"),
    ("QALA For Financial Investments - Preferred Shares", "preferred"),
    ("O G Capital For Investments SPAC", "spac"),
    ("Grova Special Purpose Acquisition Company", "spac"),
    ("Faisal Islamic Bank of Egypt - In US Dollars", "fx_class"),
    ("Saudi Egyptian Investment & Finance $", "fx_class"),
    ("Commercial International Bank-Egypt (CIB)", "ordinary"),
    ("Orascom Investment Holding", "ordinary"),
])
def test_classify_security(name, kind):
    assert es.classify_security(name) == kind


def test_listing_file_loads_and_matches_known_counts():
    rows = es.load_listing()
    assert len(rows) == 261
    kinds = pd.Series([r[3] for r in rows]).value_counts().to_dict()
    assert kinds["fund"] == 2 and kinds["etf"] == 1 and kinds["spac"] == 2


def test_isin_overrides_point_at_the_positions_that_needed_them():
    assert es.ISIN_OVERRIDES["EGS696Z1C017"] == "KASABF"
    assert es.ISIN_OVERRIDES["EGS07911C018"] == "KORA"


# --------------------------------------------------------------------------
# Escalation: unsourced answers must never reach the database
# --------------------------------------------------------------------------

from ops import escalation as esc  # noqa: E402


def test_identity_answer_without_source_is_rejected():
    """Plausibility is the failure mode: a fluent unsourced answer is dropped."""
    v = esc.validate_identity({"ticker": "KASABF", "security_type": "fund",
                               "security_type_source": ""})
    assert v["accepted"] == {}
    assert any("no source" in r for r in v["rejected"])


def test_identity_answer_with_source_is_accepted():
    v = esc.validate_identity({"ticker": "KASABF", "security_type": "fund",
                               "security_type_source": "https://www.egx.com.eg/x",
                               "listing_date": "2021-03-01",
                               "listing_date_source": "https://example.com/y"})
    assert v["accepted"] == {"security_type": "fund", "listing_date": "2021-03-01"}


def test_identity_unknown_is_acceptable_but_records_nothing():
    v = esc.validate_identity({"ticker": "X", "security_type": "UNKNOWN",
                               "listing_date": "UNKNOWN"})
    assert v["accepted"] == {} and v["rejected"] == []


def test_identity_invalid_type_is_rejected_even_with_a_source():
    v = esc.validate_identity({"ticker": "X", "security_type": "cryptocurrency",
                               "security_type_source": "https://a.b/c"})
    assert v["accepted"] == {}


def test_split_answer_requires_valid_cause_and_source():
    assert esc.validate_split({"ticker": "ORAS", "cause": "split",
                               "cause_source": "not a url"})["accepted"] == {}
    assert esc.validate_split({"ticker": "ORAS", "cause": "magic",
                               "cause_source": "https://a.b/c"})["accepted"] == {}
    ok = esc.validate_split({"ticker": "ORAS", "cause": "dual_currency_listing",
                             "cause_source": "https://a.b/c", "ratio": "UNKNOWN"})
    assert ok["accepted"] == {"cause": "dual_currency_listing", "source": "https://a.b/c"}


def test_prompts_demand_sources_and_offer_unknown():
    for p in (esc.identity_prompt("KASABF", "Fund", "EGS696Z1C017"),
              esc.split_triage_prompt("ORAS", 750.0, 71.05)):
        assert "source" in p.lower() and "UNKNOWN" in p and "JSON" in p


def test_norm_is_for_matching_not_display():
    assert es._norm("Alexandria Flour Mills Co. S.A.E") == es._norm("ALEXANDRIA FLOUR MILLS")
