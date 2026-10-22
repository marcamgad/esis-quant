"""Rule v2: gates, sell conditions, book limits, recording."""

from ops import system_picks as sp


def _f(**kw):
    base = {"q": {"period": "Q2 2026", "value": 50.0}, "growth": {"change": 0.2, "label": "+20% y/y"},
            "de": 0.5, "financial": False, "negative_equity": False, "ocf": 10.0, "value": 5e6,
            "capital_change": None, "val": {"pe": 8.0, "pb": 1.5}, "ttm": 120.0,
            "last2": [50.0, 40.0], "last2_periods": ["Q2 2026", "Q1 2026"]}
    base.update(kw)
    return base


def test_all_gates_pass_on_a_clean_company():
    assert all(ok for _, ok, _ in sp._gates(_f()))


def test_each_gate_can_fail_alone():
    cases = {
        "G1": _f(q={"period": "Q2 2026", "value": -1.0}),
        "G2": _f(growth={"change": 12.0, "label": "small base"}),
        "G3": _f(de=2.5),
        "G4": _f(ocf=-3.0),
        "G5": _f(value=200_000),
        "G6": _f(capital_change="2026-10-01"),
        "G7": _f(negative_equity=True),
        "G8": _f(val={"pe": 40.0, "pb": 1.5}),
        "G9": _f(val={"pe": 8.0, "pb": 6.0}),
    }
    for gate, facts in cases.items():
        failed = [g for g, ok, _ in sp._gates(facts) if not ok]
        assert len(failed) == 1 and failed[0].startswith(gate), (gate, failed)


def test_financials_exempt_from_leverage_and_cash_flow_but_not_book_value():
    fin = _f(de=8.0, ocf=-500.0, financial=True)
    assert all(ok for _, ok, _ in sp._gates(fin))
    assert not all(ok for _, ok, _ in sp._gates(_f(financial=True, val={"pe": 8.0, "pb": 3.0})))


def test_one_bad_quarter_is_not_a_sell_but_a_bad_year_is():
    assert sp._sell_reasons(_f(last2=[-7.0, 46.9], ttm=17.8)) == []            # CRST in Oct 2026
    assert sp._sell_reasons(_f(last2=[-43.4, -37.7], ttm=None))                # two losses in a row
    assert sp._sell_reasons(_f(last2=[5.0, 3.0], ttm=-10.0))                   # a losing year
    assert sp._sell_reasons(_f(de=3.5))                                        # over-borrowed
    assert sp._sell_reasons(_f(de=3.5, financial=True)) == []                  # a bank is not


def test_signals_fit_the_book_and_do_not_repeat():
    sig = sp.signals()
    assert set(sig) >= {"buy", "wait", "sell", "trim", "hold", "rejected"} and len(sig["buy"]) <= sp.MAX_BUYS
    spend = sum(b["size"]["value_egp"] for b in sig["buy"])
    proceeds = sum(x["shares"] * x["price"] for x in sig["sell"] + sig["trim"])
    assert spend <= sig["cash"] + proceeds + 1
    for b in sig["buy"]:
        assert all(ok for _, ok, _ in b["gates"]) and b["size"]["value_egp"] / sig["account"] <= sp.NAME_CAP
    sp.record(sig)
    assert sp.record(sig) == []
