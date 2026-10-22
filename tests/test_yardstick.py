"""Cash yardstick arithmetic."""

import pytest

from ops import yardstick as ys


def test_money_compounds_at_the_cash_yield():
    assert ys._grow(10_000, "2026-01-01", "2027-01-01", 0.195) == pytest.approx(11_950, rel=1e-3)
    assert ys._grow(10_000, "2026-01-01", "2025-01-01", 0.195) == 10_000          # never shrinks backwards


def test_hurdle_is_cash_plus_costs_and_falls_with_size():
    small, big = ys.hurdle(1000, 20), ys.hurdle(5000, 20)
    for h in (small, big):
        assert h["total"] == pytest.approx(h["cash"] + h["fees"] + h["slippage"])
        assert h["cash"] > 0 and h["slippage"] == pytest.approx(2 * ys.SLIPPAGE)
    assert small["fees"] > big["fees"]                                           # flat fee bites small orders


def test_section_renders():
    assert "VS DOING NOTHING" in ys.format_section()
