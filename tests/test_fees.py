"""
Date-aware fee schedule. The critical properties: the stamp duty appears only
from its effective date, and the flat per-order fee makes SMALL orders
disproportionately expensive -- which a percentage-only model cannot express.
"""
import pandas as pd
import pytest

from backtest import fees


def test_stamp_duty_starts_on_its_effective_date():
    before = fees.stamp_duty(1000.0, as_of="2026-07-28")
    on = fees.stamp_duty(1000.0, as_of="2026-07-29")
    assert before["rate"] == 0.0                  # not payable before the law
    assert on["rate"] == pytest.approx(0.0005)    # payable from the session it took effect
    assert on["fee_egp"] == pytest.approx(0.50)


def test_same_day_stamp_duty_is_half():
    std = fees.stamp_duty(1000.0, as_of="2026-08-01", same_day=False)
    intr = fees.stamp_duty(1000.0, as_of="2026-08-01", same_day=True)
    assert intr["rate"] == pytest.approx(std["rate"] / 2)


def test_broker_flat_fee_dominates_small_orders():
    """A flat EGP charge on a 40 EGP order is a huge percentage -- the failure a
    percentage-only model hides. Asserted structurally against the schedule so
    recalibrating the rate does not require editing this test."""
    small = fees.broker_fee(40.0, as_of="2026-08-01")
    big = fees.broker_fee(10_000.0, as_of="2026-08-01")
    flat, pct = fees._lookup(fees.THNDR_SCHEDULE, "2026-08-01")
    assert flat > 0                                # there IS a flat component
    assert small["pct_of_order"] > 0.05            # >5% on a tiny order
    assert big["pct_of_order"] == pytest.approx(pct + flat / 10_000.0, abs=1e-9)
    assert small["pct_of_order"] > big["pct_of_order"] * 40


def test_current_broker_schedule_matches_measured_fills():
    """KNOWN ANSWER, from 8 real Thndr contract notes on 2026-08-04. This is the
    only fee figure in the system derived from actual fills rather than from an
    unverified operator estimate, so it is pinned: if the schedule is edited and
    stops reproducing what was really charged, that must fail loudly."""
    observed = [(117.78, 5.15), (109.56, 5.14), (100.44, 5.13), (154.20, 5.20),
                (144.48, 5.19), (119.00, 5.15), (147.00, 5.19), (109.35, 5.09)]
    for notional, charged in observed:
        modelled = fees.one_side_cost(notional, as_of="2026-08-04",
                                      spread_per_side=0.0)["total_egp"]
        assert modelled == pytest.approx(charged, abs=0.05)


def test_legacy_schedule_applies_to_old_backtest_dates():
    """Costing a 2020 trade must use the rate payable in 2020, not today's."""
    old = fees.broker_fee(1000.0, as_of="2020-06-01")
    new = fees.broker_fee(1000.0, as_of="2026-08-01")
    assert old["flat"] == 0.0
    assert new["flat"] > 0.0                       # a flat leg exists from 2026
    assert fees.stamp_duty(1000.0, as_of="2020-06-01")["rate"] == 0.0


def test_splitting_a_book_multiplies_the_flat_fee():
    """One 1,000 EGP order vs seven ~143 EGP orders: same notional, more cost."""
    one = fees.book_round_trip_cost([1000.0], as_of="2026-08-01")
    seven = fees.book_round_trip_cost([1000.0 / 7] * 7, as_of="2026-08-01")
    assert seven["total_egp"] > one["total_egp"]
    assert seven["n_orders"] == 14                       # 7 buys + 7 sells
    assert seven["flat_share_of_cost"] > 0.5             # flat fees dominate


def test_round_trip_charges_both_sides():
    rt = fees.round_trip_cost(1000.0, as_of="2026-08-01")
    one = fees.one_side_cost(1000.0, as_of="2026-08-01")
    assert rt["total_egp"] == pytest.approx(2 * one["total_egp"])
