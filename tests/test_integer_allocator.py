"""
Integer allocator: known-answer cases. The central one is that it deploys cash
a per-name flooring rule would strand, because it prices idle cash as the
tracking error it actually is.
"""
import numpy as np
import pandas as pd
import pytest

from portfolio.integer_allocator import (tracking_error, greedy_integer_allocation)


def test_tracking_error_zero_when_perfectly_matched():
    sh = pd.Series({"A": 5.0, "B": 5.0})
    px = pd.Series({"A": 100.0, "B": 100.0})
    tw = pd.Series({"A": 0.5, "B": 0.5})
    assert tracking_error(sh, px, tw, cash=0.0) == pytest.approx(0.0, abs=1e-12)


def test_idle_cash_is_counted_as_tracking_error():
    """Holding cash against a 100%-invested target must score WORSE than being
    invested -- this is the property per-name flooring misses."""
    px = pd.Series({"A": 100.0})
    tw = pd.Series({"A": 1.0})
    invested = tracking_error(pd.Series({"A": 10.0}), px, tw, cash=0.0)
    half_cash = tracking_error(pd.Series({"A": 5.0}), px, tw, cash=500.0)
    assert invested == pytest.approx(0.0, abs=1e-12)
    assert half_cash > 0.2          # 50% cash -> big error


def test_deploys_cash_that_per_name_flooring_would_strand():
    """The real 1K case: every per-name gap is under one share, so flooring buys
    nothing; the allocator should still deploy, because a whole share beats cash."""
    px = pd.Series({"A": 96.0, "B": 90.0, "C": 80.0})
    tw = pd.Series({"A": 0.34, "B": 0.33, "C": 0.33})
    held = pd.Series({"A": 2.0, "B": 2.0, "C": 2.0})     # 532 invested
    cash = 339.0
    # per-name gaps here are each well under one share of the respective name
    res = greedy_integer_allocation(tw, px, cash, current_shares=held)
    assert res["n_buys"] >= 1                            # flooring would buy 0
    assert res["deployed"] > 0
    assert res["te_after"] < res["te_before"]            # strictly better book


def test_never_overspends_cash():
    px = pd.Series({"A": 100.0, "B": 250.0})
    tw = pd.Series({"A": 0.5, "B": 0.5})
    res = greedy_integer_allocation(tw, px, cash=310.0)
    spent = sum(int(n) * float(px[t]) for t, n in res["orders"].items())
    assert spent <= 310.0 + 1e-9
    assert res["cash_after"] >= -1e-9


def test_builds_a_sane_book_from_all_cash():
    px = pd.Series({"A": 20.0, "B": 50.0, "C": 10.0})
    tw = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    res = greedy_integer_allocation(tw, px, cash=1000.0)
    final = res["final_shares"]
    value = (final * px.reindex(final.index)).fillna(0.0)
    w = value / (value.sum() + res["cash_after"])
    # resulting weights should be close to target and most cash deployed
    assert abs(w["A"] - 0.5) < 0.08
    assert res["deployed"] > 900


def test_min_improvement_blocks_cosmetic_churn():
    px = pd.Series({"A": 100.0})
    tw = pd.Series({"A": 1.0})
    held = pd.Series({"A": 10.0})
    # already perfectly allocated; a huge threshold must prevent any buying
    res = greedy_integer_allocation(tw, px, cash=100.0, current_shares=held,
                                    min_improvement_bps=1e6)
    assert res["n_buys"] == 0


def test_exclude_never_buys_banned_name_and_wont_overconcentrate():
    """An excluded name is never bought, and -- importantly -- its weight is NOT
    forced onto the survivors: overshooting A/B to replace C would be a bigger
    tracking error than holding that cash, and the allocator must see that."""
    px = pd.Series({"A": 50.0, "B": 50.0, "C": 50.0})
    tw = pd.Series({"A": 1/3, "B": 1/3, "C": 1/3})
    banned = greedy_integer_allocation(tw, px, cash=300.0, exclude=["C"])
    assert "C" not in banned["orders"].index           # never bought
    assert banned["deployed"] > 0                      # still deploys usefully
    assert banned["te_after"] < banned["te_before"]    # strictly better book
    # A and B end at their 1/3 targets rather than being pushed to 1/2 each
    final = banned["final_shares"]
    assert int(final.get("A", 0)) == int(final.get("B", 0)) == 2


def test_exclude_uses_freed_cash_when_it_actually_helps():
    """When an allowed name is still UNDER target, the freed cash should go there."""
    px = pd.Series({"A": 10.0, "C": 10.0})
    tw = pd.Series({"A": 0.5, "C": 0.5})
    held = pd.Series({"A": 0.0, "C": 0.0})
    banned = greedy_integer_allocation(tw, px, cash=100.0, current_shares=held, exclude=["C"])
    assert "C" not in banned["orders"].index
    assert banned["orders"].get("A", 0) >= 1           # bought the allowed name


def test_deployment_does_not_degrade_as_capital_grows():
    """Regression: a FIXED absolute churn threshold stopped the allocator earlier
    the more capital it had (21% idle at 1,000 EGP but 46% at 10,000), because
    one share's squared-weight improvement shrinks with book size. Idle fraction
    must not INCREASE with capital."""
    px = pd.Series({f"T{i}": 10.0 + i for i in range(12)})
    tw = pd.Series({f"T{i}": 1.0 / 12 for i in range(12)})
    idle = {}
    for cap in (1_000.0, 5_000.0, 10_000.0):
        r = greedy_integer_allocation(tw, px, cap)
        idle[cap] = r["cash_after"] / cap
    assert idle[10_000.0] <= idle[1_000.0] + 0.02   # bigger book, no worse
    assert idle[10_000.0] < 0.15                    # and most capital deployed


def test_churn_threshold_still_blocks_pointless_trades():
    """The scale fix must not disable the churn guard entirely."""
    px = pd.Series({"A": 100.0})
    tw = pd.Series({"A": 1.0})
    held = pd.Series({"A": 10.0})
    r = greedy_integer_allocation(tw, px, cash=100.0, current_shares=held,
                                  min_improvement_bps=1e9)
    assert r["n_buys"] == 0
