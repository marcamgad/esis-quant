"""Execution model: trades fill at the typical price after the signal, not at its close."""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import fill_adjustment


def _setup():
    hold = pd.bdate_range("2026-08-02", periods=3)
    r = pd.DataFrame({"A": [0.10, 0.0, 0.0], "B": [-0.05, 0.0, 0.0]}, index=hold)
    ratio = pd.DataFrame({"A": [1.05, 1, 1], "B": [0.98, 1, 1]}, index=hold)
    return hold, r, ratio


def test_unchanged_book_is_untouched():
    hold, r, ratio = _setup()
    adj = fill_adjustment(pd.Series({"A": 0.0, "B": 0.0}), hold, r, ratio)
    assert (adj == 0).all()


def test_buying_after_a_rally_costs_the_part_of_the_move_already_gone():
    hold, r, ratio = _setup()
    # buy 100% of A, which rose 10% on the fill day with a typical price 5% above close/1.10*...
    adj = fill_adjustment(pd.Series({"A": 1.0}), hold, r, ratio, lag=1)
    # close-fill model credits +10%; typical fill earns close/typical - 1 = 1/1.05 - 1
    assert adj.iloc[0] == pytest.approx((1 / 1.05 - 1) - 0.10)
    assert adj.iloc[1:].abs().sum() == 0


def test_selling_late_keeps_the_fill_day_move():
    hold, r, ratio = _setup()
    adj = fill_adjustment(pd.Series({"B": -1.0}), hold, r, ratio, lag=1)
    # sold at typical = 0.98 * close_T+1 = 0.98 * 0.95 * close_T
    assert adj.iloc[0] == pytest.approx(0.98 * 0.95 - 1)


def test_two_session_lag_holds_the_old_book_one_more_day():
    hold, r, ratio = _setup()
    dw = pd.Series({"A": 1.0, "B": -1.0})                 # switch B -> A
    adj = fill_adjustment(dw, hold, r, ratio, lag=2)
    # day 1: still in B, not A: correction = -(dw * r) = -(0.10 + 0.05)
    assert adj.iloc[0] == pytest.approx(-0.15)
    assert adj.iloc[1] == pytest.approx(0.0)              # flat day 2, ratio 1: no effect


def test_slippage_is_charged_on_traded_weight():
    hold, r, ratio = _setup()
    flat = pd.DataFrame(0.0, index=hold, columns=["A", "B"])
    one = pd.DataFrame(1.0, index=hold, columns=["A", "B"])
    adj = fill_adjustment(pd.Series({"A": 0.5, "B": -0.5}), hold, flat, one, slippage=0.01)
    assert adj.iloc[0] == pytest.approx(-0.01)
