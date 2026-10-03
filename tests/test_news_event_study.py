"""
Stage 4 tests, on synthetic price panels so the answers are known exactly.

The property that matters most: the score must never include the move on the day
the news came out. An event study that enters on the publication day scores a
price change nobody could have traded, and it would make almost any set of calls
look skilful.
"""

import numpy as np
import pandas as pd

from analysis import news_event_study as es


def panel(n=80, jump_day=None, jump=0.0, after=0.0):
    idx = pd.bdate_range("2026-01-01", periods=n)
    rets = pd.DataFrame({"AAA": np.zeros(n), "BBB": np.zeros(n)}, index=idx)
    if jump_day is not None:
        rets.iloc[jump_day, 0] = jump            # the news-day move
        # drift over exactly the 5 sessions after entry (entry = jump_day)
        rets.iloc[jump_day + 1:jump_day + 6, 0] = after
    market = rets.mean(axis=1)
    return rets, market


def test_entry_is_strictly_after_publication():
    rets, mkt = panel()
    d = rets.index[10]
    fa = es.forward_abnormal(rets, mkt, "AAA", d)
    assert fa["entry"] == rets.index[11]


def test_news_day_move_is_never_scored():
    """A +20% jump on the entry session appears only in 'gap', not in ar_5."""
    rets, mkt = panel(jump_day=11, jump=0.20, after=0.0)
    fa = es.forward_abnormal(rets, mkt, "AAA", rets.index[10])
    assert fa["gap"] > 0.05
    assert abs(fa["ar_5"]) < 1e-12


def test_post_entry_drift_is_scored_against_the_market():
    rets, mkt = panel(jump_day=11, jump=0.0, after=0.01)
    fa = es.forward_abnormal(rets, mkt, "AAA", rets.index[10])
    # stock +1%/day for 5 days, market is half of that (two-stock average)
    stock = 1.01 ** 5 - 1
    market = 1.005 ** 5 - 1
    assert abs(fa["ar_5"] - (stock - market)) < 1e-9


def test_missing_prices_are_not_treated_as_zero_reaction():
    rets, mkt = panel()
    rets.iloc[11:20, 0] = np.nan
    fa = es.forward_abnormal(rets, mkt, "AAA", rets.index[10])
    assert "ar_5" not in fa


def test_unknown_ticker_and_end_of_panel():
    rets, mkt = panel()
    assert es.forward_abnormal(rets, mkt, "ZZZ", rets.index[10]) == {}
    assert "ar_20" not in es.forward_abnormal(rets, mkt, "AAA", rets.index[-5])


def test_score_is_positive_for_a_right_call_in_either_direction():
    rets, mkt = panel(jump_day=11, after=-0.01)
    ev = pd.DataFrame([{"ticker": "AAA", "date": str(rets.index[10].date()), "event_type": "earnings_down",
                        "direction": -1, "strength": 2, "confidence": "high"}])
    m = es.measure(ev, rets, mkt)
    assert m.iloc[0]["score_5"] > 0


def test_verdict_rules():
    assert es.verdict(10, 0.05, 0.001, 1.0) == "INSUFFICIENT"
    assert es.verdict(40, 0.01, 0.01, 1.0) == "PASS"
    assert es.verdict(40, 0.01, 0.20, 1.0) == "NULL"
    assert es.verdict(40, -0.01, 1.0, 0.01) == "WRONG"
    assert es.verdict(40, 0.01, 0.06, 1.0) == "NULL"      # corrected p just misses


def test_placebo_excludes_windows_around_real_events():
    """A real event's own reaction must not leak into the random baseline."""
    rets, mkt = panel(n=200)
    rets.iloc[100:106, 0] = 0.05                   # big move right after the event
    ev = pd.DataFrame([{"ticker": "AAA", "date": str(rets.index[99].date()), "direction": 1,
                        "score_5": 0.0}])
    pm = es.placebo_means(ev, rets, rets.mean(axis=1), n_draws=300, seed=1)
    # windows within +/-20 sessions of the event are excluded, so no draw sees the move
    assert pm.max() < 0.05
