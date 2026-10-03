"""Index event study: date logic and scoring on known synthetic panels."""

import numpy as np
import pandas as pd

from analysis import index_event_study as ies
from data.egx_index_events import REVIEWS, events


def test_every_review_has_valid_dates_and_sources():
    for ann, eff, added, removed, src, complete in REVIEWS:
        assert pd.Timestamp(ann) <= pd.Timestamp(eff)
        assert (pd.Timestamp(eff) - pd.Timestamp(ann)).days <= 20
        assert src and (added or removed)
        assert not set(added) & set(removed)


def test_events_carry_side():
    ev = pd.DataFrame(events())
    assert set(ev["side"]) == {1, -1}
    assert len(ev) == sum(len(a) + len(r) for _, _, a, r, _, _ in REVIEWS)


def test_entry_is_after_announcement_and_exit_after_effective():
    idx = pd.bdate_range("2024-07-01", periods=40)
    ann, eff, exit_p = ies.positions(idx, "2024-07-30", "2024-08-01")
    assert idx[ann] == pd.Timestamp("2024-07-31")          # strictly after announcement
    assert idx[eff] == pd.Timestamp("2024-08-01")
    assert idx[exit_p] == pd.Timestamp("2024-08-02")       # one session after effective


def test_same_day_announcement_still_gives_a_one_session_window():
    idx = pd.bdate_range("2024-07-01", periods=40)
    ann, eff, exit_p = ies.positions(idx, "2024-08-01", "2024-08-01")
    assert exit_p == ann + 1


def test_scores_follow_predicted_signs():
    idx = pd.bdate_range("2024-07-01", periods=80)
    rets = pd.DataFrame(0.0, index=idx, columns=["ADD", "DEL", "X"])
    ann, eff, exit_p = ies.positions(idx, "2024-07-30", "2024-08-01")
    rets.iloc[ann + 1:exit_p + 1, 0] = 0.02      # added stock rises in the pressure window
    rets.iloc[ann + 1:exit_p + 1, 1] = -0.02     # removed stock falls
    market = rets.mean(axis=1)
    ev = pd.DataFrame([
        {"ticker": "ADD", "announced": "2024-07-30", "effective": "2024-08-01", "side": 1},
        {"ticker": "DEL", "announced": "2024-07-30", "effective": "2024-08-01", "side": -1},
    ])
    m = ies.measure(ev, rets, market).set_index("ticker")
    assert m.loc["ADD", "pressure_score"] > 0
    assert m.loc["DEL", "pressure_score"] > 0


def test_egx70_events_exclude_moves_to_and_from_egx30():
    """A promotion to EGX30 is not an EGX70 removal for this study, and vice versa."""
    from data.egx_index_events import events70, EGX70_REVIEWS
    ev = pd.DataFrame(events70())
    sep = ev[ev["effective"] == "2026-09-01"]
    assert "SKPC" not in set(sep["ticker"])     # left EGX70 to join EGX30
    assert "OIH" not in set(sep["ticker"])      # joined EGX70 after leaving EGX30
    assert "TALM" in set(sep[sep["side"] == -1]["ticker"])   # pure EGX70 removal
    total = sum(len(a) + len(r) for _, _, a, r, _, _ in EGX70_REVIEWS)
    assert len(ev) < total


def test_window_ar_rejects_untraded_stock():
    s = np.full(30, np.nan)
    assert ies._window_ar(s, np.zeros(30), 5, 10) is None
