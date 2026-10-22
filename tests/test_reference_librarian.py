"""Agent 3 (reference librarian) + read-time split correction. Offline, temp DB only."""

import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from data import ca_factors as caf
from ops import reference_librarian as rl


def _series(values, start="2026-06-01"):
    idx = pd.bdate_range(start, periods=len(values), freq="C", weekmask="Sun Mon Tue Wed Thu")
    return pd.Series(values, index=idx, dtype=float)


# --- read-time correction ------------------------------------------------------

def test_split_is_applied_while_the_jump_is_present():
    s = _series([100, 101, 50.5, 51, 52])          # 2:1 on day 3, vendor never adjusted
    ev = pd.DataFrame({"ticker": ["X"], "date": [s.index[2]], "ratio": [2.0]})
    out, applied = caf.adjust_series(s, ev)
    assert applied == [s.index[2]]
    assert out.iloc[1] == pytest.approx(50.5) and out.pct_change().abs().max() < 0.03


def test_split_self_disables_once_the_vendor_adjusts():
    """The property that makes it safe to keep forever: no double adjustment."""
    s = _series([50, 50.5, 50.5, 51, 52])          # vendor already back-adjusted
    ev = pd.DataFrame({"ticker": ["X"], "date": [s.index[2]], "ratio": [2.0]})
    out, applied = caf.adjust_series(s, ev)
    assert applied == [] and out.equals(s)


# --- detection --------------------------------------------------------------------

def test_detect_respects_the_gap_aware_limit():
    w = pd.DataFrame({"A": _series([10, 10, 13, 13, 13]).values,       # +30% in one session: flagged
                      "B": _series([10, 10, 10, 10, 10]).values},
                     index=_series([0] * 5).index)
    w.loc[w.index[1:4], "B"] = np.nan
    w.loc[w.index[4], "B"] = 14                                            # +40% over 4 sessions: allowed
    d = rl.detect(w)
    assert list(d["ticker"]) == ["A"] and d["kind"].iloc[0] == "jump"


def test_bad_tick_is_one_outlier_between_agreeing_neighbours():
    w = pd.DataFrame({"A": _series([10, 10, 20, 10, 10]).values}, index=_series([0] * 5).index)
    d = rl.detect(w)
    assert list(d["kind"]) == ["bad_tick"]                                 # the reversal is not a 2nd event


def test_match_window_allows_sources_to_record_late():
    jump = pd.Timestamp("2026-06-29")
    assert rl._match(jump, 1.986, [(pd.Timestamp("2026-07-08"), 2.0)])    # HDBK: recorded 9 days later
    assert rl._match(jump, 1.986, [(pd.Timestamp("2026-08-30"), 2.0)]) is None
    assert rl._match(jump, 1.986, [(pd.Timestamp("2026-07-08"), 1.5)]) is None


def test_corroborate_requires_two_sources(monkeypatch):
    cands = pd.DataFrame([{"ticker": "H", "date": pd.Timestamp("2026-06-29"), "ret": -0.496,
                           "gap_sessions": 1, "implied_ratio": 1.986, "kind": "jump"}])
    monkeypatch.setattr(rl, "yahoo_splits", lambda t: pd.Series([2.0], index=[pd.Timestamp("2026-07-08")]))
    monkeypatch.setattr(rl, "sa_last_split", lambda t: (pd.Timestamp("2026-07-08"), 2.0))
    assert rl.corroborate(cands, pause=0)["verdict"].iloc[0] == "corroborated"
    monkeypatch.setattr(rl, "sa_last_split", lambda t: (pd.Timestamp("2026-07-08"), 1.5))   # disagrees
    assert rl.corroborate(cands, pause=0)["verdict"].iloc[0] == "one_source"
    monkeypatch.setattr(rl, "yahoo_splits", lambda t: pd.Series(dtype=float))
    monkeypatch.setattr(rl, "sa_last_split", lambda t: None)
    assert rl.corroborate(cands, pause=0)["verdict"].iloc[0] == "unexplained"


def test_librarian_cannot_write_prices():
    import sqlite3
    from ops import agent_guard as ag
    raw = sqlite3.connect(":memory:")
    raw.executescript("CREATE TABLE daily_ohlcv (c REAL); CREATE TABLE corporate_actions (x);")
    conn = ag.guarded(raw, rl.WRITABLE, rl.AGENT)
    with pytest.raises(ag.WriteBoundaryViolation):
        ag.run(conn, "UPDATE daily_ohlcv SET c = 1", agent=rl.AGENT)
    ag.run(conn, "INSERT INTO corporate_actions VALUES (1)", agent=rl.AGENT)
