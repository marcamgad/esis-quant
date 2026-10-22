"""Post-earnings drift study: surprise definition, timing, scoring. Temp DB only."""

import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from analysis import earnings_drift as ed


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "ed.db"))
    database.init_db()
    from data.fundamentals import _ensure_table
    with get_connection() as conn:
        _ensure_table(conn)
    return config.DB_PATH


def _quarters(values, start="2021-03-31"):
    ends = pd.date_range(start, periods=len(values), freq="QE")
    rows = []
    for pe, v in zip(ends, values):
        q = (pe.month - 1) // 3 + 1
        lag = 120 if q == 4 else 90
        rows.append(("X", "income", f"Q{q} {pe.year}", str(pe.date()),
                     str((pe + pd.Timedelta(days=lag)).date()), "Net Income", float(v), "q", "quarterly"))
    with get_connection() as conn:
        conn.executemany("INSERT INTO fundamentals (ticker, statement, fiscal_period, period_end, available_from, "
                         "line_item, value, fetched_at, frequency) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return ends


NOISE = [0, 1, -1, 2, -2, 1, 0, -1, 2, 1, -2, 0, 1, -1, 0, 2, -1]


def test_sue_compares_with_the_same_quarter_a_year_earlier(temp_db):
    # strong seasonality (Q4 always ~100 higher) must NOT read as a surprise
    base = [10, 10, 10, 110] * 4
    vals = [b + n for b, n in zip(base, NOISE)] + [60]          # last: Q1 jumps ~11 -> 60
    _quarters(vals)
    ev = ed.statement_events().set_index("period_end")
    assert ev.iloc[:-1]["sue"].abs().max() < 3                 # Q4 jumps are not surprises
    last = ev.iloc[-1]
    assert last["e"] == 60 and last["e_prior"] == vals[-5] and last["sue"] > 10


def test_sue_needs_four_prior_changes(temp_db):
    _quarters([1, 2, 3, 4, 5, 6, 7])                            # 3 changes before the last
    assert ed.statement_events().empty


def test_event_enters_strictly_after_availability(temp_db):
    _quarters([10 + n for n in NOISE[:12]] + [40])
    ev = ed.statement_events().iloc[[-1]]
    pe = ev["period_end"].iloc[0]
    idx = pd.bdate_range(pe, periods=200, freq="C", weekmask="Sun Mon Tue Wed Thu")
    r = pd.DataFrame({"X": 0.0}, index=idx)
    meas = ed.measure(ev, r, pd.Series(0.0, index=idx))
    entry = idx[meas["entry_pos"].iloc[0]]
    assert entry > ev["event_date"].iloc[0]


def test_score_is_signed_by_the_surprise():
    idx = pd.bdate_range("2024-01-01", periods=200)
    r = pd.DataFrame({"UP": 0.0, "DN": 0.0}, index=idx)
    r.iloc[51:111] = [0.01, -0.01]                              # good news drifts up, bad news down
    ev = pd.DataFrame([{"ticker": "UP", "period_end": idx[0], "event_date": idx[49], "sue": 2.0},
                       {"ticker": "DN", "period_end": idx[0], "event_date": idx[49], "sue": -2.0}])
    m = ed.measure(ev, r, pd.Series(0.0, index=idx)).set_index("ticker")
    assert m.loc["UP", "score_60"] > 0.5 and m.loc["DN", "score_60"] > 0.4
