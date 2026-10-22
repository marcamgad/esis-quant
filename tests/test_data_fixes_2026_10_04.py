"""Bad-tick removal, holiday calendar, macro store never blanks a column."""

import pandas as pd

from data import ca_factors as ca
from data import egx_holidays as eh


def test_bad_tick_is_replaced_only_while_it_is_a_spike():
    idx = pd.to_datetime(["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30"])
    spike = pd.Series([240.0, 242.4, 463.0, 231.9], index=idx)
    out, fixed = ca.drop_bad_ticks(spike, [idx[2]])
    assert fixed == [idx[2]] and out.iloc[2] == 242.4
    real = pd.Series([240.0, 242.4, 250.0, 251.0], index=idx)               # vendor fixed it
    out, fixed = ca.drop_bad_ticks(real, [idx[2]])
    assert fixed == [] and out.iloc[2] == 250.0


def test_rights_issue_factor_applies_at_the_ex_date():
    idx = pd.to_datetime(["2025-11-05", "2025-11-06", "2025-11-09", "2025-11-10"])
    s = pd.Series([3.88, 3.70, 0.774, 0.72], index=idx)
    ev = pd.DataFrame({"ticker": ["MCRO"], "date": [pd.Timestamp("2025-11-09")], "ratio": [4.723]})
    out, applied = ca.adjust_series(s, ev)
    assert applied and abs(out.iloc[2] / out.iloc[1] - 1) < 0.03            # no -79% day left


def test_sessions_skip_holidays():
    hol = {pd.Timestamp("2025-10-09")}                                       # 6 Oct moved to Thu 9 Oct
    assert not eh.is_session("2025-10-09", hol) and not eh.is_session("2025-10-10", hol)  # holiday, Friday
    assert eh.is_session("2025-10-08", hol)
    assert eh.sessions_between("2025-10-08", "2025-10-12", hol) == 1        # only Sun 12 Oct


def test_macro_store_keeps_values_a_fetch_did_not_get():
    from data.collectors import DataCollector
    from data.database import get_connection
    dc = DataCollector.__new__(DataCollector)
    d = pd.Timestamp("1999-01-03")
    dc._store_macro(pd.DataFrame({"egx30_level": [1234.5]}, index=[d]))
    dc._store_macro(pd.DataFrame({"usd_egp": [3.4]}, index=[d]))            # a Yahoo-only run
    with get_connection() as conn:
        row = conn.execute("SELECT egx30_level, usd_egp FROM macro_daily WHERE date='1999-01-03'").fetchone()
        conn.execute("DELETE FROM macro_daily WHERE date='1999-01-03'")
    assert row[0] == 1234.5 and row[1] == 3.4
