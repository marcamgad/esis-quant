"""
EGX holiday calendar: which Sun-Thu weekdays the exchange is closed.

WHY. The system knew only "Sun-Thu are sessions". Every Egyptian holiday --
Eid al-Fitr, Eid al-Adha (4-5 days each), Sham el-Nessim, the national days the
government moves to a Thursday -- looked like a missing day of data: the
sentinel counted a stale feed, and "N sessions later" in a backtest or the paper
test could land on a closed day.

PAST (2015 -> last stored session): DERIVED. A Sun-Thu weekday is a holiday when
NONE of three independent sources shows trading on it:
  * Mubasher's daily EGX30 history (macro_daily.egx30_level);
  * the price-derived calendar (data.calendar.valid_trading_days);
  * >= 10 names with volume in daily_ohlcv.
Each source alone has gaps (Mubasher's CSV skips days; the price calendar drops
days whose rows were carried forward), but a day missing from all three was a
closure. Result: 15-21 a year, and 2025 matches the official list exactly (Eid
al-Fitr 30 Mar-2 Apr, Sham el-Nessim 20-21 Apr, Eid al-Adha 5-9 Jun, 23 Jul moved
to Thu 24 Jul, 6 Oct moved to Thu 9 Oct, ...).

FUTURE: EXPECTED. Fixed-date holidays plus Islamic holidays from the astronomical
calendar. Islamic dates can shift by a day on moon sighting, and the government
often moves a mid-week holiday to the Thursday, so these are marked 'expected'
and must be confirmed when the cabinet announces them (`confirm`).

    py -3.11 -m data.egx_holidays            # derive past + show upcoming
"""

import sys

import pandas as pd

from data.database import get_connection

WEEKMASK = "Sun Mon Tue Wed Thu"

# (date, name). Weekend dates are listed for completeness; only Sun-Thu matter.
EXPECTED = [
    ("2026-10-06", "Armed Forces Day (may move to Thu 8 Oct)"),
    ("2027-01-07", "Coptic Christmas"),
    ("2027-01-25", "Revolution Day / Police Day"),
    ("2027-03-09", "Eid al-Fitr (approx.)"), ("2027-03-10", "Eid al-Fitr (approx.)"),
    ("2027-03-11", "Eid al-Fitr (approx.)"),
    ("2027-04-25", "Sinai Liberation Day"),
    ("2027-05-02", "Coptic Easter"), ("2027-05-03", "Sham el-Nessim"),
    ("2027-05-16", "Arafat Day (approx.)"), ("2027-05-17", "Eid al-Adha (approx.)"),
    ("2027-05-18", "Eid al-Adha (approx.)"), ("2027-05-19", "Eid al-Adha (approx.)"),
    ("2027-06-06", "Islamic New Year (approx.)"),
    ("2027-06-30", "30 June Revolution"),
    ("2027-08-15", "Prophet's Birthday (approx.)"),
    ("2027-10-06", "Armed Forces Day"),
]


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS egx_holidays (
        date TEXT PRIMARY KEY, name TEXT, status TEXT, updated_at TEXT)""")


def derive() -> dict:
    """Write past holidays from the three session sources."""
    from data.calendar import valid_trading_days
    with get_connection() as conn:
        _ensure(conn)
        idx = pd.to_datetime(pd.read_sql("SELECT date FROM macro_daily WHERE egx30_level IS NOT NULL",
                                          conn)["date"])
        vol = pd.read_sql("SELECT date, SUM(volume > 0) AS v FROM daily_ohlcv GROUP BY date", conn)
    traded = pd.to_datetime(vol.loc[vol["v"] >= 10, "date"])
    sess = pd.DatetimeIndex(idx).union(valid_trading_days()).union(pd.DatetimeIndex(traded))
    start, end = pd.Timestamp("2015-01-05"), sess.max()
    wd = pd.bdate_range(start, end, freq="C", weekmask=WEEKMASK)
    hol = wd.difference(sess)
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.executemany("INSERT INTO egx_holidays VALUES (?,?,?,?) ON CONFLICT(date) DO UPDATE SET "
                         "status=CASE WHEN egx_holidays.status='confirmed' THEN 'confirmed' ELSE "
                         "excluded.status END, updated_at=excluded.updated_at",
                         [(str(d.date()), None, "derived", now) for d in hol])
        conn.executemany("INSERT OR IGNORE INTO egx_holidays VALUES (?,?,?,?)",
                         [(d, n, "expected", now) for d, n in EXPECTED if pd.Timestamp(d) > end])
    return {"derived": len(hol), "through": str(end.date()),
            "per_year": pd.Series(hol.year).value_counts().sort_index().to_dict()}


def confirm(date: str, name: str = "", closed: bool = True) -> None:
    """Record an announced holiday (or that an expected one is NOT a holiday)."""
    with get_connection() as conn:
        _ensure(conn)
        if closed:
            conn.execute("INSERT OR REPLACE INTO egx_holidays VALUES (?,?,'confirmed',?)",
                         (date, name or None, pd.Timestamp.now().isoformat(timespec="seconds")))
        else:
            conn.execute("DELETE FROM egx_holidays WHERE date=?", (date,))


def holidays() -> set:
    with get_connection() as conn:
        try:
            _ensure(conn)
            return {pd.Timestamp(r[0]) for r in conn.execute("SELECT date FROM egx_holidays")}
        except Exception:  # noqa: BLE001
            return set()


def is_session(d, hol: set = None) -> bool:
    d = pd.Timestamp(d).normalize()
    return d.dayofweek not in (4, 5) and d not in (holidays() if hol is None else hol)


def sessions_between(a, b, hol: set = None) -> int:
    """EGX sessions in (a, b]: Sun-Thu minus known holidays."""
    a, b = pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()
    if b <= a:
        return 0
    hol = holidays() if hol is None else hol
    return int(sum(is_session(d, hol) for d in pd.date_range(a + pd.Timedelta(days=1), b, freq="D")))


def upcoming(days: int = 60) -> pd.DataFrame:
    now = pd.Timestamp.now().normalize()
    with get_connection() as conn:
        _ensure(conn)
        return pd.read_sql("SELECT * FROM egx_holidays WHERE date >= ? AND date <= ? ORDER BY date", conn,
                           params=(str(now.date()), str((now + pd.Timedelta(days=days)).date())))


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print(derive())
    print(upcoming(120).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
