"""Forward paper test of the index-addition rule (runs on the session DB copy)."""

import numpy as np

from ops import index_paper_trade as ipt
from data.database import get_connection


def test_sep_2026_review_replays_end_to_end():
    ipt.announce("2026-08-16", "2026-09-01", ["MFPC", "ALCN", "CLHO", "SKPC"], ["ARCC"], "test replay")
    ipt.update()
    r = ipt.report()
    tab = r["table"]
    sep = tab[tab["announced"] == "2026-08-16"]
    assert set(sep["ticker"]) == {"MFPC", "ALCN", "CLHO", "SKPC"}          # removals are never bought
    closed = sep[sep["status"] == "closed"]
    assert len(closed) >= 3
    for row in closed.itertuples():
        assert row.entry_date > "2026-08-16"                             # strictly after the announcement
        assert row.exit_date >= "2026-09-01"
        assert np.isfinite(row.net_abnormal)
        assert row.net_ret < row.gross_ret                               # fees always cost
    assert r["verdict"].startswith("pending")                            # one review is not a verdict
    with get_connection() as conn:
        conn.execute("DELETE FROM index_paper_trades WHERE announced='2026-08-16'")
        conn.execute("DELETE FROM index_paper_reviews WHERE announced='2026-08-16'")


def test_announcement_headlines_are_detected():
    yes = ["البورصة: انضمام شركات كيما وإيديتا وأوراسكوم لمؤشر EGX30",
           "EGX shuffles 6 new companies in and 7 out in latest EGX30 review",
           "البورصة تنتهي من المراجعة الدورية نصف السنوية لمؤشرات السوق"]
    no = ["EGX30 slips 0.35% as Egyptian investors record net sales", "أرباح طلعت مصطفى ترتفع 43%"]
    assert all(ipt.ANNOUNCE_HEADLINE.search(h) for h in yes)
    assert not any(ipt.ANNOUNCE_HEADLINE.search(h) for h in no)
