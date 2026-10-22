"""
Forward paper test of the EGX30 index-addition rule.

WHY. Every result in this system is a backtest. The index-addition effect passed
its pre-registered test on 73 events (2017-2026), and then FAILED to replicate on
34 events recovered on 2026-10-03 (score -0.22%, p 0.66, additions -1.1%). The
cheapest way to settle it is forward: write the rule down now, apply it to the
next reviews as they are announced, and score it with fills a person could get.
No money moves; nothing here places an order.

THE RULE (registered 2026-10-03, before the March 2027 review):
  * at each EGX30 review, paper-BUY every ADDED stock (no shorting on EGX, so
    removals are not traded);
  * size 5,000 EGP per name (below ~2k the flat fee eats the effect);
  * entry: typical price (H+L+C)/3 of the FIRST session after the announcement,
    plus 0.57% slippage (measured on the live account);
  * exit: typical price of the first session after the effective date (or one
    session after entry, whichever is later), minus 0.57%;
  * Thndr fees both ways (backtest.fees.round_trip_cost);
  * benchmark: equal-weight market return over the same sessions.
SCORING. After 3 reviews (Mar 2027, Sep 2027, Mar 2028) the rule is judged on
the mean net abnormal return per trade. Positive with at least 2 of 3 reviews
positive -> worth small real money; otherwise -> drop it. No re-tuning of the
rule between reviews; a changed rule restarts the count.

Announcements are entered by hand (they come as an EGX press release, ~2 weeks
before the effective date since the July 2026 rule change); `detect()` flags
likely announcement headlines from the news feed for the morning brief.

    py -3.11 -m ops.index_paper_trade announce 2027-02-15 2027-03-01 TICK1,TICK2 TICK3,TICK4 "EGX release"
    py -3.11 -m ops.index_paper_trade update     # fill entries/exits as prices arrive
    py -3.11 -m ops.index_paper_trade report
"""

import json
import re
import sys

import numpy as np
import pandas as pd

from data.database import get_connection

SIZE_EGP = 5000.0
SLIPPAGE = 0.0057
RULE_VERSION = "v1-2026-10-03"
REVIEWS_TO_JUDGE = 3

# Either order: "انضمام ... لمؤشر EGX30" and "EGX30 ... review" both occur.
ANNOUNCE_HEADLINE = re.compile(
    r"(?=.*(EGX\s*30|إيجي\s*إكس\s*30|المؤشر الرئيسي|المؤشر الثلاثيني))"
    r"(?=.*(مراجع|انضمام|تنضم|ينضم|دخول|خروج|استبعاد|\breview|reshuffl|\bjoin|\badd))"
    r"|(المراجعة الدورية|semi-annual review|periodic review).{0,60}(مؤشر|EGX|indices)", re.I)


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS index_paper_reviews (
        announced TEXT PRIMARY KEY, effective TEXT, added TEXT, removed TEXT,
        source TEXT, rule_version TEXT, entered_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS index_paper_trades (
        announced TEXT, ticker TEXT, size_egp REAL,
        entry_date TEXT, entry_px REAL, exit_date TEXT, exit_px REAL,
        fees_egp REAL, gross_ret REAL, net_ret REAL, market_ret REAL, net_abnormal REAL,
        status TEXT, updated_at TEXT, PRIMARY KEY (announced, ticker))""")


def announce(announced: str, effective: str, added, removed=(), source: str = "") -> dict:
    added = [t.strip().upper() for t in added if t.strip()]
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        _ensure(conn)
        conn.execute("INSERT OR REPLACE INTO index_paper_reviews VALUES (?,?,?,?,?,?,?)",
                     (announced, effective, json.dumps(added), json.dumps(list(removed)),
                      source, RULE_VERSION, now))
        for t in added:
            conn.execute("INSERT OR IGNORE INTO index_paper_trades (announced, ticker, size_egp, status, updated_at) "
                         "VALUES (?,?,?, 'waiting entry', ?)", (announced, t, SIZE_EGP, now))
    return {"review": announced, "paper_buys": added}


def _sessions() -> pd.DatetimeIndex:
    from data.calendar import valid_trading_days
    return valid_trading_days()


def _typical(ticker: str) -> pd.Series:
    with get_connection() as conn:
        px = pd.read_sql("SELECT date, high, low, close, volume FROM daily_ohlcv WHERE ticker=? AND close>0",
                         conn, params=(ticker,), parse_dates=["date"]).drop_duplicates("date", keep="last")
    px = px.set_index("date").sort_index()
    ok = (px["high"] > px["low"]) & (px["volume"].fillna(0) > 0)
    return pd.Series(np.where(ok, (px["high"] + px["low"] + px["close"]) / 3, px["close"]), index=px.index)


def _market_ret(start, end) -> float:
    """Equal-weight mean close-to-close return of all names over (start, end]."""
    from backtest.engine import load_price_returns
    r = load_price_returns()
    w = r[(r.index > start) & (r.index <= end)]
    return float(np.prod(1 + w.mean(axis=1, skipna=True).fillna(0)) - 1) if len(w) else np.nan


def update() -> dict:
    from backtest.fees import round_trip_cost
    sess = _sessions()
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        _ensure(conn)
        revs = {r[0]: r[1] for r in conn.execute("SELECT announced, effective FROM index_paper_reviews")}
        open_ = conn.execute("SELECT announced, ticker FROM index_paper_trades WHERE status != 'closed'").fetchall()
    done = []
    for ann, t in open_:
        eff = revs[ann]
        i_ann = sess.searchsorted(pd.Timestamp(ann), side="right")
        i_eff = sess.searchsorted(pd.Timestamp(eff), side="left")
        i_exit = max(i_ann + 1, i_eff + 1)
        if i_ann >= len(sess):
            continue
        tp = _typical(t)
        d_in = sess[i_ann]
        if d_in not in tp.index:
            continue                                   # no print yet (or the stock did not trade)
        p_in = float(tp[d_in]) * (1 + SLIPPAGE)
        fields = {"entry_date": str(d_in.date()), "entry_px": p_in, "status": "open"}
        if i_exit < len(sess) and sess[i_exit] in tp.index:
            d_out = sess[i_exit]
            p_out = float(tp[d_out]) * (1 - SLIPPAGE)
            gross = p_out / p_in - 1
            fees = round_trip_cost(SIZE_EGP)["total_egp"]
            net = gross - fees / SIZE_EGP
            mkt = _market_ret(d_in, d_out)
            fields.update(exit_date=str(d_out.date()), exit_px=p_out, fees_egp=fees, gross_ret=gross,
                          net_ret=net, market_ret=mkt, net_abnormal=net - mkt, status="closed")
        sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
        with get_connection() as conn:
            conn.execute(f"UPDATE index_paper_trades SET {sets} WHERE announced=? AND ticker=?",
                         (*fields.values(), now, ann, t))
        done.append((ann, t, fields["status"]))
    return {"updated": done}


def report() -> dict:
    with get_connection() as conn:
        _ensure(conn)
        tr = pd.read_sql("SELECT * FROM index_paper_trades ORDER BY announced, ticker", conn)
    closed = tr[tr["status"] == "closed"]
    by_rev = closed.groupby("announced")["net_abnormal"].mean() if len(closed) else pd.Series(dtype=float)
    out = {"rule": RULE_VERSION, "trades": len(tr), "closed": len(closed),
           "reviews_closed": len(by_rev), "by_review": by_rev.round(4).to_dict()}
    if len(closed):
        out.update(mean_net_abnormal=float(closed["net_abnormal"].mean()),
                   mean_net=float(closed["net_ret"].mean()),
                   net_egp=float((closed["net_ret"] * closed["size_egp"]).sum()),
                   hit=float((closed["net_abnormal"] > 0).mean()))
    if len(by_rev) >= REVIEWS_TO_JUDGE:
        ok = out["mean_net_abnormal"] > 0 and (by_rev > 0).sum() >= 2
        out["verdict"] = "KEEP (small real money)" if ok else "DROP"
    else:
        out["verdict"] = f"pending ({len(by_rev)}/{REVIEWS_TO_JUDGE} reviews closed)"
    out["table"] = tr
    return out


def detect(days: int = 21) -> pd.DataFrame:
    """Headlines that look like an EGX30 review announcement, for the brief."""
    since = (pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat()
    with get_connection() as conn:
        try:
            h = pd.read_sql("SELECT headline, url, published_at, first_seen_at FROM news_items "
                            "WHERE first_seen_at >= ?", conn, params=(since,))
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return h[h["headline"].fillna("").str.contains(ANNOUNCE_HEADLINE)].drop_duplicates("headline")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if argv[:1] == ["announce"] and len(argv) >= 4:
        removed = argv[4].split(",") if len(argv) > 4 else []
        print(announce(argv[1], argv[2], argv[3].split(","), removed, argv[5] if len(argv) > 5 else ""))
    elif argv[:1] == ["update"]:
        print(update())
    elif argv[:1] == ["detect"]:
        print(detect().to_string())
    else:
        r = report()
        tab = r.pop("table")
        print(r)
        if len(tab):
            print(tab[["announced", "ticker", "entry_date", "exit_date", "net_ret", "net_abnormal", "status"]]
                  .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
