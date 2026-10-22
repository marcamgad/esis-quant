"""
Agent 3 -- REFERENCE LIBRARIAN. Corporate actions confirmed by two sources.

THE PROBLEM IT FIXES (measured 2026-10-03). The adjusted price series still held
145 single-session moves beyond what EGX's daily limit allows. Many were
splits and bonus issues the vendor never adjusted -- HDBK 2:1 showed as a -49.6%
day, EFID 1.5:1 as -32.7% -- and every one of them reads as a crash to the
covariance model, the regime model and every risk number.

WHAT COUNTS AS CONFIRMED. A jump is a corporate action only when THREE things
agree on the ratio (within 8%):
  1. the price series itself (the implied ratio across the jump),
  2. Yahoo's split record (yfinance actions),
  3. stockanalysis.com's split record (S&P Global data).
OR (since 2026-10-04) when the EXCHANGE itself re-based the price -- an open
beyond the daily limit in Mubasher's raw history (exchange_reference_events) --
in the window and the same direction. That third source is what confirms rights
issues at par and mergers, which no split record lists.
Sources record the event up to ~9 sessions after the actual price drop (HDBK:
drop 29 Jun, recorded 8 Jul), so the date window runs from 7 days before to 20
days after the jump. Two sources -> written to corporate_actions as
SPLIT_CORROBORATED and applied at read time (data/ca_factors.py). One source,
none, or a reversing spike -> staged for a person; nothing is applied.

EGX LIMIT, GAP-AWARE. A move is only suspicious if it exceeds what the daily
limit allows over the sessions it spans: (1.2^g - 1) up, (1 - 0.8^g) down, for a
stock that last traded g sessions earlier, plus a 2% tolerance.

WRITE BOUNDARY (engine-enforced): corporate_actions, corporate_actions_staging,
agent_findings. It never writes a price.

    py -3.11 -m ops.reference_librarian               # scan, corroborate, report
    py -3.11 -m ops.reference_librarian approve HDBK 2026-06-29 2.0   # your call on a staged one
"""

import logging
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

from data.database import get_connection
from ops import agent_guard as ag

logger = logging.getLogger("esis_quant.reference_librarian")

AGENT = "reference_librarian"
WRITABLE = {"corporate_actions", "corporate_actions_staging", "agent_findings"}
DAILY_LIMIT = 0.20
TOLERANCE = 0.02
RATIO_TOL = 0.08
WINDOW_BEFORE, WINDOW_AFTER = 7, 20      # days around the price jump to look for a source record
REVERSAL_WINDOW = 3
MATERIAL_RATIO = 1.2                     # source-driven pass corrects only splits/bonuses >= x1.2


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS corporate_actions_staging (
        ticker TEXT, date TEXT, implied_ratio REAL, gap_sessions INTEGER, kind TEXT,
        yahoo TEXT, stockanalysis TEXT, verdict TEXT, status TEXT DEFAULT 'pending',
        created_at TEXT, PRIMARY KEY (ticker, date))""")
    ag.ensure_tables(conn)


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def _price_panel():
    from data.calendar import valid_trading_days
    from data.ca_factors import adjust_panel
    with get_connection() as conn:
        px = pd.read_sql("SELECT ticker, date, adjusted_close FROM daily_ohlcv WHERE adjusted_close > 0",
                         conn, parse_dates=["date"])
    wide = px.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    wide = wide.loc[wide.index.isin(valid_trading_days(wide))]
    return adjust_panel(wide)          # already-confirmed events are not re-flagged


def detect(wide: pd.DataFrame = None, since=None) -> pd.DataFrame:
    """Every move beyond the gap-aware EGX limit, classified jump / bad_tick."""
    wide = _price_panel() if wide is None else wide
    pos = {d: i for i, d in enumerate(wide.index)}
    rows = []

    def beyond(r, g):
        return r > (1 + DAILY_LIMIT) ** g - 1 + TOLERANCE or -r > 1 - (1 - DAILY_LIMIT) ** g + TOLERANCE

    for t in wide.columns:
        s = wide[t].dropna()
        if len(s) < 2:
            continue
        idx = [pos[d] for d in s.index]
        px = s.to_numpy()
        # GAP = real EGX sessions spanned, not panel rows. The calendar drops days
        # the vendor carried forward without volume (9-10 Aug 2026), so a 3-session
        # move would otherwise be judged against a 1-session limit -- which put ~30
        # names on a false "unexplained +25%" list for 11 Aug.
        gaps = [1] + [max(1, idx[i] - idx[i - 1], _egx_weekdays(s.index[i - 1], s.index[i]))
                      for i in range(1, len(s))]
        flagged = [False] + [beyond(px[i] / px[i - 1] - 1, gaps[i]) for i in range(1, len(s))]
        for i in range(1, len(s)):
            if not flagged[i]:
                continue
            d = s.index[i]
            if since is not None and d < pd.Timestamp(since):
                continue
            # A BAD TICK is a single price out of line with BOTH neighbours while
            # the neighbours agree with each other. Two such days in a row (OCPH
            # 28-30 Jul 2026: -49%, +91%, -50% around a real 2:1 split) cannot be
            # told apart by prices alone -> 'ambiguous', for a person.
            def spike(k):
                return (0 < k < len(s) - 1 and flagged[k] and flagged[k + 1]
                        and not beyond(px[k + 1] / px[k - 1] - 1, gaps[k] + gaps[k + 1]))
            if spike(i) and (spike(i - 1) or spike(i + 1)):
                kind = "ambiguous"
            elif spike(i):
                kind = "bad_tick"
            elif spike(i - 1):
                continue                                  # the reversal of a bad tick, not an event
            else:
                kind = "jump"
            rows.append({"ticker": t, "date": d, "ret": float(px[i] / px[i - 1] - 1),
                         "gap_sessions": int(gaps[i]), "implied_ratio": float(px[i - 1] / px[i]),
                         "kind": kind})
    return pd.DataFrame(rows)


def _egx_weekdays(a, b) -> int:
    """Sun-Thu days in (a, b]."""
    days = pd.date_range(pd.Timestamp(a) + pd.Timedelta(days=1), pd.Timestamp(b), freq="D")
    return int(sum(d.dayofweek not in (4, 5) for d in days))


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def yahoo_splits(ticker: str) -> pd.Series:
    import yfinance as yf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sp = yf.Ticker(f"{ticker}.CA").splits
        except Exception:  # noqa: BLE001
            return pd.Series(dtype=float)
    if sp is None or sp.empty:
        return pd.Series(dtype=float)
    sp.index = sp.index.tz_localize(None).normalize()
    return sp[sp > 0]


def sa_last_split(ticker: str):
    """(date, ratio) of the last split stockanalysis records, or None."""
    from data.fundamentals import _fetch
    try:
        html = _fetch(f"https://stockanalysis.com/quote/egx/{ticker}/statistics/")
    except Exception:  # noqa: BLE001
        return None
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
    m = re.search(r"Last Split Date\s+([A-Z][a-z]{2} \d{1,2}, \d{4})", txt)
    k = re.search(r"Split Ratio\s+([\d.]+)", txt)
    if not (m and k):
        return None
    return pd.Timestamp(m.group(1)), float(k.group(1))


# THIRD SOURCE (2026-10-04): the EXCHANGE's own reference price. When a corporate
# action takes effect -- split, bonus, rights issue at par, merger -- EGX opens the
# stock at a reference price it computes, and that open sits BEYOND the 20% daily
# limit from the previous close (POUL: closed 38.20, re-opened at 10.36 after its
# merger). Ordinary trading cannot open there: in Mubasher's raw history the open
# is the previous close. So an open more than REF_GAP away from the prior close is
# the exchange confirming an event, with ratio prior close / open. It catches what
# the split records miss entirely: rights issues at par (MCRO, COPR, AMES) and
# mergers (POUL), which Yahoo and stockanalysis never list.
REF_GAP = 0.22
# Our vendor series is sometimes back-adjusted for later events, so its jump can
# differ from the exchange's ratio; the exchange event establishes THAT an action
# happened in the window, and the factor applied is our own series' jump.
REF_RATIO_TOL = 0.30
_MUB_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}


def exchange_reference_events(ticker: str) -> list:
    """[(date, prior_close/open)] where the exchange re-based the price."""
    import io
    import urllib.request
    try:
        page = urllib.request.urlopen(urllib.request.Request(
            f"https://www.mubasher.info/markets/EGX/stocks/{ticker}", headers=_MUB_UA), timeout=30
        ).read().decode("utf-8", "replace")
        m = re.search(r'historical-data-url="([^"]+\.csv)"', page)
        if not m:
            return []
        raw = urllib.request.urlopen(urllib.request.Request(m.group(1), headers=_MUB_UA), timeout=60).read()
    except Exception:  # noqa: BLE001
        return []
    d = pd.read_csv(io.BytesIO(raw), header=None, names=["ts", "open", "high", "low", "close", "volume"])
    d["date"] = pd.to_datetime(d["ts"].str.slice(0, 10), errors="coerce")
    d = d.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    prev = d["close"].shift(1)
    gap = d["open"] / prev - 1
    hit = d[(gap.abs() > REF_GAP) & (d["open"] > 0)]
    return [(dt, float(prev.loc[dt] / r.open)) for dt, r in hit.iterrows()]


def _match(date, ratio, events) -> tuple:
    """(source_date, source_ratio) of the event matching a jump, or None."""
    for d, r in events:
        if pd.Timestamp(date) - pd.Timedelta(days=WINDOW_BEFORE) <= d <= \
                pd.Timestamp(date) + pd.Timedelta(days=WINDOW_AFTER):
            if r > 0 and abs(r / ratio - 1) <= RATIO_TOL:
                return d, r
    return None


def _match_ref(date, ratio, refs) -> tuple:
    """An exchange re-base in the window, same direction, ratio within REF_RATIO_TOL."""
    for d, r in refs:
        if pd.Timestamp(date) - pd.Timedelta(days=WINDOW_AFTER) <= d <= \
                pd.Timestamp(date) + pd.Timedelta(days=WINDOW_AFTER):
            if r > 1 and ratio > 1 and abs(np.log(r) - np.log(ratio)) <= np.log(1 + REF_RATIO_TOL) \
                    + 0.25 * abs(np.log(ratio)):
                return d, r
    return None


def corroborate(cands: pd.DataFrame, pause: float = 0.7) -> pd.DataFrame:
    """Add yahoo / stockanalysis matches and a verdict to each candidate."""
    out = []
    for t, g in cands.groupby("ticker"):
        ysp = [(d, float(r)) for d, r in yahoo_splits(t).items()]
        sa = sa_last_split(t)
        sas = [sa] if sa else []
        refs = exchange_reference_events(t)
        time.sleep(pause)
        for c in g.itertuples():
            y = _match(c.date, c.implied_ratio, ysp)
            s = _match(c.date, c.implied_ratio, sas)
            x = _match_ref(c.date, c.implied_ratio, refs)
            # stockanalysis keeps only the LATEST split, so for an older Yahoo
            # event it has nothing to say: Yahoo's exact ratio + the price step
            # beyond the daily limit are then the two pieces of evidence.
            sa_silent = y is not None and (sa is None or abs((sa[0] - y[0]).days) > 3)
            ratio = (y or s)[1] if (y or s) else None
            if c.kind == "bad_tick":
                verdict = c.kind
            elif y and (s or sa_silent):
                verdict = "corroborated"
            elif x and c.implied_ratio > 1:
                # the exchange re-based the price + our series jumped the same way
                verdict, ratio = "corroborated", float(c.implied_ratio)
            elif c.kind == "ambiguous":
                verdict = c.kind
            elif y or s:
                verdict = "one_source"
            else:
                verdict = "unexplained"
            out.append({**c._asdict(),
                        "yahoo": f"{y[0].date()} x{y[1]:g}" if y else None,
                        "stockanalysis": (f"{s[0].date()} x{s[1]:g}" if s else
                                          f"EGX reference {x[0].date()} x{x[1]:.3g}" if x else None),
                        "source_ratio": ratio, "verdict": verdict})
    return pd.DataFrame(out).drop(columns=["Index"], errors="ignore")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def source_driven(wide: pd.DataFrame, skip: set, pause: float = 0.3) -> pd.DataFrame:
    """PASS 2: start from the split RECORDS, not the price jumps.

    The limit-based scan misses splits whose drop fits inside a multi-session
    limit (MOIN 1.3:1 and GGCC 1.5:1 both fell after a weekend). For every
    Yahoo split inside the price history, look for a price step with the same
    ratio in the window and require stockanalysis to agree when it covers that
    event (it records only the latest split)."""
    rows = []
    for t in wide.columns:
        s = wide[t].dropna()
        if len(s) < 30:
            continue
        ysp = yahoo_splits(t)
        time.sleep(pause)
        if ysp.empty:
            continue
        ysp = ysp[(ysp.index >= s.index[0]) & (ysp.index <= s.index[-1] + pd.Timedelta(days=WINDOW_AFTER))]
        if ysp.empty:
            continue
        sa = sa_last_split(t)
        steps = (s.shift(1) / s).dropna()
        for d, r in ysp.items():
            # Small bonuses (x1.03-x1.15) are indistinguishable from ordinary daily
            # moves and barely distort risk if missed; the first version "matched"
            # 96 of them to random 2-10% days. Only material events are corrected.
            if 1 / MATERIAL_RATIO < r < MATERIAL_RATIO:
                continue
            win = steps[(steps.index >= d - pd.Timedelta(days=WINDOW_AFTER)) &
                        (steps.index <= d + pd.Timedelta(days=WINDOW_BEFORE))]
            if win.empty:
                continue
            # the matching step must be THE largest move in the window, not any move
            jd = np.log(win).abs().idxmax()
            step = float(win.loc[jd])
            if abs(np.log(step) - np.log(r)) > 0.25 * abs(np.log(r)):
                continue                          # vendor already adjusted, or no clean match
            if (t, jd) in skip:
                continue
            hit = win.loc[[jd]]
            # stockanalysis records only the LATEST split: it speaks to this event
            # only when its record is for the same date.
            sa_same = sa is not None and abs((sa[0] - d).days) <= 3
            sa_ok = sa_same and abs(sa[1] / r - 1) <= RATIO_TOL
            verdict = ("corroborated" if sa_ok else "one_source" if sa_same
                       else "corroborated_yahoo_price")
            rows.append({"ticker": t, "date": jd, "ret": float(1 / hit.iloc[-1] - 1), "gap_sessions": 1,
                         "implied_ratio": float(hit.iloc[-1]), "kind": "jump",
                         "yahoo": f"{d.date()} x{r:g}", "stockanalysis": f"{sa[0].date()} x{sa[1]:g}" if sa_ok else None,
                         "source_ratio": float(r), "verdict": verdict})
    return pd.DataFrame(rows)


def run(since=None, full: bool = True) -> dict:
    wide = _price_panel()
    cands = detect(wide, since=since)
    tab = corroborate(cands) if not cands.empty else pd.DataFrame()
    if full:
        seen = set(zip(tab["ticker"], pd.to_datetime(tab["date"]))) if not tab.empty else set()
        extra = source_driven(wide, seen)
        if not extra.empty:
            tab = pd.concat([tab, extra], ignore_index=True)
    if tab.empty:
        return {"candidates": 0, "table": tab}
    # Older splits: stockanalysis only records the LATEST one, so it cannot speak
    # to them. Yahoo's exact ratio matching an actual price step (within 8%) is
    # then the two independent pieces of evidence -- the vendor's record and
    # the market's own prices.
    tab.loc[tab["verdict"] == "corroborated_yahoo_price", "verdict"] = "corroborated"
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    with get_connection() as conn:
        held = {r[0] for r in conn.execute("SELECT ticker FROM user_positions")}
    with get_connection() as raw:
        _ensure(raw)
        conn = ag.guarded(raw, WRITABLE, AGENT)
        for r in tab.itertuples():
            d = pd.Timestamp(r.date).strftime("%Y-%m-%d")
            ag.run(conn, "INSERT INTO corporate_actions_staging (ticker, date, implied_ratio, gap_sessions, "
                         "kind, yahoo, stockanalysis, verdict, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?) "
                         "ON CONFLICT(ticker, date) DO UPDATE SET implied_ratio=excluded.implied_ratio, "
                         "yahoo=excluded.yahoo, stockanalysis=excluded.stockanalysis, verdict=excluded.verdict, "
                         "status=CASE WHEN corporate_actions_staging.status IN ('approved','rejected') "
                         "THEN corporate_actions_staging.status ELSE excluded.status END",
                   (r.ticker, d, r.implied_ratio, r.gap_sessions, r.kind, r.yahoo, r.stockanalysis,
                    r.verdict, "applied" if r.verdict == "corroborated" else "pending", now), AGENT)
            if r.verdict == "corroborated":
                ag.run(conn, "INSERT OR REPLACE INTO corporate_actions (ticker, date, action_type, ratio, "
                             "amount_egp) VALUES (?, ?, 'SPLIT_CORROBORATED', ?, NULL)",
                       (r.ticker, d, float(r.source_ratio)), AGENT)
        recent = tab[pd.to_datetime(tab["date"]) >= pd.Timestamp.now() - pd.Timedelta(days=365)]
        findings = []
        for r in recent[recent["verdict"] != "corroborated"].itertuples():
            sev = "high" if r.ticker in held else "medium"
            findings.append({"severity": sev, "kind": f"price_{r.verdict}", "subject": f"{r.ticker}@{r.date.date()}",
                             "summary": f"{r.ticker} {r.ret:+.0%} on {r.date.date()} "
                                        f"({r.verdict.replace('_', ' ')}; implied ratio {r.implied_ratio:.3f})",
                             "fix_hint": "check the EGX disclosure for a split/bonus/capital change; "
                                         "approve with: py -3.11 -m ops.reference_librarian approve "
                                         f"{r.ticker} {r.date.date()} RATIO"})
        ag.record_findings(conn, AGENT, findings)
    return {"candidates": len(tab), "table": tab}


def approve(ticker: str, date: str, ratio: float) -> None:
    """YOUR decision on a staged event: record it as confirmed (applied at read time)."""
    with get_connection() as conn:
        _ensure(conn)
        conn.execute("INSERT OR REPLACE INTO corporate_actions (ticker, date, action_type, ratio, amount_egp) "
                     "VALUES (?, ?, 'SPLIT_CORROBORATED', ?, NULL)", (ticker, date, float(ratio)))
        conn.execute("UPDATE corporate_actions_staging SET status='approved' WHERE ticker=? AND date=?",
                     (ticker, date))


def format_report(res: dict) -> str:
    t = res["table"]
    L = ["=" * 70, f"REFERENCE LIBRARIAN — {res['candidates']} move(s) beyond the EGX daily limit", "=" * 70]
    if t is None or t.empty:
        return "\n".join(L + ["  none"])
    L.append("  " + ", ".join(f"{k}: {v}" for k, v in t["verdict"].value_counts().items()))
    recent = t[pd.to_datetime(t["date"]) >= pd.Timestamp.now() - pd.Timedelta(days=365)].sort_values("date")
    L.append(f"\n  last 12 months ({len(recent)}):")
    for r in recent.itertuples():
        L.append(f"   {r.ticker:6} {pd.Timestamp(r.date).date()} {r.ret:+7.1%} gap {r.gap_sessions}  "
                 f"implied x{r.implied_ratio:.3f}  yahoo={r.yahoo or '-':16} sa={r.stockanalysis or '-':16} "
                 f"-> {r.verdict}")
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    if argv[:1] == ["approve"]:
        approve(argv[1], argv[2], float(argv[3]))
        print(f"recorded {argv[1]} {argv[2]} x{argv[3]} as confirmed")
        return 0
    print(format_report(run()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
