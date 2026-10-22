"""
Agent 4 -- DISCLOSURE READER (rules stage). Reported numbers from news, point-in-time.

WHAT IT DOES. Turns a results headline into a row:
    "أرباح السويدي إليكتريك ترتفع إلى 10.6 مليار جنيه خلال النصف الأول من 2026"
 -> SWDY | net_profit | 10,600 EGP m | H1 2026 | direction + | first_seen 2026-07-..
stamped with the time the system FIRST SAW it. That is a point-in-time fact
nobody can sell you: when the market could have known the number.

WHY RULES FIRST. The audit plan's own Arabic encoder needs 20-50M tokens to
pretrain; the corpus on 2026-10-03 is ~0.3M words of Arabic (about 1% of that).
Training now would memorise a tiny corpus. This module therefore (1) delivers
the extraction with deterministic rules today, (2) defines the single interface
a learned model will later plug into -- extract(task, text) -> rows, every call
logged to model_calls -- and (3) keeps the labelling queue whose held-out half
is the bar any model must beat.

WHAT IT REFUSES TO GUESS.
  * targets and budgets (تستهدف / مستهدف / موازنة / target) are guidance, not results
  * real-estate "مبيعات" are contracted PRE-SALES, not revenue -> metric 'presales'
  * a non-EGP/USD currency (SAR, AED/درهم) flags a probable wrong company link
  * a period it cannot read stays NULL; the row is kept with lower confidence

    py -3.11 -m data.disclosure_reader            # extract all, validate vs statements
    py -3.11 -m data.disclosure_reader label      # label the next items (active learning)
"""

import json
import logging
import re
import sys
import time

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.disclosure_reader")

METHOD = "figures_rules_v1"
AGENT = "disclosure_reader"
WRITABLE = {"news_figures", "model_calls", "label_queue", "agent_findings"}

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")
REAL_ESTATE = {"TMGH", "PHDC", "OCDI", "EMFD", "MASR", "HELI", "ORHD", "MNHD", "PRMH", "ELKA", "AMER",
               "ZMID", "UNIT", "TANM", "CIRA"}  # developers whose "sales" are contracted pre-sales

# Targets, budgets AND analyst forecasts ("بنوك استثمار: ... تقفز بأرباح" is an
# investment-bank estimate, not a reported result -- found in the 2019 archive).
GUIDANCE = re.compile(r"تستهدف|مستهدف|يستهدف|موازن|توقع|تقديرات|محلل|بنوك\s+استثمار|"
                      r"target|guidance|budget|forecast|expects|estimates?|analysts?", re.I)
DIVIDEND = re.compile(r"للسهم|كوبون|توزيع\S*\s+(?:أرباح|كوبون)|dividend|per share", re.I)

METRICS = [   # order matters: the first match wins
    ("net_loss", r"خسائر|خسارة|صافي\s+الخسار|net\s+loss|losses|\bloss\b"),
    ("net_profit", r"صافي\s+(?:ال)?(?:أرباح|ربح)|أرباح|ربح|net\s+(?:profit|income)|profits?|earnings"),
    ("revenue", r"إيرادات|الإيرادات|revenues?|turnover"),
    ("sales", r"مبيعات|المبيعات|\bsales\b"),
]
SCALE = [(r"مليار|مليارات|billion|\bbn\b|\bb\b", 1000.0), (r"مليون|ملايين|million|\bmn\b|\bmln\b|\bm\b", 1.0),
         (r"ألف|آلاف|thousand|\bk\b", 0.001)]
CURRENCY = [("EGP", r"جنيه|جنيها|ج\.م|EGP|\bLE\b|L\.E"), ("USD", r"دولار|USD|US\$|\$"),
            ("SAR", r"ريال|SAR"), ("AED", r"درهم|AED"), ("EUR", r"يورو|EUR|€")]
UP = re.compile(r"ترتفع|ارتفاع|ارتفعت|تقفز|قفزة|قفز|نمو|تنمو|زيادة|تزيد|تتضاعف|تتخطى|تتجاوز|"
                r"rise|rises|rose|up\b|grow|grows|jump|surge|hike|higher|increase", re.I)
DOWN = re.compile(r"تتراجع|تراجع|انخفاض|تنخفض|هبوط|تهبط|fall|falls|fell|drop|decline|lower|down\b|shrink", re.I)
PERIODS = [
    ("Q1", r"الربع\s+الأول|first\s+quarter|\bQ1\b|1Q"), ("Q2", r"الربع\s+الثاني|second\s+quarter|\bQ2\b|2Q"),
    ("Q3", r"الربع\s+الثالث|third\s+quarter|\bQ3\b|3Q"), ("Q4", r"الربع\s+الرابع|fourth\s+quarter|\bQ4\b|4Q"),
    ("H1", r"النصف\s+الأول|first\s+half|\bH1\b|1H|6\s*أشهر|ستة\s+أشهر|six\s+months"),
    ("9M", r"تسعة\s+أشهر|9\s*أشهر|nine\s+months|\b9M\b"),
    ("FY", r"العام\s+المالي|السنة\s+المالية|عام\s+\d{4}|خلال\s+\d{4}|في\s+\d{4}|\bFY\b|full[-\s]year|in\s+\d{4}"),
]
_NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
AMOUNT = re.compile(_NUM + r"\s*(?:\S{0,3}\s*)?(" + "|".join(p for p, _ in SCALE) + r")", re.I)
LE_AMOUNT = re.compile(r"(?:LE|EGP|SAR|AED|US\$|\$)\s?" + _NUM + r"\s?(bn|b|mn|m|mln|million|billion)?\b", re.I)
PCT = re.compile(_NUM + r"\s*%|%\s*" + _NUM)
PRIOR = re.compile(r"(?:مقابل|مقارنة\s+ب|vs\.?|versus|compared\s+(?:with|to)|from)\s*(?:\S+\s+){0,2}?" + _NUM, re.I)
YEAR = re.compile(r"(20\d{2})(?:\s*/\s*(20\d{2}))?")
BASIS = [("consolidated", r"المجمعة|مجمعة|consolidated"), ("standalone", r"المستقلة|مستقلة|standalone")]


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _scale(word: str) -> float:
    for pat, mult in SCALE:
        if re.fullmatch(pat, word.strip(), re.I) or re.search(pat, word, re.I):
            return mult
    return 1.0


def _period_end(label: str, year: int):
    if not label or not year:
        return None
    ends = {"Q1": f"{year}-03-31", "Q2": f"{year}-06-30", "Q3": f"{year}-09-30", "Q4": f"{year}-12-31",
            "H1": f"{year}-06-30", "9M": f"{year}-09-30", "FY": f"{year}-12-31"}
    return ends.get(label)


def extract_figures(text: str, ticker: str = None, seen=None) -> list:
    """Rules backend for task 'figures'. Returns 0 or 1 dict per headline (the
    headline's lead number); never raises on odd text."""
    if not text:
        return []
    t = text.translate(_AR_DIGITS)
    t = re.sub(r"&nbsp;|‏|‎", " ", t)
    kind = "guidance" if GUIDANCE.search(t) else ("dividend" if DIVIDEND.search(t) else "result")
    metric = next((m for m, p in METRICS if re.search(p, t, re.I)), None)
    if metric is None:
        return []
    if metric == "sales" and ticker in REAL_ESTATE:
        metric = "presales"
    value, currency = None, None
    m = LE_AMOUNT.search(t)
    if m:
        value = _num(m.group(1)) * (_scale(m.group(2)) if m.group(2) else 1.0)
        currency = next((c for c, p in CURRENCY if re.search(p, m.group(0), re.I)), "EGP")
    else:
        for m in AMOUNT.finditer(t):
            # skip the number if it is the prior-period figure or a percentage
            if PRIOR.search(t[max(0, m.start() - 25):m.start() + 1]):
                continue
            value = _num(m.group(1)) * _scale(m.group(2))
            tail = t[m.end():m.end() + 15]
            currency = next((c for c, p in CURRENCY if re.search(p, tail, re.I)), None)
            break
    pct = None
    pm = PCT.search(t)
    if pm:
        pct = _num(pm.group(1) or pm.group(2))
    direction = 1 if UP.search(t) else (-1 if DOWN.search(t) else 0)
    if pct is not None and direction:
        pct = abs(pct) * direction
    if metric == "net_loss" and value is not None:
        value = -abs(value)
    period = next((lab for lab, p in PERIODS if re.search(p, t, re.I)), None)
    ym = YEAR.search(t)
    year = int(ym.group(2) or ym.group(1)) if ym else None
    split_year = bool(ym and ym.group(2))            # "2025/2026": a fiscal year ending June 2026
    if split_year and period is None:
        period = "FY"
    if year is None and period and seen is not None:      # undated period: the latest one before it was seen
        s = pd.Timestamp(seen)
        for y in (s.year, s.year - 1):
            pe = _period_end(period, y)
            if pe and pd.Timestamp(pe) < s:
                year = y
                break
    prior = None
    pr = PRIOR.search(t)
    if pr:
        prior = _num(pr.group(1))
        sc = AMOUNT.search(t[pr.start():pr.end() + 15])
        if prior is not None and sc:
            prior *= _scale(sc.group(2))
        if metric == "net_loss" and prior is not None:
            prior = -abs(prior)
    basis = next((b for b, p in BASIS if re.search(p, t)), None)
    if value is None and pct is None:
        return []
    conf = 0.9
    if currency not in (None, "EGP", "USD"):
        conf = 0.3               # SAR/AED: almost always a foreign namesake linked by mistake
    if period is None:
        conf -= 0.3
    if kind != "result":
        conf -= 0.2
    pend = f"{year}-06-30" if (split_year and period == "FY") else _period_end(period, year)
    return [{"kind": kind, "metric": metric, "value_m": value, "currency": currency,
             "pct_change": pct, "period": period, "fiscal_year": year,
             "period_end": pend, "prior_m": prior, "basis": basis,
             "confidence": round(max(conf, 0.0), 2)}]


BACKENDS = {"figures": {"rules": extract_figures}}


def extract(task: str, text: str, backend: str = "rules", log: bool = True, **ctx) -> list:
    """THE interface. Any backend -- rules today, your own encoder later, manual
    review -- returns rows of the same schema, and every call is logged so a
    backend swap is a config change whose effect can be measured."""
    t0 = time.perf_counter()
    rows = BACKENDS[task][backend](text, **ctx)
    if log:
        _log_call(task, backend, text, rows, (time.perf_counter() - t0) * 1000)
    return rows


_CALL_BUFFER = []


def _log_call(task, backend, text, rows, ms):
    _CALL_BUFFER.append((pd.Timestamp.now().isoformat(timespec="seconds"), task, backend,
                         METHOD if backend == "rules" else backend, len(text or ""), len(rows),
                         float(np.mean([r.get("confidence", 0) for r in rows])) if rows else None,
                         round(ms, 3)))


def _ensure(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS news_figures (
        news_id INTEGER, ticker TEXT, kind TEXT, metric TEXT, value_m REAL, currency TEXT,
        pct_change REAL, period TEXT, fiscal_year INTEGER, period_end TEXT, prior_m REAL,
        basis TEXT, confidence REAL, method TEXT, published_at TEXT, first_seen_at TEXT,
        extracted_at TEXT, PRIMARY KEY (news_id, ticker, method))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS model_calls (
        called_at TEXT, task TEXT, backend TEXT, version TEXT, text_len INTEGER,
        n_rows INTEGER, mean_conf REAL, latency_ms REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS label_queue (
        news_id INTEGER, ticker TEXT, task TEXT, split TEXT, priority REAL, status TEXT DEFAULT 'todo',
        label_json TEXT, labelled_at TEXT, PRIMARY KEY (news_id, ticker, task))""")
    from ops.agent_guard import ensure_tables
    ensure_tables(conn)


def run() -> dict:
    """Extract figures from every linked headline; store point-in-time rows."""
    from ops import agent_guard as ag
    with get_connection() as conn:
        items = pd.read_sql("SELECT i.id, l.ticker, i.headline, i.summary, i.published_at, i.first_seen_at, "
                            "l.method AS link FROM news_items i JOIN news_links l ON l.news_id = i.id", conn)
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    out = []
    for r in items.itertuples():
        for f in extract("figures", r.headline, ticker=r.ticker, seen=r.first_seen_at or r.published_at):
            if r.link in ("query", "roundup"):
                f["confidence"] = min(f["confidence"], 0.4)       # weak company link
            out.append((r.id, r.ticker, f["kind"], f["metric"], f["value_m"], f["currency"], f["pct_change"],
                        f["period"], f["fiscal_year"], f["period_end"], f["prior_m"], f["basis"],
                        f["confidence"], METHOD, r.published_at, r.first_seen_at, now))
    with get_connection() as raw:
        _ensure(raw)
        conn = ag.guarded(raw, WRITABLE, AGENT)
        conn.executemany("INSERT OR REPLACE INTO news_figures VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", out)
        conn.executemany("INSERT INTO model_calls VALUES (?,?,?,?,?,?,?,?)", _CALL_BUFFER)
        _CALL_BUFFER.clear()
    return {"headlines": len(items), "figures": len(out)}


def build_label_queue(n_test: int = 300, seed: int = 7) -> dict:
    """Split items once into a locked TEST set (never trained on) and a TRAIN pool
    ordered for active learning: lowest-confidence extractions first."""
    from ops import agent_guard as ag
    with get_connection() as conn:
        _ensure(conn)
        have = conn.execute("SELECT COUNT(*) FROM label_queue").fetchone()[0]
        f = pd.read_sql("SELECT news_id, ticker, confidence FROM news_figures", conn)
        items = pd.read_sql("SELECT DISTINCT l.news_id, l.ticker FROM news_links l", conn)
    if have:
        return {"already_built": have}
    items = items.merge(f, on=["news_id", "ticker"], how="left")
    items["priority"] = 1 - items["confidence"].fillna(0.5)       # unsure first
    rng = np.random.default_rng(seed)
    test_idx = set(rng.choice(len(items), size=min(n_test, len(items)), replace=False))
    rows = [(int(r.news_id), r.ticker, "figures", "test" if i in test_idx else "train", float(r.priority))
            for i, r in enumerate(items.itertuples())]
    with get_connection() as raw:
        conn = ag.guarded(raw, WRITABLE, AGENT)
        conn.executemany("INSERT OR IGNORE INTO label_queue (news_id, ticker, task, split, priority) "
                         "VALUES (?,?,?,?,?)", rows)
    return {"queued": len(rows), "test": len(test_idx)}


def validate() -> pd.DataFrame:
    """Extracted net profit vs the statement figure for the same company and
    period, and how many days before the statement's assumed availability the
    press had it."""
    with get_connection() as conn:
        nf = pd.read_sql("SELECT * FROM news_figures WHERE kind='result' AND metric IN ('net_profit','net_loss') "
                         "AND confidence >= 0.6 AND period IS NOT NULL AND fiscal_year IS NOT NULL "
                         "AND value_m IS NOT NULL AND (currency IS NULL OR currency='EGP')", conn)
        st = pd.read_sql("SELECT ticker, fiscal_period, period_end, available_from, value FROM fundamentals "
                         "WHERE statement='income' AND line_item IN ('Net Income','Net Income to Common')", conn)
    if nf.empty:
        return nf
    # map our period labels to statement periods: Q1-Q4 -> quarterly rows, FY -> FY rows
    st["label"] = st["fiscal_period"].str.extract(r"^(Q[1-4]|FY)")[0]
    st["year"] = st["fiscal_period"].str.extract(r"(\d{4})")[0].astype(float)
    m = nf.merge(st, left_on=["ticker", "period", "fiscal_year"], right_on=["ticker", "label", "year"],
                 how="inner", suffixes=("", "_st"))
    if m.empty:
        return m
    m["ratio"] = m["value_m"] / m["value"]
    m["agrees"] = (m["ratio"] - 1).abs() <= 0.05
    seen = pd.to_datetime(m["first_seen_at"].fillna(m["published_at"]), errors="coerce", utc=True).dt.tz_localize(None)
    pub = pd.to_datetime(m["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    m["press_days_after_period"] = (pub - pd.to_datetime(m["period_end"])).dt.days
    m["days_before_assumed"] = (pd.to_datetime(m["available_from"]) - pub).dt.days
    m["seen_lag_days"] = (seen - pub).dt.days
    return m.drop_duplicates(["news_id", "ticker"])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    if argv[:1] == ["label"]:
        from data.disclosure_labeller import interactive
        return interactive()
    r = run()
    q = build_label_queue()
    print(f"headlines {r['headlines']}, figures extracted {r['figures']}; label queue {q}")
    v = validate()
    if len(v):
        print(f"validated against statements: {len(v)} matches, {v['agrees'].mean():.0%} agree within 5%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
