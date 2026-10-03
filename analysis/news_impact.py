"""
News impact assessment -- stage 3 of the news pipeline.

WHAT IT DOES. For every news item linked to an EGX ticker it records:
  event_type   what kind of news it is (earnings, dividend, capital increase...)
  direction    +1 favourable / 0 neutral or ambiguous / -1 unfavourable, FOR THAT
               COMPANY, after reading the company's own reported numbers
  strength     1-3, how much this kind of event usually matters
  confidence   high / medium / low, how much the call itself can be trusted
  rationale    one sentence that states the reason, citing the numbers used

WHY RULES AND NOT A LANGUAGE MODEL. The operator does not want paid model keys,
and a local model is weak at Egyptian business Arabic. Rules are also the better
first version for a reason that has nothing to do with cost: every call can be
traced to a keyword and a number, so when stage 4 finds a rule is wrong, it is
obvious which rule and why. A model's wrong call is just wrong.

WHY THE SAME HEADLINE CAN BE GOOD FOR ONE COMPANY AND BAD FOR ANOTHER. That is
the point of reading fundamentals, and the operator's explicit requirement:
  capital increase   profitable and growing -> funding growth       (favourable)
                     loss-making            -> funding losses, dilution (unfav.)
  new loan / bonds   cash-generative        -> routine financing     (neutral)
                     loss-making            -> more leverage on losses (unfav.)
  dividend           covered by profit      -> favourable
                     loss-making            -> unsustainable          (neutral)

POINT-IN-TIME, AGAIN. Fundamentals are read with `data.fundamentals.as_of(ticker,
published_at)`, i.e. only what had been published by the news date (period end +
120 days). Reading today's fundamentals to judge a March headline would let the
assessment "know" results that came out in May, and stage 4 would then score a
lookahead as skill.

WHAT THIS IS NOT. It is not yet evidence that the calls predict prices. That is
stage 4's job: forward returns after each call versus a random-date baseline, per
event type. Until a type passes that test, a "favourable" label is a documented
judgement, not a signal, and it must not feed the optimiser.
"""

import logging
import re

import numpy as np
import pandas as pd

from data.database import get_connection

logger = logging.getLogger("esis_quant.news_impact")

METHOD = "rules_v1"

# Revenue on stockanalysis EGX pages is published in EGP millions (COMI's 139,166
# is ~139bn). Used only to size a contract value against revenue, and the
# rationale says the ratio assumes it.
STATEMENT_UNIT_EGP = 1e6

# --------------------------------------------------------------------------
# Event taxonomy. (event_type, base_direction, strength, [patterns]).
# Order matters: earlier types win when a headline matches several, so
# earnings outrank revenue ("deeper into the red despite a 4x revenue surge").
# --------------------------------------------------------------------------

def _p(*alts):
    return re.compile("|".join(alts), re.I)


EVENTS = [
    ("earnings_down", -1, 2, _p(
        r"\b(profits?|earnings|net income)\b(?:[^.]|(?<=\d)\.(?=\d)){0,40}\b(fall|falls|fell|drop|drops|dropped|declin\w*|plunge\w*|slump\w*|tumble\w*|shrink\w*|lower)\b",
        r"\b(loss|losses)\b(?:[^.]|(?<=\d)\.(?=\d)){0,25}\b(widen\w*|deepen\w*|swell\w*|rise|rises|grow\w*)\b",
        r"\bswings? (deeper )?(in)?to (a )?(net )?loss\b", r"\binto the red\b", r"\bnet loss\b",
        r"تراجع\s*(صافي\s*)?(أرباح|ارباح)", r"انخفاض\s*(صافي\s*)?(أرباح|ارباح)",
        r"(تعمق|ارتفاع|زيادة)\s*(صافي\s*)?(خسائر|الخسائر)", r"صافي\s*خسار", r"تحول\w*\s*(إلى|الى)\s*(الخسار|خسار)")),
    ("earnings_up", +1, 2, _p(
        r"\b(profits?|earnings|net income)\b(?:[^.]|(?<=\d)\.(?=\d)){0,40}\b(rise|rises|rose|jump\w*|surge\w*|soar\w*|climb\w*|grow\w*|up|higher|doubl\w*|tripl\w*|record)\b",
        r"\bdoubles? (its )?(fy[\s\d/]*)?(net )?profit\b", r"\bswings? (back )?to (a )?profit\b", r"\bturns? profitable\b",
        r"(ارتفاع|نمو|قفز\w*|زيادة)\s*(صافي\s*)?(أرباح|ارباح)", r"تضاعف\s*(صافي\s*)?(أرباح|ارباح)",
        r"(أرباح|ارباح)\s*\S*\s*(ترتفع|تقفز|تنمو|تتضاعف)", r"تحول\w*\s*(إلى|الى)\s*(الربح|ربحية)")),
    ("index_drop", -1, 2, _p(
        r"\b(dropped|removed|excluded|deleted|exits?)\b(?:[^.]|(?<=\d)\.(?=\d)){0,20}\b(from )?(the )?(egx\s?\d+|index)\b",
        r"(استبعاد|خروج|حذف)\s*\S*\s*(من\s*)?(مؤشر|مؤشرات)")),
    ("index_add", +1, 2, _p(
        r"\b(added|joins?|joined|included|inclusion)\b(?:[^.]|(?<=\d)\.(?=\d)){0,20}\b(to |in(to)? )?(the )?(egx\s?\d+|index)\b",
        r"(انضمام|إدراج|ادراج|دخول)\s*\S*\s*(إلى|الى|في)?\s*(مؤشر|مؤشرات)")),
    # A trading halt for hitting the daily price limit is exchange routine, usually
    # after a sharp move in EITHER direction -- not a regulatory sanction. The first
    # rules called "trading halted on 11 stocks for exceeding price limits" a
    # legal/unfavourable event.
    ("price_limit_halt", 0, 1, _p(r"\bprice limits?\b", r"\bcircuit breaker\b", r"الحدود\s*السعرية",
                                  r"الحد\s*(الأقصى|السعري)")),
    # A major holder or insider adding shares. First rules misread "major
    # shareholder raises stake to 15.3%" as debt financing.
    ("stake_increase", +1, 1, _p(
        r"\b(raises?|increases?|boosts?|lifts?)\b(?:[^.]|(?<=\d)\.(?=\d)){0,20}\b(its |his |her )?(stake|holding|shareholding)\b",
        r"(يرفع|يزيد|ترفع|تزيد|رفع|زيادة)\s*(حصته|حصتها|حصة|ملكيته|ملكيتها)")),
    ("stake_decrease", -1, 1, _p(
        r"\b(cuts?|reduces?|lowers?|trims?)\b(?:[^.]|(?<=\d)\.(?=\d)){0,20}\b(its |his |her )?(stake|holding|shareholding)\b",
        r"(يخفض|يقلص|تخفض|تقلص|خفض|تقليص)\s*(حصته|حصتها|حصة|ملكيته|ملكيتها)")),
    ("legal_regulatory", -1, 2, _p(
        r"\b(fine[ds]?|penalt\w*|lawsuit|sued|investigation|probe|suspend\w*|suspension|halt\w*|violation|sanction\w*)\b",
        r"غرامة", r"إيقاف\s*(التداول|نشاط)", r"دعوى", r"تحقيق", r"مخالفات?", r"عقوبات?")),
    ("rating_down", -1, 2, _p(r"\bdowngrad\w*\b", r"خفض\s*(التصنيف|توصية)")),
    ("rating_up", +1, 2, _p(r"\bupgrad\w*\b", r"رفع\s*(التصنيف|توصية)")),
    ("fair_value", 0, 1, _p(r"\bfair value\b", r"\btarget price\b", r"\bprice target\b", r"القيمة\s*العادلة", r"السعر\s*المستهدف")),
    ("capital_increase", 0, 2, _p(
        r"\bcapital increase\b", r"\brights? issue\b", r"\bincrease (its )?(paid[- ]in |issued )?capital\b",
        r"زيادة\s*رأس\s*(المال|مال)", r"زيادة\s*رأسمال")),
    ("bonus_split", 0, 1, _p(r"\bbonus shares?\b", r"\bstock split\b", r"\bshare split\b",
                             r"أسهم\s*مجانية", r"تجزئة\s*(السهم|القيمة)")),
    ("buyback", +1, 1, _p(r"\bbuy-?back\b", r"\bshare repurchase\b", r"\btreasury shares\b", r"أسهم\s*خزينة")),
    ("dividend", +1, 1, _p(r"\bdividends?\b", r"\bcoupon\b", r"توزيع\w*\s*(أرباح|ارباح|نقدية)", r"كوبون")),
    ("divestment", 0, 1, _p(r"\bsells? (its |a |part of its )?(stake|shares|holding)\b", r"\bsale of (its )?(stake|shares)\b",
                            r"\bdivest\w*\b", r"\bexits? (its )?(stake|investment)\b", r"بيع\s*(حصة|حصتها|أسهم)", r"تخارج")),
    ("acquisition", 0, 1, _p(r"\bacqui\w+\b", r"\b(buys|takes|takes a) (a )?stake\b", r"\bmerger\b",
                             r"استحواذ", r"شراء\s*حصة", r"اندماج")),
    ("debt_financing", 0, 1, _p(r"\bloans?\b", r"\bbonds?\b", r"\bsukuk\b", r"\bfinancing (facility|deal)\b",
                                r"\bsecuriti[sz]ation\b", r"\bcredit facility\b",
                                r"قرض", r"تمويل", r"سندات", r"صكوك", r"توريق")),
    # "awarded" alone matched "awarded Padma Shri for Literature" -- it must be
    # awarded something commercial.
    ("contract_win", +1, 1, _p(r"\bwins?\b(?:[^.]|(?<=\d)\.(?=\d)){0,30}\b(contract|tender|bid|deal|project|order)\b",
                               r"\bawarded\b(?:[^.]|(?<=\d)\.(?=\d)){0,30}\b(contract|tender|project|deal|order)\b",
                               r"\bsigns?\b(?:[^.]|(?<=\d)\.(?=\d)){0,30}\b(contract|agreement|deal|mou|memorandum)\b",
                               r"تفوز", r"فوز\w*\s*(بمناقصة|بعقد|بمشروع)", r"توقيع\s*(عقد|اتفاقية|مذكرة)", r"مناقصة")),
    ("revenue_up", +1, 1, _p(
        r"\b(revenues?|sales|turnover)\b(?:[^.]|(?<=\d)\.(?=\d)){0,30}\b(rise|rises|rose|jump\w*|surge\w*|grow\w*|expand\w*|up|higher|doubl\w*)\b",
        r"(ارتفاع|نمو|زيادة)\s*(الإيرادات|إيرادات|المبيعات|مبيعات)")),
    ("revenue_down", -1, 1, _p(
        r"\b(revenues?|sales|turnover)\b(?:[^.]|(?<=\d)\.(?=\d)){0,30}\b(fall|falls|fell|drop\w*|declin\w*|shrink\w*|lower)\b",
        r"(تراجع|انخفاض)\s*(الإيرادات|إيرادات|المبيعات|مبيعات)")),
    ("expansion", +1, 1, _p(r"\bexpan\w+\b", r"\bnew (plant|factory|facility|project|branch)\b", r"\blaunch\w*\b",
                            r"\binaugurat\w*\b", r"توسع\w*", r"(مصنع|مشروع|فرع)\s*جديد", r"إطلاق", r"افتتاح")),
    ("management", 0, 1, _p(r"\bappoint\w*\b", r"\b(chairman|ceo|managing director|board)\b", r"\bresign\w*\b",
                            r"تعيين", r"رئيس\s*مجلس", r"استقالة", r"العضو\s*المنتدب")),
    ("ipo_listing", 0, 1, _p(r"\bipo\b", r"\blisting\b", r"\bgoes public\b", r"\bdebut\w*\b", r"طرح\s*(عام|أولي)", r"قيد\s*(أسهم|الشركة)")),
]

# Words that intensify an event. Applied to strength only, never to direction.
_INTENSE = _p(r"\bdoubl\w*\b", r"\btripl\w*\b", r"\b\d+x\b", r"\brecord\b", r"\bsurge\w*\b", r"\bplunge\w*\b",
              r"تضاعف", r"قياسي", r"قفز")

_NUM = r"(\d[\d,]*\.?\d*)"
_FAIR_VALUE = [
    re.compile(r"(?:fair value|target price|price target)[^\d]{0,40}" + _NUM, re.I),
    re.compile(r"(?:القيمة\s*العادلة|السعر\s*المستهدف)[^\d]{0,40}" + _NUM),
]
_MONEY = [
    (re.compile(r"(?:EGP|LE)\s*" + _NUM + r"\s*(bn|billion|mn|million)", re.I), "en"),
    (re.compile(_NUM + r"\s*(bn|billion|mn|million)\s*(?:EGP|LE|pounds)", re.I), "en"),
    (re.compile(_NUM + r"\s*(مليار|مليون)\s*جنيه"), "ar"),
]


def classify(text: str) -> list:
    """All (event_type, base_direction, strength) matched, in taxonomy order."""
    return [(name, d, s) for name, d, s, pat in EVENTS if pat.search(text)]


def _money_egp(text: str):
    for pat, _lang in _MONEY:
        m = pat.search(text)
        if m:
            v = float(m.group(1).replace(",", ""))
            unit = m.group(2).lower()
            mult = 1e9 if unit in ("bn", "billion", "مليار") else 1e6
            return v * mult
    return None


def _fair_value(text: str):
    for pat in _FAIR_VALUE:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------
# Point-in-time context
# --------------------------------------------------------------------------

class Context:
    """Fundamentals and price as they were known on a given date."""

    def __init__(self):
        with get_connection() as conn:
            self.f = pd.read_sql(
                "SELECT ticker, statement, fiscal_period, period_end, available_from, line_item, value "
                # rules_v1 reads fiscal years only; quarterly rows would count
                # quarters as loss "years".
                "FROM fundamentals WHERE available_from IS NOT NULL AND fiscal_period LIKE 'FY%'",
                conn)
            self.px = pd.read_sql("SELECT ticker, date, adjusted_close FROM daily_ohlcv "
                                  "WHERE adjusted_close IS NOT NULL", conn, parse_dates=["date"])
        self.f["available_from"] = pd.to_datetime(self.f["available_from"])
        self.px = self.px.sort_values("date")
        self._cache = {}

    def fundamentals(self, ticker: str, when) -> dict:
        when = pd.Timestamp(when)
        key = (ticker, when.to_period("M"))
        if key in self._cache:
            return self._cache[key]
        d = self.f[(self.f["ticker"] == ticker) & (self.f["available_from"] <= when)]
        out = {}
        if not d.empty:
            inc = d[d["statement"] == "income"]
            ends = sorted(inc["period_end"].dropna().unique(), reverse=True)
            if ends:
                cur = inc[inc["period_end"] == ends[0]].set_index("line_item")["value"]
                out["period_end"] = ends[0]
                for k, col in (("revenue", "Revenue"), ("net_income", "Net Income"),
                               ("rev_growth", "Revenue Growth"), ("op_income", "Operating Income")):
                    v = cur.get(col)
                    out[k] = None if v is None or pd.isna(v) else float(v)
                ni = (inc[inc["line_item"] == "Net Income"].dropna(subset=["value"])
                      .sort_values("period_end", ascending=False)["value"].tolist())
                streak = 0
                for v in ni:
                    if v < 0:
                        streak += 1
                    else:
                        break
                out["loss_streak"] = streak
            cf = d[(d["statement"] == "cashflow") & (d["line_item"] == "Operating Cash Flow")]
            if not cf.empty:
                out["op_cash_flow"] = float(cf.sort_values("period_end").iloc[-1]["value"])
            bal = d[d["statement"] == "balance"]
            if not bal.empty:
                last = bal[bal["period_end"] == bal["period_end"].max()].set_index("line_item")["value"]
                for k, col in (("cash", "Cash & Short-Term Investments"), ("total_assets", "Total Assets")):
                    v = last.get(col)
                    out[k] = None if v is None or pd.isna(v) else float(v)
        self._cache[key] = out
        return out

    def price(self, ticker: str, when):
        g = self.px[(self.px["ticker"] == ticker) & (self.px["date"] <= pd.Timestamp(when))]
        return float(g["adjusted_close"].iloc[-1]) if len(g) else None


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------

def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def assess(text: str, ticker: str, when, fund: dict, price=None, link_method: str = "text") -> dict:
    """One call for one (news item, ticker). Pure function: easy to test."""
    events = classify(text)
    base = {"event_type": "other", "direction": 0, "strength": 0, "confidence": "low",
            "rationale": "No recognised event in the headline; recorded for coverage only."}
    if not events:
        return base

    # Conflict check within earnings: both up and down wording -> ambiguous.
    types = [e[0] for e in events]
    if "earnings_up" in types and "earnings_down" in types:
        return {**base, "event_type": "earnings_mixed", "strength": 2,
                "rationale": "Headline contains both profit-up and profit-down wording; not called."}

    etype, direction, strength = events[0]
    ni, rev, growth = fund.get("net_income"), fund.get("revenue"), fund.get("rev_growth")
    streak, ocf = fund.get("loss_streak", 0), fund.get("op_cash_flow")
    have_f = ni is not None
    profitable = have_f and ni > 0
    loss_making = have_f and (ni < 0 or streak >= 1)
    fstr = (f"latest known year to {fund.get('period_end','?')}: net income {_fmt(ni)}, "
            f"revenue {_fmt(rev)}" + (f", growth {growth:+.0%}" if growth is not None else "")
            + (f", {streak} loss year(s) in a row" if streak else "")) if have_f else "no fundamentals known at that date"

    reason = ""
    if etype == "capital_increase":
        if loss_making:
            direction, reason = -1, f"Raising equity while loss-making usually funds losses and dilutes holders ({fstr})."
        elif profitable and (growth or 0) > 0:
            direction, reason = +1, f"Profitable and growing, so new capital more likely funds expansion ({fstr})."
        else:
            reason = f"Effect depends on use of proceeds, not stated ({fstr})."
    elif etype == "debt_financing":
        if loss_making:
            direction, reason = -1, f"New borrowing adds leverage to a loss-making company ({fstr}); debt levels not in data."
        elif profitable and (ocf or 0) > 0:
            reason = f"Financing is routine for a profitable, cash-generating company ({fstr})."
        else:
            reason = f"Effect unclear without debt levels ({fstr})."
    elif etype == "dividend":
        if loss_making:
            direction, reason = 0, f"A payout while loss-making is not covered by earnings ({fstr})."
        elif profitable:
            reason = f"Payout covered by profit ({fstr})."
        else:
            reason = "Dividend with no fundamentals to check coverage."
    elif etype == "earnings_up":
        reason = f"Reported profit improvement ({fstr})."
        if streak >= 1:
            strength, reason = 3, f"Profit improvement after {streak} loss year(s): a possible turnaround ({fstr})."
    elif etype == "earnings_down":
        reason = f"Reported profit deterioration ({fstr})."
        if streak >= 2:
            reason = f"Continues a {streak}-year loss run ({fstr})."
    elif etype == "revenue_up":
        if loss_making:
            direction, reason = 0, f"Revenue growth, but the company still loses money ({fstr})."
        else:
            reason = f"Revenue growth ({fstr})."
    elif etype == "revenue_down":
        reason = f"Revenue decline ({fstr})."
    elif etype in ("contract_win", "expansion"):
        val = _money_egp(text)
        if val and rev:
            ratio = val / (rev * STATEMENT_UNIT_EGP)
            if ratio >= 0.25:
                strength = 3
            reason = (f"Deal worth ~{val/1e6:,.0f}M EGP, about {ratio:.0%} of annual revenue "
                      f"(assuming statements in EGP millions). ")
        reason += f"Growth news; " + ("company is profitable" if profitable else
                                      "company is loss-making, so execution risk is higher" if loss_making else
                                      "no fundamentals to size it") + f" ({fstr})."
    elif etype == "fair_value":
        fv = _fair_value(text)
        if fv and price:
            up = fv / price - 1
            direction = +1 if up > 0.10 else -1 if up < -0.10 else 0
            reason = f"Published fair value {fv:,.2f} vs price {price:,.2f} on the day ({up:+.0%})."
        else:
            reason = "Fair-value headline without a parseable figure or price."
    elif etype in ("index_add", "index_drop"):
        reason = ("Index inclusion brings tracker buying." if direction > 0 else
                  "Index exclusion forces tracker selling.") + f" ({fstr})."
    elif etype == "legal_regulatory":
        reason = f"Legal or regulatory action ({fstr})."
    elif etype in ("stake_increase", "stake_decrease", "buyback"):
        who = {"stake_increase": "A major holder is adding shares, a sign of insider confidence",
               "stake_decrease": "A major holder is reducing its position",
               "buyback": "The company is buying back its own shares"}[etype]
        reason = f"{who} ({fstr})."
    elif etype in ("rating_up", "rating_down"):
        reason = ("Rating or recommendation raised" if direction > 0 else
                  "Rating or recommendation cut") + f" ({fstr})."
    elif etype == "price_limit_halt":
        reason = "Routine trading halt for hitting the daily price limit; direction depends on the move."
    else:
        reason = f"{etype.replace('_', ' ').capitalize()}: no clear direction on its own ({fstr})."

    if _INTENSE.search(text):
        strength = max(strength, 3) if direction != 0 else strength

    # Confidence: how much the CALL can be trusted, separate from how big the event is.
    score = 2
    if link_method == "query":
        # The item came back from a search for the company but does not name it.
        # Two of the first four wrong calls on held positions were exactly this
        # (a market-wide halt, an Indian literature award), so a search-only link
        # can never produce more than a low-confidence call.
        score = -1
    if not have_f and etype in ("capital_increase", "debt_financing", "dividend", "revenue_up"):
        score -= 1          # the call needed numbers it did not have
    if len({e[1] for e in events if e[1] != 0}) > 1:
        score -= 1          # the headline also matched an opposite-direction event
    if direction == 0:
        score = min(score, 1)
    confidence = {2: "high", 1: "medium"}.get(score, "low")

    return {"event_type": etype, "direction": int(direction), "strength": int(strength),
            "confidence": confidence, "rationale": reason.strip()}


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_assessments (
            news_id       INTEGER NOT NULL,
            ticker        TEXT NOT NULL,
            method        TEXT NOT NULL,
            published_at  TEXT,
            first_seen_at TEXT,
            link_method   TEXT,
            event_type    TEXT,
            direction     INTEGER,
            strength      INTEGER,
            confidence    TEXT,
            rationale     TEXT,
            price_at_news REAL,
            fund_period   TEXT,
            assessed_at   TEXT,
            PRIMARY KEY (news_id, ticker, method)
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_assess_ticker ON news_assessments (ticker, published_at)")


def run(since: str = "2020-01-01", include_roundups: bool = False) -> dict:
    """Assess every linked item published since `since`. Idempotent per method."""
    ctx = Context()
    with get_connection() as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT i.id, l.ticker, l.method, i.headline, i.summary, i.published_at, i.first_seen_at "
            "FROM news_links l JOIN news_items i ON i.id = l.news_id "
            "WHERE i.published_at >= ? AND (? OR l.method <> 'roundup')",
            (since, 1 if include_roundups else 0)).fetchall()
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    out = []
    for nid, t, lm, h, s, pub, seen in rows:
        when = pub or seen
        fund = ctx.fundamentals(t, when)
        price = ctx.price(t, when)
        a = assess(f"{h}. {s or ''}", t, when, fund, price, lm)
        out.append((nid, t, METHOD, pub, seen, lm, a["event_type"], a["direction"], a["strength"],
                    a["confidence"], a["rationale"], price, fund.get("period_end"), now))
    with get_connection() as conn:
        _ensure_table(conn)
        conn.executemany(
            "INSERT OR REPLACE INTO news_assessments (news_id, ticker, method, published_at, first_seen_at, "
            "link_method, event_type, direction, strength, confidence, rationale, price_at_news, "
            "fund_period, assessed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", out)
    df = pd.DataFrame(out, columns=["news_id", "ticker", "method", "published_at", "first_seen_at",
                                    "link_method", "event_type", "direction", "strength", "confidence",
                                    "rationale", "price", "fund_period", "assessed_at"])
    return {"assessed": len(df),
            "by_event": df["event_type"].value_counts().to_dict(),
            "by_direction": df["direction"].value_counts().to_dict(),
            "by_confidence": df["confidence"].value_counts().to_dict()}


def dedupe_events(df: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    """Collapse repeated coverage of one event into a single row.

    AMOC's H1 2026 results appeared as five near-identical articles on one day,
    and the first report summed them into a tone of +14 -- counting how many
    outlets ran the story, not what happened. Same ticker + same event type within
    `window_days` is one event; the most confident, strongest article represents
    it. Stage 4 must score events, not articles, for the same reason.
    """
    if df.empty:
        return df
    d = df.copy()
    d["_dt"] = pd.to_datetime(d["date"])
    if "cr" not in d.columns:
        d["cr"] = d["confidence"].map({"high": 0, "medium": 1, "low": 2})
    d = d.sort_values(["ticker", "event_type", "_dt"])
    keep = []
    for _, g in d.groupby(["ticker", "event_type"], sort=False):
        start, best = None, None
        for idx, r in g.iterrows():
            if start is None or (r["_dt"] - start).days > window_days:
                if best is not None:
                    keep.append(best)
                start, best = r["_dt"], idx
            else:
                cur = d.loc[best]
                if (r["cr"], -r["strength"]) < (cur["cr"], -cur["strength"]):
                    best = idx
        if best is not None:
            keep.append(best)
    return d.loc[keep].drop(columns=["_dt"])


def holdings_report(days: int = 120, per_ticker: int = 5) -> str:
    """Recent calls on the positions you hold, strongest and most confident first."""
    since = (pd.Timestamp.now() - pd.Timedelta(days=days)).isoformat()
    with get_connection() as conn:
        _ensure_table(conn)
        held = [r[0] for r in conn.execute("SELECT ticker FROM user_positions")]
        if not held:
            return "No holdings."
        df = pd.read_sql(
            "SELECT a.ticker, substr(a.published_at,1,10) date, a.event_type, a.direction, a.strength, "
            "a.confidence, a.link_method, a.rationale, i.headline FROM news_assessments a "
            "JOIN news_items i ON i.id = a.news_id WHERE a.method = ? AND a.published_at >= ? "
            "AND a.ticker IN (%s) AND a.event_type <> 'other'" % ",".join("?" * len(held)),
            conn, params=(METHOD, since, *held))
    arrow = {1: "▲ favourable", 0: "■ neutral", -1: "▼ unfavourable"}
    L = ["NEWS IMPACT — your holdings, last %d days (rules_v1, NOT yet validated by stage 4)" % days,
         "Low-confidence calls (search-only links) are hidden. Duplicate coverage of one event counts once.", ""]
    conf_rank = {"high": 0, "medium": 1, "low": 2}
    df = df[df["confidence"] != "low"]
    for t in held:
        d = df[df["ticker"] == t].copy()
        if d.empty:
            L.append(f"{t}: no classified news in the window")
            L.append("")
            continue
        d["cr"] = d["confidence"].map(conf_rank)
        ev = dedupe_events(d)
        net = int((ev["direction"] * ev["strength"]).sum())
        L.append(f"{t}  (net tone {net:+d} across {len(ev)} distinct event(s), {len(d)} articles)")
        d = ev.sort_values(["date", "cr", "strength"], ascending=[False, True, False]).head(per_ticker)
        for r in d.itertuples():
            L.append(f"  {r.date}  {arrow[r.direction]:<15} {r.event_type:<17} strength {r.strength}  "
                     f"conf {r.confidence}{' (search link)' if r.link_method == 'query' else ''}")
            L.append(f"      \"{r.headline[:100]}\"")
            L.append(f"      why: {r.rationale[:160]}")
        L.append("")
    return "\n".join(L)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    r = run()
    print(f"assessed {r['assessed']} (ticker, item) pairs")
    print("by event:", r["by_event"])
    print("by direction:", r["by_direction"], "| by confidence:", r["by_confidence"])
    print()
    print(holdings_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
