"""
LLM escalation for findings a SQL query cannot resolve -- with a human carrying it.

WHAT GETS ESCALATED. Most research-agent findings are resolved by the data
itself. Two kinds need someone to read prose:
  * security identity   -- "KASABF is a share?" took a web search to learn it is
                           a closed-end fund; "why does KORA have no history?"
                           took another to learn it listed on 2026-06-11.
  * price disagreements -- ORAS reads 750.00 on one source and 71.05 on another;
                           SEIGA 0.95 vs 63.29. Almost certainly an unadjusted
                           split or a second share class, and only a filing or
                           announcement says which.

WHY NO API CALL. The operator does not want accounts or paid keys, so this
module never calls a model. It writes a prompt pack -- one self-contained prompt
per question -- to paste into any chat assistant (ChatGPT, Claude, Gemini), and
it ingests the answers pasted back. The human is the transport. That is slower
than an API, and it has one real advantage: nothing reaches the database without
a person having read it.

WHY EVERY PROMPT DEMANDS A SOURCE AND OFFERS "UNKNOWN". A chat model asked a
factual question about a thinly covered Egyptian security will produce a fluent
answer whether or not it knows one. So each prompt (1) requires a source URL per
fact, (2) makes UNKNOWN an explicit, acceptable answer, and (3) asks for a fixed
JSON shape. `ingest_answer` then REJECTS any fact without a URL. An unsourced
answer is not stored, however plausible it reads -- plausibility is the failure
mode, not the safeguard.

WHAT IS NEVER ESCALATED. "Which stock should I buy?" has no verifiable answer
and no source can settle it; selection has been tested six ways on this data
(four null, one significantly negative). This module only asks questions whose
answers can be checked.
"""

import datetime
import json
import logging
import os
import re

import pandas as pd

logger = logging.getLogger("esis_quant.escalation")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "reports")

VALID_TYPES = {"ordinary", "fund", "etf", "rights", "preferred", "spac",
               "fx_class", "UNKNOWN"}
VALID_CAUSES = {"split", "bonus_issue", "dual_currency_listing", "share_class",
                "data_error", "UNKNOWN"}

_URL = re.compile(r"^https?://\S+\.\S+", re.I)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------

def identity_prompt(ticker: str, name: str = None, isin: str = None) -> str:
    who = f"ticker {ticker}" + (f", ISIN {isin}" if isin else "") + (f', listed as "{name}"' if name else "")
    return f"""You are verifying reference data for an Egyptian Exchange (EGX) security: {who}.

Answer ONLY from sources you can cite. For every field give a source URL.
If you cannot verify a field from a source, set it to "UNKNOWN". UNKNOWN is a
correct and acceptable answer; a guess is not. Do not infer from the name.

Return exactly this JSON and nothing else:
{{
  "ticker": "{ticker}",
  "security_type": "ordinary | fund | etf | rights | preferred | spac | fx_class | UNKNOWN",
  "security_type_source": "https://...",
  "listing_date": "YYYY-MM-DD or UNKNOWN",
  "listing_date_source": "https://...",
  "pricing": "continuous | nav | UNKNOWN",
  "pricing_source": "https://...",
  "holds_basket_of_equities": "yes | no | UNKNOWN",
  "notes": "one sentence, or empty"
}}"""


def split_triage_prompt(ticker: str, price_a: float, price_b: float,
                        source_a: str = "stockanalysis.com", source_b: str = "Yahoo Finance",
                        date: str = None) -> str:
    when = f" on {date}" if date else ""
    ratio = price_a / price_b if price_b else float("nan")
    return f"""EGX ticker {ticker} shows {price_a:,.4f} EGP on {source_a} and {price_b:,.4f} on {source_b}{when}
(ratio {ratio:,.3f}). Explain the difference using ONLY sourced facts.

Which of these is the cause?
  split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN

Give a source URL for the cause. If it is a split or bonus issue, give the ratio
and the ex-date. If you cannot verify, answer UNKNOWN -- that is acceptable.

Return exactly this JSON and nothing else:
{{
  "ticker": "{ticker}",
  "cause": "split | bonus_issue | dual_currency_listing | share_class | data_error | UNKNOWN",
  "cause_source": "https://...",
  "ratio": "e.g. 10:1, or UNKNOWN",
  "ex_date": "YYYY-MM-DD or UNKNOWN",
  "which_source_is_correct": "{source_a} | {source_b} | both (different instruments) | UNKNOWN",
  "notes": "one sentence, or empty"
}}"""


# --------------------------------------------------------------------------
# Building the pack
# --------------------------------------------------------------------------

def _open_questions() -> dict:
    """Identity questions for held non-ordinary/unknown securities, and triage
    questions for every flagged price disagreement."""
    from data.database import get_connection
    q = {"identity": [], "split": [], "human_only": []}

    with get_connection() as conn:
        held = [r[0] for r in conn.execute("SELECT ticker FROM user_positions")]
        ref = {}
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(stocks)")}
        sel = ["ticker", "name_english"] + [c for c in ("isin", "security_type") if c in cols]
        for r in conn.execute(f"SELECT {', '.join(sel)} FROM stocks"):
            ref[r[0]] = dict(zip(sel, r))
        led_cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_trade_ledger)")}
        assumed = []
        if "price_source" in led_cols:
            assumed = conn.execute(
                "SELECT ticker, action, shares, price, traded_at FROM user_trade_ledger "
                "WHERE price_source = 'assumed'").fetchall()

    for t in held:
        r = ref.get(t, {})
        if not r.get("name_english") or (r.get("security_type") not in (None, "ordinary")):
            q["identity"].append({"ticker": t, "name": r.get("name_english"),
                                  "isin": r.get("isin")})

    try:
        from data.stockanalysis import fetch_egx_listing, cross_check_prices
        chk = cross_check_prices(fetch_egx_listing())
        for r in chk[chk["flag"]].itertuples():
            q["split"].append({"ticker": r.ticker, "price_a": float(r.price),
                               "price_b": float(r.yahoo), "date": str(r.date)[:10]})
    except Exception as exc:  # noqa: BLE001
        logger.warning("price cross-check unavailable for escalation: %s", exc)

    # Assumed fills are NOT an LLM question: only your broker statement knows.
    for r in assumed:
        q["human_only"].append(f"{r[1]} {r[2]:g} {r[0]} booked at {r[3]:.4f} "
                               f"on {str(r[4])[:10]} -- send the real fill price")
    return q


def build_pack(path: str = None) -> dict:
    """Write the prompt pack as Markdown. Returns counts and the file path."""
    q = _open_questions()
    today = datetime.date.today().isoformat()
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = path or os.path.join(REPORT_DIR, f"escalations_{today}.md")

    L = [f"# ESIS-QUANT escalation pack — {today}", "",
         "Paste each prompt into a chat assistant (ChatGPT, Claude, Gemini). Copy the",
         "JSON it returns into a file, then run:", "",
         "```", "py -3.11 -m ops.escalation ingest answers.json", "```", "",
         "Answers without a source URL are rejected on ingest. UNKNOWN is fine.", ""]

    n = 0
    if q["identity"]:
        L += ["## Security identity", ""]
        for item in q["identity"]:
            n += 1
            L += [f"### {n}. {item['ticker']}", "", "```", identity_prompt(**item), "```", ""]
    if q["split"]:
        L += ["## Price disagreements (split / share-class triage)", ""]
        for item in q["split"]:
            n += 1
            L += [f"### {n}. {item['ticker']}", "", "```",
                  split_triage_prompt(item["ticker"], item["price_a"], item["price_b"],
                                      date=item["date"]), "```", ""]
    if q["human_only"]:
        L += ["## Needs your broker statement (not a question for an AI)", ""]
        L += [f"- {s}" for s in q["human_only"]] + [""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return {"path": path, "identity": len(q["identity"]), "split": len(q["split"]),
            "human_only": len(q["human_only"])}


# --------------------------------------------------------------------------
# Ingesting answers
# --------------------------------------------------------------------------

def _sourced(ans: dict, field: str) -> bool:
    val = str(ans.get(field, "")).strip()
    if val in ("", "UNKNOWN"):
        return False
    return bool(_URL.match(str(ans.get(f"{field}_source", "")).strip()))


def validate_identity(ans: dict) -> dict:
    """Keep only fields that are both valid and sourced."""
    out, rejected = {}, []
    t = str(ans.get("security_type", "")).strip()
    if t in VALID_TYPES and t != "UNKNOWN":
        (out.__setitem__("security_type", t) if _sourced(ans, "security_type")
         else rejected.append("security_type: no source URL"))
    elif t and t != "UNKNOWN":
        rejected.append(f"security_type: invalid value {t!r}")
    d = str(ans.get("listing_date", "")).strip()
    if d and d != "UNKNOWN":
        try:
            iso = pd.Timestamp(d).date().isoformat()
            (out.__setitem__("listing_date", iso) if _sourced(ans, "listing_date")
             else rejected.append("listing_date: no source URL"))
        except Exception:  # noqa: BLE001
            rejected.append(f"listing_date: unparseable {d!r}")
    return {"accepted": out, "rejected": rejected}


def validate_split(ans: dict) -> dict:
    c = str(ans.get("cause", "")).strip()
    if c not in VALID_CAUSES:
        return {"accepted": {}, "rejected": [f"cause: invalid value {c!r}"]}
    if c == "UNKNOWN":
        return {"accepted": {}, "rejected": ["cause: UNKNOWN (nothing to record)"]}
    if not _sourced(ans, "cause"):
        return {"accepted": {}, "rejected": ["cause: no source URL"]}
    acc = {"cause": c, "source": ans["cause_source"].strip()}
    for k in ("ratio", "ex_date", "which_source_is_correct"):
        v = str(ans.get(k, "")).strip()
        if v and v != "UNKNOWN":
            acc[k] = v
    return {"accepted": acc, "rejected": []}


def ingest_answer(ans: dict) -> dict:
    """Validate one pasted answer and record what survives.

    Writes go through the research agent's guarded connection, so escalation can
    only touch `stocks` and `research_findings` -- never a price, mark, position
    or order, whatever an answer claims.
    """
    from data.database import get_connection
    from ops.research_agent import _GuardedConnection, _ensure_findings_table, _fingerprint

    ticker = str(ans.get("ticker", "")).strip()
    if not ticker:
        return {"ok": False, "reason": "answer has no ticker"}
    kind = "split" if "cause" in ans else "identity"
    v = validate_split(ans) if kind == "split" else validate_identity(ans)
    now = pd.Timestamp.now().isoformat(timespec="seconds")

    with get_connection() as raw:
        conn = _GuardedConnection(raw)
        _ensure_findings_table(conn)
        if kind == "identity" and v["accepted"]:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(stocks)")}
            if "listing_date" not in cols:
                conn.execute("ALTER TABLE stocks ADD COLUMN listing_date TEXT")
            sets = ", ".join(f"{k} = ?" for k in v["accepted"])
            conn.execute(f"UPDATE stocks SET {sets} WHERE ticker = ?",
                         (*v["accepted"].values(), ticker))
        if v["accepted"]:
            summary = (f"{ticker}: escalation answer recorded -- "
                       + ", ".join(f"{k}={val}" for k, val in v["accepted"].items()))
            conn.execute(
                "INSERT INTO research_findings (first_seen, last_seen, fingerprint, "
                "severity, kind, subject, summary, detail, needs_human) "
                "VALUES (?,?,?,?,?,?,?,?,0) ON CONFLICT(fingerprint) DO UPDATE SET "
                "last_seen=excluded.last_seen, summary=excluded.summary, detail=excluded.detail",
                (now, now, _fingerprint(f"escalation_{kind}", ticker), "info",
                 f"escalation_{kind}", ticker, summary, json.dumps(ans)[:1900]))
    return {"ok": bool(v["accepted"]), "ticker": ticker, "kind": kind, **v}


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["ingest"] and len(argv) == 2:
        with open(argv[1], encoding="utf-8") as fh:
            data = json.load(fh)
        for ans in (data if isinstance(data, list) else [data]):
            r = ingest_answer(ans)
            print(f"{r.get('ticker','?'):8s} {r.get('kind','?'):9s} "
                  f"accepted={r.get('accepted', {})} rejected={r.get('rejected', [])}")
        return 0
    r = build_pack()
    print(f"escalation pack: {r['path']}")
    print(f"  identity questions: {r['identity']} | split triage: {r['split']} | "
          f"need your broker: {r['human_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
