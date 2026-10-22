"""
Hand-labelling for the disclosure reader (active learning).

Shows the next item -- TRAIN items lowest-confidence first, so every label
teaches the most -- with the rules' extraction filled in. Enter accepts it;
otherwise type corrections as key=value pairs. TEST items are labelled too
but are never used for training: they are the bar a model must beat.

    py -3.11 -m data.disclosure_reader label
    keys: metric=net_profit|net_loss|revenue|sales|presales|none  value=10600
          period=Q1|Q2|Q3|Q4|H1|9M|FY  year=2026  kind=result|guidance|dividend|other
          s = skip, q = quit
"""

import json

import pandas as pd

from data.database import get_connection
from data.disclosure_reader import extract_figures


def _next(split: str = None):
    q = ("SELECT q.news_id, q.ticker, q.split, i.headline, i.first_seen_at FROM label_queue q "
         "JOIN news_items i ON i.id = q.news_id WHERE q.status = 'todo'")
    params = ()
    if split:
        q += " AND q.split = ?"
        params = (split,)
    with get_connection() as conn:
        return conn.execute(q + " ORDER BY q.split = 'test', q.priority DESC LIMIT 1", params).fetchone()


def interactive(split: str = None) -> int:
    done = 0
    while True:
        r = _next(split)
        if not r:
            print("queue empty")
            return 0
        guess = (extract_figures(r["headline"], r["ticker"], r["first_seen_at"]) or [{}])[0]
        print("\n" + "-" * 70)
        print(f"[{r['split']}] {r['ticker']}  {r['headline']}")
        print("rules say:", {k: guess.get(k) for k in ("kind", "metric", "value_m", "period", "fiscal_year")})
        ans = input("enter=accept | corrections k=v ... | s=skip | q=quit > ").strip()
        if ans == "q":
            print(f"labelled {done}")
            return 0
        status, label = "labelled", dict(guess)
        if ans == "s":
            status = "skipped"
        elif ans:
            for kv in ans.split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    k = {"value": "value_m", "year": "fiscal_year"}.get(k, k)
                    label[k] = None if v == "none" else (float(v) if k in ("value_m", "fiscal_year") else v)
        with get_connection() as conn:
            conn.execute("UPDATE label_queue SET status=?, label_json=?, labelled_at=? "
                         "WHERE news_id=? AND ticker=? AND task='figures'",
                         (status, json.dumps(label, ensure_ascii=False),
                          pd.Timestamp.now().isoformat(timespec="seconds"), r["news_id"], r["ticker"]))
        done += status == "labelled"


def score(backend=extract_figures) -> dict:
    """Accuracy of a backend on the labelled TEST split -- the pass bar."""
    with get_connection() as conn:
        rows = conn.execute("SELECT q.ticker, q.label_json, i.headline, i.first_seen_at FROM label_queue q "
                            "JOIN news_items i ON i.id=q.news_id WHERE q.split='test' AND q.status='labelled'"
                            ).fetchall()
    if not rows:
        return {"n": 0}
    hits = {"metric": 0, "value": 0, "period": 0}
    for r in rows:
        gold = json.loads(r["label_json"])
        pred = (backend(r["headline"], r["ticker"], r["first_seen_at"]) or [{}])[0]
        hits["metric"] += pred.get("metric") == gold.get("metric")
        gv, pv = gold.get("value_m"), pred.get("value_m")
        hits["value"] += (gv is None and pv is None) or (gv and pv and abs(pv / gv - 1) < 0.01)
        hits["period"] += pred.get("period") == gold.get("period")
    return {"n": len(rows), **{k: v / len(rows) for k, v in hits.items()}}
