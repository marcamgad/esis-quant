"""
Daily news brief (CONTEXT ONLY -- explicitly not a trading signal).

WHY CONTEXT, NOT SIGNAL. The DB has real Arabic EGX news with per-article
sentiment, but the sentiment model scores F1 ~0.33 on real news (barely above
chance) and cross-sectional selection signals already measured NEGATIVE in the
backtest. Feeding this into the live book would lower reliability, not raise it.
So the brief surfaces recent headlines for the names you actually hold, as
reading material for YOU -- the human -- to stay informed. The system's book is
unchanged by it.

Matches articles whose tickers_mentioned contains one of the book's tickers,
newest first. Freshness is reported honestly (the scrape may be days stale).
"""

import pandas as pd

from data.database import get_connection


def recent_news(tickers, days: int = 21, limit: int = 8) -> pd.DataFrame:
    """Recent articles mentioning any of `tickers`, newest first."""
    if not tickers:
        return pd.DataFrame()
    with get_connection() as conn:
        like = " OR ".join(["tickers_mentioned LIKE ?"] * len(tickers))
        params = [f"%{t}%" for t in tickers]
        df = pd.read_sql(
            f"SELECT published_at, source, headline_arabic, tickers_mentioned, "
            f"sentiment_score FROM news_articles WHERE ({like}) "
            f"ORDER BY published_at DESC LIMIT ?", conn, params=params + [limit],
            parse_dates=["published_at"])
    return df


def news_freshness() -> dict:
    with get_connection() as conn:
        row = conn.execute("SELECT MAX(published_at), COUNT(*) FROM news_articles").fetchone()
    latest = pd.to_datetime(row[0]) if row and row[0] else None
    stale_days = (pd.Timestamp.now() - latest).days if latest is not None else None
    return {"latest": latest, "total": row[1] if row else 0, "stale_days": stale_days}


def format_news_brief(tickers, limit: int = 6) -> str:
    fresh = news_freshness()
    df = recent_news(tickers, limit=limit)
    lines = []
    if fresh["latest"] is None:
        return "    no news in the database yet."
    lines.append(f"    latest scrape: {fresh['latest'].date()} "
                 f"({fresh['stale_days']} days ago, {fresh['total']} articles total)")
    if df.empty:
        lines.append("    no recent headlines mention your current holdings.")
    else:
        for r in df.itertuples():
            tags = str(r.tickers_mentioned or "").replace(" ", "")
            s = r.sentiment_score
            senti = f"{s:+.2f}" if isinstance(s, (int, float)) and s == s else "n/a"
            head = str(r.headline_arabic or "")[:70]
            lines.append(f"      {pd.Timestamp(r.published_at).date()} [{tags[:20]}] (s={senti}) {head}")
    lines.append("    CONTEXT ONLY -- the sentiment model is too weak (F1 0.33) to trade on.")
    return "\n".join(lines)
