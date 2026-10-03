"""
Daily sentiment scoring pipeline (spec 4.5). Scores recent articles with
the trained ArabicSentimentTransformer, aggregates to one score per stock
per day, and stores rolling sentiment features into daily_sentiment.

INTERPRETATION NOTE on the aggregation formula: the spec gives
  weighted_sentiment = sum(score_i * weight_i * credibility_i) / sum(weight_i * credibility_i)
without defining `weight_i` separately from `credibility_i` (which is
already the per-article source-credibility weight from spec 4.3). Here,
weight_i is implemented as a linear recency-decay weight within the
48-hour lookback window (a fresher article counts more) -- a reasonable,
explicit fill-in for an otherwise underspecified term, not a guess dressed
up as spec.

Runs in "no scored articles" mode gracefully when no trained checkpoint
exists yet (the real state until a Kaggle-trained best_sentiment_model.pt
is placed at config.SENTIMENT_MODEL_CHECKPOINT_PATH) -- logs a warning
and returns without crashing, rather than pretending to score anything.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch

import config
from data.database import get_connection
from models.sentiment.model import ArabicSentimentTransformer
from models.sentiment.tokenizer import ArabicFinancialTokenizer

logger = logging.getLogger("esis_quant.sentiment.pipeline")

NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2


def score_to_polarity(probs: np.ndarray) -> float:
    """Continuous polarity in [-1, +1]: P(POSITIVE) - P(NEGATIVE). Richer
    than collapsing straight to the discrete argmax class, since it uses
    the full predicted distribution (e.g. a confident-neutral prediction
    scores near 0, distinct from a genuinely uncertain 33/33/33 spread).
    """
    return float(probs[POSITIVE] - probs[NEGATIVE])


def aggregate_weighted_sentiment(scores, recency_weights, credibilities):
    """weighted_sentiment = sum(score_i * weight_i * credibility_i) / sum(weight_i * credibility_i)
    (spec 4.5). Returns None if the combined weight sums to <= 0 (e.g. no
    articles, or every article aged out past the lookback window).
    """
    scores = np.asarray(scores, dtype=float)
    combined_weight = np.asarray(recency_weights, dtype=float) * np.asarray(credibilities, dtype=float)
    denom = combined_weight.sum()
    if denom <= 0:
        return None
    return float((scores * combined_weight).sum() / denom)


class SentimentPipeline:
    def __init__(self, model=None, tokenizer=None, checkpoint_path: str = None):
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            self._try_load_checkpoint(checkpoint_path or config.SENTIMENT_MODEL_CHECKPOINT_PATH)

    def _try_load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning(
                "No trained sentiment checkpoint at %s -- generate_daily_scores() will skip "
                "scoring until a Kaggle-trained model is downloaded and placed there.", path,
            )
            self.model = None
            self.tokenizer = None
            return

        checkpoint = torch.load(path, map_location="cpu")
        tokenizer = ArabicFinancialTokenizer(vocab_size=checkpoint.get("vocab_size"))
        tokenizer.vocab = checkpoint["tokenizer_vocab"]
        tokenizer.merges = [tuple(pair) for pair in checkpoint["tokenizer_merges"]]
        tokenizer._merge_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}

        model = ArabicSentimentTransformer(vocab_size=len(tokenizer.vocab))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        self.model = model
        self.tokenizer = tokenizer
        logger.info("Loaded sentiment checkpoint from %s (vocab size %d)", path, len(tokenizer.vocab))

    def score_article(self, text: str) -> dict:
        """Returns {"sentiment_class", "confidence", "score", "magnitude"}
        or None if no trained model is available.
        """
        if self.model is None or self.tokenizer is None:
            return None
        enc = self.tokenizer.encode(text)
        input_ids = torch.tensor([enc["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([enc["attention_mask"]], dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)[0].numpy()

        predicted_class = int(probs.argmax())
        confidence = float(probs[predicted_class])
        score = score_to_polarity(probs)
        return {"sentiment_class": predicted_class, "confidence": confidence, "score": score, "magnitude": confidence}

    def _fetch_recent_articles(self, ticker: str, as_of: pd.Timestamp, lookback_hours: int):
        cutoff = as_of - pd.Timedelta(hours=lookback_hours)
        # Bound the scan in SQL with a LENIENT date-only lower bound (uses
        # idx_news_published_at) instead of reloading the entire table once
        # per ticker -- an N-times full scan that blows up as the archive
        # grows (PERF-BOTTLE-01). The bound is intentionally date-only and
        # lower-bound-only: stored timestamps may be ISO ("...T08:00") or
        # space-separated, and a tight string range comparison across those
        # formats can wrongly exclude same-day rows (the 'T' > ' ' ordering
        # trap). A date-only floor can only ever OVER-include, never exclude
        # a valid row; the precise [cutoff, as_of] window and ticker
        # membership are then applied in pandas on parsed datetimes.
        with get_connection() as conn:
            rows = pd.read_sql(
                "SELECT headline_arabic, body_arabic, published_at, source, tickers_mentioned "
                "FROM news_articles WHERE published_at IS NOT NULL AND published_at >= ?",
                conn, params=(cutoff.strftime("%Y-%m-%d"),),
            )
        if rows.empty:
            return rows

        rows["published_at"] = pd.to_datetime(rows["published_at"])
        mentions = rows["tickers_mentioned"].fillna("").str.split(",")
        mask = mentions.apply(lambda tickers: ticker in tickers) & (rows["published_at"] >= cutoff) & (rows["published_at"] <= as_of)
        return rows.loc[mask]

    def _aggregate_daily_score(self, articles: pd.DataFrame, as_of: pd.Timestamp, lookback_hours: int):
        scored = []
        for _, article in articles.iterrows():
            text = ((article["headline_arabic"] or "") + " " + (article["body_arabic"] or "")).strip()
            result = self.score_article(text)
            if result is None:
                continue
            age_hours = (as_of - article["published_at"]).total_seconds() / 3600.0
            recency_weight = max(0.0, 1.0 - age_hours / lookback_hours)
            credibility = config.SENTIMENT_SOURCE_CREDIBILITY.get(
                article["source"], config.SENTIMENT_SOURCE_CREDIBILITY["other"]
            )
            scored.append({**result, "recency_weight": recency_weight, "credibility": credibility})

        if not scored:
            return None, 0

        scored_df = pd.DataFrame(scored)
        weighted_sentiment = aggregate_weighted_sentiment(
            scored_df["score"], scored_df["recency_weight"], scored_df["credibility"],
        )
        return weighted_sentiment, len(scored_df)

    def _load_recent_daily_sentiment(self, ticker: str, before_date: pd.Timestamp, n_days: int):
        with get_connection() as conn:
            df = pd.read_sql(
                "SELECT date, sentiment_score FROM daily_sentiment WHERE ticker = ? AND date < ? "
                "ORDER BY date DESC LIMIT ?",
                conn, params=(ticker, before_date.strftime("%Y-%m-%d"), n_days),
            )
        return df["sentiment_score"].to_numpy()[::-1]  # oldest-first

    def generate_daily_scores(self, tickers=None, as_of: pd.Timestamp = None) -> pd.DataFrame:
        """For each ticker: fetch articles from the last
        config.SENTIMENT_NEWS_LOOKBACK_HOURS hours, score + aggregate to
        today's sentiment_score, compute the rolling features (5d avg,
        momentum, volume, surprise), and upsert into daily_sentiment.
        Returns the computed rows as a DataFrame (also stored to the DB).
        """
        tickers = tickers or [t["ticker"] for t in config.EGX_SEED_TICKERS]
        as_of = as_of or pd.Timestamp.now()
        lookback_hours = config.SENTIMENT_NEWS_LOOKBACK_HOURS
        window = config.SENTIMENT_ROLLING_WINDOW_DAYS

        if self.model is None:
            logger.warning("generate_daily_scores: no trained model loaded, skipping scoring entirely")
            return pd.DataFrame(columns=["ticker", "date", "sentiment_score", "sentiment_5d_avg",
                                          "sentiment_momentum", "sentiment_volume", "sentiment_surprise"])

        rows = []
        for ticker in tickers:
            articles = self._fetch_recent_articles(ticker, as_of, lookback_hours)
            sentiment_score, sentiment_volume = self._aggregate_daily_score(articles, as_of, lookback_hours)
            if sentiment_score is None:
                continue

            history = self._load_recent_daily_sentiment(ticker, as_of.normalize(), window)
            trailing = np.concatenate([history, [sentiment_score]])
            sentiment_5d_avg = float(trailing[-window:].mean())
            sentiment_momentum = sentiment_score - sentiment_5d_avg

            if len(history) >= 2:
                trailing_mean, trailing_std = history.mean(), history.std()
                sentiment_surprise = (sentiment_score - trailing_mean) / trailing_std if trailing_std > 0 else 0.0
            else:
                sentiment_surprise = 0.0

            rows.append({
                "ticker": ticker, "date": as_of.normalize(),
                "sentiment_score": sentiment_score, "sentiment_5d_avg": sentiment_5d_avg,
                "sentiment_momentum": sentiment_momentum, "sentiment_volume": sentiment_volume,
                "sentiment_surprise": float(sentiment_surprise),
            })

        result = pd.DataFrame(rows)
        if not result.empty:
            self._store(result)
        logger.info("generate_daily_scores: scored %d/%d tickers", len(result), len(tickers))
        return result

    def backfill_daily_sentiment(self, tickers=None) -> pd.DataFrame:
        """Production backfill: score EVERY stored article with a ticker
        mention at its own historical publish date and aggregate into a full
        daily_sentiment time series -- as opposed to generate_daily_scores,
        which only scores a live rolling 48h window ending "now" (right for a
        daily cron, useless for building history). This is what turns the
        accumulated news corpus into the multi-day series the Layer 4 charts
        and Layer 5 DML need.

        Aggregation per (ticker, publish-date): credibility-weighted mean of
        the article polarity scores (recency weight is 1 within a day). Then
        per ticker, over its dated series: 5-day rolling average, momentum
        (today - 5d avg), article volume, and surprise (z-score vs the
        trailing history). Upserts into daily_sentiment. Returns the rows.
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("backfill_daily_sentiment: no trained model loaded -- nothing to score")
            return pd.DataFrame()

        tickers = set(tickers or [t["ticker"] for t in config.EGX_SEED_TICKERS])
        window = config.SENTIMENT_ROLLING_WINDOW_DAYS

        with get_connection() as conn:
            articles = pd.read_sql(
                "SELECT headline_arabic, body_arabic, published_at, source, tickers_mentioned "
                "FROM news_articles WHERE published_at IS NOT NULL "
                "AND tickers_mentioned IS NOT NULL AND tickers_mentioned != ''",
                conn,
            )
        if articles.empty:
            logger.warning("backfill_daily_sentiment: no ticker-tagged articles to score")
            return pd.DataFrame()

        articles["published_at"] = pd.to_datetime(articles["published_at"])
        # score each article once, then explode to (ticker, date, score, credibility)
        exploded = []
        for _, a in articles.iterrows():
            text = ((a["headline_arabic"] or "") + " " + (a["body_arabic"] or "")).strip()
            result = self.score_article(text)
            if result is None:
                continue
            credibility = config.SENTIMENT_SOURCE_CREDIBILITY.get(
                a["source"], config.SENTIMENT_SOURCE_CREDIBILITY["other"]
            )
            date = a["published_at"].normalize()
            for tk in (a["tickers_mentioned"] or "").split(","):
                if tk in tickers:
                    exploded.append({"ticker": tk, "date": date, "score": result["score"], "credibility": credibility})

        if not exploded:
            logger.warning("backfill_daily_sentiment: no article mentioned a universe ticker")
            return pd.DataFrame()

        ex = pd.DataFrame(exploded)

        # credibility-weighted daily aggregate per (ticker, date)
        def _agg(g):
            denom = g["credibility"].sum()
            score = (g["score"] * g["credibility"]).sum() / denom if denom > 0 else g["score"].mean()
            return pd.Series({"sentiment_score": score, "sentiment_volume": len(g)})

        daily = ex.groupby(["ticker", "date"]).apply(_agg, include_groups=False).reset_index()

        # rolling features per ticker over its own dated series
        rows = []
        for ticker, g in daily.sort_values("date").groupby("ticker"):
            g = g.reset_index(drop=True)
            s = g["sentiment_score"]
            avg5 = s.rolling(window, min_periods=1).mean()
            for i in range(len(g)):
                hist = s.iloc[:i].to_numpy()  # strictly prior days
                surprise = 0.0
                if len(hist) >= 2 and hist.std() > 0:
                    surprise = float((s.iloc[i] - hist.mean()) / hist.std())
                rows.append({
                    "ticker": ticker, "date": g["date"].iloc[i],
                    "sentiment_score": float(s.iloc[i]), "sentiment_5d_avg": float(avg5.iloc[i]),
                    "sentiment_momentum": float(s.iloc[i] - avg5.iloc[i]),
                    "sentiment_volume": int(g["sentiment_volume"].iloc[i]),
                    "sentiment_surprise": surprise,
                })

        result = pd.DataFrame(rows)
        if not result.empty:
            self._store(result)
        logger.info("backfill_daily_sentiment: scored %d articles -> %d (ticker, date) sentiment rows across %d tickers",
                    len(articles), len(result), result["ticker"].nunique() if not result.empty else 0)
        return result

    def _store(self, df: pd.DataFrame):
        rows = [
            (r["ticker"], r["date"].strftime("%Y-%m-%d"), r["sentiment_score"], r["sentiment_5d_avg"],
             r["sentiment_momentum"], int(r["sentiment_volume"]), r["sentiment_surprise"])
            for _, r in df.iterrows()
        ]
        with get_connection() as conn:
            conn.executemany(
                """INSERT INTO daily_sentiment
                   (ticker, date, sentiment_score, sentiment_5d_avg, sentiment_momentum, sentiment_volume, sentiment_surprise)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ticker, date) DO UPDATE SET
                     sentiment_score=excluded.sentiment_score, sentiment_5d_avg=excluded.sentiment_5d_avg,
                     sentiment_momentum=excluded.sentiment_momentum, sentiment_volume=excluded.sentiment_volume,
                     sentiment_surprise=excluded.sentiment_surprise""",
                rows,
            )
