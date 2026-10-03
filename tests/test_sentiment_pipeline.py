import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from models.sentiment.pipeline import SentimentPipeline, score_to_polarity, aggregate_weighted_sentiment
from models.sentiment.dataset import MANUAL_BOOTSTRAP_EXAMPLES
from models.sentiment.tokenizer import ArabicFinancialTokenizer
from models.sentiment.trainer import train, time_based_split


def test_score_to_polarity_positive_negative_neutral():
    assert score_to_polarity(np.array([0.0, 0.0, 1.0])) == 1.0  # all-positive
    assert score_to_polarity(np.array([1.0, 0.0, 0.0])) == -1.0  # all-negative
    assert score_to_polarity(np.array([0.0, 1.0, 0.0])) == 0.0  # all-neutral
    assert np.isclose(score_to_polarity(np.array([0.2, 0.3, 0.5])), 0.3)


def test_aggregate_weighted_sentiment_matches_manual_calc():
    scores = [1.0, -1.0, 0.5]
    recency_weights = [1.0, 0.5, 1.0]
    credibilities = [0.85, 0.80, 1.0]
    result = aggregate_weighted_sentiment(scores, recency_weights, credibilities)

    combined = [1.0 * 0.85, 0.5 * 0.80, 1.0 * 1.0]
    expected = sum(s * w for s, w in zip(scores, combined)) / sum(combined)
    assert np.isclose(result, expected)


def test_aggregate_weighted_sentiment_none_when_zero_total_weight():
    assert aggregate_weighted_sentiment([1.0], [0.0], [0.85]) is None
    assert aggregate_weighted_sentiment([], [], []) is None


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_esis.db"))
    database.init_db()
    return config.DB_PATH


def test_pipeline_with_no_checkpoint_skips_scoring_gracefully(temp_db, tmp_path):
    missing_path = str(tmp_path / "does_not_exist.pt")
    pipeline = SentimentPipeline(checkpoint_path=missing_path)
    assert pipeline.model is None
    assert pipeline.tokenizer is None

    result = pipeline.generate_daily_scores(tickers=["AAA"])
    assert result.empty  # no crash, just nothing scored


def _insert_article(headline, published_at, tickers_mentioned, source="almalnews"):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO news_articles (source, headline_arabic, body_arabic, url, published_at, tickers_mentioned)
               VALUES (?, ?, '', ?, ?, ?)""",
            (source, headline, f"https://example.com/{published_at}-{tickers_mentioned}", published_at, tickers_mentioned),
        )


@pytest.fixture
def tiny_trained_pipeline(monkeypatch):
    """A real (not mocked) tiny trained model + tokenizer, built the same
    way tests/test_sentiment_trainer.py's smoke test does -- gives
    generate_daily_scores() a genuine model to run through, without
    depending on a real Kaggle checkpoint existing.
    """
    monkeypatch.setattr(config, "SENTIMENT_D_MODEL", 32)
    monkeypatch.setattr(config, "SENTIMENT_N_LAYERS", 2)
    monkeypatch.setattr(config, "SENTIMENT_D_FF", 64)
    monkeypatch.setattr(config, "SENTIMENT_MAX_SEQ_LEN", 32)
    monkeypatch.setattr(config, "SENTIMENT_N_HEADS", 4)

    df = pd.DataFrame(MANUAL_BOOTSTRAP_EXAMPLES)
    df["weight"] = 2.0
    df["date"] = pd.NaT
    tokenizer = ArabicFinancialTokenizer(vocab_size=300)
    tokenizer.train_bpe(df["text"].tolist())
    train_df, val_df = time_based_split(df, val_fraction=0.15)
    model, _ = train(train_df, val_df, tokenizer, epochs=1, batch_size=8, device="cpu")
    model.eval()
    return SentimentPipeline(model=model, tokenizer=tokenizer)


def test_generate_daily_scores_end_to_end_with_real_tiny_model(temp_db, tiny_trained_pipeline):
    as_of = pd.Timestamp("2024-06-15 09:00:00")
    _insert_article("خبر إيجابي عن الشركة", (as_of - pd.Timedelta(hours=2)).isoformat(), "AAA")
    _insert_article("خبر آخر عن الشركة", (as_of - pd.Timedelta(hours=10)).isoformat(), "AAA")
    _insert_article("خبر قديم جدا", (as_of - pd.Timedelta(hours=60)).isoformat(), "AAA")  # outside 48h window
    _insert_article("خبر عن شركة أخرى", (as_of - pd.Timedelta(hours=1)).isoformat(), "BBB")  # different ticker

    result = tiny_trained_pipeline.generate_daily_scores(tickers=["AAA", "BBB"], as_of=as_of)

    aaa_row = result[result["ticker"] == "AAA"]
    assert len(aaa_row) == 1
    assert aaa_row.iloc[0]["sentiment_volume"] == 2  # only the 2 within-window AAA articles
    assert -1.0 <= aaa_row.iloc[0]["sentiment_score"] <= 1.0

    bbb_row = result[result["ticker"] == "BBB"]
    assert len(bbb_row) == 1
    assert bbb_row.iloc[0]["sentiment_volume"] == 1

    with get_connection() as conn:
        stored = pd.read_sql("SELECT * FROM daily_sentiment", conn)
    assert len(stored) == 2  # actually stored, not just computed in memory


def test_generate_daily_scores_computes_rolling_features_from_history(temp_db, tiny_trained_pipeline):
    # Seed 4 days of prior daily_sentiment history for AAA, then generate
    # today's score and check the rolling 5d avg / momentum / surprise use
    # that real history rather than being computed in a vacuum.
    with get_connection() as conn:
        for i, score in enumerate([0.1, 0.2, -0.1, 0.3]):
            date = (pd.Timestamp("2024-06-10") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            conn.execute(
                "INSERT INTO daily_sentiment (ticker, date, sentiment_score, sentiment_5d_avg, sentiment_momentum, "
                "sentiment_volume, sentiment_surprise) VALUES ('AAA', ?, ?, 0, 0, 1, 0)",
                (date, score),
            )

    as_of = pd.Timestamp("2024-06-14 09:00:00")
    _insert_article("خبر عن الشركة اليوم", (as_of - pd.Timedelta(hours=1)).isoformat(), "AAA")

    result = tiny_trained_pipeline.generate_daily_scores(tickers=["AAA"], as_of=as_of)
    row = result.iloc[0]

    history = np.array([0.1, 0.2, -0.1, 0.3])
    expected_5d_avg = np.concatenate([history, [row["sentiment_score"]]])[-5:].mean()
    assert np.isclose(row["sentiment_5d_avg"], expected_5d_avg)
    assert np.isclose(row["sentiment_momentum"], row["sentiment_score"] - expected_5d_avg)

    expected_surprise = (row["sentiment_score"] - history.mean()) / history.std()
    assert np.isclose(row["sentiment_surprise"], expected_surprise)


def test_backfill_daily_sentiment_scores_all_history(temp_db, tiny_trained_pipeline):
    # Articles spread across several historical dates (well outside any live
    # 48h window) with ticker mentions -> backfill must score them all at
    # their own publish dates and produce a full per-(ticker,date) series.
    dates = ["2024-01-05", "2024-01-08", "2024-01-09", "2024-02-01"]
    for i, d in enumerate(dates):
        _insert_article(f"خبر عن الشركة رقم {i} ارباح ونمو", f"{d}T10:00:00", "AAA")
    _insert_article("خبر اخر عن شركة ثانية", "2024-01-08T12:00:00", "BBB")

    result = tiny_trained_pipeline.backfill_daily_sentiment(tickers=["AAA", "BBB"])

    # AAA has 4 dated rows, BBB has 1 -> 5 total, all persisted
    assert len(result) == 5
    assert set(result["ticker"]) == {"AAA", "BBB"}
    with get_connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM daily_sentiment").fetchone()[0]
    assert n == 5
    # rolling 5d avg equals the running mean of AAA's own dated scores
    aaa = result[result["ticker"] == "AAA"].sort_values("date").reset_index(drop=True)
    assert np.isclose(aaa["sentiment_5d_avg"].iloc[0], aaa["sentiment_score"].iloc[0])
    # first date has no prior history -> surprise defined as 0
    assert aaa["sentiment_surprise"].iloc[0] == 0.0


def test_backfill_daily_sentiment_no_model_is_noop(temp_db, monkeypatch):
    _insert_article("خبر عن الشركة", "2024-01-05T10:00:00", "AAA")
    monkeypatch.setattr(config, "SENTIMENT_MODEL_CHECKPOINT_PATH", str(temp_db) + ".missing")
    pipeline = SentimentPipeline()
    assert pipeline.model is None
    assert pipeline.backfill_daily_sentiment(tickers=["AAA"]).empty
