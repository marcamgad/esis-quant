import numpy as np
import pandas as pd
import pytest

import config
from data import database
from data.database import get_connection
from models.sentiment.dataset import (
    label_from_returns, POSITIVE, NEGATIVE, NEUTRAL,
    compute_class_weights, MANUAL_BOOTSTRAP_EXAMPLES,
    build_distant_supervision_dataset, build_training_dataset, export_training_csv,
)


def test_label_from_returns_positive_when_either_day_exceeds_threshold():
    assert label_from_returns(0.02, 0.0) == POSITIVE  # R_T alone triggers it
    assert label_from_returns(0.0, 0.02) == POSITIVE  # R_T+1 alone triggers it


def test_label_from_returns_negative_when_either_day_below_negative_threshold():
    assert label_from_returns(-0.02, 0.0) == NEGATIVE
    assert label_from_returns(0.0, -0.02) == NEGATIVE


def test_label_from_returns_neutral_when_both_within_band():
    assert label_from_returns(0.005, -0.005) == NEUTRAL


def test_label_from_returns_positive_takes_precedence_check_boundary():
    # Exactly at the threshold should NOT trigger (strict inequality per spec).
    assert label_from_returns(0.015, 0.0) == NEUTRAL
    assert label_from_returns(0.0150001, 0.0) == POSITIVE


def test_manual_bootstrap_examples_has_50_entries_all_three_classes_present():
    assert len(MANUAL_BOOTSTRAP_EXAMPLES) == 50
    labels = [e["label"] for e in MANUAL_BOOTSTRAP_EXAMPLES]
    assert POSITIVE in labels and NEGATIVE in labels and NEUTRAL in labels
    assert all(isinstance(e["text"], str) and len(e["text"]) > 0 for e in MANUAL_BOOTSTRAP_EXAMPLES)


def test_compute_class_weights_matches_manual_formula():
    labels = np.array([0, 0, 0, 1, 2, 2])  # counts: 0->3, 1->1, 2->2
    weights = compute_class_weights(labels, n_classes=3)
    n_total = 6
    expected = np.array([n_total / (3 * 3), n_total / (3 * 1), n_total / (3 * 2)])
    np.testing.assert_allclose(weights, expected)


def test_compute_class_weights_handles_absent_class_without_division_by_zero():
    labels = np.array([0, 0, 1])  # class 2 absent
    weights = compute_class_weights(labels, n_classes=3)
    assert np.isfinite(weights).all()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test_esis.db"))
    database.init_db()
    return config.DB_PATH


def _insert_article(headline, published_at, tickers_mentioned, source="almalnews"):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO news_articles (source, headline_arabic, body_arabic, url, published_at, tickers_mentioned)
               VALUES (?, ?, '', ?, ?, ?)""",
            (source, headline, f"https://example.com/{published_at}", published_at, tickers_mentioned),
        )


def _insert_price(ticker, date, close):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO daily_ohlcv (ticker, date, open, high, low, close, volume, value_traded_egp, adjusted_close)
               VALUES (?, ?, ?, ?, ?, ?, 1000, 1000, ?)""",
            (ticker, date, close, close, close, close, close),
        )


def test_build_distant_supervision_dataset_labels_positive_from_real_return(temp_db):
    # T-1 close=100, T close=103 (+3% > 1.5% threshold) -> POSITIVE
    _insert_price("AAA", "2024-01-01", 100.0)
    _insert_price("AAA", "2024-01-02", 103.0)
    _insert_price("AAA", "2024-01-03", 103.5)
    _insert_article("خبر عن الشركة", "2024-01-02T10:00:00", "AAA")

    result = build_distant_supervision_dataset()
    assert len(result) == 1
    assert result.iloc[0]["label"] == POSITIVE
    assert result.iloc[0]["ticker"] == "AAA"
    assert result.iloc[0]["weight"] == config.SENTIMENT_SOURCE_CREDIBILITY["almalnews"]


def test_build_distant_supervision_dataset_skips_articles_missing_t_plus_1(temp_db):
    # Only T-1 and T exist -- no T+1 yet (e.g. article published today).
    _insert_price("AAA", "2024-01-01", 100.0)
    _insert_price("AAA", "2024-01-02", 103.0)
    _insert_article("خبر عن الشركة", "2024-01-02T10:00:00", "AAA")

    result = build_distant_supervision_dataset()
    assert len(result) == 0


def test_build_distant_supervision_dataset_empty_when_no_tagged_articles(temp_db):
    _insert_article("خبر عام", "2024-01-02T10:00:00", "")  # no ticker mentions
    result = build_distant_supervision_dataset()
    assert result.empty


def test_build_training_dataset_combines_distant_and_manual(temp_db):
    _insert_price("AAA", "2024-01-01", 100.0)
    _insert_price("AAA", "2024-01-02", 103.0)
    _insert_price("AAA", "2024-01-03", 103.5)
    _insert_article("خبر عن الشركة", "2024-01-02T10:00:00", "AAA")

    combined = build_training_dataset()
    assert len(combined) == 1 + len(MANUAL_BOOTSTRAP_EXAMPLES)
    manual_rows = combined[combined["source"] == "manual_synthetic"]
    assert (manual_rows["weight"] == config.SENTIMENT_MANUAL_EXAMPLE_WEIGHT).all()


def test_export_training_csv_roundtrips_all_columns(tmp_path):
    df = pd.DataFrame(MANUAL_BOOTSTRAP_EXAMPLES)
    df["weight"] = 2.0
    df["source"] = "manual_synthetic"
    df["date"] = pd.NaT
    df["ticker"] = None

    path = str(tmp_path / "export" / "sentiment_training_data.csv")
    export_training_csv(df, path)

    reloaded = pd.read_csv(path, encoding="utf-8-sig")
    assert len(reloaded) == len(df)
    assert set(["text", "label", "weight", "date", "ticker", "source"]).issubset(reloaded.columns)
    assert reloaded["text"].iloc[0] == df["text"].iloc[0]
