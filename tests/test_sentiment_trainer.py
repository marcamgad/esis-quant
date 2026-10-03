import numpy as np
import pandas as pd

from models.sentiment.trainer import (
    time_based_split, confusion_matrix, precision_recall_f1_per_class,
    macro_f1_score, expected_calibration_error, train, evaluate,
)
from models.sentiment.tokenizer import ArabicFinancialTokenizer
from models.sentiment.dataset import build_training_dataset, MANUAL_BOOTSTRAP_EXAMPLES


def test_confusion_matrix_matches_manual_tally():
    y_true = [0, 0, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0, 2]
    cm = confusion_matrix(y_true, y_pred, n_classes=3)
    expected = np.array([
        [1, 1, 0],  # true=0: predicted 0 once, 1 once
        [0, 2, 0],  # true=1: predicted 1 twice
        [1, 0, 2],  # true=2: predicted 0 once, 2 twice
    ])
    np.testing.assert_array_equal(cm, expected)


def test_precision_recall_f1_matches_hand_calc():
    # class 0: tp=1, fp=1 (from true=2), fn=1 (predicted as 1) -> precision=0.5, recall=0.5, f1=0.5
    cm = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [1, 0, 2],
    ])
    precision, recall, f1 = precision_recall_f1_per_class(cm)
    assert np.isclose(precision[0], 1 / 2)
    assert np.isclose(recall[0], 1 / 2)
    assert np.isclose(f1[0], 0.5)
    # class 1: tp=2, fp=1 (from true=0), fn=0 -> precision=2/3, recall=1.0
    assert np.isclose(precision[1], 2 / 3)
    assert np.isclose(recall[1], 1.0)
    # class 2: tp=2, fp=0, fn=1 -> precision=1.0, recall=2/3
    assert np.isclose(precision[2], 1.0)
    assert np.isclose(recall[2], 2 / 3)


def test_macro_f1_score_is_mean_of_per_class_f1():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]  # perfect predictions
    assert macro_f1_score(y_true, y_pred) == 1.0


def test_expected_calibration_error_zero_when_perfectly_calibrated():
    # 10 predictions all with confidence 0.7, and exactly 70% correct ->
    # that bin's |acc - conf| = 0, so ECE = 0.
    confidences = np.array([0.7] * 10)
    correctness = np.array([1.0] * 7 + [0.0] * 3)
    ece = expected_calibration_error(confidences, correctness, n_bins=10)
    assert np.isclose(ece, 0.0, atol=1e-9)


def test_expected_calibration_error_positive_when_overconfident():
    # All predictions claim 0.99 confidence but only 50% are correct.
    confidences = np.array([0.99] * 10)
    correctness = np.array([1.0] * 5 + [0.0] * 5)
    ece = expected_calibration_error(confidences, correctness, n_bins=10)
    assert ece > 0.4  # should reflect the ~0.49 gap


def test_time_based_split_never_puts_a_later_date_in_train_when_split():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "text": [f"t{i}" for i in range(20)],
        "label": [0] * 20,
        "weight": [1.0] * 20,
        "date": dates,
    })
    train_df, val_df = time_based_split(df, val_fraction=0.15)
    assert train_df["date"].max() < val_df["date"].min()
    assert len(train_df) + len(val_df) == 20


def test_time_based_split_keeps_undated_rows_in_train_only():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    dated = pd.DataFrame({"text": [f"t{i}" for i in range(20)], "label": [0] * 20, "weight": [1.0] * 20, "date": dates})
    undated = pd.DataFrame({"text": ["manual1", "manual2"], "label": [1, 2], "weight": [2.0, 2.0], "date": [pd.NaT, pd.NaT]})
    df = pd.concat([dated, undated], ignore_index=True)

    train_df, val_df = time_based_split(df, val_fraction=0.15)
    assert "manual1" in train_df["text"].values
    assert "manual2" in train_df["text"].values
    assert "manual1" not in val_df["text"].values
    assert "manual2" not in val_df["text"].values


def test_time_based_split_warns_and_skips_val_when_too_few_dated_rows():
    df = pd.DataFrame({
        "text": ["a", "b", "c"], "label": [0, 1, 2], "weight": [1.0, 1.0, 1.0],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })
    train_df, val_df = time_based_split(df, val_fraction=0.15)
    assert len(val_df) == 0
    assert len(train_df) == 3


def test_local_smoke_training_run_completes_and_metrics_are_well_formed(monkeypatch):
    # Not a claim of good sentiment performance (50 tiny bootstrap examples
    # can't teach a 6-layer Transformer anything generalizable) -- this
    # verifies the training loop mechanically works: loss is finite, macro
    # F1/ECE compute without crashing, and a checkpoint-worthy model comes
    # out the other end. Small architecture + 2 epochs to keep it fast.
    import config as cfg
    monkeypatch.setattr(cfg, "SENTIMENT_D_MODEL", 32)
    monkeypatch.setattr(cfg, "SENTIMENT_N_LAYERS", 2)
    monkeypatch.setattr(cfg, "SENTIMENT_D_FF", 64)
    monkeypatch.setattr(cfg, "SENTIMENT_MAX_SEQ_LEN", 32)
    monkeypatch.setattr(cfg, "SENTIMENT_N_HEADS", 4)

    df = pd.DataFrame(MANUAL_BOOTSTRAP_EXAMPLES)
    df["weight"] = 2.0
    df["date"] = pd.NaT  # matches real dataset.py behavior for manual examples

    tokenizer = ArabicFinancialTokenizer(vocab_size=300)
    tokenizer.train_bpe(df["text"].tolist())

    train_df, val_df = time_based_split(df, val_fraction=0.15)  # -> all in train, no dated rows

    trained_model, history = train(train_df, val_df, tokenizer, epochs=2, batch_size=8, device="cpu")
    assert len(history) == 2
    for epoch_stats in history:
        assert np.isfinite(epoch_stats["train_loss"])
        assert 0.0 <= epoch_stats["train_macro_f1"] <= 1.0

    eval_result = evaluate(trained_model, df, tokenizer, device="cpu", batch_size=8)
    assert 0.0 <= eval_result["macro_f1"] <= 1.0
    assert np.isfinite(eval_result["ece"])
    assert eval_result["confusion_matrix"].sum() == len(df)
