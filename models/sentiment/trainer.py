"""
Training loop, evaluation metrics, and time-based data splitting for the
Arabic sentiment Transformer (Layer 4). Every metric (confusion matrix,
precision/recall/F1, ECE) is implemented manually with NumPy -- no
sklearn.metrics.

This module is deliberately import-safe on CPU with a handful of
examples (so it can be exercised as a correctness smoke test locally) and
is also the exact training logic re-used, cell-for-cell, in the Kaggle
notebook this project ships for real GPU training once enough real
distant-supervision data accumulates (see models/sentiment/dataset.py
docstring on current data volume).
"""

import logging
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config
from models.sentiment.dataset import compute_class_weights
from models.sentiment.model import ArabicSentimentTransformer
from models.sentiment.tokenizer import ArabicFinancialTokenizer

logger = logging.getLogger("esis_quant.sentiment.trainer")


def time_based_split(df: pd.DataFrame, val_fraction: float = None):
    """Split `df` into train/val by DATE, never randomly (spec 4.4: "NEVER
    use random split -- would create data leakage"). Only rows with a real
    `date` can be time-ordered; rows with date=NaT (the manual bootstrap
    examples -- see dataset.py) have no temporal position and always go to
    train, never held out as validation (they're generic templates, not a
    genuine out-of-time generalization test).

    If fewer than 10 dated rows exist, there isn't a meaningful window to
    hold out -- logs a warning and returns an empty validation set rather
    than producing a near-meaningless 1-2-row split.
    """
    val_fraction = val_fraction if val_fraction is not None else config.SENTIMENT_VAL_FRACTION
    dated = df[df["date"].notna()].sort_values("date")
    undated = df[df["date"].isna()]

    if len(dated) < 10:
        logger.warning(
            "time_based_split: only %d dated examples available -- too few for a meaningful "
            "time-based validation split, training on everything with no held-out validation",
            len(dated),
        )
        return pd.concat([dated, undated], ignore_index=True), dated.iloc[0:0]

    split_idx = int(len(dated) * (1 - val_fraction))
    train_dated = dated.iloc[:split_idx]
    val_dated = dated.iloc[split_idx:]
    train_df = pd.concat([train_dated, undated], ignore_index=True)
    return train_df, val_dated.reset_index(drop=True)


class SentimentDataset(Dataset):
    def __init__(self, texts: list, labels: list, weights: list, tokenizer: ArabicFinancialTokenizer, max_length: int = None):
        self.texts = list(texts)
        self.labels = list(labels)
        self.weights = list(weights)
        self.tokenizer = tokenizer
        self.max_length = max_length or config.SENTIMENT_MAX_SEQ_LEN

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.texts[idx], max_length=self.max_length)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float),
        }


def warmup_cosine_schedule(optimizer, total_steps: int, warmup_ratio: float = None):
    """Warmup for `warmup_ratio` of total_steps (linear 0 -> base LR), then
    cosine decay to 0 over the remaining steps (spec 4.4).
    """
    warmup_ratio = warmup_ratio if warmup_ratio is not None else config.SENTIMENT_WARMUP_RATIO
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def confusion_matrix(y_true, y_pred, n_classes: int = None) -> np.ndarray:
    n_classes = n_classes or config.SENTIMENT_N_CLASSES
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1_per_class(cm: np.ndarray):
    """Manual per-class precision/recall/F1 from a confusion matrix
    (rows=true, cols=predicted). Returns three (n_classes,) arrays.
    """
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0.0
    return precision, recall, f1


def macro_f1_score(y_true, y_pred, n_classes: int = None) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    _, _, f1 = precision_recall_f1_per_class(cm)
    return float(f1.mean())


def expected_calibration_error(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b / N) * |acc_b - conf_b|, over `n_bins` equal-width
    confidence bins. `confidences`: predicted probability of the predicted
    class per sample. `correctness`: 1.0/0.0 whether that prediction was
    right. Standard reliability-diagram calibration metric.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    n = len(confidences)
    if n == 0:
        return float("nan")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_b = correctness[mask].mean()
        conf_b = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


def _run_epoch(model, loader, class_weights_t, device, optimizer=None, scheduler=None, grad_clip=None):
    """One pass over `loader`. If `optimizer` is given, trains (backward +
    step); otherwise evaluates only. Returns (mean_loss, y_true, y_pred, confidences).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_weight = 0.0, 0.0
    all_true, all_pred, all_conf = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            sample_weights = batch["weight"].to(device)

            logits, _ = model(input_ids, attention_mask)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            combined_weight = class_weights_t[labels] * sample_weights
            loss = (per_sample_loss * combined_weight).sum() / combined_weight.sum().clamp_min(1e-8)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or config.SENTIMENT_GRAD_CLIP_NORM)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

            total_loss += loss.item() * combined_weight.sum().item()
            total_weight += combined_weight.sum().item()
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_conf.extend(conf.detach().cpu().tolist())

    mean_loss = total_loss / total_weight if total_weight > 0 else float("nan")
    return mean_loss, np.array(all_true), np.array(all_pred), np.array(all_conf)


def train(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer: ArabicFinancialTokenizer,
          epochs: int = None, batch_size: int = None, device: str = None):
    """Full training loop: AdamW + warmup-cosine schedule, weighted
    cross-entropy (class weights * per-sample credibility/manual weight),
    gradient clipping, per-epoch train/val macro F1 + loss logging.
    Returns (model, history) where history is a list of per-epoch dicts.
    """
    epochs = epochs or config.SENTIMENT_EPOCHS
    batch_size = batch_size or config.SENTIMENT_BATCH_SIZE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class_weights = compute_class_weights(train_df["label"].to_numpy())
    class_weights_t = torch.tensor(class_weights, dtype=torch.float, device=device)
    logger.info("train: class weights = %s", class_weights)

    train_ds = SentimentDataset(train_df["text"], train_df["label"], train_df["weight"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if len(val_df) > 0:
        val_ds = SentimentDataset(val_df["text"], val_df["label"], val_df["weight"], tokenizer)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ArabicSentimentTransformer(vocab_size=len(tokenizer.vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.SENTIMENT_LR, weight_decay=config.SENTIMENT_WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = warmup_cosine_schedule(optimizer, total_steps)

    history = []
    for epoch in range(epochs):
        train_loss, train_true, train_pred, _ = _run_epoch(
            model, train_loader, class_weights_t, device, optimizer=optimizer, scheduler=scheduler,
        )
        train_f1 = macro_f1_score(train_true, train_pred)

        val_loss, val_f1 = float("nan"), float("nan")
        if val_loader is not None:
            val_loss, val_true, val_pred, _ = _run_epoch(model, val_loader, class_weights_t, device)
            val_f1 = macro_f1_score(val_true, val_pred)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                         "val_loss": val_loss, "val_macro_f1": val_f1})
        logger.info("epoch %d/%d: train_loss=%.4f train_f1=%.4f val_loss=%.4f val_f1=%.4f",
                    epoch + 1, epochs, train_loss, train_f1, val_loss, val_f1)

    return model, history


def evaluate(model, df: pd.DataFrame, tokenizer: ArabicFinancialTokenizer, device: str = None, batch_size: int = None):
    """Full evaluation: macro F1, per-class precision/recall/F1, ECE."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or config.SENTIMENT_BATCH_SIZE
    class_weights_t = torch.ones(config.SENTIMENT_N_CLASSES, device=device)  # unweighted for pure evaluation

    ds = SentimentDataset(df["text"], df["label"], df["weight"], tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    _, y_true, y_pred, confidences = _run_epoch(model, loader, class_weights_t, device)

    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1_per_class(cm)
    correctness = (y_true == y_pred).astype(float)
    ece = expected_calibration_error(confidences, correctness)

    return {
        "macro_f1": float(f1.mean()),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "confusion_matrix": cm,
        "ece": ece,
    }
