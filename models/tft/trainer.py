"""
TFT training utilities: pinball (quantile) loss, dataset wrapper, and a
walk-forward training loop with early stopping. Metrics (directional
accuracy, quantile coverage) implemented manually.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from models.tft.model import TemporalFusionTransformer

logger = logging.getLogger("esis_quant.tft.trainer")


def pinball_loss(preds: torch.Tensor, targets: torch.Tensor, quantiles) -> torch.Tensor:
    """Quantile (pinball) loss, averaged over quantiles, horizons, batch.

    preds:   (B, n_horizons, n_quantiles)
    targets: (B, n_horizons)   -- realized forward return per horizon
    QL_q(y, yhat) = max(q*(y-yhat), (q-1)*(y-yhat))

    Under-prediction (y > yhat) is penalized by q, over-prediction by (1-q):
    a high quantile (q=0.95) is punished ~19x harder for over-shooting than
    under-shooting, which is what pins it to the 95th percentile.
    """
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device).view(1, 1, -1)
    y = targets.unsqueeze(-1)  # (B, n_horizons, 1)
    error = y - preds          # (B, n_horizons, n_quantiles)
    loss = torch.maximum(q * error, (q - 1) * error)
    return loss.mean()


class TFTDataset(Dataset):
    """Wraps pre-built arrays: past (N, T, n_temporal), static (N, n_static),
    targets (N, n_horizons). Kept dead-simple -- windowing / alignment is done
    once in features.py, not per-__getitem__.
    """

    def __init__(self, past, static, targets):
        self.past = torch.as_tensor(np.asarray(past), dtype=torch.float32)
        self.static = torch.as_tensor(np.asarray(static), dtype=torch.float32)
        self.targets = torch.as_tensor(np.asarray(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.past)

    def __getitem__(self, i):
        return self.past[i], self.static[i], self.targets[i]


def directional_accuracy(preds: np.ndarray, targets: np.ndarray, median_idx: int) -> np.ndarray:
    """Fraction of samples where the sign of the MEDIAN forecast matches the
    sign of the realized return, per horizon. preds: (N,H,Q), targets: (N,H).
    Returns (H,).
    """
    med = preds[:, :, median_idx]  # (N,H)
    return (np.sign(med) == np.sign(targets)).mean(axis=0)


def quantile_coverage(preds: np.ndarray, targets: np.ndarray, quantiles) -> np.ndarray:
    """Empirical coverage of each quantile: fraction of realized targets that
    fall at or below the predicted quantile. Well-calibrated => coverage[q] ~ q.
    preds: (N,H,Q), targets: (N,H). Returns (Q,) averaged over horizons.
    """
    y = targets[:, :, None]  # (N,H,1)
    below = (y <= preds).mean(axis=(0, 1))  # (Q,)
    return below


def train(train_ds: TFTDataset, val_ds: TFTDataset, n_temporal: int, n_static: int,
          epochs: int = None, batch_size: int = None, device: str = None,
          horizons=None, quantiles=None):
    """Walk-forward TFT training: AdamW + OneCycleLR, pinball loss, gradient
    clipping, early stopping on validation loss. train_ds/val_ds are ALREADY
    time-ordered (features.py splits by date), so no shuffling across the
    train/val boundary. Returns (best_model, history).
    """
    epochs = epochs or config.TFT_EPOCHS
    batch_size = batch_size or config.TFT_BATCH_SIZE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    quantiles = list(quantiles or config.TFT_QUANTILES)
    horizons = list(horizons or config.TFT_FORECAST_HORIZONS)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) else None

    model = TemporalFusionTransformer(n_temporal, n_static, horizons=horizons, quantiles=quantiles).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TFT_LR, weight_decay=config.TFT_WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.TFT_LR, epochs=epochs, steps_per_epoch=steps_per_epoch,
    )

    best_val, best_state, patience = float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for past, static, targets in train_loader:
            past, static, targets = past.to(device), static.to(device), targets.to(device)
            preds, _ = model(past, static)
            loss = pinball_loss(preds, targets, quantiles)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TFT_GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(past)
        train_loss /= len(train_ds)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for past, static, targets in val_loader:
                    past, static, targets = past.to(device), static.to(device), targets.to(device)
                    preds, _ = model(past, static)
                    vl += pinball_loss(preds, targets, quantiles).item() * len(past)
            val_loss = vl / len(val_ds)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        logger.info("TFT epoch %d/%d: train_loss=%.5f val_loss=%.5f", epoch + 1, epochs, train_loss, val_loss)

        monitor = val_loss if val_loader is not None else train_loss
        if monitor < best_val - 1e-6:
            best_val, patience = monitor, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.TFT_EARLY_STOP_PATIENCE:
                logger.info("TFT early stopping at epoch %d (no val improvement for %d epochs)",
                            epoch + 1, config.TFT_EARLY_STOP_PATIENCE)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
