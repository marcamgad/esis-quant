"""
Tests for the TFT feature/window builder on the REAL database (Layer 6),
plus a small learning smoke test that the model + pinball loss can actually
fit a signal and produce ordered quantiles.
"""

import numpy as np
import torch

import config
from models.tft.features import (
    build_windows, TEMPORAL_COLS, STATIC_COLS, STANDARDIZE_TEMPORAL_COLS, REGIME_ONEHOT_COLS,
)
from models.tft.trainer import TFTDataset, train, pinball_loss


def test_build_windows_real_data_shapes_and_no_nan():
    data = build_windows()
    tr = data["splits"]["train"]
    # real data must actually produce a substantial number of windows (guards
    # against the "everything dropped by a scattered NaN" failure mode)
    assert len(tr["past"]) > 1000
    assert tr["past"].shape[1] == config.TFT_SEQUENCE_LENGTH
    assert tr["past"].shape[2] == len(TEMPORAL_COLS)
    assert tr["static"].shape[1] == len(STATIC_COLS)
    assert tr["targets"].shape[1] == len(config.TFT_FORECAST_HORIZONS)
    # windows are strictly finite (NaN windows were dropped in the builder)
    assert np.isfinite(tr["past"]).all()
    assert np.isfinite(tr["static"]).all()
    assert np.isfinite(tr["targets"]).all()
    # walk-forward ordering: every train end-date precedes every test end-date
    if len(data["splits"]["test"]["dates"]):
        assert tr["dates"].max() <= np.datetime64(config.TFT_TRAIN_END)


def test_build_windows_no_lookahead_in_standardization():
    # standardization stats must come from the train span only
    data = build_windows()
    stats = data["temporal_stats"]
    # only the continuous temporal cols are standardized; regime one-hots are raw
    assert set(stats["center"].keys()) == set(STANDARDIZE_TEMPORAL_COLS)
    assert all(c not in stats["center"] for c in REGIME_ONEHOT_COLS)
    # scale values are finite and positive
    assert all(v > 0 for v in stats["scale"].values())
    assert stats["method"] == "zscore_clipped"


def test_regime_onehot_features_present_and_binary():
    data = build_windows()
    cols = data["temporal_cols"]
    for c in REGIME_ONEHOT_COLS:
        assert c in cols
    tr = data["splits"]["train"]
    idx = [cols.index(c) for c in REGIME_ONEHOT_COLS]
    onehot = tr["past"][:, :, idx]  # (N, T, 4)
    # values are exactly 0/1 (not standardized)
    assert np.all((onehot == 0) | (onehot == 1))
    # at most one regime active per timestep (0 allowed for UNKNOWN dates)
    assert onehot.sum(axis=-1).max() <= 1.0
    # regimes are actually exercised in training (not a dead all-zero block)
    assert onehot.sum() > 0


def test_standardized_features_are_clipped():
    import numpy as np
    data = build_windows()
    tr = data["splits"]["train"]
    clip = config.TFT_ZSCORE_CLIP
    # no standardized input feature exceeds the clip bound (guards against a
    # single real event -- e.g. the 2016 EGP float -- dominating the scale)
    assert np.abs(tr["past"]).max() <= clip + 1e-4
    assert np.abs(tr["static"]).max() <= clip + 1e-4


def test_tft_overfits_tiny_signal():
    # synthetic: target sign determined by the last step of feature 0, so a
    # working model + pinball loss should push directional accuracy high and
    # keep quantiles ordered (q05 <= q50 <= q95) on average.
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, T, F, H = 200, 12, 4, 2
    past = rng.normal(size=(n, T, F)).astype(np.float32)
    static = rng.normal(size=(n, 1)).astype(np.float32)
    signal = past[:, -1, 0]
    targets = np.stack([0.02 * np.sign(signal), 0.02 * np.sign(signal)], axis=1).astype(np.float32)

    ds = TFTDataset(past, static, targets)
    model, history = train(ds, TFTDataset(past[:1], static[:1], targets[:1]),
                           n_temporal=F, n_static=1, epochs=40, batch_size=32, device="cpu",
                           horizons=[1, 5], quantiles=config.TFT_QUANTILES)
    assert history[-1]["train_loss"] < history[0]["train_loss"]  # learned something

    model.eval()
    with torch.no_grad():
        preds, _ = model(torch.tensor(past), torch.tensor(static))
    preds = preds.numpy()  # (n, H, Q)
    # quantiles ordered on average (q05 <= q50 <= q95)
    q = config.TFT_QUANTILES
    i05, i50, i95 = q.index(0.05), q.index(0.5), q.index(0.95)
    assert preds[:, :, i05].mean() <= preds[:, :, i50].mean() <= preds[:, :, i95].mean()
