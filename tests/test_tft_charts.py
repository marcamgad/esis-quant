"""
Tests for TFT checkpoint loading / inference wiring (Layer 6). The full chart
render is exercised manually (it rebuilds the real windows); here we verify
the checkpoint round-trips into the repo model and produces correctly-shaped
quantile forecasts. Skipped cleanly when no trained checkpoint is present.
"""

import os

import pytest
import torch

import config
from models.tft.model import TemporalFusionTransformer


@pytest.mark.skipif(not os.path.exists(config.TFT_MODEL_CHECKPOINT_PATH),
                    reason="no trained TFT checkpoint present (train on Kaggle first)")
def test_load_tft_checkpoint_and_forecast_shape():
    from visualization.tft_charts import load_tft
    model, ckpt = load_tft()
    for key in ("model_state_dict", "n_temporal", "n_static", "horizons", "quantiles",
                "temporal_cols", "static_cols"):
        assert key in ckpt
    past = torch.randn(3, config.TFT_SEQUENCE_LENGTH, ckpt["n_temporal"])
    static = torch.randn(3, ckpt["n_static"])
    with torch.no_grad():
        preds, interp = model(past, static)
    assert preds.shape == (3, len(ckpt["horizons"]), len(ckpt["quantiles"]))
    assert interp["attention"].shape == (3, config.TFT_SEQUENCE_LENGTH, config.TFT_SEQUENCE_LENGTH)
    assert torch.isfinite(preds).all()
