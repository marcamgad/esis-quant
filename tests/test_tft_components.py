"""
Tests for the from-scratch TFT components (Layer 6): GRN, VSN, interpretable
attention, pinball loss, and the full model's shapes + interpretability.
"""

import numpy as np
import torch

from models.tft.grn import GatedResidualNetwork, GatedLinearUnit
from models.tft.vsn import VariableSelectionNetwork
from models.tft.attention import InterpretableMultiHeadAttention, causal_mask
from models.tft.trainer import pinball_loss
from models.tft.model import TemporalFusionTransformer


def test_grn_output_shape_and_context():
    grn = GatedResidualNetwork(input_size=8, hidden_size=16, output_size=8, context_size=4)
    a = torch.randn(5, 12, 8)      # (B, T, in)
    c = torch.randn(5, 4)          # static context, broadcast over T
    out = grn(a, c)
    assert out.shape == (5, 12, 8)
    # without context also works
    grn2 = GatedResidualNetwork(8, 16, 8)
    assert grn2(a).shape == (5, 12, 8)


def test_glu_gates_between_zero_and_full():
    glu = GatedLinearUnit(4, 4)
    x = torch.randn(10, 4)
    out = glu(x)
    assert out.shape == (10, 4)
    assert torch.isfinite(out).all()


def test_vsn_weights_are_a_distribution_and_select_signal():
    torch.manual_seed(0)
    vsn = VariableSelectionNetwork(n_vars=4, hidden_size=8)
    x = torch.randn(6, 4)
    selected, weights = vsn(x)
    assert selected.shape == (6, 8)
    assert weights.shape == (6, 4)
    # softmax weights: non-negative, sum to 1 per sample
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(-1), torch.ones(6), atol=1e-5)


def test_attention_causal_mask_blocks_lookahead():
    torch.manual_seed(0)
    attn = InterpretableMultiHeadAttention(d_model=16, n_heads=4)
    attn.eval()  # disable dropout so attention weights form a proper distribution
    x = torch.randn(2, 7, 16)
    mask = causal_mask(7)
    out, weights = attn(x, x, x, mask=mask)
    assert out.shape == (2, 7, 16)
    assert weights.shape == (2, 7, 7)
    # no attention to future positions (strictly-upper triangle is ~0)
    upper = weights[:, torch.triu(torch.ones(7, 7), diagonal=1).bool()]
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)
    # each row is a distribution over the allowed (past+present) positions
    assert torch.allclose(weights.sum(-1), torch.ones(2, 7), atol=1e-5)


def test_pinball_loss_known_values_and_asymmetry():
    quantiles = [0.1, 0.5, 0.9]
    # under-prediction: y=1, yhat=0 -> error=+1 -> loss q*1
    preds = torch.zeros(1, 1, 3)
    targets = torch.ones(1, 1)
    loss = pinball_loss(preds, targets, quantiles)
    expected = np.mean([0.1 * 1, 0.5 * 1, 0.9 * 1])
    assert abs(loss.item() - expected) < 1e-6

    # over-prediction: y=0, yhat=1 -> error=-1 -> loss (1-q)*1
    preds2 = torch.ones(1, 1, 3)
    targets2 = torch.zeros(1, 1)
    loss2 = pinball_loss(preds2, targets2, quantiles)
    expected2 = np.mean([0.9 * 1, 0.5 * 1, 0.1 * 1])
    assert abs(loss2.item() - expected2) < 1e-6


def test_tft_forward_shapes_and_interpretability():
    torch.manual_seed(0)
    horizons, quantiles = [1, 5, 10, 20], [0.05, 0.25, 0.5, 0.75, 0.95]
    model = TemporalFusionTransformer(n_temporal=13, n_static=2, hidden_size=16,
                                      horizons=horizons, quantiles=quantiles)
    past = torch.randn(4, 60, 13)
    static = torch.randn(4, 2)
    preds, interp = model(past, static)
    assert preds.shape == (4, len(horizons), len(quantiles))
    assert interp["temporal_weights"].shape == (4, 60, 13)
    assert interp["static_weights"].shape == (4, 2)
    assert interp["attention"].shape == (4, 60, 60)
    assert torch.isfinite(preds).all()


def test_tft_runs_without_static():
    model = TemporalFusionTransformer(n_temporal=5, n_static=0, hidden_size=8)
    preds, interp = model(torch.randn(3, 20, 5), None)
    assert preds.shape[0] == 3
    assert interp["static_weights"] is None
