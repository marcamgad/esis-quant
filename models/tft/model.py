"""
Temporal Fusion Transformer (Lim et al. 2021), from scratch in PyTorch.

Data flow (encoder-side TFT with multi-horizon quantile heads):

  static features (B, n_static)
      -> Static VSN                     -> static embedding + static weights
      -> 4 GRNs                         -> contexts: c_s (selection),
                                           c_e (enrichment), c_h, c_c (LSTM init)

  past temporal features (B, T, n_temporal)
      -> Temporal VSN (context c_s)     -> selected sequence (B, T, H) + weights
      -> LSTM encoder (init from c_h/c_c)
      -> gated skip + LayerNorm         (post-LSTM GLU)
      -> Static enrichment GRN (c_e)    -> enriched (B, T, H)
      -> Interpretable self-attention   (causal mask) -> attended (B, T, H) + attn
      -> gated skip + LayerNorm
      -> position-wise GRN
      -> take LAST timestep             -> per-horizon quantile heads
                                        -> (B, n_horizons, n_quantiles)

The multi-horizon quantile output gives, per forecast horizon, a full
predictive distribution (5 quantiles) rather than a point estimate -- the
median is the central forecast and the outer quantiles are the uncertainty
band. Interpretability outputs (variable-selection weights, temporal
attention) are returned alongside for the Layer 6 charts (spec 6.3).
"""

import torch
import torch.nn as nn

import config
from models.tft.grn import GatedResidualNetwork, GatedLinearUnit
from models.tft.vsn import VariableSelectionNetwork
from models.tft.attention import InterpretableMultiHeadAttention, causal_mask


class _AddNorm(nn.Module):
    """Gated skip connection: LayerNorm(residual + GLU(x))."""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.glu = GatedLinearUnit(size, size, dropout)
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + self.glu(x))


class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_temporal: int, n_static: int,
                 hidden_size: int = None, n_heads: int = None, lstm_layers: int = None,
                 dropout: float = None, horizons=None, quantiles=None):
        super().__init__()
        H = hidden_size or config.TFT_HIDDEN_SIZE
        n_heads = n_heads or config.TFT_ATTENTION_HEADS
        lstm_layers = lstm_layers or config.TFT_LSTM_LAYERS
        dropout = config.TFT_DROPOUT if dropout is None else dropout
        self.horizons = list(horizons or config.TFT_FORECAST_HORIZONS)
        self.quantiles = list(quantiles or config.TFT_QUANTILES)
        self.hidden_size = H
        self.n_temporal = n_temporal
        self.n_static = n_static
        self.lstm_layers = lstm_layers

        # --- static path ---
        self.static_vsn = VariableSelectionNetwork(n_static, H, dropout) if n_static > 0 else None
        static_ctx_in = H if n_static > 0 else 0
        # 4 static context encoders (selection / enrichment / lstm h / lstm c)
        make_ctx = lambda: GatedResidualNetwork(H, H, H, dropout) if n_static > 0 else None
        self.ctx_selection = make_ctx()
        self.ctx_enrichment = make_ctx()
        self.ctx_h = make_ctx()
        self.ctx_c = make_ctx()

        # --- temporal path ---
        self.temporal_vsn = VariableSelectionNetwork(
            n_temporal, H, dropout, context_size=static_ctx_in or None
        )
        self.lstm = nn.LSTM(H, H, num_layers=lstm_layers, batch_first=True, dropout=0.0)
        self.post_lstm = _AddNorm(H, dropout)
        self.enrichment = GatedResidualNetwork(H, H, H, dropout, context_size=static_ctx_in or None)
        self.attention = InterpretableMultiHeadAttention(H, n_heads, dropout)
        self.post_attn = _AddNorm(H, dropout)
        self.position_wise = GatedResidualNetwork(H, H, H, dropout)
        self.post_ff = _AddNorm(H, dropout)

        # --- quantile heads: one per horizon, each outputs len(quantiles) ---
        self.quantile_heads = nn.ModuleList([
            nn.Linear(H, len(self.quantiles)) for _ in self.horizons
        ])

    def forward(self, past: torch.Tensor, static: torch.Tensor = None):
        """past: (B, T, n_temporal). static: (B, n_static) or None.
        Returns (quantile_preds (B, n_horizons, n_quantiles), interpretability dict).
        """
        B, T, _ = past.shape
        device = past.device

        # static contexts
        c_s = c_e = None
        static_weights = None
        if self.static_vsn is not None and static is not None:
            static_emb, static_weights = self.static_vsn(static)  # (B,H), (B,n_static)
            c_s = self.ctx_selection(static_emb)
            c_e = self.ctx_enrichment(static_emb)
            c_h = self.ctx_h(static_emb).unsqueeze(0).repeat(self.lstm_layers, 1, 1)  # (L,B,H)
            c_c = self.ctx_c(static_emb).unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        else:
            c_h = torch.zeros(self.lstm_layers, B, self.hidden_size, device=device)
            c_c = torch.zeros(self.lstm_layers, B, self.hidden_size, device=device)

        # temporal variable selection (context = static selection context)
        selected, temporal_weights = self.temporal_vsn(past, c_s)  # (B,T,H), (B,T,n_temporal)

        # LSTM encoder, initialized from static contexts
        lstm_out, _ = self.lstm(selected, (c_h, c_c))
        temporal = self.post_lstm(lstm_out, selected)  # gated skip over the VSN output

        # static enrichment (per timestep)
        enriched = self.enrichment(temporal, c_e)

        # interpretable self-attention over time (causal: no look-ahead)
        mask = causal_mask(T, device=device)
        attn_out, attention = self.attention(enriched, enriched, enriched, mask=mask)
        attended = self.post_attn(attn_out, enriched)

        # position-wise feed-forward + final gated skip
        ff = self.position_wise(attended)
        out = self.post_ff(ff, temporal)  # skip back to the pre-enrichment temporal

        # forecast from the LAST observed timestep
        last = out[:, -1, :]  # (B,H)
        preds = torch.stack([head(last) for head in self.quantile_heads], dim=1)  # (B, n_horizons, n_quantiles)

        interpretability = {
            "temporal_weights": temporal_weights,   # (B,T,n_temporal)
            "static_weights": static_weights,       # (B,n_static) or None
            "attention": attention,                 # (B,T,T)
        }
        return preds, interpretability
