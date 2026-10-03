"""
Interpretable Multi-Head Attention (Lim et al. 2021, eq. 14-16), from
scratch. Differs from standard multi-head attention in ONE way that makes
the attention weights interpretable: all heads SHARE a single value
projection (one V for every head), so the per-head attention matrices act on
the same values and can be averaged into a single, meaningful
attention-over-time pattern. Each head still has its own Q, K projections.

  head_h = Softmax(Q_h K_h^T / sqrt(d_k)) V        (shared V)
  output = W_o * mean_h(head_h)

A causal (look-ahead) mask can be supplied so a position never attends to
future time steps.
"""

import math

import torch
import torch.nn as nn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(n_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(d_model, self.d_k) for _ in range(n_heads)])
        self.v_layer = nn.Linear(d_model, self.d_k)  # SHARED across heads (interpretability)
        self.w_o = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """q,k,v: (B, T, d_model). mask: (T, T) or (B, T, T) with True where
        attention is DISALLOWED. Returns (output (B,T,d_model),
        attention (B,T,T) averaged over heads).
        """
        v_shared = self.v_layer(v)  # (B, T, d_k)
        head_outputs = []
        attns = []
        for h in range(self.n_heads):
            qh = self.q_layers[h](q)  # (B,T,d_k)
            kh = self.k_layers[h](k)
            scores = qh @ kh.transpose(-2, -1) / math.sqrt(self.d_k)  # (B,T,T)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)
            head_outputs.append(attn @ v_shared)  # (B,T,d_k)
            attns.append(attn)

        # average heads (shared V makes this average meaningful)
        head_mean = torch.stack(head_outputs, dim=0).mean(dim=0)  # (B,T,d_k)
        attn_mean = torch.stack(attns, dim=0).mean(dim=0)          # (B,T,T)
        output = self.w_o(head_mean)
        return output, attn_mean


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """(seq_len, seq_len) boolean mask, True in the strictly-upper triangle,
    so position t cannot attend to any position > t (no look-ahead).
    """
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
