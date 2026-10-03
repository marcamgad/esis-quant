"""
Arabic sentiment Transformer, implemented from scratch in PyTorch (no
nn.MultiheadAttention / nn.TransformerEncoderLayer -- every component
below is hand-written so the architecture matches the spec exactly:
pre-norm encoder layers, sinusoidal positional encoding, and an
attention-pooling classification head instead of a bare [CLS] token).
"""

import math

import torch
import torch.nn as nn

import config


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(...).
    Precomputed once as a non-trainable buffer, added to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    """Q = X W_Q, K = X W_K, V = X W_V, each (d_model -> d_model), split
    into n_heads of d_model/n_heads, A = softmax(QK^T / sqrt(d_k)) V,
    concat heads and project back to d_model. Padding positions (where
    attention_mask == 0) get their attention score set to -inf before the
    softmax so they receive ~0 weight, matching standard Transformer
    padding-mask handling.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B,H,T,dk)
        k = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,T,T)
        mask = attention_mask[:, None, None, :].bool()  # (B,1,1,T), broadcasts over heads and query positions
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # a fully-padded row would softmax to nan; zero it instead
        attn = self.dropout(attn)

        out = attn @ v  # (B,H,T,dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = GELU(x W_1 + b_1) W_2 + b_2, d_ff inner dimension."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Pre-norm architecture (more stable to train than post-norm):
    h = x + Attention(LayerNorm(x))
    out = h + FFN(LayerNorm(h))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        out = h + self.dropout(self.ffn(self.norm2(h)))
        return out


class AttentionPooling(nn.Module):
    """Weighted average of all token representations, weights learned via
    a scalar attention score per position (softmax over the sequence,
    masked to exclude padding) -- richer than pooling only the [CLS] slot,
    since it lets the model learn which tokens actually carry sentiment.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        scores = self.score(x).squeeze(-1)  # (B,T)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B,T)
        # A fully-padded row (attention_mask all zeros -- e.g. an empty
        # string that slips past the tokenizer) softmaxes over all -inf to
        # NaN, which would poison gradients. Zero those weights so the row
        # pools to the zero vector instead of NaN (ROBUST-ML-01). The
        # tokenizer always emits [CLS]/[SEP] so this is a defensive guard,
        # not an expected path.
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.einsum("bt,btd->bd", weights, x)
        return pooled, weights


class ArabicSentimentTransformer(nn.Module):
    """Full encoder-only Transformer for 3-class Arabic financial
    sentiment classification (Negative=0, Neutral=1, Positive=2).
    Input: input_ids (B,T) int64, attention_mask (B,T) int64/float {0,1}.
    Output: logits (B,3), pooling attention weights (B,T) (for
    interpretability -- which tokens drove the prediction).
    """

    def __init__(self, vocab_size: int = None, d_model: int = None, n_heads: int = None,
                 n_layers: int = None, d_ff: int = None, max_seq_len: int = None,
                 dropout: float = None, n_classes: int = None):
        super().__init__()
        vocab_size = vocab_size or config.SENTIMENT_VOCAB_SIZE
        d_model = d_model or config.SENTIMENT_D_MODEL
        n_heads = n_heads or config.SENTIMENT_N_HEADS
        n_layers = n_layers or config.SENTIMENT_N_LAYERS
        d_ff = d_ff or config.SENTIMENT_D_FF
        max_seq_len = max_seq_len or config.SENTIMENT_MAX_SEQ_LEN
        dropout = config.SENTIMENT_DROPOUT if dropout is None else dropout
        n_classes = n_classes or config.SENTIMENT_N_CLASSES

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=config.SENTIMENT_SPECIAL_TOKENS["[PAD]"])
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = AttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.final_norm(x)
        pooled, pool_weights = self.pool(x, attention_mask)
        logits = self.classifier(pooled)
        return logits, pool_weights
