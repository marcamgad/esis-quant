import torch

from models.sentiment.model import ArabicSentimentTransformer, SinusoidalPositionalEncoding, MultiHeadSelfAttention


def _make_small_model():
    torch.manual_seed(0)
    return ArabicSentimentTransformer(
        vocab_size=100, d_model=32, n_heads=4, n_layers=2, d_ff=64, max_seq_len=16, dropout=0.1, n_classes=3,
    )


def test_forward_pass_shapes_and_no_nan():
    model = _make_small_model()
    B, T = 4, 16
    input_ids = torch.randint(5, 100, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[:, 10:] = 0

    logits, pool_weights = model(input_ids, attention_mask)
    assert logits.shape == (B, 3)
    assert pool_weights.shape == (B, T)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(pool_weights).any()


def test_attention_pooling_weights_sum_to_one_and_ignore_padding():
    model = _make_small_model()
    B, T = 4, 16
    input_ids = torch.randint(5, 100, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[:, 10:] = 0

    _, pool_weights = model(input_ids, attention_mask)
    sums = pool_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)
    assert (pool_weights[:, 10:] < 1e-6).all()


def test_gradient_flows_to_every_parameter():
    model = _make_small_model()
    B, T = 4, 16
    input_ids = torch.randint(5, 100, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)

    logits, _ = model(input_ids, attention_mask)
    logits.sum().backward()

    n_total = 0
    n_with_grad = 0
    for p in model.parameters():
        n_total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            n_with_grad += 1
    assert n_with_grad == n_total


def test_multihead_attention_output_shape_matches_input():
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.long)
    out = attn(x, mask)
    assert out.shape == x.shape


def test_multihead_attention_fully_masked_row_does_not_produce_nan():
    # A sequence where every position is padding is a degenerate but
    # possible batch edge case (e.g. an empty string) -- softmax over an
    # all -inf row would otherwise produce NaN.
    attn = MultiHeadSelfAttention(d_model=16, n_heads=2, dropout=0.0)
    x = torch.randn(1, 5, 16)
    mask = torch.zeros(1, 5, dtype=torch.long)
    out = attn(x, mask)
    assert not torch.isnan(out).any()


def test_sinusoidal_positional_encoding_matches_formula():
    d_model, max_len = 8, 5
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)
    x = torch.zeros(1, max_len, d_model)
    out = pe_module(x)

    import math
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            expected_sin = math.sin(pos / (10000 ** (i / d_model)))
            expected_cos = math.cos(pos / (10000 ** (i / d_model)))
            assert abs(out[0, pos, i].item() - expected_sin) < 1e-5
            assert abs(out[0, pos, i + 1].item() - expected_cos) < 1e-5
