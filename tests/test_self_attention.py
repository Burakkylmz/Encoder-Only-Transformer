from __future__ import annotations

import pytest
import torch

from encoder_only_transformer.layers.attention import (
    ScaledDotProductAttention,
    SelfAttention,
)


def build_input_tensor(
    batch_size: int = 2,
    seq_len: int = 8,
    d_model: int = 64,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, d_model)


def test_self_attention_returns_expected_output_shapes() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output, attention_weights = attention(x)

    assert output.shape == (2, 8, 64)
    assert attention_weights.shape == (2, 4, 8, 8)


def test_self_attention_returns_finite_outputs() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output, attention_weights = attention(x)

    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_weights).all()


def test_self_attention_raises_error_for_non_3d_input() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    invalid_x = torch.randn(2, 8, 64, 1)

    with pytest.raises(
        ValueError,
        match="x must be a 3D tensor with shape",
    ):
        attention(invalid_x)


def test_self_attention_raises_error_for_embedding_dimension_mismatch() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    invalid_x = torch.randn(2, 8, 32)

    with pytest.raises(
        ValueError,
        match="Expected input embedding dimension 64, but got 32",
    ):
        attention(invalid_x)


def test_self_attention_raises_error_when_d_model_is_not_divisible_by_n_heads() -> None:
    with pytest.raises(
        ValueError,
        match="d_model must be divisible by n_heads",
    ):
        SelfAttention(d_model=62, n_heads=4, dropout=0.0)


def test_self_attention_exposes_expected_properties() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)

    assert attention.d_model == 64
    assert attention.n_heads == 4
    assert attention.head_dim == 16


def test_self_attention_contains_scaled_dot_product_attention_module() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)

    assert isinstance(attention._attention, ScaledDotProductAttention)


def test_split_heads_returns_expected_shape() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    split = attention._split_heads(x)

    assert split.shape == (2, 4, 8, 16)


def test_combine_heads_returns_expected_shape() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 4, 8, 16)

    combined = attention._combine_heads(x)

    assert combined.shape == (2, 8, 64)


def test_combine_heads_raises_error_for_invalid_number_of_heads() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    invalid_x = torch.randn(2, 2, 8, 16)

    with pytest.raises(
        ValueError,
        match="Expected 4 heads, but got 2",
    ):
        attention._combine_heads(invalid_x)


def test_combine_heads_raises_error_for_invalid_head_dim() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    invalid_x = torch.randn(2, 4, 8, 8)

    with pytest.raises(
        ValueError,
        match="Expected head_dim 16, but got 8",
    ):
        attention._combine_heads(invalid_x)


def test_self_attention_supports_3d_attention_mask() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)
    attention_mask = torch.ones(2, 5, 5)

    output, attention_weights = attention(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 64)
    assert attention_weights.shape == (2, 4, 5, 5)


def test_self_attention_supports_4d_attention_mask() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)
    attention_mask = torch.ones(2, 1, 5, 5)

    output, attention_weights = attention(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 64)
    assert attention_weights.shape == (2, 4, 5, 5)


def test_self_attention_mask_blocks_selected_positions() -> None:
    attention = SelfAttention(d_model=4, n_heads=2, dropout=0.0)

    x = torch.tensor(
        [
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[[[1, 0], [1, 0]]]])

    output, attention_weights = attention(x, attention_mask=attention_mask)

    assert output.shape == (1, 2, 4)
    assert attention_weights.shape == (1, 2, 2, 2)
    assert torch.allclose(attention_weights[..., 1], torch.zeros_like(attention_weights[..., 1]), atol=1e-6)


def test_self_attention_preserves_deterministic_behavior_when_dropout_is_zero() -> None:
    torch.manual_seed(42)
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output_1, weights_1 = attention(x)
    output_2, weights_2 = attention(x)

    assert torch.allclose(output_1, output_2)
    assert torch.allclose(weights_1, weights_2)