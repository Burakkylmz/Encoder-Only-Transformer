from __future__ import annotations

import pytest
import torch
from torch import nn

from encoder_only_transformer.layers.attention import SelfAttention
from encoder_only_transformer.blocks.encoder_block import EncoderBlock
from encoder_only_transformer.layers.feed_forward import PositionwiseFeedForward


def build_input_tensor(
    batch_size: int = 2,
    seq_len: int = 8,
    d_model: int = 64,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, d_model)


def test_encoder_block_returns_expected_output_shapes() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output, attention_weights = block(x)

    assert output.shape == (2, 8, 64)
    assert attention_weights.shape == (2, 4, 8, 8)


def test_encoder_block_returns_finite_outputs() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output, attention_weights = block(x)

    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_weights).all()


def test_encoder_block_raises_error_for_non_3d_input() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 8, 64, 1)

    with pytest.raises(
        ValueError,
        match="x must be a 3D tensor with shape",
    ):
        block(invalid_x)


def test_encoder_block_raises_error_for_embedding_dimension_mismatch() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 8, 32)

    with pytest.raises(
        ValueError,
        match="Expected input embedding dimension 64, but got 32",
    ):
        block(invalid_x)


def test_encoder_block_exposes_expected_properties() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )

    assert block.d_model == 64
    assert block.n_heads == 4
    assert block.ff_hidden_dim == 128


def test_encoder_block_contains_expected_submodules() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )

    assert isinstance(block._self_attention, SelfAttention)
    assert isinstance(block._feed_forward, PositionwiseFeedForward)
    assert isinstance(block._attention_dropout, nn.Dropout)
    assert isinstance(block._feed_forward_dropout, nn.Dropout)
    assert isinstance(block._attention_norm, nn.LayerNorm)
    assert isinstance(block._feed_forward_norm, nn.LayerNorm)


def test_encoder_block_supports_3d_attention_mask() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)
    attention_mask = torch.ones(2, 5, 5)

    output, attention_weights = block(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 64)
    assert attention_weights.shape == (2, 4, 5, 5)


def test_encoder_block_supports_4d_attention_mask() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)
    attention_mask = torch.ones(2, 1, 5, 5)

    output, attention_weights = block(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 64)
    assert attention_weights.shape == (2, 4, 5, 5)


def test_encoder_block_mask_blocks_selected_positions() -> None:
    block = EncoderBlock(
        d_model=4,
        n_heads=2,
        ff_hidden_dim=8,
        dropout=0.0,
    )
    x = torch.tensor(
        [
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[[[1, 0], [1, 0]]]])

    output, attention_weights = block(x, attention_mask=attention_mask)

    assert output.shape == (1, 2, 4)
    assert attention_weights.shape == (1, 2, 2, 2)
    assert torch.allclose(
        attention_weights[..., 1],
        torch.zeros_like(attention_weights[..., 1]),
        atol=1e-6,
    )


def test_encoder_block_preserves_deterministic_behavior_when_dropout_is_zero() -> None:
    torch.manual_seed(42)
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output_1, weights_1 = block(x)
    output_2, weights_2 = block(x)

    assert torch.allclose(output_1, output_2)
    assert torch.allclose(weights_1, weights_2)


def test_encoder_block_output_shape_matches_input_shape() -> None:
    block = EncoderBlock(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=256,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=3, seq_len=6, d_model=64)

    output, _ = block(x)

    assert output.shape == x.shape