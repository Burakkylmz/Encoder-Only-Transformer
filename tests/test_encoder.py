from __future__ import annotations

import pytest
import torch
from torch import nn

from encoder_only_transformer.models.encoder import Encoder
from encoder_only_transformer.blocks.encoder_block import EncoderBlock


def build_input_tensor(
    batch_size: int = 2,
    seq_len: int = 8,
    d_model: int = 64,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, d_model)


def test_encoder_returns_expected_output_shape() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=3,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output, attention_weights_per_layer = encoder(x)

    assert output.shape == (2, 8, 64)
    assert len(attention_weights_per_layer) == 3


def test_encoder_returns_attention_weights_for_each_layer_with_expected_shape() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=3,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    _, attention_weights_per_layer = encoder(x)

    assert len(attention_weights_per_layer) == 3
    for attention_weights in attention_weights_per_layer:
        assert attention_weights.shape == (2, 4, 8, 8)


def test_encoder_returns_finite_outputs() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=2,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output, attention_weights_per_layer = encoder(x)

    assert torch.isfinite(output).all()
    for attention_weights in attention_weights_per_layer:
        assert torch.isfinite(attention_weights).all()


def test_encoder_raises_error_for_non_3d_input() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=2,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 8, 64, 1)

    with pytest.raises(
        ValueError,
        match="x must be a 3D tensor with shape",
    ):
        encoder(invalid_x)


def test_encoder_raises_error_for_embedding_dimension_mismatch() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=2,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 8, 32)

    with pytest.raises(
        ValueError,
        match="Expected input embedding dimension 64, but got 32",
    ):
        encoder(invalid_x)


def test_encoder_raises_error_when_d_model_is_not_divisible_by_n_heads() -> None:
    with pytest.raises(
        ValueError,
        match="d_model must be divisible by n_heads",
    ):
        Encoder(
            d_model=62,
            n_heads=4,
            ff_hidden_dim=128,
            n_layers=2,
            dropout=0.0,
        )


def test_encoder_raises_error_when_n_layers_is_invalid() -> None:
    with pytest.raises(
        ValueError,
        match="n_layers must be greater than 0",
    ):
        Encoder(
            d_model=64,
            n_heads=4,
            ff_hidden_dim=128,
            n_layers=0,
            dropout=0.0,
        )


def test_encoder_exposes_expected_properties() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=3,
        dropout=0.0,
    )

    assert encoder.d_model == 64
    assert encoder.n_heads == 4
    assert encoder.ff_hidden_dim == 128
    assert encoder.n_layers == 3


def test_encoder_contains_expected_submodules() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=3,
        dropout=0.0,
    )

    assert isinstance(encoder._layers, nn.ModuleList)
    assert len(encoder._layers) == 3
    assert all(isinstance(layer, EncoderBlock) for layer in encoder._layers)


def test_encoder_supports_3d_attention_mask() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=2,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)
    attention_mask = torch.ones(2, 5, 5)

    output, attention_weights_per_layer = encoder(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 64)
    assert len(attention_weights_per_layer) == 2
    for attention_weights in attention_weights_per_layer:
        assert attention_weights.shape == (2, 4, 5, 5)


def test_encoder_supports_4d_attention_mask() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=2,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)
    attention_mask = torch.ones(2, 1, 5, 5)

    output, attention_weights_per_layer = encoder(x, attention_mask=attention_mask)

    assert output.shape == (2, 5, 64)
    assert len(attention_weights_per_layer) == 2
    for attention_weights in attention_weights_per_layer:
        assert attention_weights.shape == (2, 4, 5, 5)


def test_encoder_preserves_deterministic_behavior_when_dropout_is_zero() -> None:
    torch.manual_seed(42)
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        n_layers=2,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output_1, weights_1 = encoder(x)
    output_2, weights_2 = encoder(x)

    assert torch.allclose(output_1, output_2)
    assert len(weights_1) == len(weights_2)

    for layer_weights_1, layer_weights_2 in zip(weights_1, weights_2):
        assert torch.allclose(layer_weights_1, layer_weights_2)


def test_encoder_output_shape_matches_input_shape() -> None:
    encoder = Encoder(
        d_model=64,
        n_heads=4,
        ff_hidden_dim=256,
        n_layers=4,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=3, seq_len=6, d_model=64)

    output, _ = encoder(x)

    assert output.shape == x.shape