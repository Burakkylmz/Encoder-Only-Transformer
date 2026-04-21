from __future__ import annotations

import pytest
import torch
from torch import nn

from encoder_only_transformer.layers.feed_forward import PositionwiseFeedForward


def build_input_tensor(
    batch_size: int = 2,
    seq_len: int = 8,
    d_model: int = 64,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, d_model)


def test_positionwise_feed_forward_returns_expected_shape() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output = feed_forward(x)

    assert output.shape == (2, 8, 64)


def test_positionwise_feed_forward_returns_finite_outputs() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output = feed_forward(x)

    assert torch.isfinite(output).all()


def test_positionwise_feed_forward_raises_error_for_non_3d_input() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 8, 64, 1)

    with pytest.raises(
        ValueError,
        match="x must be a 3D tensor with shape",
    ):
        feed_forward(invalid_x)


def test_positionwise_feed_forward_raises_error_for_embedding_dimension_mismatch() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 8, 32)

    with pytest.raises(
        ValueError,
        match="Expected input embedding dimension 64, but got 32",
    ):
        feed_forward(invalid_x)


def test_positionwise_feed_forward_exposes_expected_properties() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )

    assert feed_forward.d_model == 64
    assert feed_forward.hidden_dim == 128


def test_positionwise_feed_forward_uses_gelu_as_default_activation() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )

    assert isinstance(feed_forward._activation, nn.GELU)


def test_positionwise_feed_forward_accepts_custom_activation() -> None:
    custom_activation = nn.ReLU()
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
        activation=custom_activation,
    )

    assert feed_forward._activation is custom_activation


def test_positionwise_feed_forward_contains_expected_submodules() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )

    assert isinstance(feed_forward._input_projection, nn.Linear)
    assert isinstance(feed_forward._output_projection, nn.Linear)
    assert isinstance(feed_forward._dropout, nn.Dropout)
    assert isinstance(feed_forward._output_dropout, nn.Dropout)


def test_positionwise_feed_forward_preserves_deterministic_behavior_when_dropout_is_zero() -> None:
    torch.manual_seed(42)
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=128,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output_1 = feed_forward(x)
    output_2 = feed_forward(x)

    assert torch.allclose(output_1, output_2)


def test_positionwise_feed_forward_output_last_dimension_matches_d_model() -> None:
    feed_forward = PositionwiseFeedForward(
        d_model=64,
        hidden_dim=256,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=3, seq_len=5, d_model=64)

    output = feed_forward(x)

    assert output.size(-1) == 64