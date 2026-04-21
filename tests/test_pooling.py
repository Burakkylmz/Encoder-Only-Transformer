from __future__ import annotations

import pytest
import torch

from encoder_only_transformer.layers.pooling import (
    FirstTokenPooling,
    MaxPooling,
    MeanPooling,
)


def build_input_tensor(
    batch_size: int = 2,
    seq_len: int = 4,
    d_model: int = 3,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, d_model)


def test_mean_pooling_returns_expected_shape_without_mask() -> None:
    pooling = MeanPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)

    output = pooling(x)

    assert output.shape == (2, 3)


def test_mean_pooling_matches_manual_mean_without_mask() -> None:
    pooling = MeanPooling()
    x = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    output = pooling(x)
    expected = torch.tensor([[3.0, 4.0]])

    assert torch.allclose(output, expected)


def test_mean_pooling_matches_masked_mean_with_padding_mask() -> None:
    pooling = MeanPooling()
    x = torch.tensor(
        [
            [
                [1.0, 10.0],
                [3.0, 30.0],
                [100.0, 1000.0],
            ]
        ]
    )
    padding_mask = torch.tensor([[1, 1, 0]])

    output = pooling(x, padding_mask=padding_mask)
    expected = torch.tensor([[2.0, 20.0]])

    assert torch.allclose(output, expected)


def test_mean_pooling_raises_error_when_all_tokens_are_masked() -> None:
    pooling = MeanPooling()
    x = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    padding_mask = torch.tensor([[0, 0]])

    with pytest.raises(
        ValueError,
        match="no valid tokens",
    ):
        pooling(x, padding_mask=padding_mask)


def test_first_token_pooling_returns_expected_shape() -> None:
    pooling = FirstTokenPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)

    output = pooling(x)

    assert output.shape == (2, 3)


def test_first_token_pooling_returns_first_token_representation() -> None:
    pooling = FirstTokenPooling()
    x = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )

    output = pooling(x)
    expected = torch.tensor([[1.0, 2.0]])

    assert torch.allclose(output, expected)


def test_first_token_pooling_ignores_padding_mask() -> None:
    pooling = FirstTokenPooling()
    x = torch.tensor(
        [
            [
                [7.0, 8.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ]
    )
    padding_mask = torch.tensor([[0, 1, 1]])

    output = pooling(x, padding_mask=padding_mask)
    expected = torch.tensor([[7.0, 8.0]])

    assert torch.allclose(output, expected)


def test_max_pooling_returns_expected_shape_without_mask() -> None:
    pooling = MaxPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)

    output = pooling(x)

    assert output.shape == (2, 3)


def test_max_pooling_matches_manual_max_without_mask() -> None:
    pooling = MaxPooling()
    x = torch.tensor(
        [
            [
                [1.0, 2.0],
                [7.0, 1.0],
                [5.0, 9.0],
            ]
        ]
    )

    output = pooling(x)
    expected = torch.tensor([[7.0, 9.0]])

    assert torch.allclose(output, expected)


def test_max_pooling_matches_masked_max_with_padding_mask() -> None:
    pooling = MaxPooling()
    x = torch.tensor(
        [
            [
                [1.0, 10.0],
                [7.0, 30.0],
                [100.0, 1000.0],
            ]
        ]
    )
    padding_mask = torch.tensor([[1, 1, 0]])

    output = pooling(x, padding_mask=padding_mask)
    expected = torch.tensor([[7.0, 30.0]])

    assert torch.allclose(output, expected)


def test_max_pooling_raises_error_when_all_tokens_are_masked() -> None:
    pooling = MaxPooling()
    x = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ]
    )
    padding_mask = torch.tensor([[0, 0]])

    with pytest.raises(
        ValueError,
        match="no valid tokens",
    ):
        pooling(x, padding_mask=padding_mask)


def test_pooling_raises_error_for_non_3d_input() -> None:
    pooling = MeanPooling()
    invalid_x = torch.randn(2, 4)

    with pytest.raises(
        ValueError,
        match="x must be a 3D tensor",
    ):
        pooling(invalid_x)


def test_pooling_raises_error_for_non_2d_padding_mask() -> None:
    pooling = MeanPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)
    invalid_padding_mask = torch.ones(2, 4, 1)

    with pytest.raises(
        ValueError,
        match="padding_mask must be a 2D tensor",
    ):
        pooling(x, padding_mask=invalid_padding_mask)


def test_pooling_raises_error_for_padding_mask_shape_mismatch() -> None:
    pooling = MeanPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)
    invalid_padding_mask = torch.ones(2, 3)

    with pytest.raises(
        ValueError,
        match="Expected padding_mask shape",
    ):
        pooling(x, padding_mask=invalid_padding_mask)


def test_mean_pooling_returns_finite_outputs() -> None:
    pooling = MeanPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)

    output = pooling(x)

    assert torch.isfinite(output).all()


def test_max_pooling_returns_finite_outputs() -> None:
    pooling = MaxPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)

    output = pooling(x)

    assert torch.isfinite(output).all()


def test_first_token_pooling_returns_finite_outputs() -> None:
    pooling = FirstTokenPooling()
    x = build_input_tensor(batch_size=2, seq_len=4, d_model=3)

    output = pooling(x)

    assert torch.isfinite(output).all()