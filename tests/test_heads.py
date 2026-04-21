from __future__ import annotations

import pytest
import torch

from encoder_only_transformer.models.heads import SequenceClassificationHead
from encoder_only_transformer.layers.pooling import (
    FirstTokenPooling,
    MeanPooling,
    MaxPooling,
)


def build_input_tensor(
    batch_size: int = 2,
    seq_len: int = 5,
    d_model: int = 64,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, d_model)


def test_sequence_classification_head_returns_expected_shape() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)

    logits = head(x)

    assert logits.shape == (2, 3)


def test_sequence_classification_head_returns_finite_outputs() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)

    logits = head(x)

    assert torch.isfinite(logits).all()


def test_sequence_classification_head_uses_mean_pooling_by_default() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        dropout=0.0,
    )

    assert isinstance(head.pooling, MeanPooling)


def test_sequence_classification_head_accepts_first_token_pooling() -> None:
    pooling = FirstTokenPooling()
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        pooling=pooling,
        dropout=0.0,
    )

    assert head.pooling is pooling


def test_sequence_classification_head_accepts_max_pooling() -> None:
    pooling = MaxPooling()
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        pooling=pooling,
        dropout=0.0,
    )

    assert head.pooling is pooling


def test_sequence_classification_head_raises_error_for_non_3d_input() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 5)

    with pytest.raises(
        ValueError,
        match="x must be a 3D tensor with shape",
    ):
        head(invalid_x)


def test_sequence_classification_head_raises_error_for_embedding_dimension_mismatch() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 5, 32)

    with pytest.raises(
        ValueError,
        match="Expected input embedding dimension 64, but got 32",
    ):
        head(invalid_x)


def test_sequence_classification_head_supports_padding_mask() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        pooling=MeanPooling(),
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)
    padding_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ]
    )

    logits = head(x, padding_mask=padding_mask)

    assert logits.shape == (2, 3)


def test_sequence_classification_head_exposes_expected_properties() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=4,
        pooling=MeanPooling(),
        dropout=0.0,
    )

    assert head.d_model == 64
    assert head.num_classes == 4
    assert isinstance(head.pooling, MeanPooling)


def test_sequence_classification_head_preserves_deterministic_behavior_when_dropout_is_zero() -> None:
    torch.manual_seed(42)
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=3,
        pooling=MeanPooling(),
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=2, seq_len=5, d_model=64)

    logits_1 = head(x)
    logits_2 = head(x)

    assert torch.allclose(logits_1, logits_2)


def test_sequence_classification_head_output_last_dimension_matches_num_classes() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=7,
        pooling=MeanPooling(),
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=3, seq_len=6, d_model=64)

    logits = head(x)

    assert logits.size(-1) == 7


def test_sequence_classification_head_with_first_token_pooling_returns_expected_shape() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=2,
        pooling=FirstTokenPooling(),
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=4, seq_len=8, d_model=64)

    logits = head(x)

    assert logits.shape == (4, 2)


def test_sequence_classification_head_with_max_pooling_returns_expected_shape() -> None:
    head = SequenceClassificationHead(
        d_model=64,
        num_classes=5,
        pooling=MaxPooling(),
        dropout=0.0,
    )
    x = build_input_tensor(batch_size=3, seq_len=7, d_model=64)

    logits = head(x)

    assert logits.shape == (3, 5)