from __future__ import annotations

import pytest
import torch

from encoder_only_transformer.config.config import ModelConfig
from encoder_only_transformer.models.model import (
    EncoderForSequenceClassification,
    SequenceClassificationOutput,
)
from encoder_only_transformer.layers.pooling import FirstTokenPooling, MeanPooling


def build_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=100,
        max_seq_len=16,
        d_model=32,
        n_heads=4,
        ff_hidden_dim=64,
        dropout=0.0,
        n_layers=2,
        pad_token_id=0,
    )


def build_input_ids(
    batch_size: int = 2,
    seq_len: int = 6,
    vocab_size: int = 100,
) -> torch.Tensor:
    return torch.randint(1, vocab_size, (batch_size, seq_len))


def test_encoder_for_sequence_classification_returns_expected_output_structure() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)
    input_ids = build_input_ids(batch_size=2, seq_len=6, vocab_size=config.vocab_size)

    output = model(input_ids)

    assert isinstance(output, SequenceClassificationOutput)
    assert output.logits.shape == (2, 3)
    assert len(output.attention_weights_per_layer) == config.n_layers


def test_encoder_for_sequence_classification_returns_attention_weights_for_each_layer() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)
    input_ids = build_input_ids(batch_size=2, seq_len=6, vocab_size=config.vocab_size)

    output = model(input_ids)

    assert len(output.attention_weights_per_layer) == 2
    for attention_weights in output.attention_weights_per_layer:
        assert attention_weights.shape == (2, 4, 6, 6)


def test_encoder_for_sequence_classification_returns_finite_logits() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=4)
    input_ids = build_input_ids(batch_size=2, seq_len=5, vocab_size=config.vocab_size)

    output = model(input_ids)

    assert torch.isfinite(output.logits).all()


def test_encoder_for_sequence_classification_raises_error_for_non_2d_input_ids() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)
    invalid_input_ids = torch.randint(1, config.vocab_size, (2, 5, 1))

    with pytest.raises(
        ValueError,
        match="input_ids must be a 2D tensor with shape",
    ):
        model(invalid_input_ids)


def test_encoder_for_sequence_classification_raises_error_for_non_2d_padding_mask() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)
    input_ids = build_input_ids(batch_size=2, seq_len=5, vocab_size=config.vocab_size)
    invalid_padding_mask = torch.ones(2, 5, 1)

    with pytest.raises(
        ValueError,
        match="padding_mask must be a 2D tensor with shape",
    ):
        model(input_ids, padding_mask=invalid_padding_mask)


def test_encoder_for_sequence_classification_raises_error_for_padding_mask_shape_mismatch() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)
    input_ids = build_input_ids(batch_size=2, seq_len=5, vocab_size=config.vocab_size)
    invalid_padding_mask = torch.ones(2, 4)

    with pytest.raises(
        ValueError,
        match="Expected padding_mask shape",
    ):
        model(input_ids, padding_mask=invalid_padding_mask)


def test_encoder_for_sequence_classification_uses_default_padding_mask_from_pad_token_id() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)

    input_ids = torch.tensor(
        [
            [5, 7, 9, 0, 0],
            [3, 4, 8, 6, 0],
        ]
    )

    output = model(input_ids)

    assert output.logits.shape == (2, 3)
    assert len(output.attention_weights_per_layer) == config.n_layers


def test_encoder_for_sequence_classification_supports_explicit_padding_mask() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)

    input_ids = torch.tensor(
        [
            [5, 7, 9, 11, 13],
            [3, 4, 8, 6, 2],
        ]
    )
    padding_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ]
    )

    output = model(input_ids, padding_mask=padding_mask)

    assert output.logits.shape == (2, 3)
    assert len(output.attention_weights_per_layer) == config.n_layers


def test_encoder_for_sequence_classification_exposes_expected_properties() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=5)

    assert model.config == config
    assert model.num_classes == 5


def test_encoder_for_sequence_classification_accepts_custom_pooling() -> None:
    config = build_model_config()
    pooling = FirstTokenPooling()
    model = EncoderForSequenceClassification(
        config=config,
        num_classes=3,
        pooling=pooling,
    )
    input_ids = build_input_ids(batch_size=2, seq_len=6, vocab_size=config.vocab_size)

    output = model(input_ids)

    assert output.logits.shape == (2, 3)
    assert model._classification_head.pooling is pooling


def test_encoder_for_sequence_classification_uses_mean_pooling_by_default() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)

    assert isinstance(model._classification_head.pooling, MeanPooling)


def test_encoder_for_sequence_classification_builds_expected_attention_mask_shape() -> None:
    padding_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ]
    )

    attention_mask = EncoderForSequenceClassification._build_attention_mask(padding_mask)

    assert attention_mask.shape == (2, 1, 1, 4)


def test_encoder_for_sequence_classification_preserves_deterministic_behavior_when_dropout_is_zero() -> None:
    torch.manual_seed(42)

    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=3)
    input_ids = build_input_ids(batch_size=2, seq_len=6, vocab_size=config.vocab_size)

    output_1 = model(input_ids)
    output_2 = model(input_ids)

    assert torch.allclose(output_1.logits, output_2.logits)

    assert len(output_1.attention_weights_per_layer) == len(output_2.attention_weights_per_layer)
    for weights_1, weights_2 in zip(
        output_1.attention_weights_per_layer,
        output_2.attention_weights_per_layer,
    ):
        assert torch.allclose(weights_1, weights_2)


def test_encoder_for_sequence_classification_output_last_dimension_matches_num_classes() -> None:
    config = build_model_config()
    model = EncoderForSequenceClassification(config=config, num_classes=7)
    input_ids = build_input_ids(batch_size=3, seq_len=5, vocab_size=config.vocab_size)

    output = model(input_ids)

    assert output.logits.size(-1) == 7