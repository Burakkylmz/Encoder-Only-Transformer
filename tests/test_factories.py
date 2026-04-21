from __future__ import annotations

import pytest

from encoder_only_transformer.config.config import (
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from encoder_only_transformer.factories.factories import ModelFactory
from encoder_only_transformer.models.model import EncoderForSequenceClassification
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


def build_training_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        epochs=5,
    )


def build_project_config() -> ProjectConfig:
    return ProjectConfig(
        model=build_model_config(),
        training=build_training_config(),
    )


def test_create_sequence_classification_model_returns_expected_instance() -> None:
    model_config = build_model_config()

    model = ModelFactory.create_sequence_classification_model(
        model_config=model_config,
        num_classes=3,
    )

    assert isinstance(model, EncoderForSequenceClassification)


def test_create_sequence_classification_model_uses_given_model_config() -> None:
    model_config = build_model_config()

    model = ModelFactory.create_sequence_classification_model(
        model_config=model_config,
        num_classes=3,
    )

    assert model.config == model_config


def test_create_sequence_classification_model_uses_given_num_classes() -> None:
    model_config = build_model_config()

    model = ModelFactory.create_sequence_classification_model(
        model_config=model_config,
        num_classes=7,
    )

    assert model.num_classes == 7


def test_create_sequence_classification_model_uses_mean_pooling_by_default() -> None:
    model_config = build_model_config()

    model = ModelFactory.create_sequence_classification_model(
        model_config=model_config,
        num_classes=3,
    )

    assert isinstance(model._classification_head.pooling, MeanPooling)


def test_create_sequence_classification_model_accepts_custom_pooling() -> None:
    model_config = build_model_config()
    pooling = FirstTokenPooling()

    model = ModelFactory.create_sequence_classification_model(
        model_config=model_config,
        num_classes=3,
        pooling=pooling,
    )

    assert model._classification_head.pooling is pooling


def test_create_sequence_classification_model_raises_type_error_for_invalid_model_config() -> None:
    with pytest.raises(
        TypeError,
        match="model_config must be an instance of ModelConfig",
    ):
        ModelFactory.create_sequence_classification_model(
            model_config="invalid",  # type: ignore[arg-type]
            num_classes=3,
        )


def test_create_sequence_classification_model_from_project_config_returns_expected_instance() -> None:
    project_config = build_project_config()

    model = ModelFactory.create_sequence_classification_model_from_project_config(
        project_config=project_config,
        num_classes=3,
    )

    assert isinstance(model, EncoderForSequenceClassification)


def test_create_sequence_classification_model_from_project_config_uses_project_model_config() -> None:
    project_config = build_project_config()

    model = ModelFactory.create_sequence_classification_model_from_project_config(
        project_config=project_config,
        num_classes=3,
    )

    assert model.config == project_config.model


def test_create_sequence_classification_model_from_project_config_uses_given_num_classes() -> None:
    project_config = build_project_config()

    model = ModelFactory.create_sequence_classification_model_from_project_config(
        project_config=project_config,
        num_classes=5,
    )

    assert model.num_classes == 5


def test_create_sequence_classification_model_from_project_config_accepts_custom_pooling() -> None:
    project_config = build_project_config()
    pooling = FirstTokenPooling()

    model = ModelFactory.create_sequence_classification_model_from_project_config(
        project_config=project_config,
        num_classes=3,
        pooling=pooling,
    )

    assert model._classification_head.pooling is pooling


def test_create_sequence_classification_model_from_project_config_raises_type_error_for_invalid_project_config() -> None:
    with pytest.raises(
        TypeError,
        match="project_config must be an instance of ProjectConfig",
    ):
        ModelFactory.create_sequence_classification_model_from_project_config(
            project_config="invalid",  # type: ignore[arg-type]
            num_classes=3,
        )