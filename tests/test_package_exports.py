from __future__ import annotations

import encoder_only_transformer as eot

from encoder_only_transformer.layers.attention import (
    ScaledDotProductAttention,
    SelfAttention,
)
from encoder_only_transformer.config.config import (
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    YamlConfigLoader,
    load_config,
)
from encoder_only_transformer.layers.embeddings import (
    EncoderInputEmbedding,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)
from encoder_only_transformer.models.encoder import Encoder
from encoder_only_transformer.blocks.encoder_block import EncoderBlock
from encoder_only_transformer.factories.factories import ModelFactory
from encoder_only_transformer.layers.feed_forward import PositionwiseFeedForward
from encoder_only_transformer.models.heads import SequenceClassificationHead
from encoder_only_transformer.models.model import (
    EncoderForSequenceClassification,
    SequenceClassificationOutput,
)
from encoder_only_transformer.layers.pooling import (
    BasePooling,
    FirstTokenPooling,
    MaxPooling,
    MeanPooling,
)


def test_package_exports_expected_public_symbols_in___all__() -> None:
    expected_exports = {
        "ConfigError",
        "ConfigFileError",
        "ConfigValidationError",
        "ModelConfig",
        "ProjectConfig",
        "TrainingConfig",
        "YamlConfigLoader",
        "load_config",
        "TokenEmbedding",
        "SinusoidalPositionalEncoding",
        "EncoderInputEmbedding",
        "ScaledDotProductAttention",
        "SelfAttention",
        "PositionwiseFeedForward",
        "EncoderBlock",
        "Encoder",
        "BasePooling",
        "MeanPooling",
        "FirstTokenPooling",
        "MaxPooling",
        "SequenceClassificationHead",
        "SequenceClassificationOutput",
        "EncoderForSequenceClassification",
        "ModelFactory",
    }

    assert set(eot.__all__) == expected_exports


def test_package_exports_config_symbols() -> None:
    assert eot.ConfigError is ConfigError
    assert eot.ConfigFileError is ConfigFileError
    assert eot.ConfigValidationError is ConfigValidationError
    assert eot.ModelConfig is ModelConfig
    assert eot.ProjectConfig is ProjectConfig
    assert eot.TrainingConfig is TrainingConfig
    assert eot.YamlConfigLoader is YamlConfigLoader
    assert eot.load_config is load_config


def test_package_exports_embedding_symbols() -> None:
    assert eot.TokenEmbedding is TokenEmbedding
    assert eot.SinusoidalPositionalEncoding is SinusoidalPositionalEncoding
    assert eot.EncoderInputEmbedding is EncoderInputEmbedding


def test_package_exports_attention_symbols() -> None:
    assert eot.ScaledDotProductAttention is ScaledDotProductAttention
    assert eot.SelfAttention is SelfAttention


def test_package_exports_encoder_symbols() -> None:
    assert eot.PositionwiseFeedForward is PositionwiseFeedForward
    assert eot.EncoderBlock is EncoderBlock
    assert eot.Encoder is Encoder


def test_package_exports_pooling_symbols() -> None:
    assert eot.BasePooling is BasePooling
    assert eot.MeanPooling is MeanPooling
    assert eot.FirstTokenPooling is FirstTokenPooling
    assert eot.MaxPooling is MaxPooling


def test_package_exports_head_and_model_symbols() -> None:
    assert eot.SequenceClassificationHead is SequenceClassificationHead
    assert eot.SequenceClassificationOutput is SequenceClassificationOutput
    assert eot.EncoderForSequenceClassification is EncoderForSequenceClassification


def test_package_exports_factory_symbol() -> None:
    assert eot.ModelFactory is ModelFactory