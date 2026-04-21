from __future__ import annotations

import torch
import pytest

from encoder_only_transformer.config.config import ModelConfig
from encoder_only_transformer.layers.embeddings import (
    EncoderInputEmbedding,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)


def build_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=1000,
        max_seq_len=32,
        d_model=64,
        n_heads=4,
        ff_hidden_dim=128,
        dropout=0.1,
        n_layers=2,
        pad_token_id=0,
    )


def test_token_embedding_returns_expected_shape() -> None:
    embedding = TokenEmbedding(vocab_size=1000, d_model=64)
    input_ids = torch.randint(0, 1000, (2, 10))

    output = embedding(input_ids)

    assert output.shape == (2, 10, 64)


def test_token_embedding_raises_error_for_non_2d_input() -> None:
    embedding = TokenEmbedding(vocab_size=1000, d_model=64)
    invalid_input_ids = torch.randint(0, 1000, (2, 10, 3))

    with pytest.raises(
        ValueError,
        match="input_ids must be a 2D tensor",
    ):
        embedding(invalid_input_ids)


def test_positional_encoding_preserves_input_shape() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        d_model=64,
        max_seq_len=32,
        dropout=0.0,
    )
    x = torch.randn(2, 10, 64)

    output = positional_encoding(x)

    assert output.shape == (2, 10, 64)


def test_positional_encoding_changes_zero_input() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        d_model=64,
        max_seq_len=32,
        dropout=0.0,
    )
    x = torch.zeros(2, 10, 64)

    output = positional_encoding(x)

    assert not torch.allclose(output, x)


def test_positional_encoding_raises_error_for_non_3d_input() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        d_model=64,
        max_seq_len=32,
        dropout=0.0,
    )
    invalid_x = torch.randn(2, 10)

    with pytest.raises(
        ValueError,
        match="Input tensor must be 3D",
    ):
        positional_encoding(invalid_x)


def test_positional_encoding_raises_error_for_embedding_dimension_mismatch() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        d_model=64,
        max_seq_len=32,
        dropout=0.0,
    )
    x = torch.randn(2, 10, 32)

    with pytest.raises(
        ValueError,
        match="Expected input embedding dimension 64, but got 32",
    ):
        positional_encoding(x)


def test_positional_encoding_raises_error_when_seq_len_exceeds_max_seq_len() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        d_model=64,
        max_seq_len=8,
        dropout=0.0,
    )
    x = torch.randn(2, 10, 64)

    with pytest.raises(
        ValueError,
        match="exceeds max_seq_len 8",
    ):
        positional_encoding(x)


def test_positional_encoding_registers_buffer_not_parameter() -> None:
    positional_encoding = SinusoidalPositionalEncoding(
        d_model=64,
        max_seq_len=32,
        dropout=0.0,
    )

    named_buffers = dict(positional_encoding.named_buffers())
    named_parameters = dict(positional_encoding.named_parameters())

    assert "_positional_encoding" in named_buffers
    assert "_positional_encoding" not in named_parameters


def test_encoder_input_embedding_returns_expected_shape() -> None:
    config = build_model_config()
    embedding_layer = EncoderInputEmbedding(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 16))

    output = embedding_layer(input_ids)

    assert output.shape == (4, 16, config.d_model)


def test_encoder_input_embedding_contains_expected_submodules() -> None:
    config = build_model_config()
    embedding_layer = EncoderInputEmbedding(config)

    assert isinstance(embedding_layer._token_embedding, TokenEmbedding)
    assert isinstance(
        embedding_layer._positional_encoding,
        SinusoidalPositionalEncoding,
    )


def test_encoder_input_embedding_raises_error_for_non_2d_input() -> None:
    config = build_model_config()
    embedding_layer = EncoderInputEmbedding(config)
    invalid_input_ids = torch.randint(0, config.vocab_size, (2, 8, 3))

    with pytest.raises(
        ValueError,
        match="input_ids must be a 2D tensor",
    ):
        embedding_layer(invalid_input_ids)