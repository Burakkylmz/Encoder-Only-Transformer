from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from encoder_only_transformer.config.config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Converts token ids into dense vector representations.

    Input shape:
        (batch_size, seq_len)

    Output shape:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")

        self._d_model = d_model
        self._embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be a 2D tensor with shape (batch_size, seq_len)."
            )

        token_embeddings = self._embedding(input_ids)
        return token_embeddings * math.sqrt(self._d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal positional information to token embeddings.

    Input shape:
        (batch_size, seq_len, d_model)

    Output shape:
        (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self._d_model = d_model
        self._max_seq_len = max_seq_len
        self._dropout = nn.Dropout(p=dropout)

        positional_encoding = self._build_positional_encoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
        )

        self.register_buffer(
            name="_positional_encoding",
            tensor=positional_encoding,
            persistent=False,
        )

    @staticmethod
    def _build_positional_encoding(d_model: int, max_seq_len: int) -> Tensor:
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        positional_encoding = torch.zeros(
            max_seq_len,
            d_model,
            dtype=torch.float32,
        )

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        return positional_encoding.unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Input tensor must be 3D with shape (batch_size, seq_len, d_model)."
            )

        batch_size, seq_len, d_model = x.shape

        if d_model != self._d_model:
            raise ValueError(
                f"Expected input embedding dimension {self._d_model}, but got {d_model}."
            )

        if seq_len > self._max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len {self._max_seq_len}."
            )

        positional_slice = self._positional_encoding[:, :seq_len, :]
        output = x + positional_slice

        return self._dropout(output)


class EncoderInputEmbedding(nn.Module):
    """
    Composes token embeddings and positional encoding into a single input representation layer.

    Input shape:
        (batch_size, seq_len)

    Output shape:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self._token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )
        self._positional_encoding = SinusoidalPositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        token_embeddings = self._token_embedding(input_ids)
        return self._positional_encoding(token_embeddings)