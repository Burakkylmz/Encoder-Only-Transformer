from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from encoder_only_transformer.config.config import ModelConfig
from encoder_only_transformer.layers.embeddings import EncoderInputEmbedding
from encoder_only_transformer.layers.pooling import BasePooling
from encoder_only_transformer.models.encoder import Encoder
from encoder_only_transformer.models.heads import SequenceClassificationHead


@dataclass(frozen=True, slots=True)
class SequenceClassificationOutput:
    logits: Tensor
    attention_weights_per_layer: list[Tensor]


class EncoderForSequenceClassification(nn.Module):
    """
    Full encoder-only model for sequence classification.

    Pipeline:
        1. Token Embedding + Positional Encoding
        2. Stacked Encoder
        3. Sequence Pooling + Classification Head

    Input shape:
        input_ids: (batch_size, seq_len)

    Optional padding mask shape:
        padding_mask: (batch_size, seq_len)

    Output:
        SequenceClassificationOutput
            logits: (batch_size, num_classes)
            attention_weights_per_layer:
                list[Tensor] where each tensor has shape
                (batch_size, n_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        config: ModelConfig,
        num_classes: int,
        pooling: BasePooling | None = None,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0.")

        self._config = config
        self._num_classes = num_classes

        self._input_embedding = EncoderInputEmbedding(config)
        self._encoder = Encoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            ff_hidden_dim=config.ff_hidden_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
        self._classification_head = SequenceClassificationHead(
            d_model=config.d_model,
            num_classes=num_classes,
            pooling=pooling,
            dropout=config.dropout,
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(
        self,
        input_ids: Tensor,
        padding_mask: Tensor | None = None,
    ) -> SequenceClassificationOutput:
        self._validate_input_ids(input_ids)

        effective_padding_mask = self._resolve_padding_mask(
            input_ids=input_ids,
            padding_mask=padding_mask,
        )
        attention_mask = self._build_attention_mask(effective_padding_mask)

        embedded = self._input_embedding(input_ids)
        encoded, attention_weights_per_layer = self._encoder(
            embedded,
            attention_mask=attention_mask,
        )
        logits = self._classification_head(
            encoded,
            padding_mask=effective_padding_mask,
        )

        return SequenceClassificationOutput(
            logits=logits,
            attention_weights_per_layer=attention_weights_per_layer,
        )

    def _resolve_padding_mask(
        self,
        input_ids: Tensor,
        padding_mask: Tensor | None,
    ) -> Tensor:
        if padding_mask is None:
            return (input_ids != self._config.pad_token_id).to(dtype=torch.long)

        self._validate_padding_mask(input_ids=input_ids, padding_mask=padding_mask)
        return padding_mask.to(dtype=torch.long)

    @staticmethod
    def _build_attention_mask(padding_mask: Tensor) -> Tensor:
        """
        Converts a 2D padding mask of shape:
            (batch_size, seq_len)

        into a 4D attention mask of shape:
            (batch_size, 1, 1, seq_len)

        This mask is broadcast across attention heads and query positions.
        """
        return padding_mask.unsqueeze(1).unsqueeze(2)

    @staticmethod
    def _validate_input_ids(input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be a 2D tensor with shape (batch_size, seq_len)."
            )

    @staticmethod
    def _validate_padding_mask(input_ids: Tensor, padding_mask: Tensor) -> None:
        if padding_mask.ndim != 2:
            raise ValueError(
                "padding_mask must be a 2D tensor with shape (batch_size, seq_len)."
            )

        if padding_mask.shape != input_ids.shape:
            raise ValueError(
                f"Expected padding_mask shape {tuple(input_ids.shape)}, but got {tuple(padding_mask.shape)}."
            )