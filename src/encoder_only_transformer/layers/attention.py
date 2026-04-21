from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ScaledDotProductAttention(nn.Module):
    """
    Computes scaled dot-product attention.

    Expected shapes:
        query: (batch_size, n_heads, seq_len, head_dim)
        key:   (batch_size, n_heads, seq_len, head_dim)
        value: (batch_size, n_heads, seq_len, head_dim)

    Output:
        context:           (batch_size, n_heads, seq_len, head_dim)
        attention_weights: (batch_size, n_heads, seq_len, seq_len)
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self._dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_inputs(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        head_dim = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        if attention_mask is not None:
            prepared_mask = self._prepare_attention_mask(attention_mask)
            attention_scores = attention_scores.masked_fill(prepared_mask == 0, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self._dropout(attention_weights)

        context = torch.matmul(attention_weights, value)

        return context, attention_weights

    @staticmethod
    def _prepare_attention_mask(attention_mask: Tensor) -> Tensor:
        """
        Normalizes attention mask shape for broadcasting.

        Accepted input shapes:
            3D: (batch_size, seq_len, seq_len)
            4D: (batch_size, 1 or n_heads, seq_len, seq_len)

        Returned shape:
            4D: (batch_size, 1 or n_heads, seq_len, seq_len)
        """
        if attention_mask.ndim == 3:
            return attention_mask.unsqueeze(1)

        return attention_mask

    @staticmethod
    def _validate_inputs(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
    ) -> None:
        if query.ndim != 4:
            raise ValueError("query must be a 4D tensor.")
        if key.ndim != 4:
            raise ValueError("key must be a 4D tensor.")
        if value.ndim != 4:
            raise ValueError("value must be a 4D tensor.")

        if query.shape != key.shape:
            raise ValueError("query and key must have the same shape.")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape.")

        if query.size(-1) <= 0:
            raise ValueError("head_dim must be greater than 0.")

        if attention_mask is not None and attention_mask.ndim not in (3, 4):
            raise ValueError("attention_mask must be a 3D or 4D tensor.")

class SelfAttention(nn.Module):
    """
    Multi-head self-attention layer built on top of scaled dot-product attention.

    Input shape:
        x: (batch_size, seq_len, d_model)

    Output:
        output:            (batch_size, seq_len, d_model)
        attention_weights: (batch_size, n_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if n_heads <= 0:
            raise ValueError("n_heads must be greater than 0.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self._d_model = d_model
        self._n_heads = n_heads
        self._head_dim = d_model // n_heads

        self._query_projection = nn.Linear(d_model, d_model)
        self._key_projection = nn.Linear(d_model, d_model)
        self._value_projection = nn.Linear(d_model, d_model)
        self._output_projection = nn.Linear(d_model, d_model)

        self._attention = ScaledDotProductAttention(dropout=dropout)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_input(x)

        query = self._query_projection(x)
        key = self._key_projection(x)
        value = self._value_projection(x)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        context, attention_weights = self._attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        combined_context = self._combine_heads(context)
        output = self._output_projection(combined_context)

        return output, attention_weights

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        Converts:
            (batch_size, seq_len, d_model)
        into:
            (batch_size, n_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape

        x = x.view(batch_size, seq_len, self._n_heads, self._head_dim)
        x = x.transpose(1, 2)

        return x

    def _combine_heads(self, x: Tensor) -> Tensor:
        """
        Converts:
            (batch_size, n_heads, seq_len, head_dim)
        into:
            (batch_size, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, head_dim = x.shape

        if n_heads != self._n_heads:
            raise ValueError(
                f"Expected {self._n_heads} heads, but got {n_heads}."
            )
        if head_dim != self._head_dim:
            raise ValueError(
                f"Expected head_dim {self._head_dim}, but got {head_dim}."
            )

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, self._d_model)

        return x

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x must be a 3D tensor with shape (batch_size, seq_len, d_model)."
            )

        if x.size(-1) != self._d_model:
            raise ValueError(
                f"Expected input embedding dimension {self._d_model}, but got {x.size(-1)}."
            )