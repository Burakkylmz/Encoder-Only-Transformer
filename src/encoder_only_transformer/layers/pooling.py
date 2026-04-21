from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class BasePooling(nn.Module, ABC):
    """
    Abstract base class for sequence pooling strategies.

    Input shape:
        x: (batch_size, seq_len, d_model)

    Optional mask shape:
        padding_mask: (batch_size, seq_len)

    Output shape:
        (batch_size, d_model)
    """

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_input(x)

        if padding_mask is not None:
            self._validate_padding_mask(x=x, padding_mask=padding_mask)

        return self._pool(x=x, padding_mask=padding_mask)

    @abstractmethod
    def _pool(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x must be a 3D tensor with shape (batch_size, seq_len, d_model)."
            )

    @staticmethod
    def _validate_padding_mask(x: Tensor, padding_mask: Tensor) -> None:
        if padding_mask.ndim != 2:
            raise ValueError(
                "padding_mask must be a 2D tensor with shape (batch_size, seq_len)."
            )

        expected_shape = x.shape[:2]
        if padding_mask.shape != expected_shape:
            raise ValueError(
                f"Expected padding_mask shape {expected_shape}, but got {tuple(padding_mask.shape)}."
            )


class MeanPooling(BasePooling):
    """
    Computes mask-aware mean pooling over the sequence dimension.

    If no padding_mask is provided, averages across all token positions.
    """

    def _pool(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if padding_mask is None:
            return x.mean(dim=1)

        mask = padding_mask.to(dtype=x.dtype).unsqueeze(-1)
        masked_x = x * mask
        valid_token_counts = mask.sum(dim=1)

        if torch.any(valid_token_counts == 0):
            raise ValueError(
                "MeanPooling received at least one sequence with no valid tokens."
            )

        return masked_x.sum(dim=1) / valid_token_counts


class FirstTokenPooling(BasePooling):
    """
    Returns the representation of the first token in the sequence.

    This is commonly used in encoder-style models that rely on a special
    classification token or first-token representation.
    """

    def _pool(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        _ = padding_mask  # kept for interface consistency
        return x[:, 0, :]


class MaxPooling(BasePooling):
    """
    Computes mask-aware max pooling over the sequence dimension.

    If no padding_mask is provided, takes the maximum across all token positions.
    """

    def _pool(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        if padding_mask is None:
            return x.max(dim=1).values

        if torch.any(padding_mask.sum(dim=1) == 0):
            raise ValueError(
                "MaxPooling received at least one sequence with no valid tokens."
            )

        mask = padding_mask.unsqueeze(-1).to(dtype=torch.bool)
        masked_x = x.masked_fill(~mask, float("-inf"))

        return masked_x.max(dim=1).values