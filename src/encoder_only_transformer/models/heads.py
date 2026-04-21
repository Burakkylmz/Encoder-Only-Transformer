from __future__ import annotations

from torch import Tensor, nn

from encoder_only_transformer.layers.pooling import BasePooling, MeanPooling


class SequenceClassificationHead(nn.Module):
    """
    Classification head for encoder outputs.

    Pipeline:
        1. Pool sequence representations into a single vector
        2. Apply dropout
        3. Project into class logits

    Input shape:
        x: (batch_size, seq_len, d_model)

    Optional padding mask shape:
        padding_mask: (batch_size, seq_len)

    Output shape:
        logits: (batch_size, num_classes)
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        pooling: BasePooling | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self._d_model = d_model
        self._num_classes = num_classes
        self._pooling = pooling if pooling is not None else MeanPooling()

        self._dropout = nn.Dropout(p=dropout)
        self._classifier = nn.Linear(d_model, num_classes)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def pooling(self) -> BasePooling:
        return self._pooling

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        self._validate_input(x)

        pooled_representation = self._pooling(x, padding_mask=padding_mask)
        pooled_representation = self._dropout(pooled_representation)

        return self._classifier(pooled_representation)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x must be a 3D tensor with shape (batch_size, seq_len, d_model)."
            )

        if x.size(-1) != self._d_model:
            raise ValueError(
                f"Expected input embedding dimension {self._d_model}, but got {x.size(-1)}."
            )