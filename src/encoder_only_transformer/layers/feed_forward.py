from __future__ import annotations

from torch import Tensor, nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network used inside Transformer blocks.

    This module applies the same two-layer MLP independently to each token position.

    Input shape:
        x: (batch_size, seq_len, d_model)

    Output shape:
        (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self._d_model = d_model
        self._hidden_dim = hidden_dim
        self._activation = activation if activation is not None else nn.GELU()

        self._input_projection = nn.Linear(d_model, hidden_dim)
        self._dropout = nn.Dropout(p=dropout)
        self._output_projection = nn.Linear(hidden_dim, d_model)
        self._output_dropout = nn.Dropout(p=dropout)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)

        x = self._input_projection(x)
        x = self._activation(x)
        x = self._dropout(x)
        x = self._output_projection(x)
        x = self._output_dropout(x)

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