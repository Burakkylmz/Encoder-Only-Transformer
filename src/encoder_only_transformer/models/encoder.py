from __future__ import annotations

from torch import Tensor, nn

from encoder_only_transformer.blocks.encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    Stacked Transformer encoder composed of multiple EncoderBlock instances.

    Input shape:
        x: (batch_size, seq_len, d_model)

    Output:
        output:               (batch_size, seq_len, d_model)
        all_attention_weights:
            list[Tensor] where each tensor has shape
            (batch_size, n_heads, seq_len, seq_len)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_hidden_dim: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if n_heads <= 0:
            raise ValueError("n_heads must be greater than 0.")
        if ff_hidden_dim <= 0:
            raise ValueError("ff_hidden_dim must be greater than 0.")
        if n_layers <= 0:
            raise ValueError("n_layers must be greater than 0.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self._d_model = d_model
        self._n_heads = n_heads
        self._ff_hidden_dim = ff_hidden_dim
        self._n_layers = n_layers

        self._layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def ff_hidden_dim(self) -> int:
        return self._ff_hidden_dim

    @property
    def n_layers(self) -> int:
        return self._n_layers

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        self._validate_input(x)

        attention_weights_per_layer: list[Tensor] = []

        for layer in self._layers:
            x, attention_weights = layer(x, attention_mask=attention_mask)
            attention_weights_per_layer.append(attention_weights)

        return x, attention_weights_per_layer

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x must be a 3D tensor with shape (batch_size, seq_len, d_model)."
            )

        if x.size(-1) != self._d_model:
            raise ValueError(
                f"Expected input embedding dimension {self._d_model}, but got {x.size(-1)}."
            )