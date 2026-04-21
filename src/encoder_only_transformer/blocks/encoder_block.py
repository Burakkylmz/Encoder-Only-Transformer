from __future__ import annotations

from torch import Tensor, nn

from encoder_only_transformer.layers.attention import SelfAttention
from encoder_only_transformer.layers.feed_forward import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    """
    A single Transformer encoder block.

    Structure:
        1. Self-Attention
        2. Residual Connection + LayerNorm
        3. Position-wise Feed Forward
        4. Residual Connection + LayerNorm

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
        ff_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if n_heads <= 0:
            raise ValueError("n_heads must be greater than 0.")
        if ff_hidden_dim <= 0:
            raise ValueError("ff_hidden_dim must be greater than 0.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self._d_model = d_model
        self._n_heads = n_heads
        self._ff_hidden_dim = ff_hidden_dim

        self._self_attention = SelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self._attention_dropout = nn.Dropout(p=dropout)
        self._attention_norm = nn.LayerNorm(d_model)

        self._feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            hidden_dim=ff_hidden_dim,
            dropout=dropout,
        )
        self._feed_forward_dropout = nn.Dropout(p=dropout)
        self._feed_forward_norm = nn.LayerNorm(d_model)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @property
    def ff_hidden_dim(self) -> int:
        return self._ff_hidden_dim

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_input(x)

        attention_output, attention_weights = self._self_attention(
            x=x,
            attention_mask=attention_mask,
        )
        x = self._attention_norm(x + self._attention_dropout(attention_output))

        feed_forward_output = self._feed_forward(x)
        x = self._feed_forward_norm(
            x + self._feed_forward_dropout(feed_forward_output)
        )

        return x, attention_weights

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "x must be a 3D tensor with shape (batch_size, seq_len, d_model)."
            )

        if x.size(-1) != self._d_model:
            raise ValueError(
                f"Expected input embedding dimension {self._d_model}, but got {x.size(-1)}."
            )