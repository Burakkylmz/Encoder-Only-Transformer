from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

from encoder_only_transformer.models.model import EncoderForSequenceClassification
from encoder_only_transformer.training.metrics import (
    ClassificationBatchStats,
    SequenceClassificationMetrics,
)


@dataclass(slots=True)
class SequenceClassificationBatch:
    input_ids: Tensor
    labels: Tensor
    padding_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.size(0))


@dataclass(frozen=True, slots=True)
class StepResult:
    loss: float
    batch_size: int
    accuracy: float
    correct: int


@dataclass(frozen=True, slots=True)
class EpochResult:
    average_loss: float
    accuracy: float
    num_batches: int
    num_examples: int


class SequenceClassificationTrainer:
    """
    Trainer for encoder-only sequence classification models.

    Responsibilities:
        1. Run train/eval steps
        2. Compute loss
        3. Compute classification metrics
        4. Move batches to the target device
        5. Aggregate epoch-level statistics
    """

    def __init__(
        self,
        model: EncoderForSequenceClassification,
        optimizer: Optimizer,
        loss_fn: nn.Module | None = None,
        device: str | torch.device = "cpu",
        max_grad_norm: float | None = None,
    ) -> None:
        if not isinstance(model, EncoderForSequenceClassification):
            raise TypeError(
                "model must be an instance of EncoderForSequenceClassification."
            )

        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be an instance of torch.optim.Optimizer.")

        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than 0 when provided.")

        self._model = model
        self._optimizer = optimizer
        self._loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self._device = torch.device(device)
        self._max_grad_norm = max_grad_norm

        self._model.to(self._device)

    @property
    def model(self) -> EncoderForSequenceClassification:
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def device(self) -> torch.device:
        return self._device

    def train_step(self, batch: SequenceClassificationBatch) -> StepResult:
        self._validate_batch(batch)
        batch = self._move_batch_to_device(batch)

        self._model.train()
        self._optimizer.zero_grad(set_to_none=True)

        output = self._model(
            input_ids=batch.input_ids,
            padding_mask=batch.padding_mask,
        )
        loss = self._compute_loss(logits=output.logits, labels=batch.labels)
        batch_stats = self._compute_batch_stats(logits=output.logits, labels=batch.labels)

        loss.backward()

        if self._max_grad_norm is not None:
            clip_grad_norm_(self._model.parameters(), max_norm=self._max_grad_norm)

        self._optimizer.step()

        return StepResult(
            loss=float(loss.detach().item()),
            batch_size=batch.batch_size,
            accuracy=batch_stats.accuracy,
            correct=batch_stats.correct,
        )

    def train_epoch(
        self,
        batches: Iterable[SequenceClassificationBatch],
    ) -> EpochResult:
        total_loss = 0.0
        total_examples = 0
        num_batches = 0
        batch_stats_list: list[ClassificationBatchStats] = []

        for batch in batches:
            step_result = self.train_step(batch)

            total_loss += step_result.loss * step_result.batch_size
            total_examples += step_result.batch_size
            num_batches += 1
            batch_stats_list.append(
                ClassificationBatchStats(
                    correct=step_result.correct,
                    total=step_result.batch_size,
                )
            )

        if num_batches == 0:
            raise ValueError("train_epoch received no batches.")

        metric_result = SequenceClassificationMetrics.aggregate_batch_stats(batch_stats_list)

        return EpochResult(
            average_loss=total_loss / total_examples,
            accuracy=metric_result.accuracy,
            num_batches=num_batches,
            num_examples=total_examples,
        )

    @torch.no_grad()
    def evaluate(
        self,
        batches: Iterable[SequenceClassificationBatch],
    ) -> EpochResult:
        self._model.eval()

        total_loss = 0.0
        total_examples = 0
        num_batches = 0
        batch_stats_list: list[ClassificationBatchStats] = []

        for batch in batches:
            self._validate_batch(batch)
            batch = self._move_batch_to_device(batch)

            output = self._model(
                input_ids=batch.input_ids,
                padding_mask=batch.padding_mask,
            )
            loss = self._compute_loss(logits=output.logits, labels=batch.labels)
            batch_stats = self._compute_batch_stats(logits=output.logits, labels=batch.labels)

            total_loss += float(loss.detach().item()) * batch.batch_size
            total_examples += batch.batch_size
            num_batches += 1
            batch_stats_list.append(batch_stats)

        if num_batches == 0:
            raise ValueError("evaluate received no batches.")

        metric_result = SequenceClassificationMetrics.aggregate_batch_stats(batch_stats_list)

        return EpochResult(
            average_loss=total_loss / total_examples,
            accuracy=metric_result.accuracy,
            num_batches=num_batches,
            num_examples=total_examples,
        )

    def _compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        if logits.ndim != 2:
            raise ValueError("logits must be a 2D tensor with shape (batch_size, num_classes).")

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D tensor with shape (batch_size,).")

        if logits.size(0) != labels.size(0):
            raise ValueError("logits and labels must have the same batch size.")

        return self._loss_fn(logits, labels)

    @staticmethod
    def _compute_batch_stats(logits: Tensor, labels: Tensor) -> ClassificationBatchStats:
        return SequenceClassificationMetrics.compute_batch_stats(
            logits=logits,
            labels=labels,
        )

    def _move_batch_to_device(
        self,
        batch: SequenceClassificationBatch,
    ) -> SequenceClassificationBatch:
        return SequenceClassificationBatch(
            input_ids=batch.input_ids.to(self._device),
            labels=batch.labels.to(self._device),
            padding_mask=(
                batch.padding_mask.to(self._device)
                if batch.padding_mask is not None
                else None
            ),
        )

    @staticmethod
    def _validate_batch(batch: SequenceClassificationBatch) -> None:
        if batch.input_ids.ndim != 2:
            raise ValueError(
                "batch.input_ids must be a 2D tensor with shape (batch_size, seq_len)."
            )

        if batch.labels.ndim != 1:
            raise ValueError(
                "batch.labels must be a 1D tensor with shape (batch_size,)."
            )

        if batch.input_ids.size(0) != batch.labels.size(0):
            raise ValueError("batch.input_ids and batch.labels must have the same batch size.")

        if batch.padding_mask is not None:
            if batch.padding_mask.ndim != 2:
                raise ValueError(
                    "batch.padding_mask must be a 2D tensor with shape (batch_size, seq_len)."
                )

            if batch.padding_mask.shape != batch.input_ids.shape:
                raise ValueError(
                    "batch.padding_mask must have the same shape as batch.input_ids."
                )