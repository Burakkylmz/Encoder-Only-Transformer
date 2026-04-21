from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ClassificationBatchStats:
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            raise ValueError("total must be greater than 0 to compute accuracy.")

        return self.correct / self.total


@dataclass(frozen=True, slots=True)
class ClassificationMetricResult:
    accuracy: float
    num_examples: int


class SequenceClassificationMetrics:
    """
    Utility class for sequence classification metrics.

    Current support:
        - accuracy
    """

    @staticmethod
    def compute_batch_stats(
        logits: Tensor,
        labels: Tensor,
    ) -> ClassificationBatchStats:
        SequenceClassificationMetrics._validate_logits_and_labels(
            logits=logits,
            labels=labels,
        )

        predictions = torch.argmax(logits, dim=-1)
        correct = int((predictions == labels).sum().item())
        total = int(labels.size(0))

        return ClassificationBatchStats(correct=correct, total=total)

    @staticmethod
    def compute_accuracy(
        logits: Tensor,
        labels: Tensor,
    ) -> ClassificationMetricResult:
        batch_stats = SequenceClassificationMetrics.compute_batch_stats(
            logits=logits,
            labels=labels,
        )

        return ClassificationMetricResult(
            accuracy=batch_stats.accuracy,
            num_examples=batch_stats.total,
        )

    @staticmethod
    def aggregate_batch_stats(
        batch_stats_list: Iterable[ClassificationBatchStats],
    ) -> ClassificationMetricResult:
        total_correct = 0
        total_examples = 0

        for batch_stats in batch_stats_list:
            if not isinstance(batch_stats, ClassificationBatchStats):
                raise TypeError(
                    "All items in batch_stats_list must be instances of ClassificationBatchStats."
                )

            total_correct += batch_stats.correct
            total_examples += batch_stats.total

        if total_examples == 0:
            raise ValueError("aggregate_batch_stats received no examples.")

        return ClassificationMetricResult(
            accuracy=total_correct / total_examples,
            num_examples=total_examples,
        )

    @staticmethod
    def _validate_logits_and_labels(
        logits: Tensor,
        labels: Tensor,
    ) -> None:
        if logits.ndim != 2:
            raise ValueError(
                "logits must be a 2D tensor with shape (batch_size, num_classes)."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must be a 1D tensor with shape (batch_size,)."
            )

        if logits.size(0) != labels.size(0):
            raise ValueError("logits and labels must have the same batch size.")

        if logits.size(-1) <= 0:
            raise ValueError("num_classes must be greater than 0.")