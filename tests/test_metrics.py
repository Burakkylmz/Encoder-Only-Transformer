from __future__ import annotations

import pytest
import torch

from encoder_only_transformer.training.metrics import (
    ClassificationBatchStats,
    ClassificationMetricResult,
    SequenceClassificationMetrics,
)


def test_classification_batch_stats_computes_accuracy_correctly() -> None:
    stats = ClassificationBatchStats(correct=8, total=10)

    assert stats.accuracy == 0.8


def test_classification_batch_stats_raises_error_when_total_is_zero() -> None:
    stats = ClassificationBatchStats(correct=0, total=0)

    with pytest.raises(
        ValueError,
        match="total must be greater than 0",
    ):
        _ = stats.accuracy


def test_compute_batch_stats_returns_expected_values() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.5],  # pred -> 0
            [0.1, 2.5, 0.3],  # pred -> 1
            [0.2, 0.4, 1.8],  # pred -> 2
            [2.0, 1.0, 0.5],  # pred -> 0
        ]
    )
    labels = torch.tensor([0, 1, 2, 1])

    result = SequenceClassificationMetrics.compute_batch_stats(
        logits=logits,
        labels=labels,
    )

    assert isinstance(result, ClassificationBatchStats)
    assert result.correct == 3
    assert result.total == 4
    assert result.accuracy == 0.75


def test_compute_accuracy_returns_expected_metric_result() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.5],  # pred -> 0
            [0.2, 1.7],  # pred -> 1
            [1.3, 0.7],  # pred -> 0
        ]
    )
    labels = torch.tensor([0, 1, 1])

    result = SequenceClassificationMetrics.compute_accuracy(
        logits=logits,
        labels=labels,
    )

    assert isinstance(result, ClassificationMetricResult)
    assert result.accuracy == pytest.approx(2 / 3)
    assert result.num_examples == 3


def test_compute_batch_stats_raises_error_for_non_2d_logits() -> None:
    logits = torch.randn(2, 3, 4)
    labels = torch.randint(0, 3, (2,))

    with pytest.raises(
        ValueError,
        match="logits must be a 2D tensor",
    ):
        SequenceClassificationMetrics.compute_batch_stats(
            logits=logits,
            labels=labels,
        )


def test_compute_batch_stats_raises_error_for_non_1d_labels() -> None:
    logits = torch.randn(2, 3)
    labels = torch.randint(0, 3, (2, 1))

    with pytest.raises(
        ValueError,
        match="labels must be a 1D tensor",
    ):
        SequenceClassificationMetrics.compute_batch_stats(
            logits=logits,
            labels=labels,
        )


def test_compute_batch_stats_raises_error_for_batch_size_mismatch() -> None:
    logits = torch.randn(4, 3)
    labels = torch.randint(0, 3, (3,))

    with pytest.raises(
        ValueError,
        match="logits and labels must have the same batch size",
    ):
        SequenceClassificationMetrics.compute_batch_stats(
            logits=logits,
            labels=labels,
        )


def test_compute_batch_stats_raises_error_when_num_classes_is_invalid() -> None:
    logits = torch.randn(4, 0)
    labels = torch.randint(0, 1, (4,))

    with pytest.raises(
        ValueError,
        match="num_classes must be greater than 0",
    ):
        SequenceClassificationMetrics.compute_batch_stats(
            logits=logits,
            labels=labels,
        )


def test_aggregate_batch_stats_returns_expected_metric_result() -> None:
    batch_stats = [
        ClassificationBatchStats(correct=3, total=4),
        ClassificationBatchStats(correct=2, total=3),
        ClassificationBatchStats(correct=5, total=5),
    ]

    result = SequenceClassificationMetrics.aggregate_batch_stats(batch_stats)

    assert isinstance(result, ClassificationMetricResult)
    assert result.num_examples == 12
    assert result.accuracy == pytest.approx(10 / 12)


def test_aggregate_batch_stats_raises_type_error_for_invalid_item() -> None:
    batch_stats = [
        ClassificationBatchStats(correct=3, total=4),
        "invalid",
    ]

    with pytest.raises(
        TypeError,
        match="All items in batch_stats_list must be instances of ClassificationBatchStats",
    ):
        SequenceClassificationMetrics.aggregate_batch_stats(batch_stats)  # type: ignore[arg-type]


def test_aggregate_batch_stats_raises_error_for_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="aggregate_batch_stats received no examples",
    ):
        SequenceClassificationMetrics.aggregate_batch_stats([])


def test_compute_accuracy_matches_compute_batch_stats_accuracy() -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0],  # pred -> 0
            [0.5, 2.0],  # pred -> 1
            [3.0, 0.2],  # pred -> 0
            [0.3, 1.1],  # pred -> 1
        ]
    )
    labels = torch.tensor([0, 1, 1, 1])

    batch_stats = SequenceClassificationMetrics.compute_batch_stats(
        logits=logits,
        labels=labels,
    )
    metric_result = SequenceClassificationMetrics.compute_accuracy(
        logits=logits,
        labels=labels,
    )

    assert metric_result.accuracy == pytest.approx(batch_stats.accuracy)
    assert metric_result.num_examples == batch_stats.total