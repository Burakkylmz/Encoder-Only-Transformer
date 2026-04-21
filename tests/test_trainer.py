from __future__ import annotations

import pytest
import torch
from torch.optim import SGD

from encoder_only_transformer.config.config import ModelConfig
from encoder_only_transformer.models.model import EncoderForSequenceClassification
from encoder_only_transformer.training.trainer import (
    EpochResult,
    SequenceClassificationBatch,
    SequenceClassificationTrainer,
    StepResult,
)


def build_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=100,
        max_seq_len=16,
        d_model=32,
        n_heads=4,
        ff_hidden_dim=64,
        dropout=0.0,
        n_layers=2,
        pad_token_id=0,
    )


def build_model(num_classes: int = 3) -> EncoderForSequenceClassification:
    config = build_model_config()
    return EncoderForSequenceClassification(
        config=config,
        num_classes=num_classes,
    )


def build_batch(
    batch_size: int = 4,
    seq_len: int = 6,
    vocab_size: int = 100,
    num_classes: int = 3,
    with_padding_mask: bool = True,
) -> SequenceClassificationBatch:
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_classes, (batch_size,))

    padding_mask = None
    if with_padding_mask:
        padding_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    return SequenceClassificationBatch(
        input_ids=input_ids,
        labels=labels,
        padding_mask=padding_mask,
    )


def clone_model_parameters(model: EncoderForSequenceClassification) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def parameters_changed(
    before: list[torch.Tensor],
    after: list[torch.Tensor],
) -> bool:
    return any(
        not torch.allclose(p_before, p_after)
        for p_before, p_after in zip(before, after)
    )


def test_sequence_classification_batch_exposes_batch_size() -> None:
    batch = build_batch(batch_size=5)

    assert batch.batch_size == 5


def test_trainer_raises_type_error_for_invalid_model() -> None:
    optimizer = SGD(build_model().parameters(), lr=0.1)

    with pytest.raises(
        TypeError,
        match="model must be an instance of EncoderForSequenceClassification",
    ):
        SequenceClassificationTrainer(  # type: ignore[arg-type]
            model="invalid",
            optimizer=optimizer,
        )


def test_trainer_raises_type_error_for_invalid_optimizer() -> None:
    model = build_model()

    with pytest.raises(
        TypeError,
        match="optimizer must be an instance of torch.optim.Optimizer",
    ):
        SequenceClassificationTrainer(  # type: ignore[arg-type]
            model=model,
            optimizer="invalid",
        )


def test_trainer_raises_value_error_for_invalid_max_grad_norm() -> None:
    model = build_model()
    optimizer = SGD(model.parameters(), lr=0.1)

    with pytest.raises(
        ValueError,
        match="max_grad_norm must be greater than 0",
    ):
        SequenceClassificationTrainer(
            model=model,
            optimizer=optimizer,
            max_grad_norm=0.0,
        )


def test_train_step_returns_step_result_with_expected_values() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batch = build_batch(batch_size=4, num_classes=3)
    result = trainer.train_step(batch)

    assert isinstance(result, StepResult)
    assert result.batch_size == 4
    assert isinstance(result.loss, float)
    assert result.loss >= 0.0
    assert isinstance(result.accuracy, float)
    assert 0.0 <= result.accuracy <= 1.0
    assert isinstance(result.correct, int)
    assert 0 <= result.correct <= result.batch_size
    assert result.accuracy == pytest.approx(result.correct / result.batch_size)


def test_train_step_updates_model_parameters() -> None:
    torch.manual_seed(42)

    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batch = build_batch(batch_size=4, num_classes=3)

    before = clone_model_parameters(model)
    trainer.train_step(batch)
    after = clone_model_parameters(model)

    assert parameters_changed(before, after)


def test_train_step_raises_error_for_invalid_input_ids_rank() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batch = SequenceClassificationBatch(
        input_ids=torch.randint(1, 100, (4, 6, 1)),
        labels=torch.randint(0, 3, (4,)),
    )

    with pytest.raises(
        ValueError,
        match="batch.input_ids must be a 2D tensor",
    ):
        trainer.train_step(batch)


def test_train_step_raises_error_for_invalid_labels_rank() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batch = SequenceClassificationBatch(
        input_ids=torch.randint(1, 100, (4, 6)),
        labels=torch.randint(0, 3, (4, 1)),
    )

    with pytest.raises(
        ValueError,
        match="batch.labels must be a 1D tensor",
    ):
        trainer.train_step(batch)


def test_train_step_raises_error_for_batch_size_mismatch_between_inputs_and_labels() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batch = SequenceClassificationBatch(
        input_ids=torch.randint(1, 100, (4, 6)),
        labels=torch.randint(0, 3, (3,)),
    )

    with pytest.raises(
        ValueError,
        match="batch.input_ids and batch.labels must have the same batch size",
    ):
        trainer.train_step(batch)


def test_train_step_raises_error_for_invalid_padding_mask_rank() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batch = SequenceClassificationBatch(
        input_ids=torch.randint(1, 100, (4, 6)),
        labels=torch.randint(0, 3, (4,)),
        padding_mask=torch.ones(4, 6, 1),
    )

    with pytest.raises(
        ValueError,
        match="batch.padding_mask must be a 2D tensor",
    ):
        trainer.train_step(batch)


def test_train_step_raises_error_for_padding_mask_shape_mismatch() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batch = SequenceClassificationBatch(
        input_ids=torch.randint(1, 100, (4, 6)),
        labels=torch.randint(0, 3, (4,)),
        padding_mask=torch.ones(4, 5),
    )

    with pytest.raises(
        ValueError,
        match="batch.padding_mask must have the same shape as batch.input_ids",
    ):
        trainer.train_step(batch)


def test_train_epoch_returns_epoch_result_with_expected_values() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batches = [
        build_batch(batch_size=4, num_classes=3),
        build_batch(batch_size=3, num_classes=3),
    ]

    result = trainer.train_epoch(batches)

    assert isinstance(result, EpochResult)
    assert result.num_batches == 2
    assert result.num_examples == 7
    assert result.average_loss >= 0.0
    assert 0.0 <= result.accuracy <= 1.0


def test_train_epoch_raises_error_for_empty_batches() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    with pytest.raises(
        ValueError,
        match="train_epoch received no batches",
    ):
        trainer.train_epoch([])


def test_evaluate_returns_epoch_result_with_expected_values() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batches = [
        build_batch(batch_size=4, num_classes=3),
        build_batch(batch_size=2, num_classes=3),
    ]

    result = trainer.evaluate(batches)

    assert isinstance(result, EpochResult)
    assert result.num_batches == 2
    assert result.num_examples == 6
    assert result.average_loss >= 0.0
    assert 0.0 <= result.accuracy <= 1.0


def test_evaluate_does_not_update_model_parameters() -> None:
    torch.manual_seed(42)

    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    batches = [
        build_batch(batch_size=4, num_classes=3),
        build_batch(batch_size=2, num_classes=3),
    ]

    before = clone_model_parameters(model)
    trainer.evaluate(batches)
    after = clone_model_parameters(model)

    assert not parameters_changed(before, after)


def test_evaluate_raises_error_for_empty_batches() -> None:
    model = build_model(num_classes=3)
    optimizer = SGD(model.parameters(), lr=0.1)
    trainer = SequenceClassificationTrainer(model=model, optimizer=optimizer)

    with pytest.raises(
        ValueError,
        match="evaluate received no batches",
    ):
        trainer.evaluate([])