from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from encoder_only_transformer.config.config import ProjectConfig, load_config
from encoder_only_transformer.factories.factories import ModelFactory
from encoder_only_transformer.training.trainer import (
    SequenceClassificationBatch,
    SequenceClassificationTrainer,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training, evaluation, and checkpoint options."""
    parser = argparse.ArgumentParser(description="Train encoder-only sequence classification model.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoints/sequence_classification_latest.pt"),
    )
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--valid-size", type=int, default=16)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds and deterministic flags for reproducible experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_toy_dataloader(
    *,
    config: ProjectConfig,
    size: int,
    num_classes: int,
    shuffle: bool,
) -> DataLoader:
    """Build a synthetic dataloader for quick local sequence classification experiments."""
    input_ids = torch.randint(
        low=0,
        high=config.model.vocab_size,
        size=(size, config.model.max_seq_len),
        dtype=torch.long,
    )
    labels = torch.randint(low=0, high=num_classes, size=(size,), dtype=torch.long)
    padding_mask = torch.ones_like(input_ids, dtype=torch.long)

    dataset = TensorDataset(input_ids, labels, padding_mask)
    return DataLoader(dataset, batch_size=config.training.batch_size, shuffle=shuffle)


def to_trainer_batch(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> SequenceClassificationBatch:
    """Convert a DataLoader tuple into the trainer's typed batch dataclass."""
    input_ids, labels, padding_mask = batch
    return SequenceClassificationBatch(
        input_ids=input_ids,
        labels=labels,
        padding_mask=padding_mask,
    )


def save_checkpoint(
    *,
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: AdamW,
) -> None:
    """Save model and optimizer states with the current epoch to a checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: AdamW,
    device: str,
) -> int:
    """Load model/optimizer states from checkpoint and return next epoch index."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"]) + 1


def main() -> None:
    """Run end-to-end training with optional resume, evaluation, and checkpointing."""
    args = parse_args()
    set_seed(args.seed)

    project_config = load_config(args.config)
    model = ModelFactory.create_sequence_classification_model_from_project_config(
        project_config=project_config,
        num_classes=args.num_classes,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=project_config.training.learning_rate,
        weight_decay=project_config.training.weight_decay,
    )
    trainer = SequenceClassificationTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
    )

    train_loader = build_toy_dataloader(
        config=project_config,
        size=args.train_size,
        num_classes=args.num_classes,
        shuffle=True,
    )
    valid_loader = build_toy_dataloader(
        config=project_config,
        size=args.valid_size,
        num_classes=args.num_classes,
        shuffle=False,
    )

    total_epochs = args.epochs if args.epochs is not None else project_config.training.epochs
    start_epoch = 0

    if args.resume and args.checkpoint_path.exists():
        start_epoch = load_checkpoint(
            checkpoint_path=args.checkpoint_path,
            model=trainer.model,
            optimizer=optimizer,
            device=args.device,
        )
        print(f"Resumed from checkpoint at epoch {start_epoch}.")

    for epoch_idx in range(start_epoch, total_epochs):
        train_batches = (to_trainer_batch(batch) for batch in train_loader)
        train_result = trainer.train_epoch(train_batches)

        valid_batches = (to_trainer_batch(batch) for batch in valid_loader)
        valid_result = trainer.evaluate(valid_batches)

        print(
            f"epoch={epoch_idx + 1}/{total_epochs} "
            f"train_loss={train_result.average_loss:.4f} "
            f"train_acc={train_result.accuracy:.4f} "
            f"valid_loss={valid_result.average_loss:.4f} "
            f"valid_acc={valid_result.accuracy:.4f}"
        )

        save_checkpoint(
            checkpoint_path=args.checkpoint_path,
            epoch=epoch_idx,
            model=trainer.model,
            optimizer=optimizer,
        )

    print(f"Training complete. Checkpoint saved at: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
