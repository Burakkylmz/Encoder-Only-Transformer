from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(Exception):
    """Base exception for all configuration-related errors."""


class ConfigFileError(ConfigError):
    """Raised when the configuration file cannot be read or parsed."""


class ConfigValidationError(ConfigError):
    """Raised when configuration content is invalid."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_heads: int
    ff_hidden_dim: int
    dropout: float
    n_layers: int
    pad_token_id: int

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ModelConfig:
        required_fields = (
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_heads",
            "ff_hidden_dim",
            "dropout",
            "n_layers",
            "pad_token_id",
        )
        _ensure_required_fields(data=data, required_fields=required_fields, section_name="model")

        return cls(
            vocab_size=int(data["vocab_size"]),
            max_seq_len=int(data["max_seq_len"]),
            d_model=int(data["d_model"]),
            n_heads=int(data["n_heads"]),
            ff_hidden_dim=int(data["ff_hidden_dim"]),
            dropout=float(data["dropout"]),
            n_layers=int(data["n_layers"]),
            pad_token_id=int(data["pad_token_id"]),
        )

    def _validate(self) -> None:
        if self.vocab_size <= 0:
            raise ConfigValidationError("model.vocab_size must be greater than 0.")

        if self.max_seq_len <= 0:
            raise ConfigValidationError("model.max_seq_len must be greater than 0.")

        if self.d_model <= 0:
            raise ConfigValidationError("model.d_model must be greater than 0.")

        if self.n_heads <= 0:
            raise ConfigValidationError("model.n_heads must be greater than 0.")

        if self.d_model % self.n_heads != 0:
            raise ConfigValidationError(
                "model.d_model must be divisible by model.n_heads."
            )

        if self.ff_hidden_dim <= 0:
            raise ConfigValidationError("model.ff_hidden_dim must be greater than 0.")

        if not 0.0 <= self.dropout < 1.0:
            raise ConfigValidationError("model.dropout must be in the range [0.0, 1.0).")

        if self.n_layers <= 0:
            raise ConfigValidationError("model.n_layers must be greater than 0.")

        if self.pad_token_id < 0:
            raise ConfigValidationError("model.pad_token_id cannot be negative.")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TrainingConfig:
        required_fields = (
            "batch_size",
            "learning_rate",
            "weight_decay",
            "epochs",
        )
        _ensure_required_fields(data=data, required_fields=required_fields, section_name="training")

        return cls(
            batch_size=int(data["batch_size"]),
            learning_rate=float(data["learning_rate"]),
            weight_decay=float(data["weight_decay"]),
            epochs=int(data["epochs"]),
        )

    def _validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigValidationError("training.batch_size must be greater than 0.")

        if self.learning_rate <= 0:
            raise ConfigValidationError("training.learning_rate must be greater than 0.")

        if self.weight_decay < 0:
            raise ConfigValidationError("training.weight_decay cannot be negative.")

        if self.epochs <= 0:
            raise ConfigValidationError("training.epochs must be greater than 0.")


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ProjectConfig:
        model_section = _get_required_mapping(
            data=data,
            key="model",
            parent_name="root",
        )
        training_section = _get_required_mapping(
            data=data,
            key="training",
            parent_name="root",
        )

        return cls(
            model=ModelConfig.from_mapping(model_section),
            training=TrainingConfig.from_mapping(training_section),
        )


class YamlConfigLoader:
    """Loads project configuration from a YAML file."""

    def load(self, path: str | Path) -> ProjectConfig:
        config_path = Path(path)
        raw_data = self._read_yaml(config_path)
        return ProjectConfig.from_mapping(raw_data)

    @staticmethod
    def _read_yaml(path: Path) -> Mapping[str, Any]:
        if not path.exists():
            raise ConfigFileError(f"Config file not found: {path}")

        if not path.is_file():
            raise ConfigFileError(f"Config path is not a file: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                raw_data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ConfigFileError(f"Failed to parse YAML file: {path}") from exc
        except OSError as exc:
            raise ConfigFileError(f"Failed to read config file: {path}") from exc

        if raw_data is None:
            raise ConfigValidationError("Config file is empty.")

        if not isinstance(raw_data, Mapping):
            raise ConfigValidationError(
                "Config file must contain a top-level mapping."
            )

        return raw_data


def load_config(path: str | Path) -> ProjectConfig:
    """
    Convenience function for loading project configuration from YAML.
    """
    loader = YamlConfigLoader()
    return loader.load(path)


def _get_required_mapping(
    data: Mapping[str, Any],
    key: str,
    parent_name: str,
) -> Mapping[str, Any]:
    if key not in data:
        raise ConfigValidationError(f"Missing '{key}' section in {parent_name} config.")

    value = data[key]

    if not isinstance(value, Mapping):
        raise ConfigValidationError(
            f"Section '{key}' in {parent_name} config must be a mapping."
        )

    return value


def _ensure_required_fields(
    data: Mapping[str, Any],
    required_fields: tuple[str, ...],
    section_name: str,
) -> None:
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        missing_fields_str = ", ".join(missing_fields)
        raise ConfigValidationError(
            f"Missing required field(s) in '{section_name}' section: {missing_fields_str}"
        )