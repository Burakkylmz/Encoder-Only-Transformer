from __future__ import annotations

from pathlib import Path

import pytest

from encoder_only_transformer.config import (
    ConfigFileError,
    ConfigValidationError,
    ProjectConfig,
    load_config,
)


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def build_valid_yaml() -> str:
    return """
model:
  vocab_size: 30000
  max_seq_len: 128
  d_model: 128
  n_heads: 4
  ff_hidden_dim: 256
  dropout: 0.1
  n_layers: 2
  pad_token_id: 0

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 5
"""


def test_load_config_returns_project_config_for_valid_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "default.yaml"
    write_text_file(config_file, build_valid_yaml())

    config = load_config(config_file)

    assert isinstance(config, ProjectConfig)
    assert config.model.vocab_size == 30000
    assert config.model.d_model == 128
    assert config.model.n_heads == 4
    assert config.training.batch_size == 32
    assert config.training.learning_rate == 0.001


def test_load_config_raises_config_file_error_when_file_does_not_exist(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.yaml"

    with pytest.raises(ConfigFileError, match="Config file not found"):
        load_config(missing_file)


def test_load_config_raises_validation_error_when_file_is_empty(tmp_path: Path) -> None:
    config_file = tmp_path / "empty.yaml"
    write_text_file(config_file, "")

    with pytest.raises(ConfigValidationError, match="Config file is empty"):
        load_config(config_file)


def test_load_config_raises_file_error_for_invalid_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    write_text_file(
        config_file,
        """
model:
  vocab_size: 30000
  max_seq_len: [128
training:
  batch_size: 32
""",
    )

    with pytest.raises(ConfigFileError, match="Failed to parse YAML file"):
        load_config(config_file)


def test_load_config_raises_validation_error_when_model_section_is_missing(tmp_path: Path) -> None:
    config_file = tmp_path / "missing_model.yaml"
    write_text_file(
        config_file,
        """
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 5
""",
    )

    with pytest.raises(ConfigValidationError, match="Missing 'model' section"):
        load_config(config_file)


def test_load_config_raises_validation_error_when_training_section_is_missing(tmp_path: Path) -> None:
    config_file = tmp_path / "missing_training.yaml"
    write_text_file(
        config_file,
        """
model:
  vocab_size: 30000
  max_seq_len: 128
  d_model: 128
  n_heads: 4
  ff_hidden_dim: 256
  dropout: 0.1
  n_layers: 2
  pad_token_id: 0
""",
    )

    with pytest.raises(ConfigValidationError, match="Missing 'training' section"):
        load_config(config_file)


def test_load_config_raises_validation_error_when_required_model_field_is_missing(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "missing_model_field.yaml"
    write_text_file(
        config_file,
        """
model:
  vocab_size: 30000
  max_seq_len: 128
  d_model: 128
  n_heads: 4
  dropout: 0.1
  n_layers: 2
  pad_token_id: 0

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 5
""",
    )

    with pytest.raises(ConfigValidationError, match="ff_hidden_dim"):
        load_config(config_file)


def test_load_config_raises_validation_error_when_d_model_is_not_divisible_by_n_heads(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "invalid_heads.yaml"
    write_text_file(
        config_file,
        """
model:
  vocab_size: 30000
  max_seq_len: 128
  d_model: 130
  n_heads: 4
  ff_hidden_dim: 256
  dropout: 0.1
  n_layers: 2
  pad_token_id: 0

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 5
""",
    )

    with pytest.raises(
        ConfigValidationError,
        match="model.d_model must be divisible by model.n_heads",
    ):
        load_config(config_file)


def test_load_config_raises_validation_error_when_learning_rate_is_invalid(
    tmp_path: Path,
) -> None:
    config_file = tmp_path / "invalid_learning_rate.yaml"
    write_text_file(
        config_file,
        """
model:
  vocab_size: 30000
  max_seq_len: 128
  d_model: 128
  n_heads: 4
  ff_hidden_dim: 256
  dropout: 0.1
  n_layers: 2
  pad_token_id: 0

training:
  batch_size: 32
  learning_rate: 0
  weight_decay: 0.0001
  epochs: 5
""",
    )

    with pytest.raises(
        ConfigValidationError,
        match="training.learning_rate must be greater than 0",
    ):
        load_config(config_file)