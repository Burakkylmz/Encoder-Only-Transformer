# Changelog

This file summarizes meaningful milestone-level changes throughout the project process.

## [2026-04-19] — Repository initialized and project foundation created

### Added

* `uv`-based project setup was created
* `pyproject.toml` was created
* YAML config structure was created
* Initial README skeleton was prepared

### Notes

* The project was positioned as an educational encoder-only transformer repository.

---

## [2026-04-19] — Core encoder architecture completed

### Added

* `TokenEmbedding`, `SinusoidalPositionalEncoding`, `EncoderInputEmbedding`
* `ScaledDotProductAttention`
* `SelfAttention`
* `PositionwiseFeedForward`
* `EncoderBlock`
* `Encoder`
* Pooling strategies
* `SequenceClassificationHead`
* `EncoderForSequenceClassification`
* `ModelFactory`

### Changed

* Core architecture was organized with a modular OOP structure

### Notes

* End-to-end working encoder-only core for sequence classification was completed.

---

## [2026-04-19] — Test coverage expanded across architecture

### Added

* Config tests
* Embedding tests
* Attention tests
* Self-attention tests
* Feed-forward tests
* Encoder block tests
* Encoder tests
* Pooling tests
* Head tests
* Full model tests
* Factory tests
* Package export tests

### Fixed

* 3D attention mask broadcasting bug was fixed
* Import paths broken after refactor were fixed

### Notes

* All existing core modules were validated with tests.

---

## [2026-04-19] — Package refactor completed

### Added

* Folder structure `config/`, `layers/`, `blocks/`, `models/`, `factories/`, `training/` was created
* Subpackage export structure was organized

### Changed

* Flat source structure was migrated to a maintainable subpackage structure
* Import paths were updated according to the new structure

### Notes

* Refactor was completed in a controlled manner before expanding the training/data layer.

---

## [2026-04-19] — Training and metrics utilities added

### Added

* `training/trainer.py`
* `training/metrics.py`
* Loss + accuracy flow
* `SequenceClassificationBatch`, `StepResult`, `EpochResult`

### Changed

* Trainer was updated to use metric computation from a separate module

### Notes

* The repository now supports not only architectural demonstration but also a basic training workflow.

---

## [2026-04-19] — Student contribution planning started

### Added

* Student contribution strategy was defined
* Documentation templates began to be adapted for the project
* Student onboarding guide was planned

### Notes

* Next stage: dataset/data pipeline, experiment script, and student feature ownership model.

