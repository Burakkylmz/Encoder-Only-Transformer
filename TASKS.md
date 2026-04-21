# TASKS

This file is used for task tracking of the **Encoder-Only-Transformer** project.

Purpose:

* to make the current project status visible,
* to clarify upcoming work,
* to clearly separate student contribution areas,
* to track progress at the milestone level.

This file does not track every small commit-level change. Only meaningful task areas, active work topics, and completed stages are monitored.

---

## Project Status

* **Current phase:** Development → Student Contribution Phase
* **Current focus:** Student task execution, dataset/data pipeline, experiment script, evaluation expansion
* **Last updated:** 2026-04-20

---

## Current Priority

At this stage, the priority order is as follows:

1. Students setting up the repository
2. Technically understanding and explaining the existing architecture
3. Distributing tasks based on branches
4. Starting student implementations
5. Maturing the experiment pipeline and documentation

---

## Student Task Assignment

This section clearly defines the tasks assigned to students.
Each student should work within their own responsibility area. Core files should not be modified simultaneously by multiple people.

---

### Student A — Data / Input Pipeline

#### Core Tasks

* [ ] create `training/datasets.py`
* [ ] implement `collate_fn`
* [ ] define padding strategy
* [ ] write a small toy dataset loader

#### Extended Tasks

* [ ] implement dynamic padding approach
* [ ] implement static padding approach
* [ ] explain the difference between dynamic vs static padding with a short note
* [ ] generate `padding_mask`
* [ ] convert mask to attention-compatible format if needed

#### Deliverables

* working data pipeline
* correct padding and mask generation
* short technical note or explanation file

---

### Student B — Training Workflow / Experiments

#### Core Tasks

* [ ] write `experiments/train_sequence_classification.py`
* [ ] build model from config
* [ ] create training loop
* [ ] add checkpoint save/load support

#### Extended Tasks

* [ ] add epoch-based logging
* [ ] output loss and accuracy
* [ ] add deterministic training option
* [ ] add simple evaluation step

#### Deliverables

* runnable training script
* training flow with logging
* checkpoint mechanism
* simple example run output

---

### Student C — Evaluation / Docs / Analysis

#### Core Tasks

* [ ] add or extend `precision`, `recall`, `f1` metrics
* [ ] write evaluation helper functions
* [ ] compare pooling strategies
* [ ] update README usage examples
* [ ] expand `docs/architecture_notes.md`

#### Extended Tasks

* [ ] write confusion matrix utility
* [ ] compare mean pooling / max pooling / first token pooling
* [ ] prepare a short experiment report
* [ ] add the report under `docs/` as an appropriate file

#### Deliverables

* extended metrics module
* analysis notes
* updated documentation
* short comparison report

---

## Student Workflow Requirements

Each student must follow the workflow below:

* [ ] clone the repository
* [ ] set up the environment
* [ ] run all existing tests
* [ ] technically explain the current architecture
* [ ] create a branch for their task area
* [ ] develop on the branch
* [ ] add or update necessary tests
* [ ] open a Pull Request
* [ ] proceed to merge process after review

---

## Backlog

This section includes the general project backlog independent of student tasks.

* [ ] select a small but educational real dataset
* [ ] add an end-to-end training example
* [ ] add full training usage example to README
* [ ] add example run output
* [ ] add checkpoint usage example
* [ ] add evaluation result example

---

## In Progress

* [ ] student onboarding is being completed
* [ ] task distribution has been clarified
* [ ] student work areas are being separated by branches
* [ ] contribution process is being aligned with guide and task documents

---

## Blocked

* [ ] real dataset selection

  * **Reason:** student ownership and feature implementations must be clarified first
  * **Needed action:** select a small, clear sequence classification dataset suitable for educational purposes

---

## Done

* [x] repository initialized
* [x] `uv` environment set up
* [x] `pyproject.toml` and YAML config structure created
* [x] `ModelConfig`, `TrainingConfig`, `ProjectConfig` designed
* [x] `TokenEmbedding` implemented
* [x] `SinusoidalPositionalEncoding` implemented
* [x] `EncoderInputEmbedding` implemented
* [x] `ScaledDotProductAttention` implemented
* [x] mask support added
* [x] `SelfAttention` implemented
* [x] `PositionwiseFeedForward` implemented
* [x] `EncoderBlock` implemented
* [x] `Encoder` implemented
* [x] pooling strategies added
* [x] `SequenceClassificationHead` implemented
* [x] `EncoderForSequenceClassification` model implemented
* [x] `ModelFactory` added
* [x] package export structure updated
* [x] folder refactor completed
* [x] `SequenceClassificationTrainer` implemented
* [x] `SequenceClassificationMetrics` implemented
* [x] trainer and metrics integration completed
* [x] all existing tests made to pass

---

## Phase Breakdown

### Phase 1 — Foundation and Architecture

* [x] problem statement written
* [x] learning use case clarified
* [x] scope defined
* [x] modular architecture separated
* [x] basic test strategy established

### Phase 2 — Core Encoder Implementation

* [x] input representation layer implemented
* [x] attention core implemented
* [x] self-attention layer implemented
* [x] feed-forward layer implemented
* [x] encoder block created
* [x] stacked encoder created
* [x] pooling and classification head added
* [x] full sequence classification model built

### Phase 3 — Refactor and Package Structure

* [x] migrated from flat structure to subpackage structure
* [x] import paths fixed
* [x] public package API updated
* [x] test suite restored to green after refactor

### Phase 4 — Training Foundation

* [x] `training/trainer.py` created
* [x] `training/metrics.py` created
* [x] loss + accuracy flow connected
* [ ] dataset / dataloader layer to be added
* [ ] experiment script to be added

### Phase 5 — Student Contribution Phase

* [ ] students will set up the repository
* [ ] students will technically explain the architecture
* [ ] tasks will be distributed to students via branches
* [ ] student implementations will begin
* [ ] Pull Request review process will be executed

### Phase 6 — Finalization

* [ ] README will be updated one final time
* [ ] CHANGELOG will be updated
* [ ] example run outputs will be added
* [ ] demo flow will be prepared
* [ ] final cleanup will be done

---

## Milestones

### Milestone 1 — Repository and Core Setup

* [x] repository created
* [x] config system completed
* [x] test infrastructure established

### Milestone 2 — Encoder Core Ready

* [x] core encoder components implemented
* [x] full model is operational
* [x] basic unit test coverage completed

### Milestone 3 — Refactor and Training Base

* [x] package refactor completed
* [x] trainer added
* [x] metrics added
* [x] all tests restored to passing

### Milestone 4 — Student Contribution Phase

* [ ] student branches created
* [ ] assignment ownership clarified
* [ ] initial implementations started
* [ ] first PRs opened

### Milestone 5 — Learning Repository Ready

* [ ] experiment script ready
* [ ] example training flow ready
* [ ] README and docs updated
* [ ] repository ready for student walkthrough and contribution

---

## Notes

The purpose of this file is to guide students and make progress visible.
For reasoning behind technical decisions, refer to `DECISIONS.md`; for project context, refer to `PROJECT_CONTEXT.md`; for working rules, refer to `GUIDE.md`; for important changes, refer to `CHANGELOG.md`.

