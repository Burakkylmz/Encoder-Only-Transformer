# Student Guide

This guide is prepared for students who will work on the **Encoder-Only-Transformer** project.
The goal is not only to write code, but to understand the existing system and contribute in a controlled and professional manner.

---

## 1. Your role in this project

This project is a learning repository for an **Encoder-only Transformer** written from scratch.
What is expected from you is not to directly add random features.

First, you are expected to:

* set up the repository locally,
* understand the current architecture,
* run the tests,
* be able to technically explain how the system works

After that, you will be asked to develop features in your assigned areas.

---

## 2. Expected working model

In this project, you are expected to proceed in the following order:

### Phase 1 — Set up the repository

* Clone the repo
* Run `uv sync`
* Run the tests
* Inspect the package structure

### Phase 2 — Understand the system

Be able to technically explain the following flow:

`input_ids -> embeddings -> encoder -> pooling -> classification head -> trainer -> metrics`

### Phase 3 — Do a short walkthrough

Explain the following topics to your instructor or team lead:

* What does the project do?
* Which folder has which responsibility?
* How does the attention layer work?
* Why is pooling necessary?
* How are trainer and metrics connected?

### Phase 4 — Work on your assigned feature

Do not directly modify the core architecture outside your assigned area.

### Phase 5 — Add tests + open PR

Your work must:

* include tests,
* be explainable,
* be developed in your own branch,
* be submitted via a PR.

---

## 3. Setup

### Requirements

* Python 3.10
* `uv`

### Setup

```bash
uv sync
```

### Run tests

```bash
uv run pytest -v
```

### Lint

```bash
uv run ruff check .
```

---

## 4. How to read the project structure

### `config/`

Config objects, YAML loading, and validation are located here.

### `layers/`

The smallest architectural components are here:

* embeddings
* attention
* feed forward
* pooling

### `blocks/`

Larger structures where layers are combined:

* `EncoderBlock`

### `models/`

Task-level and model-level structures:

* `Encoder`
* `SequenceClassificationHead`
* `EncoderForSequenceClassification`

### `factories/`

Config-based model construction logic.

### `training/`

Training loop and metrics.

### `tests/`

Contract tests for each important module.

---

## 5. How you should think in this project

The goal of this repository is not just to write “working code”.
Your contributions are expected to answer the following questions:

* Why is this change necessary?
* Is the responsibility of the code in the correct place?
* Did I add tests?
* Am I breaking the existing architecture, or extending it?
* Can another student read and understand this contribution?

---

## 6. Mandatory working rules

### 6.1 Do not work directly on the `main` branch

Each student must work on their own feature branch.

Example branch names:

* `feature/student-a-data-pipeline`
* `feature/student-b-training-script`
* `feature/student-c-metrics-docs`

### 6.2 Use meaningful commit messages

Good examples:

* `add batch collation for padded sequence classification data`
* `implement experiment script for sequence classification training`
* `extend metrics with macro f1 support`

Bad examples:

* `update`
* `fix`
* `last version`

### 6.3 Do not open a PR without adding tests

If there is a new feature, there must also be tests.

### 6.4 Update README / docs if affected

If the code changes, documentation must also be updated.

### 6.5 Do not make large unauthorized changes to core files

Be especially careful with the following files:

* `models/model.py`
* `models/encoder.py`
* `blocks/encoder_block.py`
* `layers/attention.py`

If a major refactor or behavioral change is required in these files, discuss it first.

---

## 7. Code standard

The following standards are expected in this project:

* Use type hints
* Follow OOP and SRP
* Validate inputs
* Do not add unnecessary abstractions
* Use private methods only if they are truly meaningful
* Write descriptive docstrings for public classes and functions
* Keep shape (`shape`) contracts explicit

The goal in this project:
**senior discipline, not unnecessary enterprise complexity**

---

## 8. Student Task Distribution — Encoder-Only-Transformer

## Overview

This project is divided among three students to promote modular development, clear responsibility distribution, and teamwork habits.

Each student is responsible for a specific subsystem. The goal is not only to write code, but also to understand and explain the design decisions made.

---

## Student A — Data / Input Pipeline

### Main Tasks

* Implement `datasets.py`
* Write `collate_fn`
* Design a padding strategy
* Create a small toy dataset loader

### Additional Tasks

#### 1. Dynamic vs Static Padding Comparison

* Apply fixed `max_seq_len` padding
* Apply batch-based dynamic padding
* Write a short comparison (performance, memory)

#### 2. Attention Mask Generation

* Generate `padding_mask`
* (Optional) Convert to attention-compatible format:
  `(batch_size, 1, 1, seq_len)`

### Goal

To understand how raw data is transformed into a format that the model can process.

---

## Student B — Training Workflow / Experiments

### Main Tasks

* Write `experiments/train_sequence_classification.py`
* Build the model from config
* Implement checkpoint save/load
* Create a working training loop

### Additional Tasks

#### 1. Simple Logging

* For each epoch:

  * loss
  * accuracy
* Print to console

Example:

```
Epoch 1 | loss: 0.65 | acc: 0.72
```

#### 2. Deterministic Training Option

* Ensure reproducibility:

  * `torch.manual_seed(...)`
* Add optional config flag

### Goal

To understand how a model is trained, monitored, and made reproducible.

---

## Student C — Evaluation / Docs / Analysis

### Main Tasks

* Implement `precision`, `recall`, `f1`
* Write evaluation helper functions
* Compare pooling methods
* Improve README examples
* Write architecture notes

### Additional Tasks

#### 1. Confusion Matrix Utility

* Implement using Torch or NumPy
* Generate matrix from prediction results

#### 2. Mini Experiment Report

* Compare:

  * mean pooling
  * max pooling
  * CLS token
* Write a short 1–2 page report
* Place it under `docs/`

### Goal

To understand how model performance is measured and interpreted.

---

## General Rules

* Each student must work on a separate branch
* No direct commits to the `main` branch
* Each feature must include:

  * clean code
  * type hints
  * tests if possible
* A Pull Request must be opened and reviewed

---

## Expected Output

At the end of this phase:

* The project will have:

  * a working data pipeline
  * a runnable training script
  * correct evaluation metrics
* Students will:

  * understand the full pipeline
  * have contributed to a real codebase
  * gain teamwork experience

---

## Final Note

Priority order:

* clarity
* correctness
* modularity

Fast-written but unclear code has no value.

This work is not just coding, it is an engineering practice.

---

## 9. Pre-work checklist

You should be able to answer “yes” to all of the following:

* Did I set up the repo locally?
* Did I run the tests?
* Can I explain the architectural flow?
* Do I know which folder does what?
* Is my assigned task area clear?
* Do I know which files I need to modify?

---

## 10. Pre-PR checklist

* [ ] Is my branch correct?
* [ ] Is my change aligned with my assigned task?
* [ ] Did I add tests?
* [ ] Did I run all tests?
* [ ] Are imports and package paths correct?
* [ ] Did I update documentation if needed?
* [ ] Are my commit messages meaningful?
* [ ] Does my PR description explain what and why?

---

## 11. Things to avoid in this project

* Changing too many files randomly at once
* Refactoring without understanding the core architecture
* Developing without tests
* Adding features without updating README / docs
* Copy-pasting code from the internet without understanding it
* Making changes that “seem to work” but break contracts

---

## 12. What is truly expected from you

The goal in this project is not only to “add a feature”.
What is expected from you:

* understand the existing system,
* explain it technically,
* extend it in a controlled way,
* demonstrate professional GitHub workflow discipline

This repository should be treated not as a homework repo, but as a small-scale **mentored engineering project**.

