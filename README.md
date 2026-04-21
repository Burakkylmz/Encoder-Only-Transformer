# Encoder-Only-Transformer

This is an educational project that aims to build the **Encoder-only Transformer** architecture **from scratch** and **step by step**.

The purpose of this repository is not just to use the architecture, but to truly understand it by constructing its components one by one. The project is designed with a **clean code**, **modular design**, test-driven development, and instructional explanation approach.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Why This Project Exists](#why-this-project-exists)
* [Project Goals](#project-goals)
* [Scope](#scope)
* [Out of Scope](#out-of-scope)
* [Current Status](#current-status)
* [Repository Structure](#repository-structure)
* [Setup](#setup)
* [Configuration](#configuration)
* [How It Works](#how-it-works)
* [Development Approach](#development-approach)
* [Student / Contributor Workflow](#student--contributor-workflow)
* [Roadmap](#roadmap)
* [Who Is It For](#who-is-it-for)
* [Contribution](#contribution)
* [License](#license)

---

## Project Overview

Transformer-based models are one of the fundamental building blocks of the NLP world today. However, in practice, these architectures are often used through ready-made libraries, which makes the internal mechanism of the model insufficiently visible.

This project aims to build the **Encoder-only Transformer** architecture step by step by breaking it down into its components, instead of using it as a **black box**.

The main idea here is:

> One of the best ways to truly understand an architecture is to build it from scratch within a controlled scope.

Therefore, the goal of this repository is not only to produce a working model, but also to clearly answer the following questions:

* Why is **Token Embedding** necessary?
* Why is **Positional Encoding** added?
* How does **Self-Attention** work?
* Why is **Multi-Head Attention** more powerful than a single attention mechanism?
* How is an **Encoder Block** formed?
* How is a **Stacked Encoder** built?
* Why is **Pooling** needed?
* How is an **Encoder-only Transformer** used in sequence classification tasks?
* How is a clean and testable codebase built around this architecture?

---

## Why This Project Exists

Today, many developers use the **Transformer** architecture, but fewer can truly explain its internal structure.

Although ready-made frameworks are powerful, they can create the following issues during the learning process:

* the model works, but it is unclear why it works
* components become invisible
* the connection between theory and implementation weakens
* the learner uses the architecture but cannot internalize it

This project addresses exactly this point.

The goal:

* to make the architecture visible
* to understand each component individually
* to directly connect theory with code
* to enable the learner not only to use the model but also to explain it
* to demonstrate a small but serious AI engineering codebase discipline

---

## Project Goals

This project aims to:

* build the **Encoder-only Transformer** architecture **from scratch**
* develop each main component independently and modularly
* explain not only “what” the architecture does, but also “why” it exists
* connect theoretical knowledge directly with implementation
* create a clear learning resource for students, developers, and instructors
* produce a clean, tested, and educational GitHub project that can be publicly examined

---

## Scope

The main focus of this repository is the **Encoder-only Transformer** architecture.

The current and target main components include:

* **Token Embedding**
* **Positional Encoding**
* **Scaled Dot-Product Attention**
* **Self-Attention**
* **Feed Forward Network**
* **Encoder Block**
* **Stacked Encoder**
* **Pooling**
* **Sequence Classification Head**
* **Full Encoder-based Classification Model**
* Model creation using **Factory Pattern**
* **Trainer**
* **Metrics**

This structure forms a foundation for small but meaningful **text understanding** and **sequence classification** tasks.

---

## Out of Scope

The following topics are intentionally excluded at the current stage of this repository:

* **Decoder-only generation**
* **Large-scale pretraining**
* **Full BERT reproduction**
* **Masked Language Modeling**
* **Advanced tokenizer training**
* **Production deployment**
* very large datasets and high-cost training pipelines

This decision is intentional. This repository is not an “everything NLP framework”, but a controlled-scope **learning laboratory**.

---

## Current Status

At this point, the repository has reached the following level:

* modular **Encoder-only architecture core** has been established
* main components have been implemented as separate layers
* **SequenceClassificationHead** and full model flow have been added
* model construction layer using **Factory Pattern** has been added
* training foundation has been established with **trainer.py** and **metrics.py**
* folder structure has been refactored to be more scalable
* comprehensive **unit test** structure has been established

In other words: this repository is no longer just a draft explaining architectural pieces; it has become a **working, tested, and extensible Encoder-only sequence classification foundation**.

---

## Repository Structure

```text
Encoder-Only-Transformer/
├── README.md
├── GUIDE.md
├── TASKS.md
├── PROJECT_CONTEXT.md
├── DECISIONS.md
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .python-version
│
├── config/
│   └── default.yaml
│
├── docs/
│   └── architecture_notes.md
│
├── src/
│   └── encoder_only_transformer/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── config.py
│       │
│       ├── layers/
│       │   ├── __init__.py
│       │   ├── attention.py
│       │   ├── embeddings.py
│       │   ├── feed_forward.py
│       │   └── pooling.py
│       │
│       ├── blocks/
│       │   ├── __init__.py
│       │   └── encoder_block.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoder.py
│       │   ├── heads.py
│       │   └── model.py
│       │
│       ├── factories/
│       │   ├── __init__.py
│       │   └── factories.py
│       │
│       └── training/
│           ├── __init__.py
│           ├── trainer.py
│           └── metrics.py
│
└── tests/
    ├── __init__.py
    ├── test_attention.py
    ├── test_config.py
    ├── test_encoder.py
    ├── test_encoder_block.py
    ├── test_embeddings.py
    ├── test_factories.py
    ├── test_feed_forward.py
    ├── test_heads.py
    ├── test_metrics.py
    ├── test_model.py
    ├── test_package_exports.py
    ├── test_pooling.py
    ├── test_self_attention.py
    └── test_trainer.py
```

### Folder Descriptions

* `config/`
  Runtime configuration files. Currently `default.yaml` is stored here.

* `docs/`
  Technical notes, architectural explanations, and additional documentation.

* `src/encoder_only_transformer/config/`
  Typed config objects, config loading, and validation logic.

* `src/encoder_only_transformer/layers/`
  Smallest architectural components: embedding, attention, feed forward, pooling.

* `src/encoder_only_transformer/blocks/`
  Larger structures. Currently `EncoderBlock` is located here.

* `src/encoder_only_transformer/models/`
  Encoder stack, head, and full model.

* `src/encoder_only_transformer/factories/`
  Config-driven model construction logic.

* `src/encoder_only_transformer/training/`
  Training loop and metric logic.

* `tests/`
  Unit test files for each main module.

---

## Setup

This project uses **uv** for environment and dependency management.

### Requirements

* Python **3.10**
* **uv**

### Project setup

```bash
uv sync
```

### Run tests

```bash
uv run pytest -v
```

### Run lint

```bash
uv run ruff check .
```

---

## Configuration

This project has two separate configuration layers.

### 1. `pyproject.toml`

This file is used for:

* project metadata
* dependency management
* pytest configuration
* ruff configuration

### 2. `config/default.yaml`

This file stores model and training parameters.

Example fields:

* `vocab_size`
* `max_seq_len`
* `d_model`
* `n_heads`
* `ff_hidden_dim`
* `dropout`
* `n_layers`
* `batch_size`
* `learning_rate`
* `weight_decay`
* `epochs`

This separation is intentional. Managing project metadata and experimental model parameters in separate layers is healthier than keeping them in the same place.

---

## How It Works

The main flow in the repository is as follows:

1. **input_ids** are taken
2. token representation + positional information is generated with `EncoderInputEmbedding`
3. this representation is passed into a multi-layer `Encoder`
4. encoder output is reduced to a sequence-level representation
5. logits are produced with `SequenceClassificationHead`
6. loss and metrics are calculated with `SequenceClassificationTrainer`

Conceptual flow:

```text
input_ids
  -> TokenEmbedding
  -> PositionalEncoding
  -> EncoderBlock x N
  -> Pooling
  -> Classification Head
  -> logits
  -> loss / accuracy
```

---

## Development Approach

This project is intentionally developed **step by step**.

At each stage, the following approach is followed:

1. first, the purpose of the relevant component is clarified
2. then a minimal but correct implementation is written
3. then unit tests are added
4. if necessary, it is documented with a short technical explanation
5. the next component is only started after the previous structure is well understood

The reason for this approach is:

When trying to understand an architecture, seeing too many files, too many abstractions, and too many features at once makes learning difficult. In this repository, speed is less important than **clarity**, **separation of responsibility**, and **testability**.

---

## Student / Contributor Workflow

This repository is suitable not only for individual learning, but also for a mentor-controlled contributor development process.

Recommended contributor flow:

1. clone the repository
2. set up the environment
3. pass all tests
4. technically explain the current architecture
5. work on your assigned feature area in your own branch
6. add tests
7. open a PR
8. merge after review

### Branch approach

* `main` should be protected for clean and controlled versions
* contributors should work on feature branches

Example branch names:

* `feature/data-pipeline`
* `feature/training-script`
* `feature/evaluation-metrics`

For detailed contributor and student working rules:

Refer to:

* `GUIDE.md`
* `TASKS.md`
* `DECISIONS.md`
* `CHANGELOG.md`
* `PROJECT_CONTEXT.md`

---

## Roadmap

### Completed Main Stages

* project setup
* config system
* embeddings
* attention core
* feed forward
* encoder block
* stacked encoder
* pooling
* classification head
* full model
* factory layer
* trainer
* metrics
* package refactor
* test stabilization

### Upcoming Possible Stages

* end-to-end training example script
* toy dataset / dataset utility
* evaluation script
* checkpoint save/load
* richer metrics
* sentence pair task
* ablation / pooling comparison experiments
* docs and architecture visual improvements

---

## Who Is It For

This repository is especially suitable for:

* students who want to learn **Transformer architecture**
* developers who want to connect theory with code
* instructors who want to use an educational repository in their courses
* engineers who want to understand the internals of encoder architectures before using ready-made libraries

This repository is not designed for complete beginners. It is most beneficial for those who are comfortable learning by reading code and want to deeply understand the architecture.

---

## Contribution

This is an education-focused repository.

Those who want to contribute can do so in the following areas:

* expanding test coverage
* adding training examples
* improving metrics
* improving docs and architecture notes
* bug fixes
* small experiments and analyses
* contributor workflow improvements

Before contributing, consider the educational nature of the repository. Priority should always be:

* clarity
* modularity
* testability
* instructional value

---

## License

This project will be shared under the **MIT License**.

---

## Final Note

The goal of this repository is not just to write a working model.

The real goal here is:

* to make an architecture visible
* to understand components one by one
* to demonstrate clean engineering discipline
* to gradually prepare learners for more advanced AI engineering work
* and to create a genuinely useful public learning resource

