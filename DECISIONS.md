# Decisions

This file summarizes the key technical and architectural decisions made throughout the project.

---

## 1) Project Positioning

**Decision:** The repository will be positioned not as a full BERT reproduction, but as an **Encoder-only Transformer educational foundation**.

**Reasoning:**

* The project is education-focused.
* Understanding the architecture is more important than feature parity.
* Claiming “it does everything BERT does” would be technically premature at this stage.

**Alternatives considered:**

* Full BERT clone
* Production-ready NLP library

**Consequence:**

* Scope remains controlled.
* The repository becomes more honest and educational.
* Development focus stays on architectural visibility.

**Status:** Confirmed

---

## 2) Start with Encoder-only Architecture

**Decision:** As the initial architecture, **encoder-only** is chosen instead of decoder-only.

**Reasoning:**

* Encoder-only provides a cleaner starting point for learning.
* It is suitable for classification and sentence-level understanding tasks.
* Additional complexities such as generation, sampling, and KV cache would expand the scope too early.

**Alternatives considered:**

* Decoder-only architecture
* Encoder-decoder architecture

**Consequence:**

* Architectural teaching becomes simpler.
* Choosing sequence classification as the first downstream task becomes natural.

**Status:** Confirmed

---

## 3) Build from Scratch with PyTorch

**Decision:** The core architecture will be written from scratch using **custom PyTorch modules**, instead of ready-made high-level model classes.

**Reasoning:**

* The goal is not black-box usage, but architectural visibility.
* Each component should be testable and explainable.
* Readability is important for student contributions.

**Alternatives considered:**

* Hugging Face Transformers-based wrapper approach
* Notebook-first prototyping

**Consequence:**

* Learning value increases.
* Code volume increases, but architectural transparency is gained.

**Status:** Confirmed

---

## 4) Use uv + pyproject.toml + YAML

**Decision:** Use `uv` for environment management, `pyproject.toml` for project configuration, and YAML for model/training settings.

**Reasoning:**

* Provides a clean setup for modern Python project management.
* Separating project metadata from experimental config values is more appropriate.
* Makes reproducible setup easier for students.

**Alternatives considered:**

* `venv` + `requirements.txt`
* Keeping all config values in a single file

**Consequence:**

* Setup and development workflow become more organized.
* Config system becomes more maintainable.

**Status:** Confirmed

---

## 5) Modular OOP Design with Explicit Validation

**Decision:** Code will be written using OOP with separated responsibilities, and input/shape validation will be explicit.

**Reasoning:**

* The repository is open to student contributions.
* SRP, testability, and maintainability are targeted.
* Most errors in Transformer implementations occur at the shape level.

**Alternatives considered:**

* Single-file rapid prototype
* Simplified forward flow without validation

**Consequence:**

* Code becomes slightly more verbose.
* However, maintenance, explanation, and debugging become significantly easier.

**Status:** Confirmed

---

## 6) Pooling as Strategy

**Decision:** Pooling logic is modeled as separate strategies through a `BasePooling` contract.

**Reasoning:**

* Approaches like mean / first-token / max should be comparable for downstream tasks.
* Classification head should not embed pooling logic internally.

**Alternatives considered:**

* Embedding pooling directly into the head
* Using only mean pooling

**Consequence:**

* A Strategy Pattern-like extensible structure is achieved.
* Students can more easily compare pooling behaviors.

**Status:** Confirmed

---

## 7) Separate Trainer and Metrics

**Decision:** Training logic and metric computation logic will be kept in separate modules.

**Reasoning:**

* Trainer should only manage training/evaluation flow.
* Accuracy and future metrics should be extendable independently.

**Alternatives considered:**

* Embedding accuracy computation inside the trainer

**Consequence:**

* `trainer.py` remains cleaner.
* `metrics.py` is prepared for future extension.

**Status:** Confirmed

---

## 8) Full Model Builds Attention Mask Internally

**Decision:** The full model will accept a `padding_mask` externally and internally construct the 4D attention mask used at the attention level.

**Reasoning:**

* Simplifies the user API.
* Separates padding semantics from attention implementation details.

**Alternatives considered:**

* Requiring users to directly provide a 4D attention mask

**Consequence:**

* Model usage becomes more ergonomic.
* Mask transformations are centralized internally.

**Status:** Confirmed

---

## 9) Refactor to Subpackages Before Data/Training Expansion

**Decision:** After completing the core architecture, a subpackage refactor will be performed before expanding the training/data layer.

**Reasoning:**

* Flat structure is educational at first but not scalable.
* Separation into `training/`, `layers/`, `models/`, `blocks/`, `factories/` supports growth.

**Alternatives considered:**

* Continuing with a flat structure
* Performing refactor too early

**Consequence:**

* Refactor was done at a controlled time.
* Future features will be added to a cleaner folder structure.

**Status:** Confirmed

---

## 10) Student Contributions Should Extend, Not Destabilize, the Core

**Decision:** Student contributions should primarily extend data, experiment, evaluation, and docs areas instead of modifying the core architecture.

**Reasoning:**

* The core architecture is now stable.
* Multiple people editing the same core files simultaneously may destabilize the repository.
* Educationally, understanding first and then extending in controlled areas is more effective.

**Alternatives considered:**

* Directly splitting core model files among students

**Consequence:**

* Risk of merge conflicts and architectural degradation is reduced.
* Student contributions become more manageable.

**Status:** Confirmed

