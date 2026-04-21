# Project Context

This document is prepared to explain the background, scope, technical direction, and core assumptions of the **Encoder-Only-Transformer** project.

## 1. Project Title

Encoder-Only-Transformer

## 2. Project Goal

The main goal of this project is to build the **Encoder-only Transformer** architecture from scratch in a modular and educational manner, and to enable students both to understand the architecture and to extend this structure for tasks such as sequence classification.

## 3. Business / Use Case Context

This project is an **educational and learning repository**.
Main use cases:

* instructors providing AI / NLP / Transformer training
* students who want to understand the encoder-only architecture
* developers who want to see a from-scratch architecture with PyTorch

This repository is not designed for direct production product development, but to **make the BERT-style encoder logic visible**.

## 4. Problem Statement

Transformer-based encoder models are often used through ready-made frameworks. This makes it difficult for students to clearly understand:

* the attention mechanism,
* the encoder block structure,
* the relationship between pooling and classification,
* how the model is connected to the task pipeline

This project solves this problem by building the architecture **piece by piece and in a test-driven manner**.

## 5. Target User

* students who want to deeply learn the encoder-only Transformer architecture
* participants in technical training
* student contributors who will develop features under instructor supervision

## 6. Target Workflow

Example workflow:

**Input IDs → Embedding → Encoder → Pooling → Classification Head → Trainer / Metrics → Output**

Detailed flow:

1. Token IDs enter the model.
2. Token embeddings and positional encoding are applied.
3. The sequence is passed through stacked encoder blocks.
4. A sequence-level representation is obtained via pooling.
5. The classification head produces logits.
6. The trainer calculates loss and accuracy.
7. The result is reported during the training / evaluation process.

## 7. In Scope

Within the scope of this project:

* from-scratch implementation of the encoder-only Transformer architecture
* modular PyTorch components
* sequence classification pipeline
* trainer and metric layer
* test-driven development
* student contribution workflow
* small demo / experiment scripts

## 8. Out of Scope

Within the scope of this project, the following will not be included:

* full BERT reproduction
* masked language modeling pretraining (initial phase)
* large-scale distributed training
* production deployment
* heavy tokenizer research pipeline
* decoder-only generation

## 9. Input Types

The following input types will be used in the project:

* token ID tensors
* padding mask tensors
* simple text classification dataset examples in later stages

## 10. Expected Output

The expected outputs of the system:

* classification logits
* attention weights
* loss
* accuracy
* training / evaluation summaries

## 11. Technical Direction

### Planned components

* config loader
* input embedding layer
* attention layers
* feed-forward layer
* encoder block
* stacked encoder
* pooling strategies
* classification head
* full model
* trainer
* metrics
* future dataset / experiment scripts

### Initial tech choices

* **Language:** Python 3.10
* **Framework:** Custom PyTorch implementation
* **Deep Learning Library:** PyTorch
* **Config:** `pyproject.toml` + YAML
* **Testing:** pytest
* **Linting:** ruff
* **Environment / dependency management:** uv
* **Interface:** CLI / script-based

## 12. Constraints

Constraints affecting the project:

* code should not become too complex due to educational focus
* compute cost should be kept low
* it should be runnable by students locally
* the core architecture should not be bloated with unnecessary dependencies
* scope growth should be controlled

## 13. Assumptions

Initial assumptions in the project:

* the student has basic Python and OOP knowledge
* the student can run pytest and set up the repository locally
* sequence classification is sufficiently educational as a first task
* modular design will make it easier to manage student contributions

## 14. Risks

Possible risks:

* scope creep
* students trying to extend without understanding the core architecture
* merge conflicts due to too much overlap in the same files
* documentation not being updated alongside code
* the educational repository drifting into unnecessary enterprise-level complexity

## 15. Success Criteria

Minimum criteria for the project to be considered successful:

* the repository can be set up from scratch and tests run successfully
* the encoder-only core architecture is explainable and inspectable
* the sequence classification pipeline is working
* trainer + metrics flow is working
* students can explain the repository and add certain features
* the repository is clean, organized, and readable as a public learning resource

## 16. Notes

This repository should currently be positioned not as a “BERT replacement”, but as a **strong educational foundation for BERT-style encoder systems**.

