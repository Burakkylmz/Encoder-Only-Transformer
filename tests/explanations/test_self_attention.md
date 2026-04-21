# General Purpose of the `test_self_attention.py` File

This file is a unit test file that validates the **behavioral contract** of the `SelfAttention` layer. More clearly, this file does not settle for saying “self-attention is implemented”; it tests whether this layer:

* produces the correct shape,
* rejects invalid inputs early,
* correctly processes mask support,
* preserves the head split / combine logic,
* provides deterministic behavior

It does this directly through `SelfAttention` and partially through `ScaledDotProductAttention`.

This file exists in the project because the attention layer is one of the most sensitive parts of the Transformer architecture. Errors in the embedding layer are usually caught at the shape level; on the attention side, both shape and mathematical behavior matter simultaneously. Therefore, this file solves the following problem:

> Does the self-attention layer truly comply with the expected tensor contract, mask semantics, and internal architectural assumptions?

As a layer classification, this file:

* is not production code,
* is not a service/API,
* is not a domain business rule.

The most accurate classification:

* **test layer**
* more specifically **component-level unit test**

---

# Its Role in the Project at a High Level

This file directly interacts with:

* `ScaledDotProductAttention` inside `encoder_only_transformer.layers.attention`
* `SelfAttention` inside `encoder_only_transformer.layers.attention`

The components it indirectly affects are:

* `encoder_block.py`
* `encoder.py`
* `model.py`
* `trainer.py`

Because all of these components ultimately depend on self-attention. If self-attention breaks, the encoder chain collapses; if the encoder breaks, the full model collapses.

## Likely input / output

The inputs of this test file are not real data. The test inputs are:

* synthetic feature tensors generated with `torch.randn(...)`,
* mask tensors such as `torch.ones(...)` or custom-defined masks,
* some small deterministic example tensors.

The outputs are also not application-level outputs. The output is:

* test pass/fail result,
* meaning a signal of “this contract is preserved” or “it is broken”.

## Its place in data flow and control flow

This file belongs to the quality assurance flow, not the data flow.

In the real flow:
`embedding -> self-attention -> feed-forward -> encoder block`

In the test flow:

* synthetic input is prepared,
* `SelfAttention` is called directly,
* output and internal behavior are validated.

This is an important distinction. This file does not use the system; it evaluates the system.

---

# Reviewing the File Block by Block

## Imports

```python
from __future__ import annotations

import pytest
import torch

from encoder_only_transformer.layers.attention import (
    ScaledDotProductAttention,
    SelfAttention,
)
```

### What does it do?

* `pytest`: for assertions and exception testing
* `torch`: for creating and comparing tensors
* `ScaledDotProductAttention`, `SelfAttention`: the actual classes under test

### Why is it there?

The import set is clean. The test file imports only what it needs. Notably, the test is aware not only of `SelfAttention` but also its internal component `ScaledDotProductAttention`. This shows that the test scope touches not only the external API but partially the internal composition.

### Design comment

This is not a pure black-box test approach. It is a mix of **contract test + implementation-aware test**.

---

# Helper Function: `build_input_tensor`

```python
def build_input_tensor(
    batch_size: int = 2,
    seq_len: int = 8,
    d_model: int = 64,
) -> torch.Tensor:
    return torch.randn(batch_size, seq_len, d_model)
```

## What does it do?

Provides a helper function to avoid repeatedly writing tensor creation logic in tests. It returns a 3D tensor sampled from a normal distribution.

## Why is it there?

This is good test hygiene because:

* it reduces duplication,
* clarifies intent,
* keeps test bodies readable.

## Technical meaning

Since the input contract of `SelfAttention` is `(batch_size, seq_len, d_model)`, the helper directly produces this shape. In other words, the helper reflects the public contract of the tested component.

## Edge-case

This helper produces only valid inputs. Invalid tensors are constructed directly inside tests when needed. This is the correct choice; otherwise the helper would become overly complex.

---

# `test_self_attention_returns_expected_output_shapes`

```python
def test_self_attention_returns_expected_output_shapes() -> None:
    attention = SelfAttention(d_model=64, n_heads=4, dropout=0.0)
    x = build_input_tensor(batch_size=2, seq_len=8, d_model=64)

    output, attention_weights = attention(x)

    assert output.shape == (2, 8, 64)
    assert attention_weights.shape == (2, 4, 8, 8)
```

## What does it do?

Validates the two fundamental shape contracts of the self-attention layer:

* output: `(batch_size, seq_len, d_model)`
* attention weights: `(batch_size, n_heads, seq_len, seq_len)`

## Why is it there?

This test verifies the most fundamental promise the layer makes to the rest of the architecture. All other components depend on these shapes.

## Technical comment

This test is critical. Especially the attention weights shape is important because:

* the number of heads is visible here,
* the token-to-token relationship matrix is formed here.

## Improvement note

This test is correct but does not validate numerical correctness. That is acceptable as a first-level contract test.

---

# `test_self_attention_returns_finite_outputs`

```python
assert torch.isfinite(output).all()
assert torch.isfinite(attention_weights).all()
```

## What does it do?

Tests that no NaN or infinite values are produced.

## Why is it important?

In attention implementations, the following areas are numerically sensitive:

* `softmax`
* `masked_fill(-inf)`
* projection + matmul chains

Therefore, checking for finite outputs is very valuable.

## Technical comment

This is not a superficial test. Even if shapes are correct, outputs can still be numerically broken. This test attempts to catch such issues.

## Limitation

It only checks “finite or not”. Finite does not guarantee correctness. This is acceptable.

---

# `test_self_attention_raises_error_for_non_3d_input`

This test validates that `SelfAttention` enforces its input contract. When a 4D tensor is passed, a `ValueError` is expected.

## Why is it important?

`SelfAttention` expects `(B, S, D)` input. Without rank validation, errors would occur deeper and become harder to debug.

## Design comment

This test confirms a fail-fast design decision.

---

# `test_self_attention_raises_error_for_embedding_dimension_mismatch`

```python
invalid_x = torch.randn(2, 8, 32)
...
match="Expected input embedding dimension 64, but got 32"
```

## What does it do?

Tests that an error is raised when the last dimension does not match `d_model`.

## Why is it there?

One of the most common failure points in Transformer implementations is dimension mismatch between embedding and attention layers.

## Technical concept

This is a **representation contract test**.

---

# `test_self_attention_raises_error_when_d_model_is_not_divisible_by_n_heads`

```python
with pytest.raises(
    ValueError,
    match="d_model must be divisible by n_heads",
):
    SelfAttention(d_model=62, n_heads=4, dropout=0.0)
```

## What does it do?

Tests a critical constructor invariant: `d_model % n_heads == 0`.

## Why is it there?

Because the implementation relies on:

```python
head_dim = d_model // n_heads
```

If not divisible, the layer becomes invalid.

## Strength

Very important test. It enforces failure at initialization time rather than runtime.

---

# `test_self_attention_exposes_expected_properties`

```python
assert attention.d_model == 64
assert attention.n_heads == 4
assert attention.head_dim == 16
```

## What does it do?

Tests the property API.

## Why is it there?

Ensures the class exposes expected read-only attributes for debugging and introspection.

## Honest evaluation

Useful but not critical.

---

# `test_self_attention_contains_scaled_dot_product_attention_module`

```python
assert isinstance(attention._attention, ScaledDotProductAttention)
```

## What does it do?

Validates internal composition.

## Why is it there?

This is an implementation-aware test. It verifies that `SelfAttention` internally uses `ScaledDotProductAttention`.

## Strength

Reasonable for an educational repository.

## Weakness

Relies on private attributes, creating implementation coupling.

---

# `test_split_heads_returns_expected_shape`

```python
split = attention._split_heads(x)
assert split.shape == (2, 4, 8, 16)
```

## What does it do?

Tests the `_split_heads()` transformation.

## Why is it there?

Head splitting is one of the most error-prone parts of self-attention implementations.

## Honest comment

This tests a private method, which is usually discouraged, but justified here due to the importance of tensor transformations.

---

# `test_combine_heads_returns_expected_shape`

Tests `_combine_heads()`:

* input: `(2, 4, 8, 16)`
* output: `(2, 8, 64)`

## Why is it important?

Split and combine must work together correctly.

---

# `test_combine_heads_raises_error_for_invalid_number_of_heads`

## What does it do?

Ensures `_combine_heads()` fails early when the number of heads is incorrect.

## Why is it there?

Protects internal tensor invariants.

---

# `test_combine_heads_raises_error_for_invalid_head_dim`

Similarly validates mismatch in `head_dim`.

## Why is it important?

Because `d_model = n_heads * head_dim` is a fundamental relationship.

---

# `test_self_attention_supports_3d_attention_mask`

```python
attention_mask = torch.ones(2, 5, 5)
...
assert attention_weights.shape == (2, 4, 5, 5)
```

## What does it do?

Tests support for 3D mask format.

## Why is it important?

Ensures user-friendly API. Users do not always provide 4D masks.

---

# `test_self_attention_supports_4d_attention_mask`

Tests direct support for 4D masks.

## Why is it there?

Supports advanced use cases.

---

# `test_self_attention_mask_blocks_selected_positions`

```python
attention_mask = torch.tensor([[[[1, 0], [1, 0]]]])
...
assert torch.allclose(attention_weights[..., 1], torch.zeros_like(...), atol=1e-6)
```

## What does it do?

Validates that masked positions receive zero attention weight.

## Why is it important?

This test verifies **semantic behavior**, not just shape.

## Strength

One of the most valuable tests in the file.

---

# `test_self_attention_preserves_deterministic_behavior_when_dropout_is_zero`

```python
torch.manual_seed(42)
...
assert torch.allclose(output_1, output_2)
assert torch.allclose(weights_1, weights_2)
```

## What does it do?

Ensures deterministic behavior when dropout is disabled.

## Why is it there?

Confirms absence of hidden randomness.

## Strength

Very valuable contract test.

---

# Additional Technical Analysis as a Python File

## Function signatures

Contains:

* one helper function
* multiple pytest test functions

All return `-> None`.

## Parameters

No fixtures are used.

### Pros

* readability
* independence

### Cons

* some duplication

---

## Return values

No direct returns; assertion-based success.

## Type hints

Present and correct.

## Exception handling

Uses `pytest.raises(...)` properly.

## Dependencies

Minimal.

## OOP design

Function-based pytest approach. Appropriate.

## Maintainability

Good.

## Readability

High.

## Scalability

May need splitting if expanded.

---

# Critical Evaluation

## Strengths

### 1. Tests both shape and behavior

### 2. Tests internal tensor transformations

### 3. Supports both 3D and 4D masks

### 4. Includes determinism testing

---

## Weaknesses

### 1. Private method testing

### 2. No dropout > 0 tests

### 3. Limited numerical validation

### 4. Projection layer behavior not deeply tested

---

## Potential risks

* implementation changes may break tests unnecessarily
* mask semantics may evolve
* all-masked case not tested
* dtype/device mismatch not tested

---

## Security / performance / testability

### Security

No concerns.

### Performance

Efficient.

### Testability

High.

---

# Missing but Important Points

### 1. All-masked row test

### 2. Invalid mask shape semantics

### 3. Input dtype test

### 4. Dropout train/eval behavior

---

# Questions for Senior Code Review

1. Is testing private methods necessary here?
2. Why is all-masked scenario not tested?
3. Should the name be `MultiHeadSelfAttention` instead?
4. Why no invalid mask semantic tests?
5. Should train/eval behavior be tested?
6. Is this too implementation-aware?

---

# What Problems Arise If This File Grows?

If attention expands:

* cross-attention
* causal attention
* flash attention
* custom masks

Then this file will become too large.

Possible refactor:

* `test_self_attention_shapes.py`
* `test_self_attention_masks.py`
* `test_self_attention_internals.py`

---

# Summary

## One-sentence definition

This file is a **unit test file that validates the shape, validation, mask behavior, internal tensor transformations, and deterministic execution contract of the `SelfAttention` layer**.

## Three most critical takeaways

1. **Testing self-attention requires validating both shape and mask semantics.**
2. **Private method testing can be justified when tensor transformations are critical.**
3. **Determinism and finite output are essential checks in sensitive layers like attention.**

## Key concepts for interviews

* unit test design
* `pytest.raises`
* shape contract testing
* multi-head self-attention
* head split / combine logic
* attention mask broadcasting
* `torch.allclose`
* deterministic behavior
* implementation-aware vs black-box testing

