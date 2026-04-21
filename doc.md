
### Config
- config/default.yaml
- src/encoder_only_transformer/config/config.py
- tests/test_config.py
   - RUN: uv run pytest tests/test_config.py -v

### Embeddings
- src/encoder_only_transformer/layers/embeddings.py
- tests/test_embeddings.py
   - RUN: uv run pytest tests/test_embeddings.py -v

### Attention
- src/encoder_only_transformer/layers/attention.py
- tests/test_self_attention.py.py
   - RUN: uv run pytest tests/test_self_attention.py -v

### Feed Forward
- src/encoder_only_transformer/layers/feed_forward.py
- tests/test_feed_forward.py
  - RUN: uv run pytest tests/test_feed_forward.py -v

---

### Encode Block
- src/encoder_only_transformer/blocks/encoder_block.py
- tests/test_encoder_block.py
  - RUN: uv run pytest tests/test_encoder_block.py -v

### Encoder
- src/encoder_only_transformer/models/encoder.py
- tests/test_encoder.py
  - RUN: uv run pytest tests/test_encoder.py -v

### Pooling
- src/encoder_only_transformer/layers/pooling.py
- tests/test_pooling.py
  - RUN: uv run pytest tests/test_pooling.py -v

### Head
- src/encoder_only_transformer/models/heads.py
- tests/test_heads.py
  - RUN: tests/test_heads.py


### Model
- src/encoder_only_transformer/models/model.py
- tests/test_model.py
  - RUN: uv run pytest tests/test_model.py -v


### Factory Design Pattern
- src/encoder_only_transformer/factories/factories.py
- tests/test_factories.py
  - RUN: uv run pytest tests/test_factories.py -v


### Trainer
- src/encoder_only_transformer/training/trainer.py
- tests/test_trainer.py
  - RUN: uv run pytest tests/test_trainer.py -v

### Metrics
- src/encoder_only_transformer/training/metrics.py
- tests/test_metrics.py
  - RUN: uv run pytest tests/test_metrics.py -v