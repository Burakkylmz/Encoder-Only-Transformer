from __future__ import annotations

import pytest
import torch

from encoder_only_transformer.data.datasets import (
    TextClassificationDataset,
    TextClassificationSample,
    build_attention_mask,
    build_padding_mask,
    compare_padding_strategies,
    make_collate_fn,
    make_toy_dataloader,
)


PAD = 0

SAMPLES = [
    TextClassificationSample(input_ids=[1, 2, 3, 4, 5], label=0),   # length 5
    TextClassificationSample(input_ids=[6, 7],          label=1),   # length 2
    TextClassificationSample(input_ids=[8, 9, 10],      label=0),   # length 3
    TextClassificationSample(input_ids=[11],             label=1),   # length 1
]



class TestTextClassificationDataset:
    def test_len(self):
        ds = TextClassificationDataset(SAMPLES)
        assert len(ds) == 4

    def test_getitem_returns_correct_sample(self):
        ds = TextClassificationDataset(SAMPLES)
        assert ds[0].input_ids == [1, 2, 3, 4, 5]
        assert ds[0].label == 0

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            TextClassificationDataset([])



class TestCollateFnDynamic:
    def setup_method(self):
        self.collate_fn = make_collate_fn(pad_token_id=PAD, max_seq_len=None)

    def test_output_shapes(self):
        input_ids, labels, padding_mask = self.collate_fn(SAMPLES)
        batch_size = len(SAMPLES)
        max_len = max(len(s.input_ids) for s in SAMPLES)

        assert input_ids.shape == (batch_size, max_len)
        assert labels.shape == (batch_size,)
        assert padding_mask.shape == (batch_size, max_len)

    def test_padding_filled_with_pad_token(self):
        input_ids, _, _ = self.collate_fn(SAMPLES)
        assert input_ids[1, 2:].tolist() == [PAD, PAD, PAD]

    def test_real_tokens_preserved(self):
        input_ids, _, _ = self.collate_fn(SAMPLES)
        assert input_ids[0, :5].tolist() == [1, 2, 3, 4, 5]

    def test_labels_correct(self):
        _, labels, _ = self.collate_fn(SAMPLES)
        assert labels.tolist() == [0, 1, 0, 1]

    def test_padding_mask_values(self):
        _, _, padding_mask = self.collate_fn(SAMPLES)
        # İlk örnek 5 token: hepsi 1
        assert padding_mask[0].tolist() == [1, 1, 1, 1, 1]
        # İkinci örnek 2 token: 2 tane 1, 3 tane 0
        assert padding_mask[1].tolist() == [1, 1, 0, 0, 0]

    def test_dtypes(self):
        input_ids, labels, padding_mask = self.collate_fn(SAMPLES)
        assert input_ids.dtype == torch.long
        assert labels.dtype == torch.long
        assert padding_mask.dtype == torch.long

    def test_pads_to_batch_max_not_global_max(self):
        short_batch = [
            TextClassificationSample(input_ids=[1, 2], label=0),
            TextClassificationSample(input_ids=[3],    label=1),
        ]
        input_ids, _, _ = self.collate_fn(short_batch)
        assert input_ids.shape[1] == 2



class TestCollateFnStatic:
    MAX_SEQ_LEN = 8

    def setup_method(self):
        self.collate_fn = make_collate_fn(pad_token_id=PAD, max_seq_len=self.MAX_SEQ_LEN)

    def test_seq_len_always_equals_max_seq_len(self):
        input_ids, _, padding_mask = self.collate_fn(SAMPLES)
        assert input_ids.shape[1]    == self.MAX_SEQ_LEN
        assert padding_mask.shape[1] == self.MAX_SEQ_LEN

    def test_truncation(self):
        long_sample = [TextClassificationSample(input_ids=list(range(1, 20)), label=0)]
        input_ids, _, _ = self.collate_fn(long_sample)
        assert input_ids.shape[1] == self.MAX_SEQ_LEN
        assert input_ids[0].tolist() == list(range(1, self.MAX_SEQ_LEN + 1))

    def test_short_batch_still_max_seq_len(self):
        short_batch = [TextClassificationSample(input_ids=[1], label=0)]
        input_ids, _, _ = self.collate_fn(short_batch)
        assert input_ids.shape[1] == self.MAX_SEQ_LEN



class TestBuildPaddingMask:
    def test_shape_preserved(self):
        ids = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        mask = build_padding_mask(ids, pad_token_id=0)
        assert mask.shape == ids.shape

    def test_real_tokens_are_one(self):
        ids = torch.tensor([[1, 2, 0, 0]])
        mask = build_padding_mask(ids, pad_token_id=0)
        assert mask[0, 0].item() == 1
        assert mask[0, 1].item() == 1

    def test_pad_tokens_are_zero(self):
        ids = torch.tensor([[1, 2, 0, 0]])
        mask = build_padding_mask(ids, pad_token_id=0)
        assert mask[0, 2].item() == 0
        assert mask[0, 3].item() == 0

    def test_no_padding_all_ones(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        mask = build_padding_mask(ids, pad_token_id=0)
        assert mask.tolist() == [[1, 1, 1, 1]]

    def test_all_padding_all_zeros(self):
        ids = torch.tensor([[0, 0, 0]])
        mask = build_padding_mask(ids, pad_token_id=0)
        assert mask.tolist() == [[0, 0, 0]]

    def test_dtype_is_long(self):
        ids = torch.tensor([[1, 0]])
        mask = build_padding_mask(ids, pad_token_id=0)
        assert mask.dtype == torch.long



class TestBuildAttentionMask:
    def test_output_shape(self):
        padding_mask = torch.ones(4, 7, dtype=torch.long)
        attn_mask = build_attention_mask(padding_mask)
        assert attn_mask.shape == (4, 1, 1, 7)

    def test_values_unchanged(self):
        padding_mask = torch.tensor([[1, 1, 0, 0]])
        attn_mask = build_attention_mask(padding_mask)
        assert attn_mask[0, 0, 0].tolist() == [1, 1, 0, 0]

    def test_broadcastable_over_heads_and_queries(self):
        # (batch, 1, 1, seq_len) x (batch, n_heads, seq_len, seq_len) broadcast kontrolü
        batch, n_heads, seq_len = 2, 4, 6
        padding_mask = torch.ones(batch, seq_len, dtype=torch.long)
        attn_mask = build_attention_mask(padding_mask)
        scores = torch.zeros(batch, n_heads, seq_len, seq_len)
        result = scores * attn_mask
        assert result.shape == (batch, n_heads, seq_len, seq_len)



class TestMakeToyDataloader:
    def test_returns_dataloader(self):
        from torch.utils.data import DataLoader
        loader = make_toy_dataloader()
        assert isinstance(loader, DataLoader)

    def test_default_dynamic_padding(self):
        loader = make_toy_dataloader(batch_size=8)
        input_ids, labels, padding_mask = next(iter(loader))
        # Dinamik -> seq_len batch içindeki en uzun dizi
        assert input_ids.shape[1] <= 8  # toy datanın max uzunluğu

    def test_static_padding(self):
        loader = make_toy_dataloader(batch_size=8, max_seq_len=20)
        input_ids, _, _ = next(iter(loader))
        assert input_ids.shape[1] == 20

    def test_batch_size_respected(self):
        loader = make_toy_dataloader(batch_size=3)
        input_ids, labels, _ = next(iter(loader))
        assert input_ids.shape[0] == 3
        assert labels.shape[0]    == 3



class TestComparePaddingStrategies:
    def test_static_total_tokens(self):
        result = compare_padding_strategies(SAMPLES, pad_token_id=PAD, max_seq_len=10)
        assert result.static.total_tokens == 10 * len(SAMPLES)

    def test_dynamic_total_tokens(self):
        result = compare_padding_strategies(SAMPLES, pad_token_id=PAD, max_seq_len=10)
        max_len = max(len(s.input_ids) for s in SAMPLES)
        assert result.dynamic.total_tokens == max_len * len(SAMPLES)

    def test_static_has_more_or_equal_tokens(self):
        result = compare_padding_strategies(SAMPLES, pad_token_id=PAD, max_seq_len=10)
        assert result.static.total_tokens >= result.dynamic.total_tokens

    def test_padding_ratios_in_range(self):
        result = compare_padding_strategies(SAMPLES, pad_token_id=PAD, max_seq_len=10)
        assert 0.0 <= result.static.padding_ratio  <= 1.0
        assert 0.0 <= result.dynamic.padding_ratio <= 1.0