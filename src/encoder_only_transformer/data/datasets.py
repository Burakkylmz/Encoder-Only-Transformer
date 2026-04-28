

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True, slots=True)
class TextClassificationSample:
    """
    Tek bir veri örneği: token id'leri + etiket
    """
    input_ids: list[int]
    label: int


class TextClassificationDataset(Dataset):
    """
    Sequence classification için PyTorch Dataset.

    Her __getitem__ çağrısı ham bir TextClassificationSample döndürür.
    Padding burada yapılmaz, batch oluşturulurken collate_fn halleder.
    """

    def __init__(self, samples: list[TextClassificationSample]) -> None:
        if not samples:
            raise ValueError("samples listesi boş olamaz.")
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> TextClassificationSample:
        return self._samples[idx]


# Padding stratejileri
def _pad_sequence(seq: list[int], target_len: int, pad_token_id: int) -> list[int]:
    """
    Verilen uzunluğa padding uygular (sağdan)

    """
    return seq[:target_len] + [pad_token_id] * max(0, target_len - len(seq))


def make_collate_fn(
    pad_token_id: int,
    max_seq_len: int | None = None,
) -> Callable[[list[TextClassificationSample]], tuple[Tensor, Tensor, Tensor]]:


    def collate_fn(
        batch: list[TextClassificationSample],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if max_seq_len is not None:
            # Statik: sabit uzunluk
            target_len = max_seq_len
        else:
            # Dinamik: bu batch'teki en uzun dizi
            target_len = max(len(s.input_ids) for s in batch)

        padded_ids = [
            _pad_sequence(s.input_ids, target_len, pad_token_id)
            for s in batch
        ]

        input_ids = torch.tensor(padded_ids, dtype=torch.long)
        labels = torch.tensor([s.label for s in batch], dtype=torch.long)
        padding_mask = build_padding_mask(input_ids, pad_token_id)

        return input_ids, labels, padding_mask

    return collate_fn



# Attention Mask
def build_padding_mask(input_ids: Tensor, pad_token_id: int) -> Tensor:
    """
    2D padding mask üretir.
    """
    return (input_ids != pad_token_id).to(dtype=torch.long)


def build_attention_mask(padding_mask: Tensor) -> Tensor:
    return padding_mask.unsqueeze(1).unsqueeze(2)



# Toy Dataset Loader

# Küçük toy dayaset (token_id listesi, etiket)
_TOY_SAMPLES: list[tuple[list[int], int]] = [
    ([1, 2, 3, 4, 5],             0),
    ([1, 6, 7],                   1),
    ([8, 9, 10, 11, 12, 13, 14], 0),
    ([2, 3],                      1),
    ([15, 16, 17, 18],            0),
    ([19, 20, 21, 22, 23, 24],   1),
    ([5, 3, 8],                   0),
    ([11, 12, 13, 14, 15, 16, 17, 18], 1),
]


def make_toy_dataloader(
    batch_size: int = 2,
    pad_token_id: int = 0,
    max_seq_len: int | None = None,
    shuffle: bool = False,
) -> DataLoader:
    samples = [
        TextClassificationSample(input_ids=ids, label=label)
        for ids, label in _TOY_SAMPLES
    ]
    dataset = TextClassificationDataset(samples)
    collate_fn = make_collate_fn(pad_token_id=pad_token_id, max_seq_len=max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )



# Dinamik vs Statik Padding
@dataclass(frozen=True, slots=True)
class PaddingStrategyStats:
    total_tokens: int
    padding_tokens: int
    padding_ratio: float


@dataclass(frozen=True, slots=True)
class PaddingComparisonResult:
    static: PaddingStrategyStats
    dynamic: PaddingStrategyStats

    def print_report(self) -> None:
        print("=== Padding Stratejisi Karşılaştırması ===")
        print(f"{'':20} {'Statik':>10} {'Dinamik':>10}")
        print(f"{'Toplam token':20} {self.static.total_tokens:>10} {self.dynamic.total_tokens:>10}")
        print(f"{'Padding token':20} {self.static.padding_tokens:>10} {self.dynamic.padding_tokens:>10}")
        print(f"{'Padding oranı':20} {self.static.padding_ratio:>10.1%} {self.dynamic.padding_ratio:>10.1%}")
        saved = self.static.total_tokens - self.dynamic.total_tokens
        print(f"\nDinamik padding {saved} token tasarruf eder "
                f"({saved / self.static.total_tokens:.1%} azalma).")


def compare_padding_strategies(
    samples: list[TextClassificationSample],
    pad_token_id: int,
    max_seq_len: int,
) -> PaddingComparisonResult:
    lengths = [min(len(s.input_ids), max_seq_len) for s in samples]
    n = len(samples)

    static_target = max_seq_len
    static_total = static_target * n
    static_padding = sum(static_target - l for l in lengths)

    dynamic_target = max(lengths)
    dynamic_total = dynamic_target * n
    dynamic_padding = sum(dynamic_target - l for l in lengths)

    return PaddingComparisonResult(
        static=PaddingStrategyStats(
            total_tokens=static_total,
            padding_tokens=static_padding,
            padding_ratio=static_padding / static_total,
        ),
        dynamic=PaddingStrategyStats(
            total_tokens=dynamic_total,
            padding_tokens=dynamic_padding,
            padding_ratio=dynamic_padding / dynamic_total if dynamic_total > 0 else 0.0,
        ),
    )


def benchmark_padding_speed(
    samples: list[TextClassificationSample],
    pad_token_id: int,
    max_seq_len: int,
    batch_size: int = 4,
    n_runs: int = 100,
) -> None:
    """
    Statik vs dinamik padding'in DataLoader üzerindeki hızını karşılaştırır.
    """
    dataset = TextClassificationDataset(samples)

    static_fn = make_collate_fn(pad_token_id=pad_token_id, max_seq_len=max_seq_len)
    dynamic_fn = make_collate_fn(pad_token_id=pad_token_id, max_seq_len=None)

    static_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=static_fn)
    dynamic_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=dynamic_fn)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        for _ in static_loader:
            pass
    static_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_runs):
        for _ in dynamic_loader:
            pass
    dynamic_time = time.perf_counter() - t0

    print(f"=== Hız Karşılaştırması ({n_runs} iterasyon) ===")
    print(f"Statik  padding: {static_time * 1000:.1f} ms")
    print(f"Dinamik padding: {dynamic_time * 1000:.1f} ms")
    faster = "Dinamik" if dynamic_time < static_time else "Statik"
    ratio = max(static_time, dynamic_time) / min(static_time, dynamic_time)
    print(f"{faster} {ratio:.2f}x daha hızlı.")