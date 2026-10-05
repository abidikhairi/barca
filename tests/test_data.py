"""Tests for dataset splitting, tokenization and packing (fake tokenizer)."""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from barca.config import Config
from barca.data import build_tokenized_datasets, resolve_splits


class FakeTokenizer:
    """Minimal character-level tokenizer implementing the bits we use."""

    eos_token_id = 2
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    bos_token = "<bos>"
    name_or_path = "fake"

    def __call__(self, texts, truncation=True, max_length=None, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[(ord(c) % 20) + 3 for c in text] for text in texts]
        if truncation and max_length is not None:
            ids = [row[:max_length] for row in ids]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}


def _make_dataset(tmp_path, splits=True):
    rows = {"sequence": ["ACDEFGHIK", "MNPQRSTVWY", "ACDE", "GGGGWWWW", "LLLL"]}
    dataset = Dataset.from_dict(rows)
    path = tmp_path / "raw"
    if splits:
        DatasetDict({"train": dataset, "validation": dataset.select([0, 1])}).save_to_disk(path)
    else:
        dataset.save_to_disk(path)
    return str(path)


def _cfg(path, **data_overrides):
    data = {"path": path, "text_column": "sequence", "num_proc": None}
    data.update(data_overrides)
    return Config.from_dict({"data": data, "tokenizer": {"name_or_path": "fake"}})


def test_resolve_splits_from_datasetdict(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path))
    train, eval_ds, test_ds = resolve_splits(cfg)
    assert train is not None and eval_ds is not None and test_ds is None
    assert eval_ds.column_names == ["sequence"]


def test_resolve_splits_from_single_dataset(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path, splits=False), eval_fraction=0.4)
    train, eval_ds, _ = resolve_splits(cfg)
    assert len(train) + len(eval_ds) == 5
    assert 0 < len(eval_ds) < 5


def test_non_packing_truncates(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path), max_seq_length=4)
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    lengths = {len(ids) for ids in datasets["train"]["input_ids"]}
    assert lengths == {4}


def test_packing_produces_fixed_blocks(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path), max_seq_length=8, packing=True, packing_add_eos=True)
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    lengths = {len(ids) for ids in datasets["train"]["input_ids"]}
    assert lengths == {8}
    assert "labels" not in datasets["train"].column_names
    # EOS separators must appear across the packed stream.
    assert any(
        FakeTokenizer.eos_token_id in block for block in datasets["train"]["input_ids"]
    )


def test_columns_are_limited_to_model_inputs(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path), max_seq_length=5)
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    assert set(datasets["train"].column_names) <= {"input_ids", "attention_mask"}
