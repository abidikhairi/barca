"""Tests for dataset loading, splitting, tokenization and packing."""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from barca.config import Config
from barca.data import build_tokenized_datasets, is_tokenized, load_splits


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


def _sequences():
    return {"sequence": ["ACDEFGHIK", "MNPQRSTVWY", "ACDE", "GGGGWWWW", "LLLL"]}


def _make_dataset(tmp_path, splits=True, name="raw"):
    dataset = Dataset.from_dict(_sequences())
    path = tmp_path / name
    if splits:
        DatasetDict({"train": dataset, "validation": dataset.select([0, 1])}).save_to_disk(path)
    else:
        dataset.save_to_disk(path)
    return str(path)


def _make_split_root(tmp_path, tokenized=True, name="barca"):
    """Create a .../barca/{train,valid} split root (optionally pre-tokenized)."""
    root = tmp_path / name
    if tokenized:
        rows = {
            "input_ids": [[1, 4, 5, 6], [7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8]],
            "attention_mask": [[1] * 4, [1] * 3, [1] * 8],
        }
    else:
        rows = {"sequence": ["ACDEF", "MNPQ", "GGGWW"]}
    Dataset.from_dict(rows).save_to_disk(root / "train")
    Dataset.from_dict(rows).save_to_disk(root / "valid")
    return str(root)


def _cfg(path, **data_overrides):
    data = {"path": path, "text_column": "sequence", "num_proc": None}
    data.update(data_overrides)
    return Config.from_dict({"data": data, "tokenizer": {"name_or_path": "fake"}})


# --------------------------------------------------------------------------- #
# Layout A: single root
# --------------------------------------------------------------------------- #
def test_load_splits_from_datasetdict(tmp_path):
    splits = load_splits(_cfg(_make_dataset(tmp_path)))
    assert set(splits) == {"train", "eval"}
    assert len(splits["eval"]) == 2


def test_load_splits_from_single_dataset(tmp_path):
    splits = load_splits(_cfg(_make_dataset(tmp_path, splits=False), eval_fraction=0.4))
    assert len(splits["train"]) + len(splits["eval"]) == 5
    assert 0 < len(splits["eval"]) < 5


def test_non_packing_truncates(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path), max_seq_length=4)
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    assert {len(ids) for ids in datasets["train"]["input_ids"]} == {4}


def test_packing_produces_fixed_blocks(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path), max_seq_length=8, packing=True, packing_add_eos=True)
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    assert {len(ids) for ids in datasets["train"]["input_ids"]} == {8}
    assert any(
        FakeTokenizer.eos_token_id in block for block in datasets["train"]["input_ids"]
    )


def test_columns_are_limited_to_model_inputs(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path), max_seq_length=5)
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    assert set(datasets["train"].column_names) <= {"input_ids", "attention_mask"}


# --------------------------------------------------------------------------- #
# Layout B: split root with one subdirectory per split
# --------------------------------------------------------------------------- #
def test_split_root_autodetects_pretokenized(tmp_path):
    cfg = _cfg(_make_split_root(tmp_path, tokenized=True), train_split="train", eval_split="valid")
    splits = load_splits(cfg)
    assert set(splits) == {"train", "eval"}
    assert is_tokenized(splits["train"])

    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    # Passed through untouched, no text_column required.
    assert set(datasets["train"].column_names) == {"input_ids", "attention_mask"}
    assert datasets["train"]["input_ids"][0] == [1, 4, 5, 6]


def test_split_root_with_explicit_tokenized_flag(tmp_path):
    cfg = _cfg(
        _make_split_root(tmp_path, tokenized=True),
        train_split="train",
        eval_split="valid",
        tokenized=True,
    )
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    assert len(datasets["train"]) == 3
    assert len(datasets["eval"]) == 3


def test_tokenized_split_root_ignores_packing(tmp_path):
    cfg = _cfg(
        _make_split_root(tmp_path, tokenized=True),
        train_split="train",
        eval_split="valid",
        tokenized=True,
        packing=True,
    )
    datasets = build_tokenized_datasets(cfg, FakeTokenizer())
    # Pre-packed data is not re-packed; the short block stays short.
    assert len(datasets["train"]["input_ids"][1]) == 3


def test_explicit_tokenized_without_input_ids_raises(tmp_path):
    cfg = _cfg(_make_dataset(tmp_path), tokenized=True)
    try:
        build_tokenized_datasets(cfg, FakeTokenizer())
    except ValueError as exc:
        assert "lacks 'input_ids'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
