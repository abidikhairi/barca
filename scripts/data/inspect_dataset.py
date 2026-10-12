#!/usr/bin/env python
"""Inspect a dataset saved with ``datasets.save_to_disk``.

Supports both layouts:
  * a single dataset root (``DatasetDict`` or ``Dataset``), and
  * a split root holding one dataset subdirectory per split
    (e.g. ``/datasets/uniref50/barca/{train,valid}``).

Reports splits, features, row counts and length statistics. For
pre-tokenized data the ``input_ids`` lengths are reported instead of raw
sequence lengths.

Usage:
    python scripts/data/inspect_dataset.py /home/kabidi/workspace/datasets/uniref50/barca
    python scripts/data/inspect_dataset.py /data/raw --column sequence --tokenizer khairi/barca-0.8b-base
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict

from datasets import Dataset, DatasetDict, load_from_disk


def _is_dataset_dir(path: Path) -> bool:
    return (
        (path / "dataset_info.json").exists()
        or (path / "state.json").exists()
        or any(path.glob("*.arrow"))
    )


def discover(path: str) -> Dict[str, Dataset]:
    """Return ``{split_name: dataset}`` for either supported layout."""
    root = Path(path)
    if _is_dataset_dir(root):
        source = load_from_disk(str(root))
        if isinstance(source, DatasetDict):
            return {name: source[name] for name in sorted(source.keys())}
        return {root.name: source}

    splits = {
        sub.name: load_from_disk(str(sub))
        for sub in sorted(root.iterdir())
        if sub.is_dir() and _is_dataset_dir(sub)
    }
    if not splits:
        source = load_from_disk(str(root))
        return {root.name: source}
    return splits


def _length_stats(lengths: list) -> dict:
    if not lengths:
        return {}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "median": int(statistics.median(lengths)),
        "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
    }


def _print_split(name: str, dataset: Dataset, column: str, tokenizer) -> None:
    print(f"\n[{name}] {len(dataset):,} examples")
    print(f"  columns : {dataset.column_names}")
    print(f"  features: {dataset.features}")

    sample = dataset.select(range(min(2000, len(dataset))))

    if "input_ids" in dataset.column_names:
        stats = _length_stats([len(ids) for ids in sample["input_ids"]])
        print(f"  input_ids length (tokens, pre-tokenized): {stats}")
        first = sample["input_ids"][0]
        preview = first if len(first) <= 30 else first[:30] + ["..."]
        print(f"  example : {preview}")
        return

    if column not in dataset.column_names:
        print(f"  ! column {column!r} not found; skipping length stats")
        return

    sequences = sample[column]
    if tokenizer is None:
        stats = _length_stats([len(s) for s in sequences])
        print(f"  {column!r} length (characters): {stats}")
    else:
        lengths = [len(tokenizer(s)["input_ids"]) for s in sequences]
        stats = _length_stats(lengths)
        print(f"  {column!r} length (tokens, {tokenizer.name_or_path}): {stats}")

    first = sequences[0]
    preview = first if len(first) <= 120 else first[:120] + "..."
    print(f"  example : {preview!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Dataset root or split root.")
    parser.add_argument("--column", default="sequence", help="Sequence column to summarize.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional HF tokenizer id to report token lengths for raw sequences.",
    )
    args = parser.parse_args()

    splits = discover(args.path)
    print(f"Loaded {args.path} -> {len(splits)} split(s): {', '.join(splits)}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for name, dataset in splits.items():
        _print_split(name, dataset, args.column, tokenizer)


if __name__ == "__main__":
    main()
