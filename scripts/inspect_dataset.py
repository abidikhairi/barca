#!/usr/bin/env python
"""Inspect a dataset saved with ``datasets.save_to_disk``.

Prints splits, features, row counts and sequence-length statistics before you
commit to a training recipe.

Usage:
    python scripts/inspect_dataset.py /data/proteins/processed
    python scripts/inspect_dataset.py /data/proteins/processed --tokenizer khairi/barca-0.8b-base
"""

from __future__ import annotations

import argparse
import statistics
from typing import Iterable

from datasets import Dataset, DatasetDict, load_from_disk


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
    if column not in dataset.column_names:
        print(f"  ! column {column!r} not found; skipping length stats")
        return

    sample = dataset.select(range(min(2000, len(dataset))))
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
    parser.add_argument("path", help="Path passed to datasets.load_from_disk.")
    parser.add_argument("--column", default="sequence", help="Sequence column to summarize.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional HF tokenizer id to report token lengths instead of characters.",
    )
    args = parser.parse_args()

    source = load_from_disk(args.path)
    print(f"Loaded {args.path} -> {type(source).__name__}")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    splits: Iterable = (
        sorted(source.items()) if isinstance(source, DatasetDict) else [("dataset", source)]
    )
    for name, dataset in splits:
        _print_split(name, dataset, args.column, tokenizer)


if __name__ == "__main__":
    main()
