#!/usr/bin/env python
"""Split a saved dataset into train/validation/test and save the result.

Useful when the source dataset is a single ``Dataset`` (or a ``DatasetDict``
whose splits you want to re-derive).

Usage:
    python scripts/data/split_dataset.py /data/proteins/raw \
        --output /data/proteins/processed \
        --eval-fraction 0.02 --test-fraction 0.02 --seed 42
"""

from __future__ import annotations

import argparse

from datasets import Dataset, DatasetDict, load_from_disk


def split_once(dataset: Dataset, fraction: float, seed: int):
    if fraction <= 0:
        return dataset, None
    split = dataset.train_test_split(test_size=fraction, seed=seed, shuffle=True)
    return split["train"], split["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Source path for datasets.load_from_disk.")
    parser.add_argument("--output", required=True, help="Destination path for save_to_disk.")
    parser.add_argument("--source-split", default=None, help="Split to use if source is a DatasetDict.")
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = load_from_disk(args.path)
    if isinstance(source, DatasetDict):
        key = args.source_split or "train"
        if key not in source:
            raise ValueError(f"Split {key!r} not in {sorted(source.keys())}")
        dataset = source[key]
    elif isinstance(source, Dataset):
        dataset = source
    else:
        raise TypeError(f"Unsupported dataset type: {type(source).__name__}")

    print(f"Source examples: {len(dataset):,}")
    train, test = split_once(dataset, args.test_fraction, args.seed)
    train, eval_ds = split_once(train, args.eval_fraction, args.seed)

    result = DatasetDict({"train": train})
    if eval_ds is not None:
        result["validation"] = eval_ds
    if test is not None:
        result["test"] = test

    result.save_to_disk(args.output)
    print(f"Saved splits {dict(result.num_rows)} to {args.output}")


if __name__ == "__main__":
    main()
