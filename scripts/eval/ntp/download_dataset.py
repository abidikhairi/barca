#!/usr/bin/env python
"""Download a small next-token-prediction eval sample from Cosmopedia.

HF Dataset: HuggingFaceTB/cosmopedia
Task: from each subset save 10 rows to a CSV file with the following columns: 'text'

Subsets are streamed, so only the first rows of each are fetched rather than the
full dataset.

Usage:
    python scripts/eval/ntp/download_dataset.py
    python scripts/eval/ntp/download_dataset.py --output data/ntp_eval.csv --rows-per-subset 10
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

DATASET = "HuggingFaceTB/cosmopedia"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--output", default="cosmopedia_ntp.csv", help="Output CSV path.")
    p.add_argument("--rows-per-subset", type=int, default=10)
    p.add_argument("--split", default="train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subsets = get_dataset_config_names(args.dataset)
    print(f"{args.dataset}: {len(subsets)} subsets -> {', '.join(subsets)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for subset in subsets:
            stream = load_dataset(args.dataset, subset, split=args.split, streaming=True)
            n = 0
            for row in stream:
                writer.writerow([row["text"]])
                n += 1
                if n >= args.rows_per_subset:
                    break
            print(f"  {subset}: {n} rows")
            total += n

    print(f"Wrote {total} rows to {out}")


if __name__ == "__main__":
    main()
