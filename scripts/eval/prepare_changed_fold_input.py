#!/usr/bin/env python
"""Write a ColabFold FASTA for changed valid GRPO candidates only.

Candidates whose sequence is identical to the paired SFT baseline are skipped;
their SFT fold scores are reused by ``compare_grpo_sft_folds.py --reuse-identical``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sft-json", type=Path, required=True)
    parser.add_argument("--grpo-json", type=Path, required=True)
    parser.add_argument("--output-fasta", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--min-len", type=int, default=30)
    parser.add_argument("--max-len", type=int, default=512)
    return parser.parse_args()


def name(row: dict) -> str:
    return f"val{row['idx']}_{row['accession']}_gen{row['candidate']}"


def valid(row: dict, min_len: int, max_len: int) -> bool:
    return bool(row["closed"] and row["canon"] and min_len <= row["gen_len"] <= max_len)


def main() -> None:
    args = parse_args()
    sft_rows = {name(row): row for row in json.load(open(args.sft_json))["rows"]}
    grpo_rows = json.load(open(args.grpo_json))["rows"]
    changed, reused, invalid = [], [], []
    with args.output_fasta.open("w") as handle:
        for row in grpo_rows:
            candidate = name(row)
            if not valid(row, args.min_len, args.max_len):
                invalid.append(candidate)
                continue
            baseline = sft_rows.get(candidate)
            if baseline is not None and baseline["gen_seq"] == row["gen_seq"]:
                reused.append(candidate)
                continue
            changed.append(candidate)
            handle.write(f">{candidate}\n{row['gen_seq']}\n")
    json.dump(
        {"changed": changed, "reused": reused, "invalid": invalid},
        open(args.output_map, "w"),
        indent=2,
    )
    print(f"changed valid {len(changed)}; reused identical {len(reused)}; invalid {len(invalid)}")
    print(f"wrote {args.output_fasta}")


if __name__ == "__main__":
    main()
