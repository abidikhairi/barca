#!/usr/bin/env python
"""Filter a generated FASTA down to valid protein sequences.

Keeps sequences that are plausible proteins and foldable by the ESMFold API:
  * residues drawn from the 20 standard amino acids,
  * length within [--min-len, --max-len],
  * Shannon entropy of residue composition >= --min-entropy (drops low-complexity),
  * no homopolymer run longer than --max-run.

Usage:
    python scripts/generation/filter_proteins.py in.fasta out.fasta
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import List, Tuple

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--min-len", type=int, default=30)
    p.add_argument("--max-len", type=int, default=400)
    p.add_argument("--min-entropy", type=float, default=3.0, help="bits (max ~4.32 for 20 aa).")
    p.add_argument("--max-run", type=int, default=10, help="max consecutive identical residues.")
    return p.parse_args()


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read FASTA, joining wrapped/multi-line sequence data into one string."""
    records: List[Tuple[str, str]] = []
    header = None
    buf: List[str] = []
    for line in open(path, encoding="utf-8"):
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(buf)))
            header = line[1:].strip()
            buf = []
        else:
            buf.append(line.strip())
    if header is not None:
        records.append((header, "".join(buf)))
    return records


def entropy(seq: str) -> float:
    counts = Counter(seq)
    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def max_run(seq: str) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best if seq else 0


def main() -> None:
    args = parse_args()
    records = read_fasta(args.input)

    kept, dropped = [], []
    for header, seq in records:
        reasons = []
        if not set(seq) <= STANDARD:
            reasons.append("non-standard residues")
        if not (args.min_len <= len(seq) <= args.max_len):
            reasons.append(f"length {len(seq)}")
        elif entropy(seq) < args.min_entropy:
            reasons.append(f"low complexity (H={entropy(seq):.2f})")
        elif max_run(seq) > args.max_run:
            reasons.append(f"homopolymer run {max_run(seq)}")
        (dropped if reasons else kept).append((header, seq, reasons))

    with open(args.output, "w", encoding="utf-8") as fh:
        for header, seq, _ in kept:
            fh.write(f">{header} length={len(seq)}\n{seq}\n")

    print(f"input {len(records)} -> kept {len(kept)}, dropped {len(dropped)}")
    for header, seq, reasons in dropped:
        print(f"  drop {header} (len {len(seq)}): {', '.join(reasons)}")
    print(f"wrote {len(kept)} sequences to {args.output}")


if __name__ == "__main__":
    main()
