#!/usr/bin/env python
"""Fold protein sequences with ESMFold (CZ Biohub API) and average pLDDT.

Reads a FASTA file (headers formatted as ``<group>_sampleN ...``), folds each
sequence via the CZ Biohub ESMFold endpoint, parses per-residue pLDDT from the
CA B-factor column, and prints the mean pLDDT per group (checkpoint) plus per
sequence. PDBs and a summary are written to --out-dir.

Usage:
    python scripts/fold_plddt.py /tmp/proteins.fasta --out-dir /tmp/pdbs
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta", help="Input FASTA (one group per header prefix).")
    parser.add_argument("--out-dir", default="/tmp/pdbs", help="Directory for PDB files.")
    parser.add_argument("--api", default=API_URL)
    parser.add_argument("--max-len", type=int, default=400, help="Skip sequences longer than this.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout (s).")
    parser.add_argument("--sleep", type=float, default=1.0, help="Pause between requests (s).")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse PDBs already present in --out-dir instead of refolding.",
    )
    return parser.parse_args()


def read_fasta(path: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:].split()[0]
        else:
            records.append((header, line))
    return records


def group_of(header: str) -> str:
    return header.split("_sample")[0] if "_sample" in header else header


def fold(seq: str, api: str, timeout: float, retries: int) -> str:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                api, data=seq.encode(), headers={"Content-Type": "text/plain"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 413:  # too long -> no point retrying
                raise
            time.sleep(min(30, 5 * (attempt + 1)))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(min(30, 5 * (attempt + 1)))
    raise last_exc  # type: ignore[misc]


def mean_plddt(pdb: str) -> float:
    values = []
    for line in pdb.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            values.append(float(line[60:66]))
    if not values:
        raise ValueError("no CA atoms in PDB")
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    records = read_fasta(args.fasta)

    per_group: Dict[str, List[float]] = defaultdict(list)
    skipped: Dict[str, int] = defaultdict(int)
    rows: List[Tuple[str, int, float, str]] = []

    for header, seq in records:
        group = group_of(header)
        if len(seq) > args.max_len:
            skipped[group] += 1
            print(f"skip {header} (len {len(seq)} > {args.max_len})", flush=True)
            continue
        pdb_path = os.path.join(args.out_dir, f"{header}.pdb")
        if args.resume and os.path.exists(pdb_path):
            try:
                plddt = mean_plddt(open(pdb_path, encoding="utf-8").read())
                per_group[group].append(plddt)
                rows.append((header, len(seq), plddt, "cached"))
                print(f"{header} len={len(seq)} pLDDT={plddt:.4f} (cached)", flush=True)
                continue
            except Exception:  # noqa: BLE001 - refold if the cached file is bad
                pass
        try:
            pdb = fold(seq, args.api, args.timeout, args.retries)
            plddt = mean_plddt(pdb)
            with open(pdb_path, "w", encoding="utf-8") as fh:
                fh.write(pdb)
            per_group[group].append(plddt)
            rows.append((header, len(seq), plddt, "ok"))
            print(f"{header} len={len(seq)} pLDDT={plddt:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            rows.append((header, len(seq), float("nan"), f"FAILED {str(exc)[:60]}"))
            print(f"{header} len={len(seq)} FAILED: {str(exc)[:80]}", flush=True)
        time.sleep(args.sleep)

    print("\n" + "=" * 72)
    print(f"{'checkpoint':<28} {'folded':>7} {'skipped':>8} {'mean pLDDT':>11} {'(x100)':>8}")
    print("-" * 72)
    summary_lines = []
    for group in sorted(set(list(per_group) + list(skipped))):
        vals = per_group.get(group, [])
        mean = sum(vals) / len(vals) if vals else float("nan")
        line = f"{group:<28} {len(vals):>7} {skipped.get(group, 0):>8} {mean:>11.4f} {mean * 100:>8.2f}"
        print(line)
        summary_lines.append(line)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("per-sequence\n")
        for header, length, plddt, status in rows:
            fh.write(f"{header}\t{length}\t{plddt:.4f}\t{status}\n")
        fh.write("\nper-checkpoint\n")
        fh.write("\n".join(summary_lines) + "\n")
    print(f"\nPDBs + summary written to {args.out_dir}")


if __name__ == "__main__":
    main()
