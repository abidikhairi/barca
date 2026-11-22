#!/usr/bin/env python
"""Compare GRPO candidates and ColabFold pLDDT against an SFT baseline.

Both checkpoints are folded with ColabFold; this script pairs candidates by
``val<idx>_<accession>_gen<k>`` name and reports per-candidate and aggregate
pLDDT/pTM deltas, plus sequence-level reward metrics (validity, length, uniq5,
reference 5-mer overlap/F1).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as stats
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sft-json", type=Path, required=True)
    parser.add_argument("--grpo-json", type=Path, required=True)
    parser.add_argument("--sft-folds", type=Path, required=True)
    parser.add_argument("--grpo-folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("grpo_vs_sft_comparison.json"))
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--min-len", type=int, default=30)
    parser.add_argument("--reuse-identical", action="store_true", help="Reuse SFT folds for identical sequences.")
    return parser.parse_args()


def kmers(sequence: str, k: int = 5) -> set[str]:
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def ref_f1(sequence: str, reference: str) -> float:
    a, b = kmers(sequence), kmers(reference)
    return 2.0 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def fold_scores(directory: Path) -> dict[str, dict[str, float]]:
    scores = {}
    for path in sorted(glob.glob(str(directory / "*_scores_rank_001*.json"))):
        name = os.path.basename(path).split("_scores_rank_001")[0]
        payload = json.load(open(path))
        plddt = payload.get("plddt") or []
        if not plddt:
            continue
        scores[name] = {
            "plddt_mean": stats.mean(plddt),
            "ptm": float(payload.get("ptm", float("nan"))),
            "length": len(plddt),
        }
    return scores


def candidate_name(row: dict) -> str:
    return f"val{row['idx']}_{row['accession']}_gen{row['candidate']}"


def is_valid(row: dict, min_len: int, max_len: int) -> bool:
    return bool(row["closed"] and row["canon"] and min_len <= row["gen_len"] <= max_len)


def reward_summary(rows: list[dict], min_len: int, max_len: int) -> dict[str, float]:
    valid = [row for row in rows if is_valid(row, min_len, max_len)]
    return {
        "n": len(rows),
        "valid": len(valid),
        "valid_rate": len(valid) / len(rows),
        "mean_gen_len": stats.mean(row["gen_len"] for row in rows),
        "mean_valid_len": stats.mean(row["gen_len"] for row in valid) if valid else float("nan"),
        "mean_uniq5": stats.mean(row["uniq5"] for row in rows),
        "mean_ref5_cov": stats.mean(row["ref5_cov"] for row in rows),
        "mean_ref_f1": stats.mean(ref_f1(row["gen_seq"], row["ref_seq"]) for row in rows),
    }


def main() -> None:
    args = parse_args()
    sft_rows = json.load(open(args.sft_json))["rows"]
    grpo_rows = json.load(open(args.grpo_json))["rows"]
    sft_scores = fold_scores(args.sft_folds)
    grpo_scores = fold_scores(args.grpo_folds)

    paired = []
    excluded = []
    for row in grpo_rows:
        name = candidate_name(row)
        if not (args.min_len <= row["gen_len"] <= args.max_len):
            excluded.append(name)
            continue
        if name in sft_scores:
            sft = next(item for item in sft_rows if candidate_name(item) == name)
            if not (args.min_len <= sft["gen_len"] <= args.max_len):
                excluded.append(name)
                continue
            grpo_score = grpo_scores.get(name)
            reused = False
            if grpo_score is None:
                if args.reuse_identical and sft["gen_seq"] == row["gen_seq"]:
                    grpo_score = sft_scores[name]
                    reused = True
                else:
                    continue
            paired.append(
                {
                    "name": name,
                    "idx": row["idx"],
                    "accession": row["accession"],
                    "candidate": row["candidate"],
                    "sft_plddt": sft_scores[name]["plddt_mean"],
                    "grpo_plddt": grpo_score["plddt_mean"],
                    "delta_plddt": grpo_score["plddt_mean"] - sft_scores[name]["plddt_mean"],
                    "sft_ptm": sft_scores[name]["ptm"],
                    "grpo_ptm": grpo_score["ptm"],
                    "reused_sft_fold": reused,
                    "sft_len": len(sft["gen_seq"]),
                    "grpo_len": len(row["gen_seq"]),
                    "identical": sft["gen_seq"] == row["gen_seq"],
                    "identity": sum(x == y for x, y in zip(sft["gen_seq"], row["gen_seq"]))
                    / max(len(sft["gen_seq"]), len(row["gen_seq"]), 1),
                }
            )

    deltas = [row["delta_plddt"] for row in paired]
    result = {
        "sft_rewards": reward_summary(sft_rows, args.min_len, args.max_len),
        "grpo_rewards": reward_summary(grpo_rows, args.min_len, args.max_len),
        "n_paired": len(paired),
        "sft_plddt_mean": stats.mean(row["sft_plddt"] for row in paired),
        "grpo_plddt_mean": stats.mean(row["grpo_plddt"] for row in paired),
        "delta_plddt_mean": stats.mean(deltas),
        "delta_plddt_median": stats.median(deltas),
        "wins": sum(delta > 0.5 for delta in deltas),
        "losses": sum(delta < -0.5 for delta in deltas),
        "ties": sum(abs(delta) <= 0.5 for delta in deltas),
        "sft_ptm_mean": stats.mean(row["sft_ptm"] for row in paired),
        "grpo_ptm_mean": stats.mean(row["grpo_ptm"] for row in paired),
        "identical_candidates": sum(row["identical"] for row in paired),
        "excluded_out_of_range": sorted(set(excluded)),
        "reused_folds": sum(row["reused_sft_fold"] for row in paired),
        "paired": paired,
    }
    json.dump(result, open(args.output, "w"), indent=2)

    print("reward metrics (all candidates)")
    for key in ["n", "valid", "valid_rate", "mean_gen_len", "mean_valid_len", "mean_uniq5", "mean_ref5_cov", "mean_ref_f1"]:
        print(f"  {key:16s} SFT {result['sft_rewards'][key]:.4f}  GRPO {result['grpo_rewards'][key]:.4f}")
    print(f"\npaired ColabFold candidates: {len(paired)}")
    print(f"  pLDDT SFT {result['sft_plddt_mean']:.2f}  GRPO {result['grpo_plddt_mean']:.2f}  "
          f"delta {result['delta_plddt_mean']:+.2f} (median {result['delta_plddt_median']:+.2f})")
    print(f"  wins/losses/ties (|delta|>0.5): {result['wins']}/{result['losses']}/{result['ties']}")
    print(f"  pTM   SFT {result['sft_ptm_mean']:.3f} GRPO {result['grpo_ptm_mean']:.3f}")
    print(f"  identical sequences: {result['identical_candidates']}/{len(paired)}")
    print(f"  reused SFT folds: {result['reused_folds']}")
    if result["excluded_out_of_range"]:
        print(f"  excluded out-of-range: {', '.join(result['excluded_out_of_range'])}")
    print("\nper candidate")
    for row in paired:
        print(
            f"  {row['name']:<32} SFT {row['sft_plddt']:6.2f} GRPO {row['grpo_plddt']:6.2f} "
            f"delta {row['delta_plddt']:+6.2f} identity {row['identity']:.3f}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
