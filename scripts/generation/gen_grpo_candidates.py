#!/usr/bin/env python
"""Generate protein-design candidates from a Barca checkpoint with an optional LoRA adapter.

The generation settings match the SFT gen10 baseline: T 0.3, top-p 0.95, top-k 50,
repetition penalty 1.0, 1100 new tokens, seeds 0..N-1.

Usage (server, repo root):
    PYTHONPATH=src python scripts/generation/gen_grpo_candidates.py \
        --adapter outputs/barca-0.8b-grpo/checkpoint-1600 \
        --output-dir eval_out/grpo_ckpt1600
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

AA = set("ACDEFGHIKLMNPQRSTVWY")
CHAT = re.compile(
    r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*?)<\|im_end\|>",
    re.S,
)
DEFAULT_INDICES = "107,548,178,1574,834,545,220,1717,1845,1791"
GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.3,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.0,
    max_new_tokens=1100,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--base-model", default="outputs/barca-0.8b-sft-design-cont-2e-6"
    )
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter directory.")
    parser.add_argument(
        "--data", default="/home/kabidi/workspace/datasets/barca/mol-instruct"
    )
    parser.add_argument("--indices", default=DEFAULT_INDICES)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_out/grpo_candidates"))
    parser.add_argument("--tag", default="grpo")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; candidate k uses seed+k.")
    return parser.parse_args()


def kmers(sequence: str, k: int = 5) -> set[str]:
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


def extract_sequence(text: str) -> tuple[str, bool]:
    if "<protein>" not in text:
        return "", False
    return text.split("<protein>")[1].split("</protein>")[0].replace("ζ", "").strip(), "</protein>" in text


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    validation = load_from_disk(args.data)["validation"]
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.cuda().eval()
    end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")

    rows = []
    prompts = {}
    started = time.time()
    for idx in indices:
        example = validation[idx]
        match = CHAT.match(example["text"])
        if match is None:
            raise ValueError(f"cannot parse validation row {idx}")
        user, reply = match.group(1), match.group(2)
        reference = reply.split("<protein>")[1].split("</protein>")[0].replace("ζ", "")
        prompts[str(idx)] = user
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        for candidate in range(args.num_candidates):
            torch.manual_seed(args.seed + candidate)
            generated = model.generate(
                **encoded,
                eos_token_id=end_token,
                **GEN_KWARGS,
            )
            text = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=False
            )
            sequence, closed = extract_sequence(text)
            generated_kmers = kmers(sequence)
            reference_kmers = kmers(reference)
            row = {
                "idx": idx,
                "accession": example["accession"],
                "candidate": candidate,
                "seed": args.seed + candidate,
                "ref_len": len(reference),
                "ref_seq": reference,
                "gen_len": len(sequence),
                "gen_seq": sequence,
                "gen_text": text,
                "closed": closed,
                "canon": bool(sequence) and set(sequence) <= AA,
                "uniq5": len(generated_kmers) / max(1, len(sequence) - 4),
                "ref5_cov": len(generated_kmers & reference_kmers) / max(1, len(reference_kmers)),
            }
            rows.append(row)
            print(
                f"{idx:>5} {example['accession']:<12} c{candidate} ref {len(reference):>4} "
                f"gen {len(sequence):>4} closed {closed!s:<5} canon {row['canon']!s:<5}",
                flush=True,
            )

    json_path = args.output_dir / f"{args.tag}_candidates.json"
    fasta_path = args.output_dir / f"{args.tag}_candidates.fasta"
    json.dump({"rows": rows, "prompts": prompts}, open(json_path, "w"), indent=2)
    with fasta_path.open("w") as handle:
        for row in rows:
            if row["closed"] and row["canon"] and 30 <= row["gen_len"] <= 512:
                handle.write(
                    f">val{row['idx']}_{row['accession']}_gen{row['candidate']}\n{row['gen_seq']}\n"
                )
    meta = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "generation": GEN_KWARGS,
        "seeds": [args.seed + i for i in range(args.num_candidates)],
        "indices": indices,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    json.dump(meta, open(args.output_dir / f"{args.tag}_meta.json", "w"), indent=2)
    print(f"wrote {json_path}\nwrote {fasta_path}")


if __name__ == "__main__":
    main()
