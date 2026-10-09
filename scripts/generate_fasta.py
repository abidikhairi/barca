#!/usr/bin/env python
"""Generate protein sequences from one or more Barca checkpoints as FASTA.

For every checkpoint given, sample N sequences from the '<protein>' prompt and
append them to a single FASTA file, with a header identifying the checkpoint,
sample index and length.

Usage:
    python scripts/generate_fasta.py outputs/barca-0.8b-full-ft/checkpoint-* \
        --num-samples 10 --output proteins.fasta

Note: the Barca model needs CUDA (FLA Triton kernels).
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="Checkpoint paths or HF repo ids.")
    parser.add_argument("--num-samples", type=int, default=10, help="Sequences per checkpoint.")
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Sequences generated at once (kept small to fit alongside training).",
    )
    parser.add_argument("--output", default="proteins.fasta", help="FASTA output path.")
    parser.add_argument("--prompt", default="<protein>", help="Generation prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(model_id: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model


def strip_protein(text: str) -> str:
    return text.replace("<protein>", "").replace("</protein>", "").replace("ζ", "").strip()


def generate(args) -> List[tuple]:
    tokenizer = AutoTokenizer.from_pretrained(args.models[0], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    close_id = tokenizer.convert_tokens_to_ids("</protein>")

    records = []
    for model_id in args.models:
        name = os.path.basename(model_id.rstrip("/")) or model_id
        print(f"generating {args.num_samples} from {model_id}", flush=True)
        model = load_model(model_id)
        inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
        done = 0
        while done < args.num_samples:
            chunk = min(args.batch_size, args.num_samples - done)
            torch.manual_seed(args.seed + done)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    num_return_sequences=chunk,
                    eos_token_id=[tokenizer.eos_token_id, close_id],
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j, seq in enumerate(output):
                residues = strip_protein(tokenizer.decode(seq, skip_special_tokens=True))
                records.append((f"{name}_sample{done + j}", residues))
            done += chunk
        del model
        torch.cuda.empty_cache()
    return records


def main() -> None:
    args = parse_args()
    records = generate(args)

    with open(args.output, "w", encoding="utf-8") as fh:
        for header, residues in records:
            fh.write(f">{header} length={len(residues)}\n{residues}\n")

    lengths = [len(r) for _, r in records]
    print(
        f"wrote {len(records)} sequences to {args.output} "
        f"(min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f})"
    )


if __name__ == "__main__":
    main()
