#!/usr/bin/env python
"""Autocomplete a few prompts with a Barca checkpoint.

Loads any Barca-family checkpoint (HF repo id or local path) and prints
greedy/sampled continuations for a handful of prompts, so generation quality
can be eyeballed while training runs.

Usage:
    python scripts/generate.py khairi/barca-0.8b-base
    python scripts/generate.py outputs/barca-0.8b-full-ft/checkpoint-500
    python scripts/generate.py /path/to/model --device cpu --dtype float32
    python scripts/generate.py khairi/barca-0.8b-base --prompts "Once upon a time" "<protein>"
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS: List[str] = [
    "Once upon a time",
    "<protein>",
    "A potein is a chain of",
    "In the beginning, there was",
    "<protein>MSEGSSLSDTETIRFGTGLLVESATVTR",
    "The quick brown fox",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="HF repo id or local path to a checkpoint.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
        help="Compute dtype for loading the model.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 => greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per prompt.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Custom prompts (defaults to a built-in list if omitted).",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> Optional[torch.dtype]:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }[name]


def load_model(model_id: str, dtype: Optional[torch.dtype], device: str):
    kwargs = {"trust_remote_code": True}
    if dtype is not None:
        # `dtype` replaced `torch_dtype` in transformers 4.56.
        from packaging.version import Version

        import transformers

        key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
        kwargs[key] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model, resolve_dtype(args.dtype), args.device)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    print("=" * 88)
    print(f"Model : {args.model}")
    print(f"Device: {device} | dtype: {dtype} | max_new_tokens: {args.max_new_tokens}")
    print("=" * 88)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        do_sample = args.temperature > 0
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        print(f"\nPROMPT: {prompt!r}")
        for i, sequence in enumerate(output):
            text = tokenizer.decode(sequence, skip_special_tokens=False)
            print(f"  [{i}] {text}")
    print()


if __name__ == "__main__":
    main()
