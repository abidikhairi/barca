#!/usr/bin/env python
"""Autocomplete and diagnostics for a Barca checkpoint.

Loads any Barca-family checkpoint (HF repo id or local path) and either:
  * prints continuations for a few prompts, or
  * with --probe, prints next-token diagnostics (probability mass on protein
    tokens and the rank of the best protein token) so generation quality can be
    tracked while training runs.

Usage:
    python scripts/generate.py khairi/barca-0.8b-base
    python scripts/generate.py outputs/barca-0.8b-full-ft/checkpoint-500
    python scripts/generate.py outputs/barca-0.8b-full-ft/checkpoint-500 --probe
    python scripts/generate.py MODEL --prompts "Once upon a time" "<protein>ζM"

Note: the Barca model uses FLA Triton kernels for its linear-attention
layers, so generation/probing must run on CUDA. Running with --device cpu
fails with "Pointer argument ... cannot be accessed from Triton (cpu tensor?)".
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

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

# Prompts used by --probe: the second isolates "stay in protein mode".
PROBE_PROMPTS: List[str] = ["<protein>", "<protein>ζM"]


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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--do-sample", action=argparse.BooleanOptionalAction, default=True,
        help="Sample (default) or use greedy decoding.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Used when sampling.")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per prompt.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--protein-stop", action=argparse.BooleanOptionalAction, default=True,
        help="For protein-mode prompts, also stop at the '</protein>' token.",
    )
    parser.add_argument(
        "--prompts", nargs="*", default=None,
        help="Custom prompts (defaults to a built-in list if omitted).",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Print next-token diagnostics instead of generating.",
    )
    parser.add_argument(
        "--probe-base", default="khairi/barca-0.8b-base",
        help="Reference model for --probe ('' to skip the comparison).",
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
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model


def protein_token_ids(tokenizer) -> List[int]:
    """Ids of the added protein tokens (ζ-prefixed residues + markers)."""
    added = tokenizer.get_added_vocab()
    ids = sorted(
        i for token, i in added.items() if token.startswith("ζ") or token in {"<protein>", "</protein>"}
    )
    if not ids:  # fall back to the known barca range
        ids = list(range(248077, 248099))
    return ids


def is_protein_prompt(input_ids: List[int], protein_ids: List[int]) -> bool:
    return any(int(i) in set(protein_ids) for i in input_ids)


def next_token_probe(model, tokenizer, protein_ids: List[int], prompts: List[str]) -> Dict[str, dict]:
    residue_ids = [i for i in protein_ids if tokenizer.convert_ids_to_tokens(i).startswith("ζ")]
    results: Dict[str, dict] = {}
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(input_ids=inputs.input_ids).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        best = max(residue_ids, key=lambda i: logits[i].item())
        results[text] = {
            "p_protein": float(probs[residue_ids].sum().item()),
            "best_token": tokenizer.convert_ids_to_tokens(best),
            "best_rank": int((logits > logits[best]).sum().item()),
            "top1": tokenizer.convert_ids_to_tokens(int(logits.argmax())),
        }
    return results


def run_probe(args, tokenizer, dtype, device) -> None:
    prompts = args.prompts if args.prompts else PROBE_PROMPTS
    protein_ids = protein_token_ids(tokenizer)

    tables = {}
    if args.probe_base:
        base = load_model(args.probe_base, dtype, device)
        tables[args.probe_base] = next_token_probe(base, tokenizer, protein_ids, prompts)
        del base
        torch.cuda.empty_cache()
    target = load_model(args.model, dtype, device)
    tables[args.model] = next_token_probe(target, tokenizer, protein_ids, prompts)
    del target

    print("=" * 100)
    print("Next-token diagnostics after the prompt (lower rank = closer to emitting protein)")
    print("=" * 100)
    print(f"{'model':<44} {'prompt':<14} {'P(protein)':>11} {'best':>6} {'rank':>8}  top1")
    for model_name, table in tables.items():
        for prompt, row in table.items():
            short = model_name if len(model_name) <= 43 else "..." + model_name[-40:]
            print(
                f"{short:<44} {prompt:<14} {row['p_protein']:>11.5f} {row['best_token']:>6} "
                f"{row['best_rank']:>8}  {row['top1']}"
            )
    print()


def run_generation(args, tokenizer, dtype, device) -> None:
    model = load_model(args.model, dtype, device)
    dev = next(model.parameters()).device
    dtype_actual = next(model.parameters()).dtype
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    protein_ids = protein_token_ids(tokenizer)
    close_id = tokenizer.convert_tokens_to_ids("</protein>")

    print("=" * 88)
    print(f"Model : {args.model}")
    print(f"Device: {dev} | dtype: {dtype_actual} | max_new_tokens: {args.max_new_tokens}")
    print("=" * 88)

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(dev)
        stop_ids = [tokenizer.eos_token_id]
        if (
            args.protein_stop
            and close_id is not None
            and close_id != tokenizer.unk_token_id
            and is_protein_prompt(inputs.input_ids[0].tolist(), protein_ids)
        ):
            stop_ids.append(close_id)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature if args.do_sample else None,
                top_p=args.top_p if args.do_sample else None,
                top_k=args.top_k if args.do_sample else None,
                repetition_penalty=args.repetition_penalty,
                num_return_sequences=args.num_samples,
                eos_token_id=stop_ids if len(stop_ids) > 1 else tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        print(f"\nPROMPT: {prompt!r}")
        for i, sequence in enumerate(output):
            print(f"  [{i}] {tokenizer.decode(sequence, skip_special_tokens=False)}")
    print()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(args.dtype)
    if args.probe:
        run_probe(args, tokenizer, dtype, args.device)
    else:
        run_generation(args, tokenizer, dtype, args.device)


if __name__ == "__main__":
    main()
