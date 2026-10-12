#!/usr/bin/env python
"""Next-token-prediction loss and perplexity, overall and per group.

Reads text from a CSV file or a ``datasets.save_to_disk`` directory, tokenizes
each row independently (truncated to --max-length, no packing), and computes the
token-level cross-entropy of the model's next-token predictions. Perplexity is
``exp(sum NLL / number of predicted tokens)`` per group, so long rows weigh more
than short ones; the first token of a row has no prediction and is not scored.

Groups come from --group-column (e.g. ``subset`` in the Cosmopedia CSV). Without
it, everything is reported as a single ``all`` group.

Usage:
    python scripts/eval/ntp/evaluate.py khairi/barca-0.8b-base \
        --data data/eval/ntp/cosmopedia_100.csv --group-column subset
    python scripts/eval/ntp/evaluate.py khairi/barca-0.8b-base \
        --data /home/kabidi/workspace/datasets/uniref50/barca_5m/valid --max-length 515

Note: the Barca model needs CUDA (FLA Triton kernels).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model", help="HF repo id or local path to a checkpoint.")
    p.add_argument("--data", required=True, help="CSV file or load_from_disk directory.")
    p.add_argument("--split", default="train", help="Split to use if --data is a DatasetDict.")
    p.add_argument("--text-column", default="text")
    p.add_argument("--group-column", default=None, help="Report per value of this column.")
    p.add_argument("--max-length", type=int, default=1024, help="Truncate rows to this many tokens.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None, help="Only evaluate the first N rows.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    p.add_argument("--output-json", default=None, help="Also write the results to this JSON file.")
    return p.parse_args()


def load_rows(args: argparse.Namespace):
    """Return a datasets.Dataset from a CSV file or a save_to_disk directory."""
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

    path = Path(args.data)
    if path.is_file():
        ds = load_dataset("csv", data_files=str(path), split="train")
    else:
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            ds = ds[args.split]
    assert isinstance(ds, Dataset)
    if args.text_column not in ds.column_names:
        raise SystemExit(f"column {args.text_column!r} not in {ds.column_names}")
    if args.group_column and args.group_column not in ds.column_names:
        raise SystemExit(f"column {args.group_column!r} not in {ds.column_names}")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    return ds


def load_model(model_id: str, dtype: str, device: str):
    from packaging.version import Version

    import transformers

    key = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
    kwargs = {"trust_remote_code": True, key: getattr(torch, dtype)}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), device


@torch.no_grad()
def evaluate(model, tokenizer, texts: List[str], groups: List[str], args, device: str):
    """Return {group: [nll_sum, token_count, row_count]}."""
    encoded = tokenizer(
        texts, truncation=True, max_length=args.max_length, add_special_tokens=True
    )["input_ids"]
    # Sort by length so batches pad as little as possible.
    order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    totals: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0, 0])
    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        longest = max(len(encoded[i]) for i in idx)
        input_ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
        attention = torch.zeros((len(idx), longest), dtype=torch.long)
        for row, i in enumerate(idx):
            ids = encoded[i]
            input_ids[row, : len(ids)] = torch.tensor(ids)  # right padding
            attention[row, : len(ids)] = 1
        input_ids, attention = input_ids.to(device), attention.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        for row, i in enumerate(idx):
            n = len(encoded[i])
            if n < 2:
                continue
            nll = F.cross_entropy(
                logits[row, : n - 1].float(), input_ids[row, 1:n], reduction="sum"
            )
            bucket = totals[groups[i]]
            bucket[0] += nll.item()
            bucket[1] += n - 1
            bucket[2] += 1
        del logits
        if device.startswith("cuda"):
            torch.cuda.empty_cache()  # shapes vary per batch; avoid allocator growth
        done = min(start + args.batch_size, len(order))
        if (start // args.batch_size) % 25 == 0:
            mem = f"  gpu {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB peak" if device.startswith("cuda") else ""
            print(f"  {done}/{len(order)} rows{mem}", flush=True)
    return totals


def summarize(totals: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for group, (nll, tokens, rows) in sorted(totals.items()):
        loss = nll / tokens
        out[group] = {"rows": rows, "tokens": tokens, "loss": loss, "ppl": math.exp(loss)}
    nll = sum(v[0] for v in totals.values())
    tokens = sum(v[1] for v in totals.values())
    rows = sum(v[2] for v in totals.values())
    if len(totals) > 1:
        out["ALL"] = {
            "rows": rows, "tokens": tokens, "loss": nll / tokens, "ppl": math.exp(nll / tokens),
        }
    return out


def main() -> None:
    args = parse_args()
    ds = load_rows(args)
    texts = [str(t) for t in ds[args.text_column]]
    groups = [str(g) for g in ds[args.group_column]] if args.group_column else ["all"] * len(texts)
    print(f"{len(texts)} rows from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, device = load_model(args.model, args.dtype, args.device)
    totals = evaluate(model, tokenizer, texts, groups, args, device)
    results = summarize(totals)

    width = max(len(g) for g in results)
    print(f"\nmodel: {args.model}   max_length: {args.max_length}")
    print(f"{'group':<{width}}  {'rows':>6}  {'tokens':>9}  {'loss':>8}  {'ppl':>10}")
    for group, r in results.items():
        print(f"{group:<{width}}  {r['rows']:>6}  {r['tokens']:>9}  {r['loss']:>8.4f}  {r['ppl']:>10.3f}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(
            json.dumps({"model": args.model, "data": args.data, "results": results}, indent=2)
        )


if __name__ == "__main__":
    main()
