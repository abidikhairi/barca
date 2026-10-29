#!/usr/bin/env python
"""Build a chat-format SFT dataset from the Mol-Instructions protein-design subtask.

Input is ``protein_design.json`` from ``Protein-oriented_Instructions.zip``
(zjunlp/Mol-Instructions). Each row is turned into one chat conversation using the
model's own chat template:

    user:       <instruction>\\n\\n<input>
    assistant:  <prose>\\n<protein>ζMζI...</protein>

The plain amino-acid sequence in the fenced block of the original output is rewritten
into the Barca protein format (one ζ-prefixed token per residue, wrapped in
``<protein>`` ... ``</protein>``). Rows whose output does not hold exactly one sequence
of the 20 canonical amino acids are dropped.

The result is a pre-tokenized ``DatasetDict`` (``train`` / ``validation``) with
``input_ids`` and ``attention_mask``, so the loss covers the whole conversation (question
and answer). With ``--assistant-mask`` it also gets ``labels`` that are -100 on the prompt
(user turn and the assistant header including the empty think block), so only the reply and
its closing ``<|im_end|>`` are scored. Note that the repo's collator rebuilds labels from
``input_ids`` and ignores such a column. The raw conversation is kept in ``text``. Validation is split by protein accession so no protein appears in both splits.

Usage:
    python scripts/data/prepare_mol_instruct_design.py \\
        --json /home/kabidi/workspace/datasets/barca/mol-instruct/raw/protein/Protein-oriented_Instructions/protein_design.json \\
        --tokenizer outputs/barca-0.8b-calibration-ft \\
        --output /home/kabidi/workspace/datasets/barca/mol-instruct
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
FENCE = re.compile(r"```[^\n]*\n?([^`]*?)\n?```", re.S)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--json", required=True, help="Path to protein_design.json.")
    p.add_argument("--tokenizer", required=True, help="Barca checkpoint (chat template + tokens).")
    p.add_argument("--output", required=True, help="save_to_disk directory.")
    p.add_argument("--val-size", type=int, default=2000, help="Validation rows (by accession).")
    p.add_argument("--max-length", type=int, default=2048, help="Drop rows longer than this.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--assistant-mask", action="store_true",
        help="Also write a `labels` column with the prompt masked to -100.",
    )
    return p.parse_args()


def to_protein_tokens(seq: str) -> str:
    return "<protein>" + "".join("ζ" + aa for aa in seq) + "</protein>"


def convert_output(output: str):
    """Return (assistant text, reason). Reason is None when the row is usable."""
    blocks = FENCE.findall(output)
    if len(blocks) != 1:
        return None, f"{len(blocks)} fenced blocks"
    seq = blocks[0].strip()
    if not seq or set(seq) - AMINO_ACIDS:
        return None, "non-canonical residues"
    text = FENCE.sub(lambda _: to_protein_tokens(seq), output, count=1)
    return text.strip(), None


def main() -> None:
    args = parse_args()
    rows = json.load(open(args.json))
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    drops: Counter = Counter()
    kept = []
    for r in rows:
        reply, reason = convert_output(r["output"])
        if reason:
            drops[reason] += 1
            continue
        user = f'{r["instruction"].strip()}\n\n{r["input"].strip()}'.strip()
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
        prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        # The reply (+ <|im_end|>) is what we score; the final newline is not.
        assert full.startswith(prompt), "chat template prefix mismatch"
        kept.append((r["metadata"].get("protein_accession", ""), prompt, full))

    texts = [f for _, _, f in kept]
    prompts = [p for _, p, _ in kept]
    full_ids = tok(texts, add_special_tokens=False)["input_ids"]
    prompt_lens = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]

    examples = []
    for (acc, _, text), ids, n_prompt in zip(kept, full_ids, prompt_lens):
        if len(ids) > args.max_length:
            drops[f"longer than {args.max_length} tokens"] += 1
            continue
        example = {
            "accession": acc,
            "text": text,
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "length": len(ids),
        }
        if args.assistant_mask:
            labels = [-100] * n_prompt + ids[n_prompt:]
            if tok.convert_ids_to_tokens(ids[-1]) == "Ċ":  # newline after <|im_end|>
                labels[-1] = -100
            example["labels"] = labels
        examples.append(example)

    # Hold out whole accessions for validation.
    accessions = sorted({e["accession"] for e in examples})
    random.Random(args.seed).shuffle(accessions)
    val_acc, rows_in_val = set(), 0
    count = Counter(e["accession"] for e in examples)
    for a in accessions:
        if rows_in_val >= args.val_size:
            break
        val_acc.add(a)
        rows_in_val += count[a]
    train = [e for e in examples if e["accession"] not in val_acc]
    val = [e for e in examples if e["accession"] in val_acc]

    ds = DatasetDict(
        {"train": Dataset.from_list(train), "validation": Dataset.from_list(val)}
    )
    ds.save_to_disk(args.output)

    n_tok = sum(e["length"] for e in examples)
    print(f"input rows {len(rows)} -> kept {len(examples)}; dropped: {dict(drops)}")
    print(f"train {len(train)}  validation {len(val)} ({len(val_acc)} accessions held out)")
    lens = sorted(e["length"] for e in examples)
    print(f"tokens/row mean {n_tok / len(examples):.0f} median {lens[len(lens) // 2]} max {lens[-1]}")
    print(f"total tokens {n_tok}")
    print("\n--- example ---\n" + examples[0]["text"][:900])


if __name__ == "__main__":
    main()
