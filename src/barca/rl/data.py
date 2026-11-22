"""Build GRPO prompt datasets from the prepared Mol-Instructions SFT data."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("barca.rl.data")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
CHAT = re.compile(
    r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n<think>\n\n</think>\n\n(.*?)<\|im_end\|>",
    re.S,
)


def parse_chat_row(text: str) -> Optional[tuple[str, str]]:
    """Return ``(user, reference_sequence)`` for a prepared protein-design row."""
    match = CHAT.match(text)
    if match is None:
        return None
    user = match.group(1).strip()
    reply = match.group(2)
    if reply.count("<protein>") != 1 or reply.count("</protein>") != 1:
        return None
    sequence = reply.split("<protein>")[1].split("</protein>")[0].replace("ζ", "").strip()
    if not sequence or set(sequence) - AMINO_ACIDS:
        return None
    return user, sequence


def length_buckets(min_len: int, max_len: int, width: int = 100) -> list[tuple[int, int]]:
    buckets = []
    start = min_len
    while start <= max_len:
        end = min(max_len, start + width - 1)
        buckets.append((start, end))
        start = end + 1
    return buckets


def select_records(
    records: list[dict],
    num_prompts: int,
    seed: int,
    min_len: int,
    max_len: int,
) -> list[dict]:
    """Deterministically sample prompts with round-robin coverage of length buckets."""
    target = min(num_prompts, len(records))
    if target <= 0:
        return []
    buckets = length_buckets(min_len, max_len)
    grouped: dict[tuple[int, int], list[dict]] = {bucket: [] for bucket in buckets}
    for record in records:
        for bucket in buckets:
            if bucket[0] <= record["ref_len"] <= bucket[1]:
                grouped[bucket].append(record)
                break
    rng = random.Random(seed)
    for bucket_records in grouped.values():
        rng.shuffle(bucket_records)

    selected: list[dict] = []
    cursor = {bucket: 0 for bucket in buckets}
    while len(selected) < target:
        progressed = False
        for bucket in buckets:
            index = cursor[bucket]
            if index < len(grouped[bucket]):
                selected.append(grouped[bucket][index])
                cursor[bucket] = index + 1
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break

    if len(selected) < target:
        chosen = {id(record) for record in selected}
        remaining = [record for record in records if id(record) not in chosen]
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])
    rng.shuffle(selected)
    return selected


def _build_dataset(records: list[dict]):
    from datasets import Dataset

    rows = [
        {
            "prompt": [{"role": "user", "content": record["user"]}],
            "reference": record["reference"],
            "ref_len": record["ref_len"],
            "accession": record["accession"],
            "idx": record["idx"],
        }
        for record in records
    ]
    return Dataset.from_list(rows)


def build_grpo_dataset(
    data_path: str,
    num_prompts: int,
    min_len: int = 30,
    max_len: int = 512,
    max_ref_len: Optional[int] = None,
    train_split: str = "train",
    validation_split: str = "validation",
    seed: int = 42,
    cache_dir: Optional[str] = None,
    reuse_cache: bool = True,
):
    """Load, filter, stratify and return a GRPO ``datasets.Dataset``."""
    from datasets import load_from_disk

    if cache_dir:
        cache_path = Path(cache_dir)
        if reuse_cache and (cache_path / "state.json").exists():
            dataset = load_from_disk(str(cache_path))
            if len(dataset) == num_prompts:
                LOGGER.info("loaded cached GRPO dataset: %s (%d prompts)", cache_path, len(dataset))
                return dataset
            LOGGER.info("cached GRPO dataset has %d prompts, rebuilding", len(dataset))

    splits = load_from_disk(data_path)
    if train_split not in splits:
        raise ValueError(f"{data_path!r} has no {train_split!r} split")
    train = splits[train_split]
    validation_accessions = set()
    if validation_split and validation_split in splits:
        validation_accessions = {value for value in splits[validation_split]["accession"] if value}

    upper = min(max_len, max_ref_len) if max_ref_len is not None else max_len
    records = []
    skipped_validation = 0
    skipped_parse = 0
    skipped_length = 0
    for index in range(len(train)):
        row = train[index]
        accession = row.get("accession", "")
        if accession and accession in validation_accessions:
            skipped_validation += 1
            continue
        parsed = parse_chat_row(row["text"])
        if parsed is None:
            skipped_parse += 1
            continue
        user, sequence = parsed
        if not (min_len <= len(sequence) <= upper):
            skipped_length += 1
            continue
        records.append(
            {
                "idx": index,
                "accession": accession,
                "user": user,
                "reference": sequence,
                "ref_len": len(sequence),
            }
        )

    if not records:
        raise ValueError("no eligible protein-design prompts found")
    selected = select_records(records, num_prompts, seed, min_len, upper)
    dataset = _build_dataset(selected)

    LOGGER.info(
        "GRPO prompts: train=%d eligible=%d selected=%d val_skipped=%d parse_skipped=%d "
        "length_skipped=%d ref_len=%.1f",
        len(train),
        len(records),
        len(dataset),
        skipped_validation,
        skipped_parse,
        skipped_length,
        sum(record["ref_len"] for record in selected) / len(selected),
    )
    if cache_dir:
        dataset.save_to_disk(cache_dir)
        LOGGER.info("saved GRPO dataset to %s", cache_dir)
    return dataset
