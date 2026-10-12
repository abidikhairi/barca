"""Tokenization, packing and offline preprocessing of protein datasets."""

from __future__ import annotations

import itertools
from typing import Dict, Optional

from datasets import Dataset, DatasetDict

from ..config import Config
from ..utils import LOGGER, is_main_process
from .loading import MODEL_INPUT_COLUMNS, is_tokenized, load_splits


# --------------------------------------------------------------------------- #
# Tokenization / packing
# --------------------------------------------------------------------------- #
def tokenize_dataset(dataset: Dataset, tokenizer, cfg: Config, desc: str = "Tokenizing") -> Dataset:
    """Tokenize protein sequences and optionally pack into fixed-length blocks."""
    data = cfg.data
    remove_columns = list(dataset.column_names)

    def _tokenize(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(
            batch[data.text_column],
            truncation=not data.packing,
            max_length=None if data.packing else data.max_seq_length,
            add_special_tokens=True,
        )

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=remove_columns,
        num_proc=data.num_proc,
        desc=desc,
    )

    if data.packing:
        tokenized = _pack(tokenized, tokenizer, cfg)

    return _keep_model_inputs(tokenized)


def _pack(dataset: Dataset, tokenizer, cfg: Config) -> Dataset:
    """Concatenate sequences and chunk the token stream into blocks."""
    block_size = cfg.data.max_seq_length
    eos_id = tokenizer.eos_token_id
    add_eos = cfg.data.packing_add_eos and eos_id is not None

    def _group(batch: Dict[str, list]) -> Dict[str, list]:
        concatenated: Dict[str, list] = {}
        for key in batch:
            flat = list(itertools.chain.from_iterable(batch[key]))
            if add_eos:
                if key == "input_ids":
                    flat = _append_between(flat, batch[key], eos_id)
                elif key == "attention_mask":
                    flat = _append_between(flat, batch[key], 1)
            concatenated[key] = flat

        total = len(next(iter(concatenated.values()))) if concatenated else 0
        total = (total // block_size) * block_size
        if total == 0:
            return {key: [] for key in concatenated}
        return {
            key: [values[i : i + block_size] for i in range(0, total, block_size)]
            for key, values in concatenated.items()
        }

    return dataset.map(_group, batched=True, num_proc=cfg.data.num_proc, desc="Packing")


def _append_between(flat: list, sequences: list, token: int) -> list:
    """Append ``token`` after each original sequence when flattening."""
    out = []
    idx = 0
    for seq in sequences:
        out.extend(flat[idx : idx + len(seq)])
        out.append(token)
        idx += len(seq)
    return out


def _keep_model_inputs(dataset: Dataset) -> Dataset:
    """Drop everything except input_ids/attention_mask/labels."""
    keep = [c for c in (*MODEL_INPUT_COLUMNS, "labels") if c in dataset.column_names]
    drop = [c for c in dataset.column_names if c not in keep]
    return dataset.remove_columns(drop) if drop else dataset


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def build_tokenized_datasets(
    cfg: Config,
    tokenizer,
    force_raw: bool = False,
    source: Optional[str] = None,
) -> Dict[str, Dataset]:
    """Return ``{train, eval?, test?}`` datasets ready for the Trainer.

    Pre-tokenized inputs are passed through (with packing ignored); raw inputs
    are tokenized/packed. ``data.tokenized`` overrides auto-detection and
    ``force_raw`` always tokenizes (used by the offline preprocessing script).
    """
    data = cfg.data
    splits = load_splits(cfg, source=source)

    datasets: Dict[str, Dataset] = {}
    for role, split in splits.items():
        already_tokenized = is_tokenized(split)
        if force_raw:
            split_tokenized = False
        elif data.tokenized is None:
            # Auto-detect per split: train may be pre-packed while eval is raw.
            split_tokenized = already_tokenized
        else:
            split_tokenized = bool(data.tokenized)
            if split_tokenized and not already_tokenized:
                raise ValueError(
                    f"data.tokenized=true but split {role!r} lacks 'input_ids' "
                    f"(columns: {split.column_names}). Set data.tokenized to null to tokenize "
                    "raw splits on the fly."
                )

        if split_tokenized:
            if data.packing:
                LOGGER.warning("data.packing=true ignored for pre-packed split %r.", role)
            datasets[role] = _keep_model_inputs(split)
        else:
            datasets[role] = tokenize_dataset(split, tokenizer, cfg, desc=f"Tokenizing {role}")
        datasets[role] = _maybe_cap(datasets[role], cfg, role)

    for name, ds in datasets.items():
        LOGGER.info("%s split: %d examples | columns=%s", name, len(ds), ds.column_names)
    return datasets


def _maybe_cap(dataset: Dataset, cfg: Config, split: str) -> Dataset:
    limit = cfg.data.max_train_samples if split == "train" else cfg.data.max_eval_samples
    if limit is not None and len(dataset) > limit:
        if cfg.data.shuffle_samples:
            dataset = dataset.shuffle(seed=cfg.run.seed)
            LOGGER.info("Randomly sampling %d examples for %s split.", limit, split)
        else:
            LOGGER.info("Capping %s split to the first %d examples.", split, limit)
        return dataset.select(range(limit))
    return dataset


def preprocess_and_save(
    cfg: Config, tokenizer, output_path: Optional[str] = None
) -> str:
    """Tokenize the configured raw dataset and save a ``DatasetDict`` to disk.

    This is the offline data-transformation entry point used by
    ``scripts/data/preprocess.py``. The result can be consumed at training time via
    ``data.tokenized_path``.
    """
    if output_path is None:
        if cfg.data.path is None:
            raise ValueError("--output is required when data.path is not set")
        output_path = f"{cfg.data.path.rstrip('/')}-tokenized"

    datasets = build_tokenized_datasets(
        cfg, tokenizer, force_raw=True, source=cfg.data.path
    )
    out = DatasetDict(datasets)
    if is_main_process():
        out.save_to_disk(output_path)
        LOGGER.info(
            "Saved tokenized dataset (%s) to %s",
            {k: len(v) for k, v in out.items()},
            output_path,
        )
    return output_path
