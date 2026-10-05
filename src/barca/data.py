"""Dataset loading and transformation for causal-LM protein training.

The expected input is a dataset saved with ``datasets.Dataset.save_to_disk`` and
therefore loadable with ``datasets.load_from_disk``. It can be either:

* a ``DatasetDict`` with named splits (``data.train_split`` / ``data.eval_split``), or
* a single ``Dataset`` that is split in-memory using ``data.eval_fraction`` /
  ``data.test_fraction``.

Sequences are tokenized for a decoder-only (causal LM) objective. With
``data.packing`` the token stream is concatenated and chunked into fixed
``max_seq_length`` blocks for maximum throughput.
"""

from __future__ import annotations

import itertools
from typing import Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk

from .config import Config
from .utils import LOGGER, is_main_process


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def resolve_splits(cfg: Config) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """Return ``(train, eval, test)`` datasets from the configured source.

    Handles both ``DatasetDict`` inputs and a single ``Dataset`` that must be
    split using the fractions in ``cfg.data``.
    """
    source = load_from_disk(cfg.data.path)
    data = cfg.data

    if isinstance(source, DatasetDict):
        names = set(source.keys())
        if data.train_split not in names:
            raise ValueError(
                f"data.train_split={data.train_split!r} not in dataset splits {sorted(names)}"
            )
        train = source[data.train_split]
        eval_ds = source.get(data.eval_split) if data.eval_split else None
        test_ds = source.get(data.test_split) if data.test_split else None

        # Fall back to holding out part of train when no eval split exists.
        if eval_ds is None and data.eval_fraction > 0:
            train, eval_ds = _split_off(train, data.eval_fraction, cfg.run.seed, name="eval")
        if test_ds is None and data.test_fraction > 0:
            source_without_test = train
            train, test_ds = _split_off(
                source_without_test, data.test_fraction, cfg.run.seed, name="test"
            )
        return train, eval_ds, test_ds

    if isinstance(source, Dataset):
        remaining = source
        eval_ds = None
        test_ds = None
        if data.eval_fraction > 0:
            remaining, eval_ds = _split_off(remaining, data.eval_fraction, cfg.run.seed, name="eval")
        if data.test_fraction > 0:
            remaining, test_ds = _split_off(
                remaining, data.test_fraction, cfg.run.seed, name="test"
            )
        if eval_ds is None and data.eval_fraction == 0:
            LOGGER.warning(
                "Source is a single Dataset and data.eval_fraction=0; training without evaluation. "
                "Set checkpointing.metric accordingly or provide an eval split."
            )
        return remaining, eval_ds, test_ds

    raise TypeError(f"Unsupported dataset type from load_from_disk: {type(source).__name__}")


def _split_off(
    dataset: Dataset, fraction: float, seed: int, name: str
) -> Tuple[Dataset, Dataset]:
    """Split ``fraction`` off ``dataset`` for ``name``; return (rest, held_out)."""
    if fraction <= 0:
        return dataset, None  # type: ignore[return-value]
    split = dataset.train_test_split(test_size=fraction, seed=seed, shuffle=True)
    LOGGER.info("Held out %d examples for %s.", len(split["test"]), name)
    return split["train"], split["test"]


# --------------------------------------------------------------------------- #
# Tokenization / packing
# --------------------------------------------------------------------------- #
def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    cfg: Config,
    desc: str = "Tokenizing",
) -> Dataset:
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

    # A causal-LM collator only needs input_ids + attention_mask; ensure types.
    keep = {"input_ids", "attention_mask"}
    drop = [c for c in tokenized.column_names if c not in keep]
    if drop:
        tokenized = tokenized.remove_columns(drop)
    return tokenized


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

    return dataset.map(
        _group,
        batched=True,
        num_proc=cfg.data.num_proc,
        desc="Packing",
    )


def _append_between(flat: list, sequences: list, token: int) -> list:
    """Append ``token`` after each original sequence when flattening."""
    out = []
    idx = 0
    for seq in sequences:
        out.extend(flat[idx : idx + len(seq)])
        out.append(token)
        idx += len(seq)
    return out


def build_tokenized_datasets(cfg: Config, tokenizer) -> Dict[str, Dataset]:
    """Build the ``{train, eval, test}`` tokenized datasets for training."""
    if cfg.data.tokenized_path:
        LOGGER.info("Loading pre-tokenized dataset from %s", cfg.data.tokenized_path)
        source = load_from_disk(cfg.data.tokenized_path)
        if not isinstance(source, DatasetDict):
            raise TypeError("data.tokenized_path must point to a DatasetDict saved by preprocessing.")
        datasets = {"train": source[cfg.data.train_split]}
        if cfg.data.eval_split and cfg.data.eval_split in source:
            datasets["eval"] = source[cfg.data.eval_split]
        if cfg.data.test_split and cfg.data.test_split in source:
            datasets["test"] = source[cfg.data.test_split]
    else:
        train, eval_ds, test_ds = resolve_splits(cfg)
        datasets = {"train": tokenize_dataset(train, tokenizer, cfg, desc="Tokenizing train")}
        if eval_ds is not None:
            datasets["eval"] = tokenize_dataset(eval_ds, tokenizer, cfg, desc="Tokenizing eval")
        if test_ds is not None:
            datasets["test"] = tokenize_dataset(test_ds, tokenizer, cfg, desc="Tokenizing test")

    datasets = {k: _maybe_cap(v, cfg, k) for k, v in datasets.items()}
    for name, ds in datasets.items():
        LOGGER.info("%s split: %d examples", name, len(ds))
    return datasets


def _maybe_cap(dataset: Dataset, cfg: Config, split: str) -> Dataset:
    limit = cfg.data.max_train_samples if split == "train" else cfg.data.max_eval_samples
    if limit is not None and len(dataset) > limit:
        LOGGER.info("Capping %s split to %d examples.", split, limit)
        return dataset.select(range(limit))
    return dataset


def preprocess_and_save(cfg: Config, tokenizer, output_path: Optional[str] = None) -> str:
    """Tokenize the configured dataset and save a ``DatasetDict`` to disk.

    This is the offline data-transformation entry point used by
    ``scripts/preprocess.py``. The result can be consumed at training time via
    ``data.tokenized_path``.
    """
    if output_path is None:
        if cfg.data.path is None:
            raise ValueError("--output is required when data.path is not set")
        output_path = f"{cfg.data.path.rstrip('/')}-tokenized"

    # Always tokenize from the raw source, ignoring any tokenized_path.
    original = cfg.data.tokenized_path
    cfg.data.tokenized_path = None
    try:
        datasets = build_tokenized_datasets(cfg, tokenizer)
    finally:
        cfg.data.tokenized_path = original

    out = DatasetDict(datasets)
    if is_main_process():
        out.save_to_disk(output_path)
        LOGGER.info("Saved tokenized dataset (%s) to %s", {k: len(v) for k, v in out.items()}, output_path)
    return output_path
