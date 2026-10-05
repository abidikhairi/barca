"""Dataset loading and transformation for causal-LM protein training.

Two dataset layouts are supported, both loadable via ``datasets.load_from_disk``:

A. **Single root** — one directory holding either a ``DatasetDict`` (named
   splits) or a single ``Dataset`` that is split in memory using
   ``data.eval_fraction`` / ``data.test_fraction``.

B. **Split root** — a directory containing one ``save_to_disk`` subdirectory
   per split, e.g. ``/datasets/uniref50/barca/{train,valid}``. Split directory
   names come from ``data.train_split`` / ``data.eval_split`` / ``data.test_split``.

The pipeline is also aware of **pre-tokenized / pre-packed** datasets: if a
split already provides ``input_ids`` (and ``attention_mask``), tokenization and
packing are skipped automatically (`data.tokenized` can force this on/off).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk

from .config import Config
from .utils import LOGGER, is_main_process


# Columns a causal-LM dataset needs after (optional) tokenization.
MODEL_INPUT_COLUMNS = ("input_ids", "attention_mask")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _requested_splits(cfg: Config) -> Dict[str, str]:
    """Map roles (train/eval/test) to split directory/split names."""
    requested = {"train": cfg.data.train_split}
    if cfg.data.eval_split:
        requested["eval"] = cfg.data.eval_split
    if cfg.data.test_split:
        requested["test"] = cfg.data.test_split
    return requested


def is_tokenized(dataset: Dataset) -> bool:
    """True if the dataset already carries model inputs (input_ids)."""
    return "input_ids" in dataset.column_names


def _looks_like_split_root(path: Path) -> bool:
    """A split root has no dataset metadata of its own; it holds split dirs."""
    if not path.is_dir():
        return False
    if (path / "dataset_info.json").exists() or (path / "state.json").exists():
        return False
    return True


def _load_split_root(path: Path, requested: Dict[str, str]) -> Optional[Dict[str, Dataset]]:
    """Layout B: load one subdirectory per requested split, else return None."""
    if not _looks_like_split_root(path):
        return None
    loaded: Dict[str, Dataset] = {}
    for role, name in requested.items():
        sub = path / name
        if not sub.is_dir():
            return None
        loaded[role] = load_from_disk(str(sub))
    return loaded


def load_splits(cfg: Config, source: Optional[str] = None) -> Dict[str, Dataset]:
    """Load the ``{train, eval?, test?}`` splits for a recipe (no tokenization)."""
    data = cfg.data
    source = source or data.tokenized_path or data.path
    if source is None:
        raise ValueError("No dataset path configured (data.path / data.tokenized_path)")
    root = Path(source)
    requested = _requested_splits(cfg)

    # Layout B: one subdirectory per split.
    from_dirs = _load_split_root(root, requested)
    if from_dirs is not None:
        LOGGER.info(
            "Loaded splits from subdirectories of %s: %s",
            root,
            ", ".join(f"{role}={name}" for role, name in requested.items()),
        )
        return from_dirs

    # Layout A: a single save_to_disk root.
    source_ds = load_from_disk(str(root))
    if isinstance(source_ds, DatasetDict):
        names = set(source_ds.keys())
        out: Dict[str, Dataset] = {}
        for role, name in requested.items():
            if name in names:
                out[role] = source_ds[name]
        if "train" not in out:
            raise ValueError(
                f"data.train_split={data.train_split!r} not in dataset splits {sorted(names)}"
            )
        if "eval" not in out and data.eval_fraction > 0:
            out["train"], out["eval"] = _split_off(
                out["train"], data.eval_fraction, cfg.run.seed, "eval"
            )
        if "test" not in out and data.test_fraction > 0:
            out["train"], out["test"] = _split_off(
                out["train"], data.test_fraction, cfg.run.seed, "test"
            )
        return out

    if isinstance(source_ds, Dataset):
        train = source_ds
        out = {}
        if data.eval_fraction > 0:
            train, out["eval"] = _split_off(train, data.eval_fraction, cfg.run.seed, "eval")
        if data.test_fraction > 0:
            train, out["test"] = _split_off(train, data.test_fraction, cfg.run.seed, "test")
        out["train"] = train
        if data.eval_fraction == 0 and data.test_fraction == 0:
            LOGGER.warning(
                "Source is a single Dataset with no eval/test fraction; training without "
                "evaluation. Set data.eval_fraction > 0 (or an eval split) to rank checkpoints."
            )
        return out

    raise TypeError(f"Unsupported dataset type from load_from_disk: {type(source_ds).__name__}")


def _split_off(dataset: Dataset, fraction: float, seed: int, name: str) -> Tuple[Dataset, Dataset]:
    """Split ``fraction`` off ``dataset`` for ``name``; return (rest, held_out)."""
    if fraction <= 0:
        return dataset, None  # type: ignore[return-value]
    split = dataset.train_test_split(test_size=fraction, seed=seed, shuffle=True)
    LOGGER.info("Held out %d examples for %s.", len(split["test"]), name)
    return split["train"], split["test"]


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
    train = splits["train"]

    pre_tokenized = data.tokenized
    if force_raw:
        pre_tokenized = False
    elif pre_tokenized is None:
        pre_tokenized = is_tokenized(train)

    if pre_tokenized and not is_tokenized(train):
        raise ValueError(
            f"data.tokenized is enabled but split 'train' lacks 'input_ids'. "
            f"Columns found: {train.column_names}"
        )

    datasets: Dict[str, Dataset] = {}
    for role, split in splits.items():
        if pre_tokenized:
            if data.packing:
                LOGGER.warning("data.packing=true ignored for pre-packed split %r.", role)
            datasets[role] = _keep_model_inputs(split)
        else:
            datasets[role] = tokenize_dataset(split, tokenizer, cfg, desc=f"Tokenizing {role}")
        datasets[role] = _maybe_cap(datasets[role], cfg, role)

    if pre_tokenized:
        LOGGER.info("Using pre-tokenized datasets (tokenization/packing skipped).")
    for name, ds in datasets.items():
        LOGGER.info("%s split: %d examples | columns=%s", name, len(ds), ds.column_names)
    return datasets


def _maybe_cap(dataset: Dataset, cfg: Config, split: str) -> Dataset:
    limit = cfg.data.max_train_samples if split == "train" else cfg.data.max_eval_samples
    if limit is not None and len(dataset) > limit:
        LOGGER.info("Capping %s split to %d examples.", split, limit)
        return dataset.select(range(limit))
    return dataset


def preprocess_and_save(
    cfg: Config, tokenizer, output_path: Optional[str] = None
) -> str:
    """Tokenize the configured raw dataset and save a ``DatasetDict`` to disk.

    This is the offline data-transformation entry point used by
    ``scripts/preprocess.py``. The result can be consumed at training time via
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
