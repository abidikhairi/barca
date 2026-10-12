"""Dataset loading for causal-LM protein training.

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

from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_from_disk

from ..config import Config
from ..utils import LOGGER


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
