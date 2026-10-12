"""Small shared helpers: seeding, dtype resolution, logging, fingerprints."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger("barca")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple, consistent console logger."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def is_main_process() -> bool:
    """True on rank 0 (or when no distributed state exists)."""
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all RNGs used by the Trainer)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a hard dep in practice
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass


def resolve_dtype(name: Optional[str]):
    """Map a dtype string to a ``torch.dtype`` (or None for 'auto')."""
    if name is None:
        return None
    normalized = name.lower()
    if normalized == "auto":
        return None
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(mapping)} or 'auto'")
    return mapping[normalized]


def config_fingerprint(cfg_dict: Dict[str, Any]) -> str:
    """Stable short hash of a config mapping (useful as a run tag)."""
    blob = json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def count_parameters(model) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_trainable_parameters(model, label: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    LOGGER.info(
        "Model ready (%s): %s trainable / %s total params (%.2f%%)",
        label,
        f"{trainable:,}",
        f"{total:,}",
        pct,
    )
