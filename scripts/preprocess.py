#!/usr/bin/env python
"""Tokenize and optionally pack a saved protein dataset for training.

This is the offline data-transformation step. It reads the dataset described by
a recipe's ``data.path``, tokenizes it with the recipe's tokenizer, and writes a
``DatasetDict`` to ``--output`` (default: ``<data.path>-tokenized``).

Point a recipe at the result with ``data.tokenized_path`` to skip on-the-fly
tokenization during training.

Usage:
    python scripts/preprocess.py --config configs/lora.yaml --output /data/proteins/tokenized
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from barca.config import Config  # noqa: E402
from barca.data import preprocess_and_save  # noqa: E402
from barca.model import load_tokenizer  # noqa: E402
from barca.utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Recipe describing data + tokenizer.")
    parser.add_argument("--output", default=None, help="Destination for the tokenized DatasetDict.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Override tokenizer.name_or_path (e.g. for a different barca checkpoint).",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = Config.from_yaml(args.config)
    if args.tokenizer:
        cfg.tokenizer.name_or_path = args.tokenizer

    tokenizer = load_tokenizer(cfg)
    output = preprocess_and_save(cfg, tokenizer, args.output)
    print(f"Saved tokenized dataset to {output}")


if __name__ == "__main__":
    main()
