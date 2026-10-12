"""Command-line parsing and recipe overrides for ``barca.training``."""

from __future__ import annotations

import argparse
from typing import Optional

from ..config import Config


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Barca protein language model.")
    parser.add_argument("--config", required=True, help="Path to a YAML recipe in configs/.")
    parser.add_argument("--output_dir", default=None, help="Override run.output_dir.")
    parser.add_argument("--run_name", default=None, help="Override run.name.")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Checkpoint dir to resume.")
    parser.add_argument("--seed", type=int, default=None, help="Override run.seed.")
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable W&B reporting for this run."
    )
    # Quick overrides for smoke tests / sweeps.
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    # Arbitrary TrainingArguments override, e.g. --set gradient_checkpointing=false
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args(argv)


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.output_dir:
        cfg.run.output_dir = args.output_dir
    if args.run_name:
        cfg.run.name = args.run_name
    if args.seed is not None:
        cfg.run.seed = args.seed
    if args.no_wandb:
        cfg.wandb.enabled = False

    for key in (
        "max_steps",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg.training[key] = value
    if args.max_train_samples is not None:
        cfg.data.max_train_samples = args.max_train_samples
    if args.max_eval_samples is not None:
        cfg.data.max_eval_samples = args.max_eval_samples
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        cfg.training[key.strip()] = _coerce(raw)
    return cfg
