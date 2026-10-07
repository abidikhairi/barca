"""Main training entry point for Barca protein-LM fine-tuning.

Usage
-----
    python -m barca.train --config configs/lora.yaml
    python -m barca.train --config configs/full_ft.yaml --output_dir runs/my-run
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config
from .data import build_tokenized_datasets
from .model import load_model, load_tokenizer
from .utils import LOGGER, config_fingerprint, ensure_dir, is_main_process, set_seed, setup_logging


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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
    return parser.parse_args(argv)


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
    return cfg


# --------------------------------------------------------------------------- #
# TrainingArguments construction
# --------------------------------------------------------------------------- #
def build_training_arguments(cfg: Config, has_eval: bool = True, selective: bool = False):
    """Merge the recipe's ``training`` section with enforced, safe defaults.

    We force evaluation and saving to happen at the same cadence so that the
    metric logged at a step always describes the checkpoint saved at that step
    (required by :class:`~barca.callbacks.TopKCheckpointCallback`).
    """
    from transformers import TrainingArguments

    training: Dict[str, Any] = dict(cfg.training)

    # The Trainer must not prune by itself; the callback owns retention.
    training.pop("save_total_limit", None)
    training.pop("output_dir", None)
    training.pop("run_name", None)
    training.pop("report_to", None)

    training["output_dir"] = cfg.run.output_dir
    training["run_name"] = cfg.run.name
    training["seed"] = cfg.run.seed
    training["report_to"] = ["wandb"] if cfg.wandb.enabled else []
    if has_eval:
        training["metric_for_best_model"] = cfg.checkpointing.metric
        training["greater_is_better"] = cfg.checkpointing.greater_is_better
    else:
        training["metric_for_best_model"] = None
        # Without evaluation we cannot rank checkpoints; avoid filling disk.
        training.setdefault("save_total_limit", 3)
    # The Trainer's internal best-model reload uses raw state-dict keys, which
    # do not match the Qwen3.5 save format (model.language_model.* vs model.*).
    # We keep top-k checkpoints and reload the best ourselves via from_pretrained.
    training["load_best_model_at_end"] = False

    # `eval_strategy` replaced `evaluation_strategy` in recent transformers.
    sig = set(inspect.signature(TrainingArguments.__init__).parameters)
    strategy_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"
    other_strategy_key = "evaluation_strategy" if strategy_key == "eval_strategy" else "eval_strategy"
    # Normalize aliases the user may have written.
    if other_strategy_key in training:
        training.setdefault(strategy_key, training.pop(other_strategy_key))

    eval_strategy = training.get(strategy_key, "steps")
    training[strategy_key] = eval_strategy
    training["save_strategy"] = eval_strategy

    if eval_strategy == "steps":
        eval_steps = training.get("eval_steps")
        save_steps = training.get("save_steps")
        if eval_steps is None and save_steps is None:
            eval_steps = 200
        elif eval_steps is None:
            eval_steps = save_steps
        elif save_steps is not None and save_steps != eval_steps:
            LOGGER.warning(
                "save_steps=%s != eval_steps=%s; forcing save_steps=eval_steps for top-k tracking.",
                save_steps,
                eval_steps,
            )
        training["eval_steps"] = eval_steps
        training["save_steps"] = eval_steps

    training.setdefault("logging_steps", 10)
    training.setdefault("logging_first_step", True)
    training.setdefault("remove_unused_columns", False)
    training.setdefault("dataloader_pin_memory", True)
    training.setdefault("lr_scheduler_type", "cosine")

    # -- cross-version adapters ------------------------------------------- #
    # `warmup_ratio` and `group_by_length` were removed/renamed in newer
    # transformers. Preserve their intent where possible and drop the rest.
    warmup_ratio = training.pop("warmup_ratio", None)
    if (
        "group_by_length" in training
        and "group_by_length" not in sig
        and "train_sampling_strategy" in sig
    ):
        if training.pop("group_by_length"):
            training.setdefault("train_sampling_strategy", "group_by_length")

    # `evaluation_strategy` <-> `eval_strategy` alias must match the installed version.
    for key in ("eval_strategy", "evaluation_strategy"):
        if key in training and key not in sig:
            training.pop(key)

    unsupported = sorted(k for k in training if k not in sig)
    for key in unsupported:
        training.pop(key)
    if unsupported:
        LOGGER.warning("Dropped arguments unsupported by this transformers version: %s", unsupported)

    if warmup_ratio is not None and "warmup_ratio" in sig:
        training["warmup_ratio"] = warmup_ratio

    # Selective token training: the optimizer only holds the shared embedding
    # matrix, so decay would shrink every row, including the frozen ones.
    if selective:
        if training.get("weight_decay", 0.0):
            LOGGER.warning(
                "Selective token training: forcing weight_decay=0 so frozen embedding "
                "rows are not decayed."
            )
        training["weight_decay"] = 0.0

    args = TrainingArguments(**training)

    # Defer warmup_ratio -> warmup_steps until the number of steps is known.
    if warmup_ratio is not None and "warmup_ratio" not in sig:
        args._barca_warmup_ratio = warmup_ratio
    return args


def resolve_warmup_steps(args, train_dataset, world_size: int) -> None:
    """Translate a deferred ``warmup_ratio`` into ``warmup_steps`` (transformers >= 5)."""
    ratio = getattr(args, "_barca_warmup_ratio", None)
    if ratio is None:
        return
    import math

    batch = max(1, args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps))
    steps_per_epoch = math.ceil(len(train_dataset) / (batch * max(1, world_size)))
    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = steps_per_epoch * args.num_train_epochs
    args.warmup_steps = int(ratio * total_steps)
    LOGGER.info(
        "Resolved warmup_ratio=%.4f to warmup_steps=%d (total_steps=%d).",
        ratio,
        args.warmup_steps,
        total_steps,
    )


# --------------------------------------------------------------------------- #
# W&B environment
# --------------------------------------------------------------------------- #
def setup_wandb_env(cfg: Config) -> None:
    if not cfg.wandb.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
    if cfg.wandb.entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb.entity)
    if cfg.wandb.notes:
        os.environ.setdefault("WANDB_NOTES", cfg.wandb.notes)
    if cfg.wandb.tags:
        os.environ.setdefault("WANDB_TAGS", ",".join(cfg.wandb.tags))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> None:
    setup_logging()
    args = parse_args(argv)
    cfg = apply_overrides(Config.from_yaml(args.config), args)
    set_seed(cfg.run.seed)
    ensure_dir(cfg.run.output_dir)

    fingerprint = config_fingerprint(cfg.to_dict())
    LOGGER.info("Run %s (config %s) -> %s", cfg.run.name, fingerprint, cfg.run.output_dir)
    if cfg.wandb.enabled:
        # Tag the run with the config fingerprint for easy lookup.
        if fingerprint not in cfg.wandb.tags:
            cfg.wandb.tags = list(cfg.wandb.tags) + [f"cfg-{fingerprint}"]
    if cfg.run.description:
        LOGGER.info("Description: %s", cfg.run.description)

    # Persist the resolved recipe next to the outputs.
    resolved_config_path = Path(cfg.run.output_dir) / "config.yaml"
    if is_main_process():
        cfg.to_yaml(resolved_config_path)

    setup_wandb_env(cfg)

    from transformers import DataCollatorForLanguageModeling, Trainer

    tokenizer = load_tokenizer(cfg)
    datasets = build_tokenized_datasets(cfg, tokenizer)

    has_eval = datasets.get("eval") is not None
    if not has_eval and cfg.checkpointing.metric.startswith("eval_"):
        raise ValueError(
            f"checkpointing.metric={cfg.checkpointing.metric!r} needs an eval split. "
            "Provide data.eval_split, or set data.eval_fraction > 0, or choose a "
            "non-eval metric."
        )
    if not has_eval:
        LOGGER.warning(
            "No eval split found: load_best_model_at_end is disabled and checkpoints "
            "cannot be ranked. Set data.eval_fraction > 0 to enable top-k checkpointing."
        )

    model = load_model(cfg, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = build_training_arguments(
        cfg, has_eval=has_eval, selective=cfg.model.selective_token_training
    )

    from .callbacks import TopKCheckpointCallback, WandbConfigCallback

    topk_callback = TopKCheckpointCallback(
        metric=cfg.checkpointing.metric,
        greater_is_better=cfg.checkpointing.greater_is_better,
        top_k=cfg.checkpointing.top_k,
        keep_last=cfg.checkpointing.keep_last,
    )
    callbacks = [
        topk_callback,
        WandbConfigCallback(
            config_file=cfg.source_path,
            watch=cfg.wandb.watch,
        ),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval"),
        data_collator=data_collator,
        callbacks=callbacks,
    )

    resolve_warmup_steps(training_args, datasets["train"], trainer.accelerator.num_processes)

    LOGGER.info("Starting training.")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Write the best checkpoint to output_dir as the final model. We reload via
    # from_pretrained because Trainer's internal loader does not understand the
    # Qwen3.5 save format (model.language_model.* vs model.*).
    final_dir = ensure_dir(cfg.run.output_dir)
    best_path = topk_callback.best_path()
    if best_path:
        LOGGER.info("Best checkpoint: %s -> saving to %s", best_path, final_dir)
        from transformers import AutoModelForCausalLM

        from .model import _from_pretrained_autodtype
        from .utils import resolve_dtype

        best_model = _from_pretrained_autodtype(
            AutoModelForCausalLM,
            best_path,
            resolve_dtype(cfg.model.dtype),
            {"trust_remote_code": cfg.model.trust_remote_code},
        )
        best_model.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if is_main_process():
        cfg.to_yaml(Path(final_dir) / "config.yaml")
    LOGGER.info("Final model saved to %s", final_dir)

    metrics = trainer.evaluate(eval_dataset=datasets.get("eval")) if datasets.get("eval") else {}
    if datasets.get("test"):
        test_metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
        metrics.update(test_metrics)
    if is_main_process():
        LOGGER.info("Final metrics: %s", metrics)

    if cfg.wandb.enabled:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except ImportError:  # pragma: no cover
            pass


if __name__ == "__main__":
    main()
