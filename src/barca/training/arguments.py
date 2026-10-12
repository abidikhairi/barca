"""TrainingArguments construction and W&B environment setup."""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict

from ..config import Config
from ..utils import LOGGER


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

    # Log the number of input tokens the model has seen (built-in Trainer counter).
    training.setdefault("include_num_input_tokens_seen", True)

    if selective:
        # Frozen embedding rows are protected by RestoreFrozenEmbeddingRowsCallback.
        LOGGER.info("Selective token training: frozen embedding rows will be restored after each step.")

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
