"""Training entry point: wires config, data, model and callbacks into a Trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import DataCollatorForLanguageModeling, Trainer

from ..config import Config
from ..data import build_tokenized_datasets
from ..modeling import load_model, load_tokenizer
from ..modeling.loading import _from_pretrained_autodtype
from ..utils import (
    LOGGER,
    config_fingerprint,
    ensure_dir,
    is_main_process,
    resolve_dtype,
    set_seed,
    setup_logging,
)
from .arguments import build_training_arguments, resolve_warmup_steps, setup_wandb_env
from .callbacks import (
    RestoreFrozenEmbeddingRowsCallback,
    TokenStatsCallback,
    TopKCheckpointCallback,
    WandbConfigCallback,
)
from .cli import apply_overrides, parse_args


def main(argv: Optional[list] = None) -> None:
    setup_logging()
    args = parse_args(argv)
    cfg = apply_overrides(Config.from_yaml(args.config), args)
    set_seed(cfg.run.seed)
    ensure_dir(cfg.run.output_dir)

    fingerprint = config_fingerprint(cfg.to_dict())
    LOGGER.info(
        "Run %s (config %s) -> %s", cfg.run.name, fingerprint, cfg.run.output_dir
    )
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
    if cfg.model.selective_token_training:
        callbacks.append(RestoreFrozenEmbeddingRowsCallback(model))
    callbacks.append(
        TokenStatsCallback(
            seq_len=cfg.data.max_seq_length,
            per_device_batch=training_args.per_device_train_batch_size,
            grad_accum=training_args.gradient_accumulation_steps,
            train_size=len(datasets["train"]),
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval"),
        data_collator=data_collator,
        callbacks=callbacks,
    )

    resolve_warmup_steps(
        training_args, datasets["train"], trainer.accelerator.num_processes
    )

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

    metrics = (
        trainer.evaluate(eval_dataset=datasets.get("eval"))
        if datasets.get("eval")
        else {}
    )
    if datasets.get("test"):
        test_metrics = trainer.evaluate(
            eval_dataset=datasets["test"], metric_key_prefix="test"
        )
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
