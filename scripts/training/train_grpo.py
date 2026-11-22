#!/usr/bin/env python
"""Run GRPO on the Barca SFT-design checkpoint.

The script builds a stratified Mol-Instructions prompt set, scores completions
with format/reference/diversity/length rewards plus an async ESMFold2 pLDDT
reward, and trains a LoRA adapter with TRL's ``GRPOTrainer``.

Usage (server, repo root):
    PYTHONPATH=src python scripts/training/train_grpo.py --config configs/grpo.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import httpx
import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from barca.rl import RewardConfig, build_grpo_dataset, make_rewards
from barca.utils import LOGGER, setup_logging

DEFAULT_CONFIG = "configs/grpo.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=None, help="Override model.name_or_path.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--max-ref-len", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--fold-url", default=None)
    parser.add_argument("--no-foldability", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def setup_wandb(config: dict[str, Any]) -> None:
    if not config.get("enabled", True):
        os.environ["WANDB_DISABLED"] = "true"
        return
    if config.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(config["project"]))
    if config.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(config["entity"]))
    if config.get("notes"):
        os.environ.setdefault("WANDB_NOTES", str(config["notes"]))
    if config.get("tags"):
        os.environ.setdefault("WANDB_TAGS", ",".join(str(tag) for tag in config["tags"]))


def check_fold_service(url: str, timeout: float = 15.0) -> None:
    try:
        response = httpx.get(f"{url.rstrip('/')}/health", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"foldability service is not reachable at {url}: {exc}") from exc
    if not payload.get("ready"):
        raise SystemExit(f"foldability service at {url} is not ready: {payload}")
    LOGGER.info(
        "foldability service ready: %s (%s, loops=%s, steps=%s)",
        url,
        payload.get("dtype"),
        payload.get("num_loops"),
        payload.get("num_sampling_steps"),
    )


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    model_cfg = dict(config.get("model", {}))
    data_cfg = dict(config.get("data", {}))
    reward_cfg = dict(config.get("rewards", {}))
    grpo_cfg = dict(config.get("grpo", {}))
    run_cfg = dict(config.get("run", {}))
    wandb_cfg = dict(config.get("wandb", {}))

    grpo_cfg.setdefault("output_dir", run_cfg.get("output_dir", "outputs/barca-0.8b-grpo"))
    grpo_cfg.setdefault("run_name", run_cfg.get("name", "barca-0.8b-grpo"))
    grpo_cfg.setdefault("seed", run_cfg.get("seed", 42))

    if args.model:
        model_cfg["name_or_path"] = args.model
    if args.max_prompts is not None:
        data_cfg["num_prompts"] = args.max_prompts
    if args.max_ref_len is not None:
        data_cfg["max_ref_len"] = args.max_ref_len
    if args.max_completion_length is not None:
        grpo_cfg["max_completion_length"] = args.max_completion_length
    if args.num_generations is not None:
        grpo_cfg["num_generations"] = args.num_generations
    if args.seed is not None:
        data_cfg["seed"] = args.seed
        grpo_cfg["seed"] = args.seed
    if args.output_dir:
        grpo_cfg["output_dir"] = args.output_dir
    if args.run_name:
        grpo_cfg["run_name"] = args.run_name
    if args.fold_url:
        reward_cfg["fold_service_url"] = args.fold_url
    if args.no_foldability:
        reward_cfg["include_foldability"] = False
    if args.no_wandb:
        wandb_cfg["enabled"] = False
    if args.log_completions:
        grpo_cfg["log_completions"] = True
    if args.no_cache:
        data_cfg["cache_dir"] = None

    resolved = {
        "model": model_cfg,
        "data": data_cfg,
        "rewards": reward_cfg,
        "grpo": grpo_cfg,
        "run": run_cfg,
        "wandb": wandb_cfg,
    }
    return resolved


def main() -> None:
    setup_logging()
    args = parse_args()
    config = resolve_config(args)

    model_path = config["model"]["name_or_path"]
    data_cfg = config["data"]
    grpo_cfg = config["grpo"]
    wandb_cfg = config["wandb"]

    seed = int(grpo_cfg.get("seed", 42))
    num_generations = int(grpo_cfg.get("num_generations", 5))
    if num_generations < 2:
        raise SystemExit("num_generations must be at least 2")
    num_prompts = int(data_cfg.get("num_prompts", 5000))
    if num_prompts < num_generations:
        raise SystemExit(
            f"num_prompts={num_prompts} must be at least num_generations={num_generations}"
        )

    reward_config = RewardConfig(
        format_weight=float(config["rewards"].get("format_weight", 0.10)),
        ref_f1_weight=float(config["rewards"].get("ref_f1_weight", 0.30)),
        foldability_weight=float(config["rewards"].get("foldability_weight", 0.35)),
        diversity_weight=float(config["rewards"].get("diversity_weight", 0.15)),
        length_weight=float(config["rewards"].get("length_weight", 0.10)),
        include_foldability=bool(config["rewards"].get("include_foldability", True)),
        min_len=int(config["rewards"].get("min_len", 30)),
        max_len=int(config["rewards"].get("max_len", 512)),
        require_zeta=bool(config["rewards"].get("require_zeta", True)),
        uniq5_target=float(config["rewards"].get("uniq5_target", 0.8)),
        fold_service_url=str(config["rewards"].get("fold_service_url", "http://127.0.0.1:8010")),
        fold_num_loops=int(config["rewards"].get("fold_num_loops", 4)),
        fold_num_sampling_steps=int(config["rewards"].get("fold_num_sampling_steps", 20)),
        fold_timeout_s=float(config["rewards"].get("fold_timeout_s", 1200.0)),
        fold_every_n_steps=int(config["rewards"].get("fold_every_n_steps", 1)),
    )
    if reward_config.include_foldability:
        check_fold_service(reward_config.fold_service_url)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is None or im_end == tokenizer.unk_token_id:
        raise SystemExit("tokenizer does not define <|im_end|>")
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.padding_side = "left"

    dataset = build_grpo_dataset(
        data_path=str(data_cfg["path"]),
        num_prompts=num_prompts,
        min_len=int(data_cfg.get("min_len", 30)),
        max_len=int(data_cfg.get("max_len", 512)),
        max_ref_len=data_cfg.get("max_ref_len"),
        train_split=str(data_cfg.get("train_split", "train")),
        validation_split=str(data_cfg.get("validation_split", "validation")),
        seed=int(data_cfg.get("seed", seed)),
        cache_dir=data_cfg.get("cache_dir"),
        reuse_cache=bool(data_cfg.get("reuse_cache", True)),
    )
    LOGGER.info("dataset columns: %s; rows=%d", dataset.column_names, len(dataset))

    dtype = torch.bfloat16 if grpo_cfg.get("bf16", True) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        attn_implementation=grpo_cfg.get("attn_implementation", "sdpa"),
    )

    lora_cfg = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 128)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(
            lora_cfg.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    reward_funcs, weights = make_rewards(reward_config)
    LOGGER.info(
        "rewards: %s weights=%s",
        [function.__name__ for function in reward_funcs],
        [round(weight, 4) for weight in weights],
    )

    generate_batch = int(grpo_cfg.get("per_device_train_batch_size", 1)) * int(
        grpo_cfg.get("gradient_accumulation_steps", 5)
    )
    if generate_batch % num_generations != 0:
        raise SystemExit(
            f"per_device_train_batch_size * gradient_accumulation_steps ({generate_batch}) "
            f"must be divisible by num_generations ({num_generations})"
        )

    max_steps = int(grpo_cfg.get("max_steps", -1))
    if args.max_steps is not None:
        max_steps = args.max_steps
    save_strategy = "no" if max_steps == 1 else str(grpo_cfg.get("save_strategy", "steps"))

    setup_wandb(wandb_cfg)
    training_args = GRPOConfig(
        output_dir=str(grpo_cfg.get("output_dir", "outputs/barca-0.8b-grpo")),
        run_name=str(grpo_cfg.get("run_name", "barca-0.8b-grpo")),
        report_to=["wandb"] if wandb_cfg.get("enabled", True) else [],
        num_generations=num_generations,
        temperature=float(grpo_cfg.get("temperature", 0.9)),
        top_p=float(grpo_cfg.get("top_p", 0.95)),
        top_k=int(grpo_cfg.get("top_k", 50)),
        repetition_penalty=float(grpo_cfg.get("repetition_penalty", 1.2)),
        max_completion_length=int(grpo_cfg.get("max_completion_length", 700)),
        beta=float(grpo_cfg.get("beta", 0.04)),
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(grpo_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 5)),
        num_train_epochs=float(grpo_cfg.get("num_train_epochs", 1.0)),
        max_steps=max_steps,
        bf16=bool(grpo_cfg.get("bf16", True)),
        gradient_checkpointing=bool(grpo_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(grpo_cfg.get("logging_steps", 1)),
        logging_first_step=True,
        save_strategy=save_strategy,
        save_steps=int(grpo_cfg.get("save_steps", 100)),
        eval_strategy="no",
        seed=seed,
        data_seed=seed,
        reward_weights=weights,
        multi_objective_aggregation=str(
            grpo_cfg.get("multi_objective_aggregation", "sum_then_normalize")
        ),
        scale_rewards=str(grpo_cfg.get("scale_rewards", "group")),
        log_completions=bool(grpo_cfg.get("log_completions", False)),
        num_completions_to_print=int(grpo_cfg.get("num_completions_to_print", 2)),
        remove_unused_columns=False,
        dataloader_num_workers=int(grpo_cfg.get("dataloader_num_workers", 0)),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()

    if max_steps != 1:
        trainer.save_model(str(training_args.output_dir))
    try:
        import wandb

        if wandb.run is not None:
            LOGGER.info("W&B run: %s", wandb.run.url)
            wandb.finish()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
