# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Fine-tuning toolkit for `khairi/barca-0.8b-base` (a Qwen3.5-based causal LM with 22 added protein tokens at ids 248077–248098: 20 ζ-prefixed amino acids + `<protein>` + `</protein>`). Runs are driven entirely by YAML recipes in `configs/`; see `README.md` for full details.

## Commands

```bash
pytest -q                                  # all tests (pythonpath=src is set in pyproject)
pytest tests/test_config.py -q             # single file (needs only PyYAML)
pytest tests/test_callbacks.py::test_name  # single test (needs transformers)
ruff check .                               # lint (line-length 100, py310)

python -m barca.train --config configs/lora.yaml [--no_wandb --max_steps N --set KEY=VALUE ...]
./scripts/train.sh configs/full_ft.yaml    # sets PYTHONPATH=src; NUM_GPUS>1 uses torchrun; PYTHON=... overrides interpreter
```

Training runs on a remote GPU server, not locally: `make sync` (rsync), `make train RECIPE=...`, `make smoke`, `make inspect`, `make pull checkpoint-1000`. Override host/paths in the gitignored `Makefile.local`. Generation/probing scripts need CUDA (the model uses FLA Triton kernels; CPU fails).

## Architecture

- `src/barca/config.py` parses and validates a recipe into typed sections (`run`, `data`, `tokenizer`, `model`, `peft`, `training`, `checkpointing`, `wandb`). Unknown keys raise. The `training` section is forwarded verbatim to `transformers.TrainingArguments`; CLI `--set KEY=VALUE` patches it.
- `src/barca/data.py` loads `datasets.save_to_disk` data (split-root or single-root layouts). Tokenization is detected **per split** (e.g. `barca_5m` train is pre-tokenized and pre-packed at length 515, valid is raw text), so keep `data.packing: false` for it.
- `src/barca/model.py` loads tokenizer/model, applies LoRA/QLoRA, and implements selective token training (`model.trainable_token_range` / `trainable_tokens` / `trainable_token_ids`): everything is frozen except chosen embedding rows (and tied `lm_head`), with gradient masking.
- `src/barca/callbacks.py`: `TopKCheckpointCallback` owns checkpoint retention (top-k by `eval_loss`; Trainer's `save_total_limit` is disabled, and `save_steps` must equal `eval_steps`); `RestoreFrozenEmbeddingRowsCallback` restores non-selected embedding rows after each step so weight decay only affects selected rows.
- `src/barca/train.py` is the entry point wiring these together; best checkpoint is reloaded and written to `run.output_dir` at the end. Checkpoints stay local; W&B gets metrics and the resolved config only.
- `configs/`: `*_1m_fp32` recipes are the 1M-sample fp32 experiments (batch sizes there were measured for memory on the GPU server); `emb_1m_fp32` is embedding-only.
- `scripts/` holds standalone utilities beyond training: `generate.py` (prompts / `--probe` next-token diagnostics), `generate_fasta.py` → `filter_proteins.py` → `fold_plddt.py` (sample sequences, filter to foldable ones, fold with the ESMFold API and average pLDDT), plus dataset tools (`inspect_dataset.py`, `split_dataset.py`, `preprocess.py`).
