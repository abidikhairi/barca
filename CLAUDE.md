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

python -m barca.training --config configs/lora.yaml [--no_wandb --max_steps N --set KEY=VALUE ...]
./scripts/training/train.sh configs/full_ft.yaml    # sets PYTHONPATH=src; NUM_GPUS>1 uses torchrun; PYTHON=... overrides interpreter
```

Training runs on a remote GPU server, not locally: `make sync` (rsync), `make train RECIPE=...`, `make smoke`, `make inspect`, `make pull checkpoint-1000`. Override host/paths in the gitignored `Makefile.local`. Generation/probing scripts need CUDA (the model uses FLA Triton kernels; CPU fails).

## Architecture

- `src/barca/config.py` parses and validates a recipe into typed sections (`run`, `data`, `tokenizer`, `model`, `peft`, `training`, `checkpointing`, `wandb`). Unknown keys raise. The `training` section is forwarded verbatim to `transformers.TrainingArguments`; CLI `--set KEY=VALUE` patches it.
- `src/barca/data/` loads `datasets.save_to_disk` data (`loading.py`: split-root or single-root layouts) and tokenizes/packs it (`processing.py`). Tokenization is detected **per split** (e.g. `barca_5m` train is pre-tokenized and pre-packed at length 515, valid is raw text), so keep `data.packing: false` for it.
- `src/barca/modeling/` loads tokenizer/model (`loading.py`), applies LoRA/QLoRA (`peft.py`), and implements selective token training (`tokens.py`; `model.trainable_token_range` / `trainable_tokens` / `trainable_token_ids`): everything is frozen except chosen embedding rows (and tied `lm_head`), with gradient masking.
- `src/barca/training/`: `run.py` is the entry point (`python -m barca.training`) wiring config, data, model and callbacks into a Trainer and saving the best checkpoint to `run.output_dir`; `cli.py` has arg parsing/overrides, `arguments.py` builds `TrainingArguments`. `callbacks.py`: `TopKCheckpointCallback` owns checkpoint retention (top-k by `eval_loss`; Trainer's `save_total_limit` is disabled, and `save_steps` must equal `eval_steps`); `RestoreFrozenEmbeddingRowsCallback` restores non-selected embedding rows after each step so weight decay only affects selected rows. Checkpoints stay local; W&B gets metrics and the resolved config only.
- `configs/`: `*_1m_fp32` recipes are the 1M-sample fp32 experiments (batch sizes there were measured for memory on the GPU server); `emb_1m_fp32` is embedding-only.
- `scripts/` is grouped into `training/` (`train.sh`), `data/` (`inspect_dataset.py`, `split_dataset.py`, `preprocess.py|sh`) and `generation/`: `generate.py` (prompts / `--probe` next-token diagnostics), `generate_fasta.py` → `filter_proteins.py` → `fold_plddt.py` (sample sequences, filter to foldable ones, fold with the ESMFold API and average pLDDT).

## Experiment notes (LoRA 2M run)

Full write-up for the paper: `docs/lora_2m_findings.md`. Things that are easy to get wrong:

- `lora_2m_fp32` (LoRA r=64, 15,625 steps, started from `outputs/barca-0.8b-emb-1m-fp32`) runs on the shared GB10 server; any GPU job of yours slows it (about 50 s/step idle, 90-100 s/step shared). Only the top-3 checkpoints by `eval_loss` survive, so earlier ones cannot be evaluated later.
- Pass generation settings explicitly when comparing checkpoints (`generate_fasta.py` defaults differ): T 0.6, top_p 0.95, top_k 50, repetition penalty 1.2, seed 0, 50 samples, batch 5, then `filter_proteins.py --max-len 400`. The shared seed means sample N is similar across checkpoints.
- Folding is local ColabFold (`--num-models 1 --num-recycle 3 --model-type auto`) using the public MSA server, which can leave requests `PENDING` for 10+ min; rerunning the same command in the same folder resumes. Outputs live under gitignored `data/gen/lora_2m_ckpt<STEP>_t06/`.
- `filter_proteins.py` does not catch tandem repeats; LRR/TPR/ankyrin proteins pass it and score high pLDDT, so check for repeats before treating a structure as a candidate.
- Findings so far: mean pLDDT plateaus at 47-52 from step 4000 with no trend (within noise); the LoRA run degrades text ability (Cosmopedia perplexity 5.995 base vs 8.7-8.8 at steps 10500-11000, mostly real weight drift, not protein-token leakage). A 3:1:1 protein/text/code calibration with KL to the base model is planned but not built.
