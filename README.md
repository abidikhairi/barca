# Barca

Training and fine-tuning toolkit for the **Barca** protein language model
family (named after the Barcid family of Carthage — most famously
**Hannibal Barca**).

This repo focuses on fine-tuning [`khairi/barca-0.8b-base`](https://huggingface.co/khairi/barca-0.8b-base)
— a Llama-style, decoder-only model — on protein sequence datasets with a
causal-language-modeling objective. It supports **full fine-tuning**, **LoRA**
and **QLoRA**, driven entirely by YAML recipes, with Weights & Biases logging
and automatic retention of the **top-3 checkpoints by evaluation loss**.

---

## Features

- **YAML recipes** (`configs/`) for every run — no code changes to sweep settings.
- **Full / LoRA / QLoRA** training selected with a single flag.
- **Tokenization + packing** utilities for large protein corpora.
- **Top-k checkpointing**: keeps only the best `k=3` checkpoints by `eval_loss`
  on disk (configurable metric and direction).
- **W&B logging** of metrics, the resolved recipe, and optional model watching.
- Works with datasets saved via `datasets.save_to_disk` / `load_from_disk`.
- Multi-GPU via `torchrun`.

---

## Layout

```
barca/
├── configs/                  # YAML training recipes
│   ├── full_ft.yaml
│   ├── lora.yaml
│   └── qlora.yaml
├── scripts/
│   ├── training/
│   │   └── train.sh          # launch (single GPU / torchrun)
│   ├── data/
│   │   ├── inspect_dataset.py  # dataset schema + sequence-length stats
│   │   ├── split_dataset.py    # (re)build train/validation/test splits
│   │   ├── preprocess.py       # tokenize/pack offline -> DatasetDict
│   │   └── preprocess.sh
│   └── generation/
│       ├── generate.py         # prompt continuation / --probe diagnostics
│       ├── generate_fasta.py   # sample sequences from checkpoints
│       ├── filter_proteins.py  # keep valid/foldable sequences
│       └── fold_plddt.py       # ESMFold + mean pLDDT
├── src/barca/
│   ├── config.py             # typed recipe parsing
│   ├── utils.py              # seeding, dtype, logging helpers
│   ├── data/                 # loading.py (splits), processing.py (tokenize/pack)
│   ├── modeling/             # loading.py, peft.py, tokens.py (selective token training)
│   └── training/             # run.py (entry), cli.py, arguments.py, callbacks.py
└── tests/                    # unit tests (config + checkpoint callback)
```

---

## Setup

The training environment is expected to already provide the required libraries
(torch, transformers, datasets, peft, accelerate, wandb, ...). To make the
package importable on the remote server, install it in editable mode:

```bash
pip install -e .            # core deps
pip install -e ".[qlora]"   # + bitsandbytes for 4-bit training
pip install -e ".[flash]"   # + flash-attention (optional)
```

No install is required either: the launch scripts prepend `src/` to
`PYTHONPATH`, so `python -m barca.training ...` works from a fresh checkout.

---

## Make targets

A `Makefile` wraps the common operations (defaults target the GPU server; copy
`Makefile.local.example` to `Makefile.local` to override):

```bash
make sync                              # rsync this repo to the server
make train                             # fine-tune with configs/lora.yaml
make train RECIPE=configs/full_ft.yaml # ...or another recipe
make train RECIPE=configs/qlora.yaml NUM_GPUS=1
make smoke                             # run configs/full_ft_smoke.yaml (2k samples)
make inspect                           # show remote dataset splits/lengths
make pull checkpoint-1000              # copy a checkpoint back into runs/
```

`make pull` skips `optimizer.pt` (the AdamW state) to keep the transfer small;
the checkpoint still contains the model weights and `trainer_state.json`.

---

## Data

Training data is any dataset written with `datasets.save_to_disk`, loadable via
`datasets.load_from_disk`. Two layouts are supported:

- **Split root** — a directory with one dataset subdirectory per split, e.g.
  `/home/kabidi/workspace/datasets/uniref50/barca_5m/{train,valid}`. The
  directory names come from `data.train_split` / `data.eval_split` /
  `data.test_split`.
- **Single root** — one directory holding either a `DatasetDict` (named splits)
  or a single `Dataset` (split in memory using `data.eval_fraction` /
  `data.test_fraction`).

Tokenization is detected **per split**. In `barca_5m`, `train` is already
pre-packed and pre-tokenized (`input_ids` + `attention_mask`, fixed length 515),
while `valid` is raw (`entry`, `organism`, `sequence`, `text`) and is tokenized
on the fly from `data.text_column` (`text`). Set `data.tokenized: true` to
require pre-tokenized splits, `false` to force tokenization, or leave it `null`
(default) to auto-detect each split. Because `train` is already packed, keep
`data.packing: false` (it is ignored with a warning if enabled).

Inspect a dataset before training:

```bash
python scripts/data/inspect_dataset.py /home/kabidi/workspace/datasets/uniref50/barca_5m
```

For raw (untokenized) sequence data, use `data.text_column` (default
`sequence`) and optionally tokenize ahead of time; reference the result from a
recipe via `data.tokenized_path`:

```bash
python scripts/data/inspect_dataset.py /data/proteins/raw \
    --column sequence --tokenizer khairi/barca-0.8b-base
python scripts/data/split_dataset.py /data/proteins/raw \
    --output /data/proteins/processed \
    --eval-fraction 0.02 --test-fraction 0.02 --seed 42
./scripts/data/preprocess.sh configs/lora.yaml --output /data/proteins/tokenized
```

---

## Training

1. Edit a recipe in `configs/` (at minimum set `data.path` and the W&B
   `project`/`entity`).
2. Launch:

```bash
# Single GPU
./scripts/training/train.sh configs/lora.yaml

# Override output directory / run name, disable W&B
./scripts/training/train.sh configs/full_ft.yaml \
    --output_dir runs/barca-full-ft-1 --run_name barca-full-ft-1 --no_wandb

# Multi-GPU
NUM_GPUS=4 ./scripts/training/train.sh configs/qlora.yaml

# Resume
./scripts/training/train.sh configs/lora.yaml --resume_from_checkpoint outputs/barca-0.8b-lora/checkpoint-800
```

Equivalent direct invocation:

```bash
python -m barca.training --config configs/lora.yaml
```

Quick overrides (useful for smoke tests and sweeps) are available on the CLI:

```bash
python -m barca.training --config configs/full_ft_smoke.yaml \
    --max_steps 200 --per_device_train_batch_size 32 --learning_rate 2e-3 \
    --max_train_samples 2000 --max_eval_samples 200 --no_wandb \
    --set gradient_checkpointing=false
```

`--set KEY=VALUE` patches any single `transformers.TrainingArguments` field.

### Recipe structure

Each YAML file has these sections: `run`, `data`, `tokenizer`, `model`, `peft`,
`training`, `checkpointing`, `wandb`. The `training` section is forwarded
verbatim to `transformers.TrainingArguments`; everything else is validated by
`barca.config` so typos fail fast. Unknown keys raise an error.

### Selective token training

To fine-tune **only** the new protein-token embeddings and leave every other
weight (including the rest of the vocabulary embeddings) frozen, set one of the
`model.trainable_*` options:

```yaml
model:
  trainable_token_range: [248077, 248098]   # inclusive token-id range
  # or: trainable_tokens: ["<protein>", "</protein>", "ζA", ...]
  # or: trainable_token_ids: [248077, 248078, ...]
```

All parameters are frozen and only the selected rows of the input embedding
(and the tied `lm_head`) are trained; gradients for every other row are masked.
In addition, `RestoreFrozenEmbeddingRowsCallback` restores the non-selected
rows from a snapshot after each optimizer step, so `weight_decay` decays only
the selected rows instead of the whole vocabulary matrix.

`barca-0.8b-base` is Qwen3.5 with 22 added protein tokens at ids
**248077–248098**: the 20 ζ-prefixed amino acids (`ζA`…`ζY`) plus
`<protein>` and `</protein>`.

### Checkpoint policy

`checkpointing.top_k: 3` keeps the three checkpoints with the lowest
`eval_loss` (set `metric` / `greater_is_better` to change this). Evaluation and
saving are forced to the same cadence (`save_steps == eval_steps`) so a saved
checkpoint always matches the metric recorded with it. The Trainer's own
`save_total_limit` is disabled — `TopKCheckpointCallback` owns retention.

Checkpoints are kept **locally only**; W&B receives metrics, not model
artifacts. The best checkpoint is reloaded at the end of training and written
to `run.output_dir` as the final model.

---

## W&B

Set credentials before launching:

```bash
export WANDB_API_KEY=...
# Optional: export WANDB_ENTITY=your-team
```

`wandb.project`, `wandb.entity`, `wandb.tags` and `wandb.notes` come from the
recipe. Every run is tagged with a short config fingerprint (`cfg-XXXXXXXX`) for
traceability, and the resolved YAML is saved to `output_dir/config.yaml` and
uploaded to the run. Pass `--no_wandb` (or `wandb.enabled: false`) to disable.

---

## Remote server notes

- The env is assumed to contain all runtime libraries; no dependency install is
  required, but `pip install -e .` is the cleanest way to expose the `barca`
  package.
- If the training interpreter is not the default `python`, point the launch
  scripts at it with `PYTHON=/path/to/python` (the GPU server uses
  `/opt/miniconda3/bin/python`).
- Set `model.attn_implementation` to `flash_attention_2` only if flash-attn is
  installed; otherwise use `sdpa` or `null`.
- Cross-node/multi-GPU launches use `torchrun` (see `scripts/training/train.sh`).
- Adjust `data.num_proc`, `dataloader_num_workers` and
  `per_device_train_batch_size` to the server's CPU/RAM/VRAM budget.

---

## Development

```bash
pytest -q
ruff check .
```

`tests/test_config.py` only needs PyYAML; `tests/test_callbacks.py` also needs
`transformers`.
