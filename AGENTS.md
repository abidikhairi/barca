# AGENTS.md

Fine-tuning toolkit for `khairi/barca-0.8b-base`, a Qwen3.5-based decoder LM (the
"Llama-style" wording in `README.md` is legacy) with 22 added protein tokens at ids
248077-248098: 20 `ζ`-prefixed amino acids plus `<protein>` / `</protein>`. Runs are driven
by YAML recipes in `configs/`. See `README.md` for the user-facing guide, `CLAUDE.md` for
architecture/experiment notes, and `docs/lora_2m_findings.md` for the LoRA 2M write-up.

## Commands

```bash
pytest -q                                  # 41 tests; pytest pythonpath=src is in pyproject
pytest tests/test_config.py -q             # config tests need only PyYAML
pytest tests/test_callbacks.py::test_name  # single test
ruff check .                               # line-length 100, py310; tracked code is clean, but
                                           # untracked scripts/generation/sweep*_sft_design.py currently fail
python -m barca.training --config configs/lora.yaml [--no_wandb --max_steps N --set KEY=VALUE]
./scripts/training/train.sh configs/lora.yaml   # sets PYTHONPATH=src, PYTORCH_CUDA_ALLOC_CONF
PYTHON=/opt/miniconda3/bin/python ./scripts/training/train.sh configs/lora.yaml
NUM_GPUS=4 ./scripts/training/train.sh configs/qlora.yaml   # torchrun
```

GRPO:
PYTHONPATH=src /opt/miniconda3/bin/python scripts/training/train_grpo.py --config configs/grpo.yaml
python scripts/serving/esmfold2_service.py --host 127.0.0.1 --port 8010  # pLDDT reward service

Training runs on the remote GPU server, not locally. `make sync`, `make smoke`,
`make train RECIPE=...`, `make inspect`, `make pull checkpoint-1000` wrap that workflow;
`Makefile.local` (gitignored) holds the SSH host/data-path overrides. CPU execution fails
because the model uses FLA Triton kernels.

## Configuration and data gotchas

- Unknown recipe keys raise, except the whole `training` section is forwarded verbatim to
  `transformers.TrainingArguments`. CLI `--set KEY=VALUE` patches only that section, not
  `data`/`model`/`peft`.
- Top-level `configs/*.yaml` are training recipes. `configs/generation/*.yaml` are sampling
  settings used by the generation/sweep scripts, not valid `--config` inputs.
- `data/loading.py` supports a split root (`<path>/{train,valid}`) or a single root; tokenization
  is detected per split. In `barca_5m`, `train` is pre-tokenized and pre-packed at length 515
  while `valid` is raw, so keep `data.packing: false`.
- Pre-tokenized splits are passed through untruncated: `data.max_seq_length` does not apply, and
  the collator pads each batch to its longest row. One long row can therefore OOM a micro-batch
  (see the header of `configs/calibration_ft.yaml`); smoke-test with `--max_steps 2`.
- Training uses `DataCollatorForLanguageModeling`, which rebuilds labels from `input_ids`; a
  `labels` column (assistant masking) is ignored and the loss covers the whole row. Expect this
  when preparing chat/SFT data such as `scripts/data/prepare_mol_instruct_design.py`.
- `checkpointing.top_k` retention is owned by `TopKCheckpointCallback`, not Trainer. Evaluation
  and saving cadence is forced equal (`save_steps == eval_steps`), `save_total_limit` is removed,
  and an `eval_*` metric requires an eval split. The best checkpoint is copied to
  `run.output_dir`; checkpoints stay local, W&B gets metrics/config only.
- Selective token training (`model.trainable_token_range|tokens|ids`) freezes everything except
  the chosen embedding rows and tied `lm_head`; `RestoreFrozenEmbeddingRowsCallback` restores the
  other rows after every optimizer step so weight decay affects only selected rows.

## Operations and experiments

- The shared GB10 server also runs the 15,625-step `lora_2m_fp32` job (~50 s/step idle,
  90-100 s/step shared). Any GPU/eval job of yours slows it; only the top-3 checkpoints survive.
- Pass generation settings explicitly when comparing checkpoints (`generate_fasta.py` defaults
  differ): T 0.6, top_p 0.95, top_k 50, repetition penalty 1.2, seed 0, 50 samples, batch 5,
  then `filter_proteins.py --max-len 400`. Sharing one seed makes sample N similar across
  checkpoints.
- Folding is local ColabFold with the public MSA server, which can sit `PENDING` for 10+ min;
  rerunning in the same directory resumes. Outputs live under gitignored `data/gen/`.
- `filter_proteins.py` does not catch tandem repeats; LRR/TPR/ankyrin proteins pass and score
  high pLDDT, so check for repeats before calling a structure a candidate.
- `scripts/eval/ntp/evaluate.py` measures next-token loss/perplexity overall and per group from
  a CSV or `save_to_disk` dataset; `scripts/eval/ntp/download_dataset.py` builds a Cosmopedia
  sample with a `subset` column.
