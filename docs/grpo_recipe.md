# GRPO recipe (working notes)

Status: **implemented and smoke-tested; full training run pending**. The reward
suite, ESMFold2 pLDDT service, prompt builder and GRPO trainer exist in the repo.
A one-step G=5 run has been tested end to end on keyeri; the full 5k-prompt run
has not been launched.

Related files:

- `configs/sft_design_continue_2e-6.yaml` — SFT starting point.
- `scripts/data/prepare_mol_instruct_design.py` — SFT data format.
- `docs/lora_2m_findings.md` — training dynamics that motivate this stage.
- Evaluation workspace: `/home/khairi/work/phd/projects/evaluation/barca`
  (`runs/exp100/`, `figures/esmfold2_exp100/`).

## 1. Goal and scope

Continue `outputs/barca-0.8b-sft-design-cont-2e-6` with GRPO so that the model
improves its **reference fidelity** and **sequence quality** without collapsing
diversity or natural-text ability. This is not a state-of-the-art folding claim
and does not include wet-lab validation.

The main hypothesis is that SFT optimizes the likelihood of the reference
sequence, while GRPO can directly optimize a multi-objective reward:
format validity, reference 5-mer overlap, ESMFold2 foldability (pLDDT), diversity,
and length.

## 2. Starting point

| Item | Value |
|---|---|
| Policy | `outputs/barca-0.8b-sft-design-cont-2e-6` |
| Reference policy | frozen copy of the same checkpoint (KL) |
| Adaptation | LoRA r=64, alpha=128 (first run); full FT is a later option |
| Tokenizer | Barca protein-aware tokenizer (`ζ`-prefixed AAs, `<protein>`, `</protein>`) |
| Prompt format | ChatML user turn (instruction + input) |

The policy and reference model are the same checkpoint at the start, so the
initial KL is zero. LoRA is preferred for the first run because it limits drift
and keeps memory low; full FT with a low learning rate is a follow-up.

## 3. Data recipe

- Source: `~/workspace/datasets/barca/mol-instruct`, `train` split
  (190,554 rows). The validation split is held out by accession.
- Select **5,000 prompts**, stratified by reference length and task, with a
  fixed seed.
- Parse each row's ChatML user turn; keep the reference sequence from the
  assistant turn, strip `ζ`, and retain canonical sequences of 30–512 aa.
- Dataset columns:
  - `prompt`: chat-templated user turn with `add_generation_prompt=True`
  - `reference`: plain amino-acid sequence
  - `accession`, `ref_len`
- Evaluation: the 100 prompts in
  `evaluation/barca/runs/exp100/exp100_indices.txt` (seed 20251120). They are
  held out from SFT and are not used for GRPO prompt sampling.
- Do not sample the validation/exp100 accessions into the GRPO training set.

## 4. Reward recipe

Rewards are computed per completion. Foldability uses a local ESMFold2 service
with reduced recycling/sampling cycles; the full 20-loop/100-step setting remains
for held-out evaluation.

| Reward | Definition | Role |
|---|---|---|
| `format` | closed `</protein>`, parseable, canonical AAs, 30–512 aa | hard validity |
| `ref_f1` | 5-mer F1: `2|A∩B|/(|A|+|B|)` against the reference | reference fidelity |
| `diversity` | `uniq5 = |K(s)|/(|s|-4)`; penalize below ~0.8 | avoid repeats/collapse |
| `length` | closeness to reference length or the 30–512 target | avoid degenerate short outputs |
| `foldability` | low-cycle ESMFold2 pLDDT via the local FastAPI service | direct structural score |

Implemented weights: `format 0.10`, `ref_f1 0.30`, `foldability 0.35`,
`diversity 0.15`, `length 0.10`. The first run uses raw-scale weights and
normalizes the total reward per group (`sum_then_normalize`). Monitor each
component separately; a rising total with collapsing `ref_f1`/`diversity` is a
reward-hacking signal.

Anti-hacking checks:

- Track `ref_f1`, `uniq5`, length, and the fraction of outputs that copy long
  reference substrings.
- Inspect 10 completions per checkpoint.
- Evaluate on the held-out 100 prompts with ESMFold2/ColabFold, not only with
  the training reward.
- Keep a KL term to the frozen SFT policy.

## 5. GRPO configuration

Proposed first run (TRL 1.6.0):

| Setting | Value |
|---|---|
| Trainer | `trl.GRPOTrainer` (fallback: a custom GRPO loop if Qwen3.5 save/load is incompatible) |
| Group size | 5 |
| Temperature | 0.8–1.0 |
| top-p | 0.95 |
| Max completion | 700 tokens |
| KL coefficient (`beta`) | 0.04 |
| Adaptation | LoRA r=64, alpha=128 |
| Learning rate | 1e-6 |
| Precision | bf16 |
| Batch | 1 prompt per device × grad accumulation 5 (one 5-prompt group per optimizer step) |
| Schedule | 1 epoch over 5k prompts, ~300–600 steps |
| Save/eval | every 100 steps; resume from checkpoint |
| Seed | 42 for training, 0 for evaluation generation |

Generation is the bottleneck. Benchmark HF `generate` on 10 prompts × group 4
before launching; use vLLM only if its version/compatibility is confirmed.
The policy and reference model are ~0.8B each; a reward model/ESM-C adds a
small footprint, but all jobs share the GB10 with other work.

Illustrative config snippet (not a committed config file):

```yaml
grpo:
  base_model: outputs/barca-0.8b-sft-design-cont-2e-6
  reference_model: outputs/barca-0.8b-sft-design-cont-2e-6
  data: mol-instruct/train
  num_prompts: 5000
  num_generations: 5
  temperature: 0.9
  top_p: 0.95
  max_completion_length: 700
  beta: 0.04
  learning_rate: 1.0e-6
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 1
  bf16: true
  gradient_checkpointing: true
  save_steps: 100
  eval_steps: 100
  lora:
    r: 64
    alpha: 128
  reward_weights: [0.10, 0.30, 0.35, 0.15, 0.10]
```

## 6. Dependencies

TRL 1.6.0 requires `transformers>=4.56.2`, `datasets>=4.7.0`,
`accelerate>=1.4.0`, and `packaging>20.0`; `peft>=0.8.0` is needed for LoRA.
The foldability reward additionally needs `httpx` and the ESMFold2 service
(`fastapi`, `uvicorn`, and the `esm` package with `biohub/ESMFold2`).

| Package | Recipe floor | keyeri (current) | local (current) |
|---|---|---|---|
| `trl` | `>=1.6.0` | 1.6.0 | 0.29.1 (needs upgrade) |
| `transformers` | `>=4.56.2` | 5.16.1 | 5.20.2 |
| `datasets` | `>=4.7.0` | 5.0.0 | 5.0.0 |
| `accelerate` | `>=1.4.0` | 1.14.0 | 1.12.0 (needs upgrade) |
| `peft` | `>=0.8.0` | 0.19.1 | 0.18.1 |
| `esm` (ESMFold2 service) | `>=3.4.1` | 3.4.3 | 3.4.0 |

The repository's core dependencies currently list `transformers>=4.40`,
`accelerate>=0.29`, and `peft>=0.11`; the GRPO run needs the stricter floors
above. The keyeri environment already satisfies them. The local environment's
`trl 0.29.1` is older than the GRPO API used in this recipe; run on keyeri or
upgrade before running locally.

## 7. Evaluation recipe

On the 100 held-out prompts:

- Generate 1 candidate per prompt (3 optionally for variance).
- Sequence: parse/validity, length, 5-mer uniqueness, reference 5-mer F1,
  ESM-C/ESM-2 likelihood.
- Folding: ESMFold2 with 20 loops, 100 diffusion steps, 1 sample, seed 0;
  truncate to 512 aa for sequences longer than 512.
- Text: small perplexity check (Cosmopedia subset) to detect catastrophic
  forgetting.
- Paired bootstrap 95% CIs against the SFT checkpoint; inspect 10 completions.
- Fold 20–50 sequences at intermediate checkpoints; fold the full 100 for the
  final comparison.

## 8. Compute and scheduling

- GB10 with unified memory; wait for the current ESMFold2/folding jobs to finish
  before launching GRPO.
- Generation dominates: 5,000 prompts × 4 completions × up to 700 tokens.
  Benchmark throughput first and reduce prompts/completion length if needed.
- Use resume-safe checkpointing and log reward components, KL, entropy, and
  sample completions.
- Do not run concurrent GPU-heavy folding and GRPO jobs on the shared server.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Reward hacking (copying, short outputs, repeats) | multi-objective reward; held-out folding eval; inspect samples |
| Catastrophic forgetting / text degradation | LoRA, KL to SFT, text-perplexity check |
| TRL/Qwen3.5 save-format incompatibility | compatibility spike; custom GRPO loop fallback |
| Folding reward too expensive | use ESM/ESM-C proxy in the loop; fold only for eval |
| GPU contention | schedule after current jobs; benchmark and cap runtime |
| Data overlap with evaluation | sample prompts from train; keep exp100 held out |

## 10. Acceptance criteria

- Format/validity at least equal to SFT on the 100-prompt evaluation.
- A significant improvement in reference 5-mer F1 (paired CI excluding zero).
- No collapse in 5-mer uniqueness or output length.
- ESMFold2 pLDDT/pTM not worse than SFT within a pre-registered tolerance.
- Text perplexity degradation bounded and documented.
- The run is reproducible from the saved config, prompt indices, and seeds.

## 11. Open questions

1. Is the primary objective reference fidelity, foldability, or diversity?
2. 5,000 prompts or a smaller 1,000-prompt first pass?
3. When to try full FT after the LoRA first run?
4. Which low-cycle ESMFold2 setting preserves enough pLDDT rank correlation with
   the full 20-loop/100-step evaluation?
5. Is a 512-aa cap acceptable for the RL completions, or should long
   sequences be included with a length penalty?
6. Should foldability score all G candidates every step, or only a subset to
   reduce GPU contention with the shared server?

## 12. Not in this document

The trainer and service are implemented in `scripts/training/train_grpo.py`,
`src/barca/rl/`, and `scripts/serving/esmfold2_service.py`. The full run is not
launched yet; use the smoke command in the README/run log before the long run.
