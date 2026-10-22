# LoRA 2M run: findings so far (working notes for the paper)

Status as of 2026-10-22: training is at about step 12,100 of 15,625. Everything below is
intermediate. Folding is complete through checkpoint 12000.

## 1. Setup

- Base model: `khairi/barca-0.8b-base` (Qwen3.5-based causal LM, 22 added protein tokens at ids
  248077-248098: 20 zeta-prefixed amino acids plus `<protein>` and `</protein>`).
- Starting point: `outputs/barca-0.8b-emb-1m-fp32` (embedding-only run on 1M sequences).
- Run: `configs/lora_2m_fp32.yaml`. LoRA r=64, alpha=128, fp32, LR 5e-5 cosine to 5e-6, 2M
  UniRef50 sequences, 15,625 steps, eval and save every 500, top-3 checkpoints by `eval_loss`
  (`TopKCheckpointCallback`).
- Speed: about 49.6 s/step on an idle GPU, 90-100 s/step when another job shares it. The server
  GPU is a GB10 with 119 GB unified memory, so any evaluation job slows training.

### Protein validation loss

| Step | Eval loss |
|---|---|
| 4000 | 2.7270 |
| 8500 | 2.7032 |
| 9000 | 2.7017 |
| 9500 | 2.7000 |
| 10000 | 2.6987 |
| 10500 | 2.6971 |
| 11000 | 2.6960 |
| 11500 | 2.6948 |
| 12000 | 2.6938 |

The loss keeps falling by about 0.001 per 500 steps. The embedding-only starting point was
2.785.

## 2. Generation and folding protocol

Generation (`scripts/generation/generate_fasta.py`), settings passed explicitly because the
script defaults differ (T 0.7, top_k 250, penalty 1.1):

- T 0.6, top_p 0.95, top_k 50, repetition penalty 1.2, seed 0, 50 samples, batch 5,
  max_new_tokens 512, prompt `<protein>`.
- Filter: `filter_proteins.py --max-len 400` (length, entropy, homopolymer run).
- The same seed is used at every checkpoint, so the same sample index gives a similar sequence
  across checkpoints. The checkpoints are therefore not independent draws.

Folding: local ColabFold, `--num-models 1 --num-recycle 3 --model-type auto`, one model per
sequence, public MSA server. About 40 sequences take 1.5-2 h on an RTX 3050 (6 GB).

Known issues:

- The MSA server sometimes leaves requests `PENDING` for 12-30 min, and one DNS failure killed
  a run. Killing and rerunning the same command resumes, skipping finished structures.
- The filter does not catch tandem repeats. LRR, TPR and ankyrin proteins pass it.
  Repeat-like heuristic used for analysis: unique 5-mer fraction below 0.8, or tandem identity
  above 0.45 at any period 3-60 for length above 70.

## 3. Structure quality by checkpoint

Mean pLDDT over all folded sequences (about 40 per checkpoint):

| Step | Folded | Mean | Median | >50 | >70 | Mean pTM |
|---|---|---|---|---|---|---|
| 1500 | | 50.5 | | | | |
| 2000 | | 45.1 | | | | |
| 4000 | | 50.0 | | | | |
| 4500 | | 52.4 | | | | |
| 5500 | | 51.4 | | | | |
| 6000 | | 50.8 | | | | |
| 6500 | | 48.4 | | | | |
| 7000 | 40 | 52.3 | 49.2 | 19 | 7 | 0.35 |
| 8500 | 39 | 48.3 | 42.6 | 16 | 5 | 0.35 |
| 9000 | 45 | 52.2 | 50.1 | 23 | 9 | 0.37 |
| 9500 | 38 | 50.2 | 41.5 | 15 | 10 | 0.36 |
| 10000 | 42 | 47.5 | 43.2 | 16 | 2 | 0.33 |
| 11000 | 40 | 48.4 | 43.6 | 14 | 5 | 0.33 |
| 11500 | 43 | 46.8 | 43.4 | 14 | 3 | 0.33 |
| 12000 | 42 | 44.9 | 40.2 | 12 | 3 | 0.33 |

(Blank cells were not recorded for those early checkpoints.)

Findings:

- Mean pLDDT plateaus at 47-52 from step 4000 on, with no trend, while eval loss keeps
  falling. The differences between checkpoints are within noise (about 40 sequences each, a
  shared seed). A real comparison needs 200+ samples per checkpoint and different seeds.
- 12000 has the lowest mean so far (44.9), but the drop from 11500 (46.8) is inside the noise.
  Its best structures are 11000/11500-style repeats (sample38, 240 aa, pLDDT 85.1, pTM 0.856, an
  LRR protein) and one weak non-repeat long fold (sample4, 357 aa, 58.7/0.627).
- Short sequences (100 aa or fewer) score higher than long ones (11000: 57.8 vs 46.4; 11500:
  53.6 vs 44.8; 12000: 59.5 vs 42.4), which partly reflects pLDDT being easier on short chains.
- The best-scoring long structures are mostly repeat families, which the filter lets through:
  LRR (9000/9500/10500/11500 sample1 or sample38), TPR (10500 sample2), ankyrin (7000
  sample4). 11500 sample38 (235 aa, pLDDT 88.2, pTM 0.859) is an LRR protein.
- Non-repeat candidates exist but are modest. 11000 sample42 (213 aa, pLDDT 72.1, pTM 0.708) has
  a packed helical bundle and a membrane-protein-like sequence (a guess from sequence, no
  database search was run). 11000 sample18 (239 aa, 65.8, 0.693) is similar in quality. 9500
  sample19 (214 aa, kinase-like motifs) is the best non-repeat-ish hit at that checkpoint.
- No BLAST or Foldseek search has been run (the sandbox blocks outbound hosts), so none of the
  candidates has been checked for novelty against natural families.

PNG renders of the top 5 per checkpoint are in `data/gen/candidates/ckpt<STEP>/` (gitignored).

## 4. Loss of natural-text ability

The LoRA run degrades the model's ability on ordinary text and code.

Six-prompt check (T 0.7, top_k 50, 70 new tokens, 2 samples each): protein-string leakage in
0/12 completions for the base model, 0/12 for emb-1m-fp32, and 5/12 for checkpoint 10500. The
base model writes working Fibonacci code and answers factual prompts. The checkpoint emits
protein strings on those prompts and has lost the code ability.

Cosmopedia perplexity (800 rows, 554,607 tokens, fp32, max length 1024):

| Subset | Base | ckpt 10500 |
|---|---|---|
| auto_math_text | 4.729 | 7.037 |
| khanacademy | 3.633 | 5.067 |
| openstax | 6.221 | 9.677 |
| stanford | 5.609 | 8.610 |
| stories | 10.371 | 14.619 |
| web_samples_v1 | 7.043 | 10.087 |
| web_samples_v2 | 7.420 | 10.226 |
| wikihow | 5.543 | 7.873 |
| **ALL** | **5.995** | **8.713 (+45%)** |

Checkpoint 11000 gives ALL 8.825 (+47%), so the drift is still growing.

Leaked-mass analysis (probability the model puts on protein tokens while reading ordinary text):

- 10500: protein-token mass 1.70%, argmax protein 1.28%, perplexity with protein tokens masked
  8.372. 11000: 1.85%, 1.38%, masked 8.444.
- Masking the protein tokens removes only about 11% of the loss gap (0.040 of 0.374 nats). The
  remaining 89% is real weight drift.
- Leakage is highest on technical subsets (openstax and auto_math_text, about 3%) and lowest on
  stories (about 0.6%).
- Text loss drifts by about 0.013 nats per 500 steps while protein eval loss improves by about
  0.001 per 500 steps.

## 5. Planned: calibration

Nothing here is built yet. It is meant to run after training finishes.

- Mix protein, text and code at 3:1:1 by tokens, with a per-batch mixing sampler.
- Text: a fresh Cosmopedia sample (different seed, excluding the 800 eval rows). Code: a
  permissively licensed Python set filtered with a syntax check.
- Keep the frozen base model as a teacher: add a KL term to its logits on text and code batches
  (computed online, no generated data needed).
- Continue from the final LoRA checkpoint at LR about 1e-5 for 1-2k steps, as a new recipe
  `configs/calibrate.yaml`.
- Guards: protein-token probability on text (`generate.py --probe`), Cosmopedia perplexity
  within about 2% of 5.995, the six-prompt check including Fibonacci, protein validation
  perplexity, and pLDDT.
- Alternatives: adapter switching at inference (lossless, since the embedding-only model leaves
  text untouched), a LoRA scale below 1, plain replay, and ablations of 4:1:1, 3:1:1 and 2:1:1
  (these need a free GPU).

## 6. Open items

- Larger, independent-seed generation test for a real checkpoint comparison.
- Tandem-repeat check in `filter_proteins.py`.
- Foldseek and BLAST on the best non-repeat candidates.
- Early-checkpoint text evaluation is not possible, because top-3 retention deleted the earlier
  checkpoints.
