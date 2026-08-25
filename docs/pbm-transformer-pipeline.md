# PBM-Held-Out Small-Transformer Pipeline

## Scientific Contract

This workflow compares two conditional language-model architectures trained
from scratch on the same PBM-excluded naturalistic caregiver-child examples:

| condition | family | nominal parameters | frozen dimensions |
| --- | --- | ---: | --- |
| `babyllama_sized_llama_58m` | decoder-only LLaMA | 58,343,936 | 16 layers, 512 hidden, 1,024 FFN, 8 heads |
| `t5_58m` | T5-style encoder-decoder | 58,540,544 | 6+6 layers, 512 hidden, 2,560 FFN, 8 heads |

The first condition reproduces the published BabyLlama **student model
dimensions**, not its training procedure. Published BabyLlama was distilled
from two larger teachers. This experiment uses no teacher and must be called
“BabyLlama-sized LLaMA trained from scratch,” not a BabyLlama replication.
The T5 condition also starts from random weights; it does not load the C4
pretrained `t5-small` checkpoint.

Both models use one 16,000-entry byte-level BPE tokenizer trained only on the
non-PBM `train_all_ages` file. Sharing the tokenizer and closely matching the
parameter count make the architecture comparison interpretable. Brown,
Manchester, and Providence never enter tokenizer training, model training,
validation, or epoch selection.

The frozen machine-readable specification is
`configs/pbm_transformers_from_scratch_v1.json`.

## Models And Age Schedule

There are 16 final models:

```text
2 architectures x 8 cumulative age cutoffs = 16
```

The cutoffs are `006-023`, `024-029`, `030-035`, `036-041`, `042-047`,
`048-053`, `054-059`, and `060-065`. Each model optimizes on its cumulative
non-PBM training file and selects an epoch using the corresponding
whole-child-disjoint validation file. It is then reinitialized and refitted
for that selected number of epochs on the cumulative development file
(`train + validation`). Only this refitted model generates PBM responses.

This policy uses all available non-PBM development speech without using PBM
outcomes for model or epoch selection. The 16 final models are the scientific
objects; the within-cell selection fits are intermediate checkpoints, not 16
additional experimental conditions.

## Generation Estimand

Each final model produces one stochastic, unconstrained-length response for
every PBM target in its matching age bin. Generation uses temperature 1,
top-p 1, and a stable seed. The response is not forced to match the child's
word count, because forced length would invalidate the utterance-effort
question.

The safety ceiling is 128 generated subword tokens. Every output includes
`eos_reached` and `max_token_censored`. A cell fails its production audit if
more than 0.1% of responses reach the ceiling. Therefore the ceiling cannot
silently make model utterances artificially short. If that gate fails, raise
the ceiling in a new versioned run and repeat the exact smoke before any
scientific comparison.

This pipeline generates one response per observed PBM target, not a
100-response semantic choice set. The existing Qwen full100 handoff remains
the response-cloud reference. These two new sources are architecture
baselines that can be overlaid or modeled as explicitly labelled candidate
types.

## Input Contract

The expected upstream directory is the audited handoff created by the
`communicative_efficiency` repository:

```text
results/transformer_training_expansion/full_20260825/
```

Preparation requires `BUILD_COMPLETE_AND_AUDITED`, rehashes every tokenizer,
train, validation, development, and PBM-target input against `manifest.json`,
checks the JSONL schema, rejects PBM rows in all learning inputs, rejects
non-PBM rows in evaluation targets, and verifies that development row counts
equal train plus validation.

Do not commit or transfer the data through Git. Put the directory under Mila
scratch and pass that exact directory to the submitter.

## Mila Runtime

Create a dedicated environment outside the production run root and install
the frozen runtime:

```bash
python3 -m venv "$SCRATCH/venvs/pbm-transformers-v1"
"$SCRATCH/venvs/pbm-transformers-v1/bin/pip" install --upgrade pip
"$SCRATCH/venvs/pbm-transformers-v1/bin/pip" install \
  -r requirements-pbm-transformers.lock.txt
"$SCRATCH/venvs/pbm-transformers-v1/bin/pip" install -e .
export PYTHON_CMD="$SCRATCH/venvs/pbm-transformers-v1/bin/python"
```

The login node can validate imports but cannot prove CUDA availability. The
exact production wrapper performs that check inside both architecture smoke
tasks.

Before pushing or submitting, run:

```bash
PYTHONPYCACHEPREFIX=/tmp/generate_baselines_transformer_pycache \
  PYTHONPATH=src python3 -m unittest discover -s tests
bash -n slurm/*.sbatch slurm/*.sh
git diff --check
```

## Smoke-Gated Production DAG

From the Mila checkout, after the handoff and environment are available:

```bash
export PYTHON_CMD="$SCRATCH/venvs/pbm-transformers-v1/bin/python"
bash slurm/submit_pbm_transformers.sh \
  "$SCRATCH/communicative_efficiency_data/transformer_training_expansion/full_20260825"
```

The submitter creates this `afterok` graph:

1. CPU input audit and shared-tokenizer training;
2. a two-task GPU smoke array, one task per architecture, using the exact
   production wrapper, production model dimensions, 1,024 training rows, and
   25 PBM targets;
3. smoke audit and `SMOKE_PASSED`;
4. production cells 0-7;
5. wave-1 audit and `WAVE1_READY`;
6. production cells 8-15;
7. wave-2 audit and `WAVE2_READY`;
8. final 16-cell merge/audit and `COMPLETE_AND_AUDITED`;
9. Slurm-state and hash report and `FINAL_REPORT_READY`.

Both selection and refit phases save epoch-boundary model, optimizer, and
scheduler state. A production cell may resume after interruption. Existing
outputs are skipped only when their complete cell audit passes.

The GPU resource contract is one task and exactly one GPU per array element.
`GPU_GRES` accepts `gpu:1` or one typed GPU such as `gpu:l40s:1`; the submitter
rejects requests for any other GPU count.

## Scorer-Ready Handoff

The final generation audit publishes:

```text
<run-root>/handoff/pbm_transformer_responses_scorer_ready.csv.gz
<run-root>/handoff/manifest.json
```

The table contains stable generated and source-example identities, PBM
provenance, age bin, `context_k3`, real target, architecture label, generated
text, generated word/token counts, decoding configuration, and censoring
flags. The handoff manifest freezes its SHA-256, row count, source-cell hashes,
target column (`generated_utterance`), context column (`context_k3`), and the
required downstream Mistral contexts (`k0`, `k3`).

Generation completion is not scoring completion. Mistral scoring remains in
`compute_surprisal_mila`, where a separate exact-wrapper smoke and audit must
pass before production scoring begins.
