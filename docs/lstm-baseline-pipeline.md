# LSTM Baseline Pipeline

This repository trains and generates LSTM baselines. It does not compute
Mistral surprisal; generated outputs are handed to `compute_surprisal_mila`
after the generation audit passes.

## Existing PBM Run

The Brown, Manchester, and Providence discovery run is already complete. It
trained 24 word-level encoder-decoder LSTMs:

```text
3 caretaker-context windows (k3, k4, k5) x 8 additive age bins
```

Each model generated a same-length child-like utterance. Those three generated
sources were subsequently scored by Mistral under scoring contexts k0-k3. Do
not rerun that proof of concept.

## Full-79 Selection

The strict-naturalistic 79-child run deliberately keeps only generation
context `k3`.

PBM means were nearly identical:

| generation context | mean k3-scored bits | mean source-minus-real gap |
| --- | ---: | ---: |
| k3 | 33.06 | 1.707 |
| k4 | 33.09 | 1.798 |
| k5 | 33.08 | 1.799 |

No context variant won consistently across the fixed-effort models. Keeping
k3 avoids selecting a discovery-sample specification after inspecting one
favorable coefficient and cuts the production model count from 24 to 8.

The remaining eight models are scientifically necessary. For each target age
bin, its LSTM trains on that bin plus all earlier bins and generates only rows
in that target bin:

```text
006-023
024-029
030-035
036-041
042-047
048-053
054-059
060-065
```

Replacing them with one global model would expose early targets to later-age
training data. Reusing one model across adjacent bins would no longer match
the additive information regime used by the n-gram controls.

Frozen full-79 specification:

- architecture: word-level encoder-decoder LSTM;
- generation context: last 3 caretaker utterances, capped at 60 word tokens;
- training scope: all strict-naturalistic children, cumulative by age bin;
- output vocabulary: child-side training tokens only;
- generation: one same-word-length sample per real child target;
- model size: 256 embedding, 512 hidden, 2 layers, dropout 0.2;
- optimization: 20 epochs, batch size 256, seed 123.

## Mila Environment

Use a Python environment on Mila in which PyTorch imports and CUDA is
available. Set `PYTHON_CMD` to that interpreter before submission. The GPU
wrapper rejects an environment without CUDA.

From the Mila login shell:

```bash
cd "$HOME/communicative_efficiency_repos/generate_baselines_mila"
export PYTHON_CMD="$HOME/venvs/generate-baselines/bin/python"
"$PYTHON_CMD" -c 'import torch; print(torch.__version__, torch.backends.cuda.is_built())'
```

The second value must be `True`, showing that this PyTorch build includes CUDA.
A login node has no allocated GPU, so `torch.cuda.is_available()` can correctly
be `False` there. The exact-wrapper smoke runs on an allocated GPU and rejects
the environment unless `torch.cuda.is_available()` is `True` in that job.
Environment creation is intentionally not hidden inside a production job.

## Local Verification Before Push

Run from the repository checkout:

```bash
PYTHONPYCACHEPREFIX=/tmp/generate_baselines_mila_pycache PYTHONPATH=src python3 -m unittest discover -s tests
bash -n slurm/*.sbatch slurm/*.sh
git diff --check
```

The optional torch test should also be run with the same PyTorch version used
on Mila.

## Submit The Full Run

The strict-naturalistic bundle is expected at:

```text
$SCRATCH/communicative_efficiency_data/big_cleaned_dataset/default_naturalistic_merged_006_023
```

Submit the complete dependency graph from the Mila login shell:

```bash
cd "$HOME/communicative_efficiency_repos/generate_baselines_mila"
export PYTHON_CMD="$HOME/venvs/generate-baselines/bin/python"
bash slurm/submit_full_79_lstm.sh "$SCRATCH/communicative_efficiency_data/big_cleaned_dataset/default_naturalistic_merged_006_023"
```

This is the only production submission command. It submits:

1. a CPU input preparation and 79-child audit;
2. a representative GPU smoke using the exact production wrapper, k3,
   `060-065`, production model dimensions, and 25 target rows;
3. wave 1 for the first four additive age-bin models;
4. a wave-1 output/checkpoint audit;
5. wave 2 for the last four additive age-bin models;
6. a wave-2 audit and final eight-model audit.

The GPU smoke writes:

```text
<run-root>/reports/smoke/smoke_report.md
<run-root>/reports/smoke/smoke_summary.json
<run-root>/SMOKE_PASSED
```

Both production waves are blocked on the smoke. Wave 2 is additionally blocked
on `WAVE1_READY`. The run is complete only when this marker exists:

```text
<run-root>/COMPLETE_AND_AUDITED
```

An empty `squeue` is not evidence of completion. Check the printed job ids with
`sacct` and retrieve the compact report directory using the exact `rsync`
command printed by the submitter. Run that retrieval command on the local
laptop, not on a Mila login node.

## Production Outputs

Each of the eight production cells writes:

- one generated CSV with row ids, child/session provenance, scoring contexts
  k1-k3, real target text, and `generated_utterance`;
- `model.pt`;
- shared `vocab.json`;
- child-only `child_output_vocab.json`;
- `train_audit.json`;
- an epoch-boundary `training_state.pt` for bounded-loss resumption;
- an output audit with checksums and row counts.

Interrupted files are published atomically. A rerun skips a cell only after
the existing generated rows, lengths, source label, checkpoint, vocabularies,
and audits validate successfully.

After retrieval and review, the generated-output manifest can be handed to
`compute_surprisal_mila`. LSTM training and generation remain complete even
before that separate Mistral scoring step begins.
