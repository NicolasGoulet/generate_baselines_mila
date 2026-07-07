# generate_baselines_mila

Cluster-ready baseline utterance generation for the communicative-efficiency
project.

This repository is intentionally small. It owns generation of baseline
utterances and scorer-ready exports; it does not own CHILDES preprocessing,
large scored outputs, Mistral scoring, or supervisor-facing reports.

## Repo Boundary

- `communicative_efficiency`: local project brain, data links, analysis
  tables, reports, design notes, and compact audits. It is not required on
  Mila for the modular smoke test.
- `compute_surprisal_mila`: neural surprisal scoring on Mila.
- `generate_baselines_mila`: baseline utterance generation on CPU or GPU
  clusters.
- `bayes_efficiency_mila`: Bayes-style `p(c | u)` likelihood scoring and
  posterior/decomposition tables.
- `child_complexity_predictors`: MLU, vocabulary, and complexity predictor
  extraction.

## Compute Lanes

CPU-first:

- random same-length baselines
- unigram, bigram, and trigram baselines
- additive age-bin count dictionaries
- manifest audits, checksums, joins, and row-count validation

CPU smoke / GPU production:

- LSTM training and generation
- small neural generator experiments
- neural parsing experiments if dependency-length predictors are added here

Mila GPU / scoring repo:

- Mistral or other large-model surprisal scoring
- large LLM response generation
- Bayes likelihood scoring if implemented with neural models

## Quick Start

Validate a manifest:

```bash
python3 -m generate_baselines_mila validate-manifest --manifest configs/ngram_additive_example.json
```

Generate CPU n-gram baselines:

```bash
python3 -m generate_baselines_mila generate-ngram --manifest configs/ngram_additive_example.json
```

Prepare and run the full strict-naturalistic 79-child n-gram baseline job on
Mila after the bundle has been extracted under scratch:

```bash
cd "$HOME/communicative_efficiency_repos/generate_baselines_mila"
sbatch --output="$SCRATCH/full79-ngram-%j.out" \
  slurm/full_79_ngram_baselines.sbatch \
  "$SCRATCH/communicative_efficiency_data/big_cleaned_dataset/default_naturalistic_merged_006_023"
```

Run the unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Run the cross-repo smoke test on Mila after cloning the three modular sibling
repos into a permanent code location under `$HOME`. The smoke outputs default
to `$SCRATCH/modular_repo_smoke/<job_id>`; keep the Git checkouts out of
scratch and put only job artifacts there.

```bash
mkdir -p "$HOME/communicative_efficiency_repos"
cd "$HOME/communicative_efficiency_repos"
git clone git@github.com:NicolasGoulet/generate_baselines_mila.git
git clone git@github.com:NicolasGoulet/bayes_efficiency_mila.git
git clone git@github.com:NicolasGoulet/child_complexity_predictors.git

cd "$HOME/communicative_efficiency_repos/generate_baselines_mila"
mkdir -p "$SCRATCH/modular_repo_smoke_logs"
sbatch --output="$SCRATCH/modular_repo_smoke_logs/modular-smoke-%j.out" slurm/modular_repos_smoke.sbatch
```

This tests `generate_baselines_mila`, `bayes_efficiency_mila`, and
`child_complexity_predictors`. It does not require cloning
`communicative_efficiency` on Mila.

After rsyncing the smoke output you need, remove the scratch job directory and
log:

```bash
bash "$SCRATCH/modular_repo_smoke/<job_id>/cleanup_after_rsync.sh"
rm -f -- "$SCRATCH/modular_repo_smoke_logs/modular-smoke-<job_id>.out"
```

## Manifest Contract

The CPU generator reads one JSON manifest. Paths may be absolute or relative to
the manifest file.

Required fields:

- `run_id`
- `train_csv`
- `target_csv`
- `output_csv`
- `text_column`
- `target_text_column`
- `id_columns`

Recommended fields:

- `age_bin_column`
- `age_bins`
- `context_column`
- `context_tail_words`
- `generators`
- `same_length`
- `samples_per_target`
- `seed`
- `carry_columns`

The output contains one row per target utterance, generator, and sample index.
Every output has a JSON audit sidecar with row counts and file checksums.

## Data Policy

Do not commit CHILDES data, generated utterance CSVs, model checkpoints, logs,
or scored outputs. Transfer large inputs/outputs with `rsync` or cluster
storage. On Mila, keep permanent Git checkouts in `$HOME`; write production
outputs, temporary files, and rsynced full datasets under `$SCRATCH`, then clean
them after they have been retrieved or are no longer needed.
