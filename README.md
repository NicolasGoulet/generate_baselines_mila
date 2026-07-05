# generate_baselines_mila

Cluster-ready baseline utterance generation for the communicative-efficiency
project.

This repository is intentionally small. It owns generation of baseline
utterances and scorer-ready exports; it does not own CHILDES preprocessing,
large scored outputs, Mistral scoring, or supervisor-facing reports.

## Repo Boundary

- `communicative_efficiency`: project brain, data links, analysis tables,
  reports, design notes, and compact audits.
- `compute_surprisal_mila`: neural surprisal scoring on Mila.
- `generate_baselines_mila`: baseline utterance generation on CPU or GPU
  clusters.
- Future `bayes_efficiency_mila`: Bayes-style `p(c | u)` likelihood scoring
  and posterior/decomposition tables.

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

Run the unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
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
storage.
