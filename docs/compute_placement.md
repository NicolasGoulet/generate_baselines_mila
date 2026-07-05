# Compute Placement

This repo separates baseline generation by compute profile so the laptop and
Mila cluster are used for the right jobs.

## CPU-Only / CPU-First

Use CPU jobs for deterministic count-based generation:

- random same-length utterances
- unigram, bigram, and trigram utterances
- additive age-bin count dictionaries
- row-count audits and checksums
- scorer-ready CSV export

These jobs should run as Slurm arrays over corpus scope, age bin, split, or
context condition when the manifests become large.

## CPU Smoke / GPU Production

Use CPU only for small smoke tests, then run production on GPU nodes:

- LSTM training and generation
- future BabyLM-style small neural generators
- neural parser experiments if dependency predictors are added

The GPU path must write explicit model artifacts and audits. Do not claim an
LSTM baseline exists unless the training command ran and the artifacts are
present.

## Scoring Elsewhere

This repo should not score Mistral surprisal. Generated utterances should be
handed to `compute_surprisal_mila` for direct neural scoring.

Bayes-style likelihood scoring, especially `p(c | u)`, should live in a
separate scoring repo because it is not just another generated baseline. It has
different assumptions, different score columns, and different audit needs.
