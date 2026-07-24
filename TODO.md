# TODO.md

Production checklist for `generate_baselines_mila`.

This repo adds baseline generation to the communicative-efficiency project. It
must preserve the scientific objects already built in `communicative_efficiency`
and must not replace previous PBM proof-of-concept outputs.

## Core Contract

- [x] Keep this repo generation-only. It generates candidate/baseline
      utterances and scorer-ready exports; it does not compute Mistral
      surprisal.
- [x] Preserve input row ids, context ids, age bins, child/session provenance,
      and source-model labels in all outputs.
- [x] Keep large data, generated outputs, logs, and checkpoints out of Git.
- [x] Keep Git limited to code, tests, docs, Slurm scripts, tiny synthetic
      fixtures, and manifest templates. Move real cleaned data, generated
      utterances, model checkpoints, and run outputs with `rsync`.
- [x] On Mila, keep the permanent Git checkouts in `$HOME` as sibling repos;
      write job outputs, temporary data, and rsynced full datasets under
      `$SCRATCH`, then remove scratch job directories after retrieval.
- [x] Provide Slurm scripts that `cd` to repo root and set `PYTHONPATH=src`.
- [x] Keep the cross-repo Mila smoke runner in this execution repo:
      `slurm/modular_repos_smoke.sbatch`. Mila smoke testing requires only the
      three modular sibling repos, not the local `communicative_efficiency`
      brain repo.
- [x] Add production manifests that point to the actual Mila-side strict
      naturalistic bundle exports, not local laptop example paths.
- [ ] Add a manifest audit command that checks required columns, duplicate row
      ids, empty target utterances, age-bin coverage, and output collision risk
      before submitting a Slurm job.

## CPU Baseline Generation

- [x] Implement manifest-driven same-length random, unigram, bigram, and
      trigram generation.
- [x] Implement additive age-bin training: target-bin generation can train on
      current plus previous bins.
- [x] Support context-tail conditioning for n-gram initial history.
- [x] Write output CSV plus JSON audit sidecar with input/output checksums.
- [x] Unit-test same-length generation, empty-target skipping, and audit
      writing.
- [ ] Add production Slurm array scripts that run one manifest per array task.
- [ ] Add full-row-count regression tests using tiny fixture manifests for all
      supported source models.
- [ ] Add scorer-ready bundle export tests that verify generated rows preserve
      target ids and context text exactly.

## GPU LSTM Generation

- [x] Implement a real LSTM training/generation path in this repo. The current
      path is generic and consumes the same manifest shape as the n-gram
      generator.
- [x] Support CPU smoke runs and GPU production runs from the same manifest.
- [x] Support additive age-bin training for LSTM, matching the n-gram
      developmental information constraints.
- [x] Write model checkpoints, vocabulary JSON, training summary, generated
      CSV, and audit sidecar.
- [x] Add tests that run without torch installed by checking manifest and error
      paths, plus optional torch tests for a tiny one-epoch smoke model.
- [ ] Add Slurm resource presets for 16GB/24GB/48GB GPU jobs.
- [x] Add a full-79 production contract that selects PBM-supported generation
      context k3, keeps all eight additive age-bin models, uses the
      encoder-decoder architecture and child-side output vocabulary, and emits
      scorer-ready provenance.
- [x] Add a Slurm DAG with CPU preparation audit, exact-wrapper GPU smoke, two
      production waves, wave readiness markers, validated resume behavior, and
      a final `COMPLETE_AND_AUDITED` marker.
- [ ] Run the full-79 k3 smoke and production DAG on Mila. Do not mark complete
      until the compact reports and final marker have been retrieved locally.

## Scientific Comparisons

- [ ] Keep PBM proof-of-concept generated outputs separate from full
      strict-naturalistic generated outputs.
- [ ] Keep random, unigram, bigram, trigram, and LSTM outputs as distinct
      `source_model` values.
- [ ] Do not mix Mistral-generated samples into this repo unless they are
      clearly labeled as response-space generation, not independent baselines.
- [ ] Add manifest-level metadata for training scope:
      `pbm_only`, `strict_naturalistic`, `leave_corpus_out`, or other explicit
      scope labels.
- [ ] Add PBM cleaned-data integration manifests using existing
      `compute_surprisal_mila/data/{Brown,Manchester,Providence}/*/chi.csv`
      as the first real-data test layer.
- [x] Add a full-79 production Slurm entrypoint that builds its manifest from
      the extracted strict-naturalistic bundle on scratch:
      `slurm/full_79_ngram_baselines.sbatch`.
- [ ] After synthetic and PBM integration tests pass, run full
      strict-naturalistic generation only from data transferred to Mila with
      `rsync`.

## Verification Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/generate_baselines_mila_pycache PYTHONPATH=src python3 -m unittest discover -s tests
bash -n slurm/*.sbatch slurm/*.sh
PYTHONPATH=src python3 -m generate_baselines_mila validate-manifest --manifest configs/ngram_additive_example.json
```
