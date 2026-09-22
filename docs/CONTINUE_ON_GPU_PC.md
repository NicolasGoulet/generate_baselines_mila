# Continue the child-trained transformer task on the GPU PC

Handoff prepared on Nicolas's laptop, 22 September 2026. This document is
self-contained: the next task does not need the laptop conversation or its
untracked Research Brain directory.

## Objective and current status

Run small GPU experiments on the personal PC to check the implementation,
memory use and model behavior. Use Mila for the eventual larger experiment.
The current transfer request means **push code and this handoff to GitHub**;
it does not authorize this laptop to submit cluster jobs or copy files to the
PC. No real transformer training, local GPU smoke, or Mila run has been verified
yet. The previously saved `pbm_transformers_test-run.json` is a fake-scheduler
unit-test product, not a real run.

The pre-existing transformer implementation is commit
`a455000568f70506d4501d62f32c7c3a24e6fd53`, verified on GitHub `main` and
`codex/pbm-transformer-generators` on 22 September. The new continuation branch
is `codex/pbm-transformer-local-smoke`; it adds explicit testing entry points
and this handoff. Fetch that branch in a separate clean checkout if the PC
already has unrelated changes. Never reset, clean, stash, or switch over them.

```bash
git clone --branch codex/pbm-transformer-local-smoke \
  https://github.com/NicolasGoulet/generate_baselines_mila.git
cd generate_baselines_mila
```

## Scientific design to preserve

- Two randomly initialized models: a BabyLlama-sized decoder-only LLaMA
  (58,343,936 parameters) and a T5 encoder-decoder (58,540,544 parameters).
  This is not fine-tuning an adult model or reproducing BabyLlama's distillation.
- Train on naturalistic caregiver-context/child-response pairs. The model
  generates an alternative **child reply**, with the actual preceding caregiver
  speech held fixed. Parent contexts include statements and requests as well
  as questions.
- The initial evaluation is 21 children from Brown, Manchester and Providence
  (PBM), excluded from tokenizer/model training, validation and epoch selection.
  Training uses the remaining original corpora plus six expansion corpora.
- Eight cumulative age stages per architecture: 16 final models. Select an
  epoch with whole-child-disjoint validation, then refit from fresh initialization
  on development data before generating held-out PBM replies.
- One free-length generated reply per target; the child's word count is not
  supplied as the desired response length. The full run uses a 128-token safety
  cap and rejects excessive censoring. Qwen's existing 100-reply clouds remain
  a separate comparison.
- Mistral k0/k3 scoring belongs to `compute_surprisal_mila` after a completed,
  audited generation handoff. It is not included in these smoke commands.

Why this matters: Qwen was prompted to speak as a child, but its broad prior
training may favor conventional wording. These child-conversation-trained
models help test how much the human–Qwen gap depends on training experience and
architecture. Neither model is a literal simulation of a child's experience.

We discussed later evaluating all 79 children using five held-out corpus groups,
with PBM as the first group. That would be up to 80 final models, potentially
reusing the first 16 if the first fold remains identical. This extension is a
proposal, not implemented or included in the current command. Decide its scope
after measuring the first experiment; never train on the children used for
the corresponding unseen-child evaluation.

## Data: prepared, outside Git

Expected directory basename: `transformer_training_expansion/full_20260825`.

The verified laptop location is:

```text
/home/apaixonada/Projects/portelance/T7_RECOVERED/Projet_1/communicative_efficiency/results/transformer_training_expansion/full_20260825
```

The original workstation location recorded in the saved run documents is:

```text
/home/apaixonada/EvaPortelance/Projet_1/communicative_efficiency/results/transformer_training_expansion/full_20260825
```

The second path is a discovery hint, not proof that it exists on the current
PC. Locate existing copies through that machine's maintained catalog/atlas
before requesting a transfer. The laptop has all 36 manifest-listed files
(about 638.8 MiB) and `BUILD_COMPLETE_AND_AUDITED`. The saved manifest reports:

| Split | Examples |
|---|---:|
| Training | 763,494 |
| Validation (whole children held out) | 175,216 |
| Development for refit | 938,710 |
| PBM evaluation | 446,508 |

`manifest.json` binds files by SHA-256. Preparation verifies those hashes and
the split/row contracts. Do not replace the bundle with differently processed
CHILDES data or put it into GitHub. GitHub supplies code, configuration, tests
and documentation; the prepared data must exist separately on the PC/Mila.

## Local GPU smoke

Use a CUDA-capable Linux environment (native or an already configured WSL
environment). Keep the virtual environment, data and all run/checkpoint files
outside the Git checkout. Inspect the actual GPU/VRAM first; do not infer that
a 4060 Ti has 16 GB, since an 8 GB version also exists.

Example environment setup; choose a fresh environment path:

```bash
python3 -m venv "$HOME/venvs/pbm-transformers-v1"
"$HOME/venvs/pbm-transformers-v1/bin/python" -m pip install \
  -r requirements-pbm-transformers.lock.txt
"$HOME/venvs/pbm-transformers-v1/bin/python" -m pip install -e .
export PYTHON_CMD="$HOME/venvs/pbm-transformers-v1/bin/python"
```

Set `HANDOFF_ROOT` to the actual verified data directory. Use an unused run
directory and invoke:

```bash
export HANDOFF_ROOT="/actual/path/to/transformer_training_expansion/full_20260825"
export RUN_ROOT="$HOME/portelance-runs/pbm-transformer-smoke-$(date +%Y%m%d_%H%M%S)"
bash scripts/run_pbm_transformer_smoke_local.sh "$HANDOFF_ROOT" "$RUN_ROOT"
```

The wrapper checks CUDA, prepares/audits inputs, trains the shared tokenizer,
runs both architecture smoke cells, audits them, and stops. There is no Slurm
submission and no automatic production phase. A fresh run directory is required;
retain failed runs and logs rather than overwriting them.

**What this smoke establishes:** the GPU/runtime, training/generation path,
artifact construction and row audits can execute. It uses real architecture
dimensions but reduced work: 1,024 training examples and 25 targets by default,
one epoch, batch size at most two, no gradient accumulation, at most 16 generated
tokens, and a relaxed censoring threshold. The saved smoke manifests are the
authority. This smoke cannot establish full-batch VRAM, production stopping
quality, or paper-ready generated language. Its timing is not a direct estimate
for the full eight-age-stage production schedule.

Inspect `reports/smoke/smoke_summary.json` and `SMOKE_PASSED` under the run
directory. The two `smoke/<architecture>/smoke/generated_responses.jsonl.gz`
files contain the generated examples; their neighboring `cell_audit.json`
files give cell-level checks. The exact smoke settings are in
`manifests/smoke_<architecture>.json`. A marker without its matching reports
and artifacts is not sufficient evidence of a successful run.

Record GPU model/VRAM, software versions, source revision, elapsed times, loss
behavior, memory measurements when available, actual generated examples and
censoring. The local execution receipt identifies the local entry point; any
legacy `exact_production_wrapper` field in the shared smoke report refers to
the corresponding Slurm wrapper, not evidence that Slurm ran on the PC.

## After the first smoke

Inspect the output before scheduling more work. If it passes, define a bounded
representative pilot to assess learning and generation at the intended context,
batch and output lengths. Use whole-child validation for development choices;
do not tune repeatedly to the held-out PBM comparison. Keep pilot deviations in
versioned manifests. A short plumbing smoke alone is not enough to decide that
child-like generation is scientifically adequate or that full production will
be quick. The current entry point intentionally performs only the technical
smoke; defining the next pilot is the next task's scientific decision.

Do not change both architecture and training-data scope merely to get a pleasing
example. Preserve the experiment's two architectures and original held-out
comparison unless Nicolas explicitly chooses a revised scientific design.

## Mila testing and eventual production

Transfer the reviewed code and immutable input bundle using the existing Mila
access workflow; check current remote runs and paths first. Validate the pinned
runtime on the assigned GPU and input hashes on Mila. Use a fresh run ID.

For **testing only**, the new wrapper submits preparation, the two-task GPU
smoke array, and the smoke audit, with dependencies between them:

```bash
export PYTHON_CMD="$SCRATCH/venvs/pbm-transformers-v1/bin/python"
bash slurm/submit_pbm_transformer_smoke_only.sh \
  "$SCRATCH/communicative_efficiency_data/transformer_training_expansion/full_20260825"
```

For a later explicitly selected **full 16-model run**, use the existing
`slurm/submit_pbm_transformers.sh`. That launcher queues production waves behind
the passing smoke audit; it is not a smoke-only command. Preserve its audits,
one-GPU-per-task contract and fresh-run rule. A PC smoke does not replace the
required Mila runtime/wrapper smoke. Keep neural checkpoints on the PC/Mila and
retrieve only the necessary compact results and generation handoff.

## Prompt to start the next task

> Continue the child-trained LLaMA/T5 experiment from docs/CONTINUE_ON_GPU_PC.md.
> This is my GPU PC. Verify the repository revision, existing work, GPU and
> prepared data. Set up a separate compatible runtime and run the local
> two-architecture technical smoke, preserving the frozen data split and model
> definitions. Keep checkpoints outside Git. Inspect losses, memory, generation
> and audit results, then give me a concrete bounded pilot recommendation before
> full production. Do not submit Mila jobs or run the all-79 extension in this
> task. If an input or runtime is missing, locate the existing artifact/setup
> before proposing regeneration or new downloads. Report actual execution,
> source revision, run paths and unresolved issues so we can continue on Mila.

## Supporting project files

- `docs/pbm-transformer-pipeline.md`: original scientific/execution contract.
- `configs/pbm_transformers_from_scratch_v1.json`: frozen architecture and defaults.
- `requirements-pbm-transformers.lock.txt`: neural runtime pins.
- `tests/test_pbm_transformer_production.py`: original fixture and fake-Slurm tests.
- `tests/test_pbm_transformer_smoke_entrypoints.py`: new local/smoke-only entry-point tests.
- `docs/TRANSFORMER_HANDOFF_VERIFICATION_20260922.md`: verification from the laptop.
