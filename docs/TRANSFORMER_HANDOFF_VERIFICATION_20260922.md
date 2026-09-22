# Transformer continuation handoff verification

Prepared on the laptop, 22 September 2026. Scope: publish the task-specific
local-PC and Mila-testing entry points and a self-contained continuation guide
to GitHub. No remote PC connection, file transfer to PC/Mila, runtime installation,
real neural training, generation or cluster submission was performed here.

## Existing code and inputs

- Base source revision: `a455000568f70506d4501d62f32c7c3a24e6fd53`.
- Live GitHub checks found that exact revision on both `main` and
  `codex/pbm-transformer-generators` before these changes.
- Original unit suite: 23 tests, 22 passed and one torch-dependent LSTM smoke
  skipped. The system Python used for fixture tests has no torch installed.
- The original 14 Slurm shell files passed syntax checks; the source checkout
  was clean and `git diff --check` passed.
- The retained training-expansion directory contains all 36 manifest-listed
  files (638.8 MiB) and `BUILD_COMPLETE_AND_AUDITED`. The saved manifest reports
  complete status, no fatal issues and no PBM training overlap. This was a
  current presence check, not a repeated full-data content audit. The pipeline
  verifies hashes and contents before use on the destination machine.

## Publication scope

Only the new entry points, their tests and the continuation documentation belong
to this publication. The laptop's unrelated uncommitted analysis changes are
preserved. Prepared CHILDES files, training checkpoints, private machine access
configuration and the broad Research Brain directory are not Git payloads.

The continuation guide includes the relevant scientific decisions and exact
data locations so the PC task can proceed without the laptop conversation.

## Verification of the new entry points

- Full suite after changes: **30 tests run, 29 passed, one torch-dependent
  LSTM smoke skipped**, using system Python without torch.
- Seven new tests cover both local smoke architectures with real synthetic
  input preparation and real output audits; missing CUDA before preparation;
  preservation of an existing run; failure without a success marker; exactly
  three Slurm stages with dependencies and no production; rejection of a
  multi-GPU request; and stopping after a failed submission while reporting the
  already-submitted preparation job. Neural execution and `sbatch` are faked.
- The local test uses an interpreter and input/output paths containing spaces.
  The real preparation check also verifies that launcher imports do not create
  bytecode in the run root before its freshness check.
- All **16** shell entry points passed `bash -n`; `git diff --check` passed.
- The new Slurm receipt identifies `production_submitted: false` and records
  the reduced smoke settings. The local receipt identifies local execution.
- No actual GPU memory, performance, generation quality, checkpoint recovery on
  hardware, or Mila runtime was tested on the laptop. Those remain PC/Mila work.

Validation command: `PYTHONPATH=src python3 -m unittest discover -s tests`
with offline model-library flags and a temporary bytecode directory. Temporary
log: `/tmp/pbm-entrypoint-tests-20260922.log`.
