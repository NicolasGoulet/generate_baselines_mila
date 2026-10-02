# PC execution workflow

Task started 2026-10-02: Nicolas wants agents to handle routine execution on the
Linux RTX 4060 Ti PC, while keeping experimental-design decisions explicit.
This extends the existing child-trained transformer handoff; it does not change
the PBM holdout, training examples, architectures, metrics or production scope.

## Division of work

The agent checks peer coordination notes, reads the experiment contract, prepares
the selected code/data/runtime, submits a bounded job, inspects logs and audits,
and reports actual results. It may repair operational defects and repeat a failed
technical check in a fresh directory. It must not reinterpret an operational
failure as permission to change the scientific experiment.

Nicolas and the scientific lead decide changes to the training/evaluation
population, held-out split, model comparison, estimands, production budget and
the proposed all-79 extension. Existing completed analyses are reused.

## Current transport and ownership

Use `ssh pc`, the configured private route to `alkan-MS-7C02`. Read the latest
`~/Documents/Codex/device-bridge/from-desktop.md` on the laptop; refresh/send
through `~/laptop_setup/projects partner`. Do not change networking or edit the
PC-owned peer note. The note-exchange timer is not a research job scheduler.

Keep the PC's existing `/home/alkan/Portelance/*` checkouts and environments
intact. The isolated execution checkout is
`/home/alkan/Portelance/.worktrees/pbm-pc-execution-20261002`.
The verified input bundle is
`/home/alkan/Portelance/pc-data/full_20260825_20261002`; all 36 files listed in
its manifest passed hash verification. The dedicated runtime is
`/home/alkan/Portelance/.venvs/pbm-transformers-20261002`, with
`torch==2.6.0`, `transformers==4.48.3`, `tokenizers==0.21.0` and
`safetensors==0.5.2`. PyTorch reports `2.6.0+cu124`, the CUDA wheel build for
the pinned public version, and the CUDA probe passed.

The prepared Samsung T7 is mounted read-write at `/media/alkan/T7` with UUID
`E2FB-205C`. Its guarded root is
`/media/alkan/T7/PORTELANCE_WORKSPACE`: laptop-authoritative inputs are under
`current/laptop`, and new PC job outputs belong only under `pc-runs`. Code and
the Python environment remain on the PC internal disk. Never overwrite an old
run. The old Linux partition `/dev/nvme0n1p6` is outside this workflow and must
not be accessed.

## Bounded launcher

`scripts/pc_job.py` provides `submit`, `status` and `stop`. Run it on the PC,
including through SSH. The submitted process belongs to the PC's user systemd
manager and survives the submitting SSH session ending. It is not restarted
automatically after a failure or reboot. After reboot an unfinished job is
reported as interrupted, not successful. This is a single-job launcher, not an
automatic experiment selector or a continuously awake Codex agent.

On 2026-10-02, an actual fixture job completed 39 tests with one expected skip.
A separate job capped at one second reached `timed_out` and preserved that
terminal result after its transient service was collected. The PC had 22 GB free
after the isolated runtime installation. No neural training or generation ran.

Implemented profiles:

- `fixture-tests`: the repository's model-free unit-test suite with system Python.
- `pbm-smoke`: the existing local two-architecture technical smoke, including
  preparation, generation and existing cell/aggregate audits. Requires the exact
  pinned runtime and prepared input manifest. It never queues the 16-model run.

The runner records the Git revision, command, environment, input-manifest hash,
times, exit status, log hash and smoke audit hashes. It refuses dirty/changed
source, reused job directories and another managed Portelance job. Smoke also
refuses other GPU compute processes and fewer than 15 GiB free before launch.
During execution it stops below 2 GiB free or at the explicit time limit. All jobs
use up to four CPU cores and a 12 GiB host-memory limit; the latter is not a GPU
VRAM limit. There are no retries, implicit dependency upgrades or model downloads.
External processes that do not use this launcher are not covered by its lock.

For T7-backed work, pass `--storage-workspace` and `--storage-uuid` together.
The runner requires the exact mounted filesystem UUID and mount root, the
laptop-authority workspace marker, a fully verified ready publication, and
matching latest receipt, immutable receipt, and manifest. It freezes the
publication `content_id` and `run_id` into the job contract, then rechecks them
before execution and while polling the child. The prepared transformer handoff
must be below `current`; the jobs root must be exactly `pc-runs`. Symlink escapes
are refused.

Each worker holds a nonblocking shared lock on `writer.lock` for its full
lifetime. A publisher's exclusive lock therefore blocks new jobs, and
`freeze.json` or readiness false also blocks execution. An unplug, remount,
UUID change, or publication change terminates the child. If the external job
directory disappears before terminal status can be saved, the failure record
and short diagnostic are retained under
`~/.local/state/portelance/job-failures`. The global one-job lock still applies.
An absent drive never causes creation of an internal fallback jobs directory,
and `--check-only` creates no job output.

Example code-validation job (agent executes; Nicolas need not copy commands):

```bash
python3 scripts/pc_job.py submit --profile fixture-tests \
  --python /usr/bin/python3 \
  --storage-workspace /media/alkan/T7/PORTELANCE_WORKSPACE \
  --storage-uuid E2FB-205C \
  --jobs-root /media/alkan/T7/PORTELANCE_WORKSPACE/pc-runs \
  --job-id fixture-UNIQUE-ID --max-seconds 180 --check-only
python3 scripts/pc_job.py submit --profile fixture-tests \
  --python /usr/bin/python3 \
  --storage-workspace /media/alkan/T7/PORTELANCE_WORKSPACE \
  --storage-uuid E2FB-205C \
  --jobs-root /media/alkan/T7/PORTELANCE_WORKSPACE/pc-runs \
  --job-id fixture-UNIQUE-ID --max-seconds 180
python3 scripts/pc_job.py status \
  --job-dir /media/alkan/T7/PORTELANCE_WORKSPACE/pc-runs/fixture-UNIQUE-ID
```

For the neural smoke, set `--profile pbm-smoke`, point `--python` at the dedicated
environment and supply `--handoff` as
`/media/alkan/T7/PORTELANCE_WORKSPACE/current/laptop/portelance/INPUTS/transformer_training_expansion/full_20260825`.
Choose an explicit maximum duration based on the technical-test scope. Read
`docs/CONTINUE_ON_GPU_PC.md` before launching. A passed smoke establishes runtime
and artifact integrity, not learned child-like language or publication evidence.
The T7 attachment and full 2,873-file checksum verification passed on 2026-10-02.
The actual T7-backed job `t7-fixtures-20261002-01` then passed 51 checks with one
expected Torch-dependent skip in 3.2 seconds. Its contract, status, execution
record and log remain under `PORTELANCE_WORKSPACE/pc-runs/`; the laptop holds
hash-verified copies in `T7_SYNC_STATE/20261002/pc-attachment/fixture-job/`.
No neural smoke, training, generation, or scoring job has run in this setup.

## Scientific sequence

1. Validate the PC job lifecycle with fixtures, including failure, timeout,
   cancellation, duplicate-run and missing-audit behavior.
2. Locate or transfer only the exact prepared input bundle and verify all its
   manifest hashes; prepare the separate pinned environment and check CUDA.
3. Run the existing bounded two-model smoke, inspect outputs and record timings,
   memory, losses and censoring. Do not extrapolate full-run cost from it alone.
4. Define a representative learning pilot with whole-child validation before
   choosing a production budget. Keep PBM evaluation out of tuning.
5. After the design/budget decision, train/generate, audit, hand off to
   `compute_surprisal_mila`, then analyze in `communicative_efficiency`.

Routine preparation and execution are agent work. Remaining design decisions
should be presented with concrete evidence and a recommendation, not commands
for Nicolas to operate manually. Stage readiness must be recorded separately:
transport verified, code tested, inputs verified, runtime verified, neural smoke
completed, pilot accepted, production completed, scoring completed.
