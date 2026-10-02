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

Use `/home/alkan/Portelance/pc-runs` for jobs and checkpoints. Code stays in Git;
processed data, outputs and weights stay outside Git. Never overwrite an old run.

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

Example code-validation job (agent executes; Nicolas need not copy commands):

```bash
python3 scripts/pc_job.py submit --profile fixture-tests \
  --python /usr/bin/python3 --jobs-root /home/alkan/Portelance/pc-runs \
  --job-id fixture-UNIQUE-ID --max-seconds 180 --check-only
python3 scripts/pc_job.py submit --profile fixture-tests \
  --python /usr/bin/python3 --jobs-root /home/alkan/Portelance/pc-runs \
  --job-id fixture-UNIQUE-ID --max-seconds 180
python3 scripts/pc_job.py status \
  --job-dir /home/alkan/Portelance/pc-runs/fixture-UNIQUE-ID
```

For the neural smoke, set `--profile pbm-smoke`, point `--python` at the dedicated
environment and supply `--handoff` with the verified `full_20260825` directory.
Choose an explicit maximum duration based on the technical-test scope. Read
`docs/CONTINUE_ON_GPU_PC.md` before launching. A passed smoke establishes runtime
and artifact integrity, not learned child-like language or publication evidence.

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
