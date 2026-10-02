#!/usr/bin/env python3
"""Bounded, disconnect-safe PC jobs. No scheduler, retries or scientific choices."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TERMINAL = {"passed", "failed", "timed_out", "disk_low", "cancelled", "busy", "launch_failed"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def git_state(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Job source checkout must be clean; preserve edits in a separate checkout")
    return git("rev-parse", "HEAD")


def command(contract, job):
    repo = Path(contract["repo"])
    if contract["profile"] == "fixture-tests":
        return [contract["python"], "-m", "unittest", "discover", "-s", "tests", "-v"]
    if contract["profile"] == "pbm-smoke":
        return ["bash", str(repo / "scripts/run_pbm_transformer_smoke_local.sh"),
                contract["handoff"], str(job / "artifacts")]
    raise ValueError("Unsupported job profile")


def validate(contract, job):
    repo = Path(contract["repo"])
    if git_state(repo) != contract["commit"]:
        raise ValueError("Source revision changed after submission")
    if shutil.disk_usage(job if job.exists() else job.parent).free < contract["min_free_gb"] * 1024**3:
        raise ValueError("Insufficient free disk for this job")
    if contract["profile"] == "fixture-tests":
        probe = subprocess.check_output([contract["python"], "-c",
            "import importlib.util; print(importlib.util.find_spec('torch') is None)"], text=True, timeout=15)
        if probe.strip() != "True":
            raise ValueError("Use a Python without torch for the model-free fixture profile")
    if contract["profile"] == "pbm-smoke":
        handoff = Path(contract["handoff"])
        if digest(handoff / "manifest.json") != contract["handoff_manifest_sha256"]:
            raise ValueError("Input manifest changed after submission")
        if not (handoff / "BUILD_COMPLETE_AND_AUDITED").is_file():
            raise ValueError("Input bundle has no audited build marker")
        if (job / "artifacts").exists():
            raise ValueError("Never restart a smoke into existing artifacts")
        # Dependencies are checked without changing any existing environment.
        pins = dict(line.strip().split("==") for line in
                    (repo / "requirements-pbm-transformers.lock.txt").read_text().splitlines()
                    if "==" in line and not line.startswith("#"))
        code = "import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in " + repr(list(pins)) + "}))"
        versions = json.loads(subprocess.check_output([contract["python"], "-c", code], text=True, timeout=30))
        if versions != pins:
            raise ValueError(f"Dedicated pinned runtime required: expected {pins}, found {versions}")
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=process_name",
                                        "--format=csv,noheader"], text=True, timeout=15)
        if any(line.strip() and "gnome-remote-desktop" not in line for line in apps.splitlines()):
            raise ValueError("Another GPU compute process is present; do not duplicate or interrupt it")


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_command(argv, cwd, env, log, max_seconds, reserve_bytes, disk_path):
    cancelled = False
    def cancel(*_):
        nonlocal cancelled
        cancelled = True
    old = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    proc = None
    started = time.monotonic()
    try:
        with Path(log).open("xb") as out:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            state = "failed"
            while proc.poll() is None:
                if cancelled:
                    state = "cancelled"
                    break
                if time.monotonic() - started >= max_seconds:
                    state = "timed_out"
                    break
                if shutil.disk_usage(disk_path).free < reserve_bytes:
                    state = "disk_low"
                    break
                time.sleep(0.2)
            else:
                state = "passed" if proc.returncode == 0 else "failed"
            stop_process(proc)
            return {"state": state, "exit_code": proc.returncode,
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
    finally:
        if proc is not None:
            stop_process(proc)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def worker(job):
    job = job.resolve()
    # Exclusive creation protects the original status even from a duplicate worker.
    try:
        with (job / "STARTED").open("x") as marker:
            marker.write(now() + "\n")
    except FileExistsError:
        print("Job already started; retain its original status", file=sys.stderr)
        return 2
    c = json.loads((job / "contract.json").read_text())
    result = {"state": "failed", "started_at": now(), "host": socket.gethostname(),
              "contract_sha256": digest(job / "contract.json")}
    lock_path = Path.home() / ".local/state/portelance/execution.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result["state"] = "busy"
                raise ValueError("Another managed Portelance job is running")
            validate(c, job)
            env = {"HOME": str(Path.home()), "PATH": "/usr/local/bin:/usr/bin:/bin",
                   "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONPATH": str(Path(c["repo"]) / "src"), "PYTHON_CMD": c["python"],
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                   "OMP_NUM_THREADS": "4", "COMMIT_SHA": c["commit"],
                   "CUDA_VISIBLE_DEVICES": "0"}
            argv = command(c, job)
            write_json(job / "execution.json", {"argv": argv, "env": env, "started_at": now()})
            write_json(job / "status.json", {**result, "state": "running"})
            result.update(run_command(argv, c["repo"], env, job / "output.log",
                                      c["max_seconds"], 2 * 1024**3, job))
            result["log_sha256"] = digest(job / "output.log")
            if result["state"] == "passed":
                if git_state(c["repo"]) != c["commit"] or digest(job / "contract.json") != result["contract_sha256"]:
                    raise ValueError("Code or job contract changed during execution")
            if result["state"] == "passed" and c["profile"] == "pbm-smoke":
                # The wrapper runs the existing scientific audits; preserve their identities.
                for rel in ["SMOKE_PASSED", "reports/smoke/smoke_summary.json"]:
                    p = job / "artifacts" / rel
                    result.setdefault("audit_sha256", {})[rel] = digest(p)
                result["meaning"] = "technical_smoke_only_not_scientific_confirmation"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["state"] == "passed":
            result["state"] = "failed"
    result["finished_at"] = now()
    write_json(job / "status.json", result)
    return 0 if result["state"] == "passed" else 1


def submit(args):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.job_id):
        raise ValueError("Use a short lowercase alphanumeric/hyphen job ID")
    if not 1 <= args.max_seconds <= 86400:
        raise ValueError("Explicit runtime must be between 1 and 86400 seconds")
    py = Path(args.python).absolute()
    # Do not resolve the venv interpreter symlink to the system interpreter.
    if not py.is_file() or not os.access(py, os.X_OK):
        raise ValueError("An executable absolute Python path is required")
    root = Path(args.jobs_root).resolve()
    if root == REPO or REPO in root.parents:
        raise ValueError("Jobs and checkpoints must be outside the code checkout")
    root.mkdir(parents=True, exist_ok=True)
    c = {"schema_version": 1, "profile": args.profile, "repo": str(REPO),
         "commit": git_state(REPO), "python": str(py), "max_seconds": args.max_seconds,
         "min_free_gb": 15 if args.profile == "pbm-smoke" else 3,
         "unit": "portelance-" + args.job_id, "submitted_at": now()}
    if args.profile == "pbm-smoke":
        if not args.handoff:
            raise ValueError("pbm-smoke requires the prepared --handoff directory")
        c["handoff"] = str(Path(args.handoff).resolve())
        c["handoff_manifest_sha256"] = digest(Path(c["handoff"]) / "manifest.json")
    job = root / args.job_id
    validate(c, job)
    job.mkdir()  # Refuse reuse, including failed runs.
    write_json(job / "contract.json", c)
    write_json(job / "status.json", {"state": "submitted", "submitted_at": now()})
    argv = ["systemd-run", "--user", "--quiet", "--collect", "--unit=" + c["unit"],
            "--property=Type=exec", "--property=RuntimeMaxSec=" + str(args.max_seconds + 120),
            "--property=TimeoutStopSec=30", "--property=KillMode=control-group",
            "--property=CPUQuota=400%", "--property=MemoryMax=12G", "--property=Nice=10",
            "/usr/bin/python3", str(REPO / "scripts/pc_job.py"), "_worker", "--job-dir", str(job)]
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        write_json(job / "status.json", {"state": "launch_failed", "error": exc.stderr, "finished_at": now()})
        raise
    print(json.dumps({"job_dir": str(job), "unit": c["unit"], "state": "submitted"}))


def status(job):
    c = json.loads((job / "contract.json").read_text())
    s = json.loads((job / "status.json").read_text())
    unit = subprocess.run(["systemctl", "--user", "show", c["unit"],
                           "--property=ActiveState,SubState,Result,ExecMainStatus"], capture_output=True, text=True)
    props = dict(line.split("=", 1) for line in unit.stdout.splitlines() if "=" in line)
    if s["state"] not in TERMINAL and props.get("ActiveState") not in {"active", "activating", "deactivating"}:
        s = {**s, "state": "interrupted_or_not_running", "saved_state": s["state"]}
    print(json.dumps({"job_dir": str(job), "status": s, "service": props}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("submit")
    p.add_argument("--profile", choices=["fixture-tests", "pbm-smoke"], required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--jobs-root", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--max-seconds", type=int, required=True)
    p.add_argument("--handoff")
    for name in ["status", "stop", "_worker"]:
        sub.add_parser(name).add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "submit":
        submit(args)
    elif args.action == "status":
        status(args.job_dir)
    elif args.action == "_worker":
        return worker(args.job_dir)
    else:
        c = json.loads((args.job_dir / "contract.json").read_text())
        subprocess.run(["systemctl", "--user", "stop", c["unit"]], check=True)
        status(args.job_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"pc_job: {exc}", file=sys.stderr)
        sys.exit(2)
