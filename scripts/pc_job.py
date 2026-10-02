#!/usr/bin/env python3
"""Bounded, disconnect-safe PC jobs. No scheduler, retries or scientific choices."""
from __future__ import annotations

import argparse
import contextlib
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
WORKSPACE_FORMAT = "portelance-workspace-v1"


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


def read_json(path, description):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid or unavailable {description}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {description}: {path}")
    return value


def mount_info(path):
    """Return findmnt's single filesystem record for path (patchable in fixtures)."""
    raw = subprocess.check_output(
        ["findmnt", "--json", "--target", str(path), "--output", "TARGET,UUID,OPTIONS"],
        text=True, timeout=15)
    records = json.loads(raw).get("filesystems", [])
    if len(records) != 1:
        raise ValueError("Storage workspace is not on exactly one mounted filesystem")
    return records[0]


def safe_descendant(path, root, description, require_exists=True):
    """Reject lexical and symlink escapes from a trusted root."""
    path, root = Path(path), Path(root)
    if not path.is_absolute() or not root.is_absolute():
        raise ValueError(f"{description} must use absolute paths")
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"{description} must stay under {root}") from None
    root_real = root.resolve(strict=True)
    resolved = path.resolve(strict=require_exists)
    try:
        resolved.relative_to(root_real)
    except ValueError:
        raise ValueError(f"{description} escapes {root} through a symlink") from None
    cursor = root
    if cursor.resolve(strict=True) != cursor:
        raise ValueError(f"{description} root must not be a symlink")
    for part in path.relative_to(root).parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            if cursor.is_symlink():
                raise ValueError(f"{description} must not contain symlink ancestors")
    return resolved


def storage_snapshot(workspace, expected_uuid, expected=None):
    """Validate the physical workspace and return its immutable publication IDs."""
    workspace = Path(workspace)
    if not workspace.is_absolute() or not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("Storage workspace must be an existing absolute directory, not a symlink")
    info = mount_info(workspace)
    mount_root = Path(info.get("target", ""))
    if not mount_root.is_absolute() or not mount_root.is_dir():
        raise ValueError("Storage mount root is unavailable")
    if workspace.name != "PORTELANCE_WORKSPACE" or not workspace.parent.samefile(mount_root):
        raise ValueError("Storage workspace must be PORTELANCE_WORKSPACE directly under its mount root")
    if workspace.resolve(strict=True) != workspace or mount_root.resolve(strict=True) != mount_root:
        raise ValueError("Storage workspace and mount root must not use symlink aliases")
    if info.get("uuid") != expected_uuid:
        raise ValueError(f"Storage UUID mismatch: expected {expected_uuid}, found {info.get('uuid')}")
    options = set((info.get("options") or "").split(","))
    if "rw" not in options:
        raise ValueError("Storage filesystem is not mounted read-write")
    marker = read_json(workspace / "workspace.json", "workspace marker")
    if (marker.get("format") != WORKSPACE_FORMAT or marker.get("drive_uuid") != expected_uuid or
            marker.get("authority") != "laptop"):
        raise ValueError("Workspace marker or physical identity mismatch")
    if (workspace / "freeze.json").exists() or (workspace / "freeze.json").is_symlink():
        raise ValueError("Workspace publication is frozen")
    readiness = read_json(workspace / "readiness.json", "workspace readiness")
    latest = read_json(workspace / "latest.json", "workspace completion receipt")
    if readiness.get("ready") is not True:
        raise ValueError("Workspace readiness is false")
    content_id, run_id = readiness.get("content_id"), readiness.get("run_id")
    if not re.fullmatch(r"[0-9a-f]{64}", content_id or "") or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id or ""):
        raise ValueError("Workspace readiness has invalid publication identifiers")
    if (latest.get("format") != WORKSPACE_FORMAT or latest.get("status") != "complete" or
            latest.get("content_id") != content_id or latest.get("run_id") != run_id):
        raise ValueError("Latest completion receipt differs from workspace readiness")
    receipt = read_json(workspace / "receipts" / f"{run_id}.complete.json", "immutable completion receipt")
    manifest = read_json(workspace / "manifests" / f"{run_id}.json", "immutable publication manifest")
    if (receipt.get("format") != WORKSPACE_FORMAT or receipt.get("status") != "complete" or
            receipt.get("content_id") != content_id or receipt.get("run_id") != run_id or
            manifest.get("format") != WORKSPACE_FORMAT or manifest.get("content_id") != content_id or
            manifest.get("drive_uuid") != expected_uuid):
        raise ValueError("Immutable manifest or receipt differs from workspace readiness")
    snapshot = {"content_id": content_id, "run_id": run_id, "mount_root": str(mount_root)}
    if expected and any(snapshot[key] != expected[key] for key in snapshot):
        raise ValueError("Storage publication identity changed after submission")
    return snapshot


def storage_runtime_check(contract, job):
    """Cheap polling guard; full receipt/manifest validation surrounds execution."""
    workspace = Path(contract["storage_workspace"])
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("Storage workspace disappeared during execution")
    info = mount_info(workspace)
    mount_root = Path(info.get("target", ""))
    if (info.get("uuid") != contract["storage_uuid"] or
            str(mount_root) != contract["storage_mount_root"] or
            "rw" not in set((info.get("options") or "").split(",")) or
            not mount_root.is_dir() or not workspace.parent.samefile(mount_root)):
        raise ValueError("Storage mount identity changed during execution")
    if (workspace / "freeze.json").exists() or (workspace / "freeze.json").is_symlink():
        raise ValueError("Workspace publication became frozen during execution")
    readiness = read_json(workspace / "readiness.json", "workspace readiness")
    if (readiness.get("ready") is not True or
            readiness.get("content_id") != contract["storage_content_id"] or
            readiness.get("run_id") != contract["storage_run_id"]):
        raise ValueError("Storage publication identity changed during execution")
    if Path(job) != workspace / "pc-runs" / contract["job_id"] or not Path(job).is_dir():
        raise ValueError("Storage job directory disappeared or changed during execution")


@contextlib.contextmanager
def storage_reader_lock(workspace):
    lock_path = Path(workspace) / "writer.lock"
    if lock_path.is_symlink():
        raise ValueError("Workspace writer lock must not be a symlink")
    try:
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Workspace publisher holds the writer lock") from None
            yield
    except FileNotFoundError:
        raise ValueError("Storage workspace disappeared before its reader lock was acquired") from None


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        if path == path.parent:
            raise ValueError("No existing filesystem ancestor for job output")
        path = path.parent
    return path


def save_result(job, result, fallback_status, fallback_log, external_error=None):
    if external_error is None:
        try:
            write_json(Path(job) / "status.json", result)
            return
        except OSError as exc:
            external_error = f"{type(exc).__name__}: {exc}"
    if external_error is not None:
        failure = {**result, "external_job_dir": str(job),
                   "external_status_error": external_error,
                   "internal_log": str(fallback_log)}
        write_json(fallback_status, failure)
        Path(fallback_log).write_text(
            f"{failure['finished_at']} {failure.get('error', failure['external_status_error'])}\n")


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


def pinned_versions_match(pins, versions):
    if pins.keys() != versions.keys():
        return False
    for name, expected in pins.items():
        actual = versions[name]
        if actual == expected:
            continue
        # CUDA wheels expose their build as a PEP 440 local suffix even though
        # the lock intentionally pins the public PyTorch version.
        if name == "torch" and "+" not in expected and actual.startswith(expected + "+"):
            local = actual[len(expected) + 1:]
            if local and re.fullmatch(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*", local):
                continue
        return False
    return True


def validate(contract, job):
    repo = Path(contract["repo"])
    if git_state(repo) != contract["commit"]:
        raise ValueError("Source revision changed after submission")
    if "storage_workspace" in contract:
        validate_storage_contract(contract, job)
    if shutil.disk_usage(nearest_existing(job)).free < contract["min_free_gb"] * 1024**3:
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
        if not pinned_versions_match(pins, versions):
            raise ValueError(f"Dedicated pinned runtime required: expected {pins}, found {versions}")
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=process_name",
                                        "--format=csv,noheader"], text=True, timeout=15)
        if any(line.strip() and "gnome-remote-desktop" not in line for line in apps.splitlines()):
            raise ValueError("Another GPU compute process is present; do not duplicate or interrupt it")


def validate_storage_contract(contract, job):
    workspace = Path(contract["storage_workspace"])
    expected = {key: contract["storage_" + key] for key in ("content_id", "run_id", "mount_root")}
    storage_snapshot(workspace, contract["storage_uuid"], expected)
    current = workspace / "current"
    if not current.is_dir():
        raise ValueError("Workspace current input directory is unavailable")
    if contract.get("handoff"):
        safe_descendant(Path(contract["handoff"]), current, "Prepared handoff")
    jobs_root = workspace / "pc-runs"
    if Path(contract["jobs_root"]) != jobs_root:
        raise ValueError("Storage job outputs must use workspace/pc-runs")
    if jobs_root.exists():
        safe_descendant(jobs_root, workspace, "Storage jobs root")
    expected_job = jobs_root / contract["job_id"]
    if Path(job) != expected_job:
        raise ValueError("Job directory differs from its frozen storage contract")
    if job.exists():
        safe_descendant(job, jobs_root, "Storage job directory")


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_command(argv, cwd, env, log, max_seconds, reserve_bytes, disk_path, poll_check=None):
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
                if poll_check:
                    poll_check()
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
    job = job.absolute()
    fallback = Path.home() / ".local/state/portelance/job-failures"
    fallback.mkdir(parents=True, exist_ok=True)
    fallback_name = job.name if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", job.name) else "invalid-job"
    fallback_status = fallback / (fallback_name + ".json")
    fallback_log = fallback / (fallback_name + ".log")
    result = {"state": "failed", "started_at": now(), "host": socket.gethostname()}
    # Preserve an existing job before any contract or status update.
    if (job / "STARTED").exists():
        print("Job already started; retain its original status", file=sys.stderr)
        return 2
    c = None
    lock_path = Path.home() / ".local/state/portelance/execution.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        c = read_json(job / "contract.json", "job contract")
        result["contract_sha256"] = digest(job / "contract.json")
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result["state"] = "busy"
                raise ValueError("Another managed Portelance job is running")
            storage_lock = (storage_reader_lock(c["storage_workspace"])
                            if "storage_workspace" in c else contextlib.nullcontext())
            with storage_lock:
                try:
                    validate(c, job)
                    # Exclusive creation protects the original status from a worker race.
                    try:
                        with (job / "STARTED").open("x") as marker:
                            marker.write(now() + "\n")
                    except FileExistsError:
                        print("Job already started; retain its original status", file=sys.stderr)
                        return 2
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
                    if "storage_workspace" in c:
                        last_storage_check = [float("-inf")]
                        def poll_check():
                            current = time.monotonic()
                            if current - last_storage_check[0] >= 1:
                                storage_runtime_check(c, job)
                                last_storage_check[0] = current
                    else:
                        poll_check = None
                    result.update(run_command(argv, c["repo"], env, job / "output.log",
                                              c["max_seconds"], 2 * 1024**3, job, poll_check))
                    result["log_sha256"] = digest(job / "output.log")
                    if result["state"] == "passed":
                        if "storage_workspace" in c:
                            validate_storage_contract(c, job)
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
                external_error = None
                if "storage_workspace" in c:
                    try:
                        storage_runtime_check(c, job)
                    except Exception as exc:
                        external_error = f"{type(exc).__name__}: {exc}"
                save_result(job, result, fallback_status, fallback_log, external_error)
                return 0 if result["state"] == "passed" else 1
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["state"] == "passed":
            result["state"] = "failed"
    result["finished_at"] = now()
    external_error = "Storage reader lock was not acquired"
    if not c or "storage_workspace" not in c:
        external_error = None
    save_result(job, result, fallback_status, fallback_log, external_error)
    return 0 if result["state"] == "passed" else 1


def submit(args):
    storage_workspace = getattr(args, "storage_workspace", None)
    storage_uuid = getattr(args, "storage_uuid", None)
    if bool(storage_workspace) != bool(storage_uuid):
        raise ValueError("--storage-workspace and --storage-uuid must be provided together")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.job_id):
        raise ValueError("Use a short lowercase alphanumeric/hyphen job ID")
    if not 1 <= args.max_seconds <= 86400:
        raise ValueError("Explicit runtime must be between 1 and 86400 seconds")
    py = Path(args.python).absolute()
    # Do not resolve the venv interpreter symlink to the system interpreter.
    if not py.is_file() or not os.access(py, os.X_OK):
        raise ValueError("An executable absolute Python path is required")
    root = Path(args.jobs_root).absolute()
    if root == REPO or REPO in root.parents:
        raise ValueError("Jobs and checkpoints must be outside the code checkout")
    c = {"schema_version": 1, "profile": args.profile, "repo": str(REPO),
         "commit": git_state(REPO), "python": str(py), "max_seconds": args.max_seconds,
         "min_free_gb": 15 if args.profile == "pbm-smoke" else 3,
         "unit": "portelance-" + args.job_id, "submitted_at": now(),
         "jobs_root": str(root), "job_id": args.job_id}
    storage_lock = contextlib.nullcontext()
    if storage_workspace:
        workspace = Path(storage_workspace).absolute()
        if root != workspace / "pc-runs":
            raise ValueError("With guarded storage, --jobs-root must be WORKSPACE/pc-runs")
        c.update(storage_workspace=str(workspace), storage_uuid=storage_uuid)
        storage_lock = storage_reader_lock(workspace)
    if args.profile == "pbm-smoke":
        if not args.handoff:
            raise ValueError("pbm-smoke requires the prepared --handoff directory")
        c["handoff"] = str(Path(args.handoff).absolute())
    job = root / args.job_id
    if job.exists():
        raise ValueError("Fresh job ID required; preserve the existing run")
    with storage_lock:
        if storage_workspace:
            snapshot = storage_snapshot(c["storage_workspace"], storage_uuid)
            c.update({"storage_" + key: value for key, value in snapshot.items()})
            if c.get("handoff"):
                safe_descendant(Path(c["handoff"]), Path(c["storage_workspace"]) / "current",
                                "Prepared handoff")
        if c.get("handoff"):
            c["handoff_manifest_sha256"] = digest(Path(c["handoff"]) / "manifest.json")
        validate(c, job)
        if args.check_only:
            print(json.dumps({"state": "preflight_passed", "contract": c, "job_dir": str(job)}))
            return
        if storage_workspace:
            expected = {key: c["storage_" + key] for key in ("content_id", "run_id", "mount_root")}
            storage_snapshot(c["storage_workspace"], storage_uuid, expected)
        root.mkdir(parents=True, exist_ok=True)
        if storage_workspace:
            safe_descendant(root, Path(c["storage_workspace"]), "Storage jobs root")
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
                           "--property=LoadState,ActiveState,SubState,Result,ExecMainStatus"], capture_output=True, text=True)
    props = dict(line.split("=", 1) for line in unit.stdout.splitlines() if "=" in line)
    if s["state"] not in TERMINAL and props.get("ActiveState") not in {"active", "activating", "deactivating"}:
        s = {**s, "state": "interrupted_or_not_running", "saved_state": s["state"]}
    if props.get("LoadState") == "not-found":
        # systemd's defaults for a collected unit are not its historical exit status.
        props = {"LoadState": "not-found", "note": "Service collected; use the saved job result"}
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
    p.add_argument("--storage-workspace", help="Exact mounted PORTELANCE_WORKSPACE path")
    p.add_argument("--storage-uuid", help="Expected physical filesystem UUID")
    p.add_argument("--check-only", action="store_true", help="Validate without creating a job or launching computation")
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
