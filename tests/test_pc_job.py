"""Exercise job outcomes and refusal gates without CUDA, SSH or systemd."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("pc_job", Path(__file__).parents[1] / "scripts/pc_job.py")
JOB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JOB)


class JobTests(unittest.TestCase):
    CONTENT = "a" * 64
    RUN_ID = "20261002T173737Z-cb7ee843-0e208530"

    def make_workspace(self, base):
        mount = Path(base) / "T7"
        workspace = mount / "PORTELANCE_WORKSPACE"
        (workspace / "current/laptop/input").mkdir(parents=True)
        (workspace / "manifests").mkdir()
        (workspace / "receipts").mkdir()
        (workspace / "writer.lock").touch()
        marker = {"format": JOB.WORKSPACE_FORMAT, "drive_uuid": "E2FB-205C",
                  "source_uuid": "laptop", "authority": "laptop"}
        readiness = {"ready": True, "run_id": self.RUN_ID, "content_id": self.CONTENT}
        latest = {"format": JOB.WORKSPACE_FORMAT, "status": "complete",
                  "run_id": self.RUN_ID, "content_id": self.CONTENT}
        manifest = {"format": JOB.WORKSPACE_FORMAT, "drive_uuid": "E2FB-205C",
                    "run_id": self.RUN_ID, "content_id": self.CONTENT}
        (workspace / "workspace.json").write_text(json.dumps(marker))
        (workspace / "readiness.json").write_text(json.dumps(readiness))
        (workspace / "latest.json").write_text(json.dumps(latest))
        (workspace / "manifests" / f"{self.RUN_ID}.json").write_text(json.dumps(manifest))
        (workspace / "receipts" / f"{self.RUN_ID}.complete.json").write_text(json.dumps(latest))
        return mount, workspace

    def mount_record(self, mount, uuid="E2FB-205C"):
        return {"target": str(mount), "uuid": uuid, "options": "rw,nosuid"}

    def run_small(self, code, timeout=5, reserve=0):
        with tempfile.TemporaryDirectory() as d:
            result = JOB.run_command([sys.executable, "-c", code], d, os.environ.copy(),
                                     Path(d) / "output.log", timeout, reserve, d)
            return result, (Path(d) / "output.log").read_text()

    def test_nonzero_exit_and_log_are_preserved(self):
        r, log = self.run_small("print('diagnostic', flush=True); raise SystemExit(7)")
        self.assertEqual((r["state"], r["exit_code"]), ("failed", 7))
        self.assertIn("diagnostic", log)

    def test_timeout_terminates_process(self):
        r, _ = self.run_small("import time; time.sleep(60)", timeout=0.2)
        self.assertEqual(r["state"], "timed_out")
        self.assertLess(r["elapsed_seconds"], 4)
        self.assertNotEqual(r["exit_code"], 0)

    def test_low_disk_terminates_process(self):
        r, _ = self.run_small("import time; time.sleep(60)", reserve=10**30)
        self.assertEqual(r["state"], "disk_low")

    def test_cancel_terminates_process(self):
        # The child requests cancellation of the supervising worker.
        r, _ = self.run_small("import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(60)")
        self.assertEqual(r["state"], "cancelled")

    def test_changed_source_refused(self):
        with tempfile.TemporaryDirectory() as d, patch.object(JOB, "git_state", return_value="new"):
            with self.assertRaisesRegex(ValueError, "revision changed"):
                JOB.validate({"repo": d, "commit": "old"}, Path(d))

    def test_existing_job_is_never_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "STARTED").write_text("prior")
            (p / "status.json").write_text('{"state":"passed"}')
            before = (p / "status.json").read_bytes()
            with patch.object(JOB.Path, "home", return_value=p):
                self.assertEqual(JOB.worker(p), 2)
            self.assertEqual((p / "status.json").read_bytes(), before)

    def test_missing_drive_before_worker_start_preserves_internal_failure(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "home"
            home.mkdir()
            missing_job = Path(d) / "unmounted/PORTELANCE_WORKSPACE/pc-runs/startup-loss"
            with patch.object(JOB.Path, "home", return_value=home):
                self.assertEqual(JOB.worker(missing_job), 1)
            saved = home / ".local/state/portelance/job-failures/startup-loss.json"
            diagnostic = saved.with_suffix(".log")
            self.assertTrue(saved.is_file())
            self.assertTrue(diagnostic.is_file())
            result = json.loads(saved.read_text())
            self.assertEqual(result["state"], "failed")
            self.assertIn("job contract", result["error"])

    def test_machine_lock_rejects_second_job(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            job = home / "job"
            job.mkdir()
            (job / "contract.json").write_text("{}")
            lock = home / ".local/state/portelance/execution.lock"
            lock.parent.mkdir(parents=True)
            with lock.open("a") as f, patch.object(JOB.Path, "home", return_value=home):
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(JOB.worker(job), 1)
                self.assertEqual(json.loads((job / "status.json").read_text())["state"], "busy")

    def test_successful_exit_cannot_replace_missing_smoke_audits(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            job = home / "job"
            job.mkdir()
            (job / "contract.json").write_text(json.dumps({"profile": "pbm-smoke", "repo": d,
                "python": sys.executable, "commit": "abc", "max_seconds": 5}))
            with patch.object(JOB.Path, "home", return_value=home), patch.object(JOB, "validate"), \
                 patch.object(JOB, "git_state", return_value="abc"), \
                 patch.object(JOB, "command", return_value=[sys.executable, "-c", "print('fake smoke')"]):
                self.assertEqual(JOB.worker(job), 1)
            status = json.loads((job / "status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertIn("FileNotFoundError", status["error"])

    def test_cuda_torch_build_matches_public_version_pin_only(self):
        pins = {"torch": "2.6.0", "transformers": "4.48.3"}
        self.assertTrue(JOB.pinned_versions_match(
            pins, {"torch": "2.6.0+cu124", "transformers": "4.48.3"}))
        self.assertFalse(JOB.pinned_versions_match(
            pins, {"torch": "2.6.1+cu124", "transformers": "4.48.3"}))
        self.assertFalse(JOB.pinned_versions_match(
            pins, {"torch": "2.6.0+cu124", "transformers": "4.48.3+local"}))

    def test_inactive_unit_never_reported_as_running(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "contract.json").write_text('{"unit":"example"}')
            (p / "status.json").write_text('{"state":"running"}')
            out = io.StringIO()
            response = subprocess.CompletedProcess([], 0, "ActiveState=inactive\nResult=success\n", "")
            with patch.object(JOB.subprocess, "run", return_value=response), contextlib.redirect_stdout(out):
                JOB.status(p)
            self.assertEqual(json.loads(out.getvalue())["status"]["state"], "interrupted_or_not_running")

    def test_collected_unit_defaults_do_not_override_saved_failure(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "contract.json").write_text('{"unit":"example"}')
            (p / "status.json").write_text('{"state":"timed_out"}')
            out = io.StringIO()
            response = subprocess.CompletedProcess([], 0,
                "LoadState=not-found\nActiveState=inactive\nResult=success\nExecMainStatus=0\n", "")
            with patch.object(JOB.subprocess, "run", return_value=response), contextlib.redirect_stdout(out):
                JOB.status(p)
            result = json.loads(out.getvalue())
            self.assertEqual(result["status"]["state"], "timed_out")
            self.assertNotIn("Result", result["service"])

    def test_check_only_creates_no_job_or_service(self):
        import argparse
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(profile="fixture-tests", python=sys.executable,
                jobs_root=d, job_id="dry-check", max_seconds=60, handoff=None, check_only=True)
            out = io.StringIO()
            with patch.object(JOB, "git_state", return_value="abc"), patch.object(JOB, "validate"), \
                 patch.object(JOB.subprocess, "run") as run, contextlib.redirect_stdout(out):
                JOB.submit(args)
            self.assertFalse((Path(d) / "dry-check").exists())
            run.assert_not_called()
            self.assertEqual(json.loads(out.getvalue())["state"], "preflight_passed")

    def test_storage_snapshot_requires_exact_uuid_and_published_identity(self):
        with tempfile.TemporaryDirectory() as d:
            mount, workspace = self.make_workspace(d)
            with patch.object(JOB, "mount_info", return_value=self.mount_record(mount, "WRONG")), \
                 self.assertRaisesRegex(ValueError, "UUID mismatch"):
                JOB.storage_snapshot(workspace, "E2FB-205C")
            with patch.object(JOB, "mount_info", return_value=self.mount_record(mount)):
                snap = JOB.storage_snapshot(workspace, "E2FB-205C")
                self.assertEqual((snap["content_id"], snap["run_id"]), (self.CONTENT, self.RUN_ID))
                changed = json.loads((workspace / "readiness.json").read_text())
                changed["content_id"] = "b" * 64
                (workspace / "readiness.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "differs"):
                    JOB.storage_snapshot(workspace, "E2FB-205C")

    def test_absent_storage_is_refused_before_jobs_root_creation(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            workspace = Path(d) / "missing" / "PORTELANCE_WORKSPACE"
            root = workspace / "pc-runs"
            args = argparse.Namespace(profile="fixture-tests", python=sys.executable,
                jobs_root=str(root), job_id="absent", max_seconds=60, handoff=None,
                storage_workspace=str(workspace), storage_uuid="E2FB-205C", check_only=False)
            with patch.object(JOB, "git_state", return_value="abc"), \
                 self.assertRaisesRegex(ValueError, "reader lock"):
                JOB.submit(args)
            self.assertFalse(root.exists())

    def test_wrong_uuid_is_refused_before_any_job_file_is_created(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            mount, workspace = self.make_workspace(d)
            root = workspace / "pc-runs"
            args = argparse.Namespace(profile="fixture-tests", python=sys.executable,
                jobs_root=str(root), job_id="wrong-uuid", max_seconds=60, handoff=None,
                storage_workspace=str(workspace), storage_uuid="E2FB-205C", check_only=False)
            with patch.object(JOB, "mount_info", return_value=self.mount_record(mount, "WRONG")), \
                 patch.object(JOB, "git_state", return_value="abc"), \
                 self.assertRaisesRegex(ValueError, "UUID mismatch"):
                JOB.submit(args)
            self.assertFalse(root.exists())

    def test_storage_paths_cannot_escape_input_or_output_roots(self):
        import argparse
        with tempfile.TemporaryDirectory() as d:
            mount, workspace = self.make_workspace(d)
            outside = Path(d) / "outside"
            outside.mkdir()
            (outside / "manifest.json").write_text("{}")
            common = dict(profile="pbm-smoke", python=sys.executable, job_id="escape",
                          max_seconds=60, storage_workspace=str(workspace),
                          storage_uuid="E2FB-205C", check_only=True)
            with patch.object(JOB, "mount_info", return_value=self.mount_record(mount)), \
                 patch.object(JOB, "git_state", return_value="abc"), \
                 self.assertRaisesRegex(ValueError, "jobs-root"):
                JOB.submit(argparse.Namespace(jobs_root=str(outside), handoff=str(outside), **common))
            with patch.object(JOB, "mount_info", return_value=self.mount_record(mount)), \
                 patch.object(JOB, "git_state", return_value="abc"), \
                 self.assertRaisesRegex(ValueError, "Prepared handoff"):
                JOB.submit(argparse.Namespace(jobs_root=str(workspace / "pc-runs"),
                                               handoff=str(outside), **common))

    def test_freeze_and_exclusive_publisher_lock_refuse_reader(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            mount, workspace = self.make_workspace(d)
            (workspace / "freeze.json").write_text("{}")
            with patch.object(JOB, "mount_info", return_value=self.mount_record(mount)), \
                 JOB.storage_reader_lock(workspace), self.assertRaisesRegex(ValueError, "frozen"):
                JOB.storage_snapshot(workspace, "E2FB-205C")
            (workspace / "freeze.json").unlink()
            with (workspace / "writer.lock").open("r+") as publisher:
                fcntl.flock(publisher, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, "publisher holds"):
                    with JOB.storage_reader_lock(workspace):
                        pass

    def test_disappearance_during_poll_terminates_child(self):
        with tempfile.TemporaryDirectory() as d:
            mount, workspace = self.make_workspace(d)
            job = workspace / "pc-runs/disappeared"
            job.mkdir(parents=True)
            contract = {"storage_workspace": str(workspace), "storage_uuid": "E2FB-205C",
                        "storage_mount_root": str(mount), "storage_content_id": self.CONTENT,
                        "storage_run_id": self.RUN_ID, "job_id": "disappeared"}
            check = lambda: JOB.storage_runtime_check(contract, job)
            with patch.object(JOB, "mount_info", return_value=self.mount_record(mount, "WRONG")), \
                 self.assertRaisesRegex(ValueError, "mount identity changed"):
                JOB.run_command([sys.executable, "-c", "import time; time.sleep(60)"], d,
                                os.environ.copy(), Path(d) / "output.log", 10, 0, d, check)

    def test_storage_flags_must_be_complete(self):
        import argparse
        args = argparse.Namespace(profile="fixture-tests", python=sys.executable,
            jobs_root="/tmp/jobs", job_id="incomplete", max_seconds=60, handoff=None,
            storage_workspace="/tmp/PORTELANCE_WORKSPACE", storage_uuid=None, check_only=True)
        with self.assertRaisesRegex(ValueError, "provided together"):
            JOB.submit(args)

    def test_worker_full_storage_validation_runs_before_and_after_success(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d) / "home"
            home.mkdir()
            mount, workspace = self.make_workspace(Path(d) / "disk")
            job = workspace / "pc-runs/postcheck"
            job.mkdir(parents=True)
            contract = {"schema_version": 1, "profile": "fixture-tests", "repo": d,
                "python": sys.executable, "commit": "abc", "max_seconds": 5,
                "unit": "portelance-postcheck", "jobs_root": str(job.parent),
                "job_id": "postcheck", "storage_workspace": str(workspace),
                "storage_uuid": "E2FB-205C", "storage_mount_root": str(mount),
                "storage_content_id": self.CONTENT, "storage_run_id": self.RUN_ID}
            (job / "contract.json").write_text(json.dumps(contract))
            original = JOB.validate_storage_contract
            with patch.object(JOB.Path, "home", return_value=home), \
                 patch.object(JOB, "mount_info", return_value=self.mount_record(mount)), \
                 patch.object(JOB, "git_state", return_value="abc"), \
                 patch.object(JOB, "command", return_value=[sys.executable, "-c", "print('ok')"]), \
                 patch.object(JOB, "validate_storage_contract", wraps=original) as full, \
                 patch.object(JOB, "validate", side_effect=lambda c, j: JOB.validate_storage_contract(c, j)):
                self.assertEqual(JOB.worker(job), 0)
            self.assertGreaterEqual(full.call_count, 2)
            self.assertEqual(json.loads((job / "status.json").read_text())["state"], "passed")


if __name__ == "__main__":
    unittest.main()
