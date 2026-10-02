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
            self.assertEqual(JOB.worker(p), 2)
            self.assertEqual((p / "status.json").read_bytes(), before)

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


if __name__ == "__main__":
    unittest.main()
