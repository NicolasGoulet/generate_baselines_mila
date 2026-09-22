"""Exercise launchers with real CPU preparation/audits and fake neural/Slurm work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_pbm_transformer_production import make_handoff


REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "scripts/run_pbm_transformer_smoke_local.sh"
SLURM = REPO / "slurm/submit_pbm_transformer_smoke_only.sh"


class SmokeEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="transformer smoke ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.handoff = make_handoff(self.root)
        self.run = self.root / "fresh run"
        self.log = self.root / "calls.jsonl"
        self.env = os.environ.copy()
        self.env.update(
            PROJECT_ROOT=str(REPO), RUN_ROOT=str(self.run), RUN_ID="fixture-smoke",
            SCRATCH=str(self.root / "scratch"), COMMIT_SHA="fixture-sha",
            GPU_GRES="gpu:1", TEST_LOG=str(self.log), TEST_SOURCE_REPO=str(REPO),
            PYTHONDONTWRITEBYTECODE="1", TEST_FAIL="",
        )

    def interpreter(self):
        executable = self.root / "fake python"
        executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["TEST_SOURCE_REPO"]) / "tests"))
from test_pbm_transformer_production import materialize_cell
from generate_baselines_mila.cli import main
args = sys.argv[3:]
command = args[0]
with open(os.environ["TEST_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
if command == os.environ.get("TEST_FAIL"):
    raise SystemExit(9)
if command == "validate-pbm-transformer-runtime":
    pass
elif command == "train-pbm-transformer-tokenizer":
    p = Path(args[args.index("--output-dir") + 1])
    p.mkdir(parents=True)
    (p / "TOKENIZER_READY").write_text("fixture\\n")
elif command == "run-pbm-transformer-cell":
    materialize_cell({"manifest_path": args[args.index("--manifest") + 1]})
else:
    raise SystemExit(main(args))
''')
        executable.chmod(0o755)
        self.env["PYTHON_CMD"] = str(executable)

    def run_local(self):
        return subprocess.run(["bash", str(LOCAL), str(self.handoff), str(self.run)],
                              cwd=self.root, env=self.env, text=True, capture_output=True)

    def calls(self):
        return [json.loads(s) for s in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_local_uses_both_smoke_manifests_and_real_cpu_audits(self):
        self.interpreter()
        result = self.run_local()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(calls[0][0], "validate-pbm-transformer-runtime")
        manifests = [c for c in calls if c[0] == "pbm-transformer-manifest"]
        self.assertEqual([c[c.index("--index") + 1] for c in manifests], ["0", "1"])
        self.assertTrue(all("--smoke" in c for c in manifests))
        self.assertEqual(sum(c[0] == "run-pbm-transformer-cell" for c in calls), 2)
        report = json.loads((self.run / "reports/smoke/smoke_summary.json").read_text())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(set(report["architectures"]), {"babyllama_sized_llama_58m", "t5_58m"})
        self.assertTrue((self.run / "SMOKE_PASSED").is_file())
        self.assertFalse((self.run / "cells").exists())
        self.assertIn("mode=local", (self.run / "LOCAL_SMOKE_EXECUTION.txt").read_text())
        self.assertIn("commit_sha=fixture-sha", (self.run / "LOCAL_SMOKE_EXECUTION.txt").read_text())

    def test_missing_cuda_stops_before_preparation(self):
        self.interpreter()
        self.env["TEST_FAIL"] = "validate-pbm-transformer-runtime"
        result = self.run_local()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([c[0] for c in self.calls()], ["validate-pbm-transformer-runtime"])
        self.assertFalse(self.run.exists())

    def test_local_refuses_existing_run_without_touching_it(self):
        self.interpreter()
        self.run.mkdir()
        marker = self.run / "keep.txt"
        marker.write_text("keep")
        self.assertNotEqual(self.run_local().returncode, 0)
        self.assertEqual(marker.read_text(), "keep")
        self.assertEqual(self.calls(), [])

    def test_failed_cell_does_not_emit_success_marker(self):
        self.interpreter()
        self.env["TEST_FAIL"] = "run-pbm-transformer-cell"
        result = self.run_local()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.run / "SMOKE_PASSED").exists())
        self.assertNotIn("audit-pbm-transformer-smoke", [c[0] for c in self.calls()])

    def scheduler(self):
        project = self.root / "cluster checkout"
        project.mkdir()
        binary = self.root / "bin"
        binary.mkdir()
        executable = binary / "sbatch"
        executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
p = Path(os.environ["TEST_LOG"])
old = p.read_text().splitlines() if p.exists() else []
with p.open("a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")
if str(len(old) + 1) == os.environ.get("TEST_FAIL"): raise SystemExit(7)
print(str(201 + len(old)) + ";test-cluster")
''')
        executable.chmod(0o755)
        self.env.update(PROJECT_ROOT=str(project), PATH=str(binary) + os.pathsep + os.environ["PATH"],
                        PYTHON_CMD=sys.executable)
        return project

    def run_slurm(self):
        return subprocess.run(["bash", str(SLURM), str(self.handoff)], cwd=self.root,
                              env=self.env, text=True, capture_output=True)

    def test_slurm_submits_only_three_stages_with_dependencies(self):
        project = self.scheduler()
        result = self.run_slurm()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("--ntasks=1" in c for c in calls))
        self.assertIn("--dependency=afterok:201", calls[1])
        self.assertIn("--array=0-1%2", calls[1])
        self.assertIn("--gres=gpu:1", calls[1])
        self.assertEqual(calls[1][-1], "smoke")
        self.assertIn("--dependency=afterok:202", calls[2])
        self.assertIn("--export=ALL,AUDIT_STAGE=smoke", calls[2])
        receipt = json.loads((project / "reports/submissions/pbm_transformer_smoke_fixture-smoke.json").read_text())
        self.assertFalse(receipt["production_submitted"])
        self.assertEqual(receipt["job_ids"]["gpu_smoke_array"], "202")
        self.assertEqual(receipt["configuration"]["max_new_tokens"], 16)
        self.assertNotEqual(self.run_slurm().returncode, 0)
        self.assertEqual(len(self.calls()), 3)

    def test_slurm_rejects_multi_gpu_before_submitting(self):
        self.scheduler()
        self.env["GPU_GRES"] = "gpu:2"
        self.assertNotEqual(self.run_slurm().returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_failed_submission_stops_and_reports_existing_job(self):
        self.scheduler()
        self.env["TEST_FAIL"] = "2"
        result = self.run_slurm()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls()), 2)
        self.assertIn("PREP_JOB=201", result.stdout)
        self.assertNotIn("Smoke-only submission queued", result.stdout)


if __name__ == "__main__":
    unittest.main()
