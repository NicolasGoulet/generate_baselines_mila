from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from generate_baselines_mila.pbm_transformers import (
    AGE_BINS,
    ARCHITECTURES,
    audit_cell_output,
    audit_pbm_transformer_run,
    load_cell_index,
    prepare_pbm_transformer_run,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_jsonl_gz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def example(index: int, *, dataset: str, age_bin: str, split: str) -> dict[str, object]:
    return {
        "example_id": f"example-{dataset}-{age_bin}-{index}",
        "dataset": dataset,
        "child_id": f"child-{index % 3}",
        "session_id": "1",
        "file": f"{dataset}/session.cha",
        "line_no": str(index + 1),
        "reference_line": str(index + 1),
        "age_months": float(age_bin.split("-")[0]),
        "age_month_floor": int(age_bin.split("-")[0]),
        "target_age_bin": age_bin,
        "context_turns": ["do you want it"],
        "context_text": "do you want it",
        "context_turn_count": 1,
        "context_word_count": 4,
        "target_text": "yes please",
        "target_word_count": 2,
        "split": split,
    }


def make_handoff(root: Path) -> Path:
    handoff = root / "handoff"
    hashes: dict[str, str] = {}
    for _, end, label in AGE_BINS:
        base = handoff / "cumulative_age_models" / f"through_{end:03d}_months"
        train = base / "train.jsonl.gz"
        validation = base / "validation.jsonl.gz"
        development = base / "development.jsonl.gz"
        target = handoff / "pbm_target_age_bins" / f"pbm_{label}.jsonl.gz"
        write_jsonl_gz(train, [example(0, dataset="Wells", age_bin=label, split="train")])
        write_jsonl_gz(validation, [example(1, dataset="Wells", age_bin=label, split="validation")])
        write_jsonl_gz(
            development,
            [
                example(0, dataset="Wells", age_bin=label, split="train"),
                example(1, dataset="Wells", age_bin=label, split="validation"),
            ],
        )
        write_jsonl_gz(target, [example(2, dataset="Brown", age_bin=label, split="held_out_pbm_evaluation")])
        for path in (train, validation, development, target):
            hashes[str(path.relative_to(handoff))] = sha256(path)
    tokenizer_train = handoff / "examples" / "train_all_ages.jsonl.gz"
    write_jsonl_gz(tokenizer_train, [example(3, dataset="Wells", age_bin=AGE_BINS[-1][2], split="train")])
    hashes[str(tokenizer_train.relative_to(handoff))] = sha256(tokenizer_train)
    manifest = {
        "status": "complete",
        "age_bins": [label for _, _, label in AGE_BINS],
        "held_out_evaluation_datasets": ["Brown", "Manchester", "Providence"],
        "pbm_dataset_overlap_with_training": [],
        "fatal_issues": [],
        "output_sha256": hashes,
    }
    (handoff / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (handoff / "BUILD_COMPLETE_AND_AUDITED").write_text("PASS\n", encoding="utf-8")
    return handoff


def materialize_cell(cell: dict[str, object], *, censored: bool = False) -> None:
    manifest_path = Path(str(cell["manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_path = Path(manifest["target_file"])
    with gzip.open(target_path, "rt", encoding="utf-8") as handle:
        targets = [json.loads(line) for line in handle]
    rows = []
    for target in targets:
        rows.append(
            {
                "generated_id": f"{manifest['architecture_id']}::{target['example_id']}::0",
                "example_id": target["example_id"],
                "dataset": target["dataset"],
                "child_id": target["child_id"],
                "session_id": target["session_id"],
                "file": target["file"],
                "line_no": target["line_no"],
                "reference_line": target["reference_line"],
                "age_months": target["age_months"],
                "target_age_bin": target["target_age_bin"],
                "context_text": target["context_text"],
                "context_k3": target["context_text"],
                "real_target_text": target["target_text"],
                "real_target_word_count": target["target_word_count"],
                "source_model": manifest["architecture_id"],
                "sample_index": 0,
                "generated_utterance": "yes",
                "generated_word_count": 1,
                "generated_token_count": 1,
                "eos_reached": not censored,
                "max_token_censored": censored,
                "generation_seed": manifest["seed"],
                "max_new_tokens": manifest["max_new_tokens"],
                "temperature": manifest["temperature"],
                "top_p": manifest["top_p"],
            }
        )
    output = Path(manifest["output_file"])
    write_jsonl_gz(output, rows)
    artifact = Path(manifest["artifact_dir"])
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "config.json").write_text("{}\n", encoding="utf-8")
    (artifact / "model.safetensors").write_bytes(b"tiny model")
    (artifact / "training_report.json").write_text(
        json.dumps({"status": "PASS", "parameter_count": 58_000_000}), encoding="utf-8"
    )


class PbmTransformerPreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.handoff = make_handoff(self.root)
        self.run_root = self.root / "run"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_architectures_are_from_scratch_and_parameter_matched(self) -> None:
        self.assertEqual(set(ARCHITECTURES), {"babyllama_sized_llama_58m", "t5_58m"})
        self.assertEqual(ARCHITECTURES["babyllama_sized_llama_58m"]["family"], "decoder_only")
        self.assertEqual(ARCHITECTURES["t5_58m"]["family"], "encoder_decoder")
        for config in ARCHITECTURES.values():
            self.assertEqual(config["initialization"], "from_scratch")
            self.assertFalse(config["teacher_distillation"])
            self.assertEqual(config["vocab_size"], 16000)
            self.assertGreater(config["nominal_parameter_count"], 58_000_000)
            self.assertLess(config["nominal_parameter_count"], 59_000_000)
        frozen = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "pbm_transformers_from_scratch_v1.json").read_text()
        )
        for architecture_id, config in ARCHITECTURES.items():
            for key, value in frozen["architectures"][architecture_id].items():
                self.assertEqual(config[key], value)

    def test_prepare_audits_handoff_and_builds_sixteen_final_cells(self) -> None:
        report = prepare_pbm_transformer_run(
            handoff_root=self.handoff,
            run_root=self.run_root,
            smoke_train_examples=2,
            smoke_target_rows=1,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["final_cell_count"], 16)
        self.assertEqual(report["smoke_cell_count"], 2)
        self.assertTrue((self.run_root / "PREPARED_AND_AUDITED").exists())
        cells = load_cell_index(self.run_root)
        self.assertEqual(len(cells), 16)
        self.assertEqual({str(cell["architecture_id"]) for cell in cells}, set(ARCHITECTURES))
        self.assertEqual({str(cell["age_bin"]) for cell in cells}, {label for _, _, label in AGE_BINS})
        manifests = [json.loads(Path(str(cell["manifest_path"])).read_text()) for cell in cells]
        self.assertTrue(all(manifest["samples_per_target"] == 1 for manifest in manifests))
        self.assertTrue(all(manifest["generation_variant"] == "unconstrained_length" for manifest in manifests))
        self.assertTrue(all(manifest["max_new_tokens"] == 128 for manifest in manifests))
        self.assertTrue(all(manifest["development_rows"] == 2 for manifest in manifests))
        self.assertTrue(all(Path(manifest["development_file"]).is_file() for manifest in manifests))
        self.assertIn("reinitialize and refit", report["training_policy"])

    def test_prepare_rejects_mutated_upstream_file(self) -> None:
        path = self.handoff / "pbm_target_age_bins" / "pbm_006-023.jsonl.gz"
        path.write_bytes(path.read_bytes() + b"mutation")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare_pbm_transformer_run(handoff_root=self.handoff, run_root=self.run_root)

    def test_cell_and_final_audits_publish_scorer_handoff(self) -> None:
        prepare_pbm_transformer_run(handoff_root=self.handoff, run_root=self.run_root)
        for cell in load_cell_index(self.run_root):
            materialize_cell(cell)
            report = audit_cell_output(Path(str(cell["manifest_path"])))
            self.assertEqual(report["status"], "PASS")
        final = audit_pbm_transformer_run(self.run_root, stage="final")
        self.assertEqual(final["status"], "PASS")
        self.assertEqual(final["output_rows"], 16)
        self.assertEqual(final["duplicate_generated_ids"], 0)
        self.assertEqual(final["max_token_censored_rows"], 0)
        self.assertTrue(Path(final["handoff_csv"]).exists())
        self.assertTrue((self.run_root / "COMPLETE_AND_AUDITED").exists())

    def test_cell_audit_rejects_artificial_length_censoring(self) -> None:
        prepare_pbm_transformer_run(
            handoff_root=self.handoff,
            run_root=self.run_root,
            max_censored_fraction=0.001,
        )
        cell = load_cell_index(self.run_root)[0]
        materialize_cell(cell, censored=True)
        report = audit_cell_output(Path(str(cell["manifest_path"])))
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("censor", " ".join(report["problems"]).lower())


class PbmTransformerSubmitDagTests(unittest.TestCase):
    def test_submitter_builds_exact_wrapper_smoke_gated_dag(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handoff = root / "handoff"
            handoff.mkdir()
            (handoff / "manifest.json").write_text("{}\n", encoding="utf-8")
            (handoff / "BUILD_COMPLETE_AND_AUDITED").write_text("PASS\n", encoding="utf-8")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "sbatch.log"
            counter = root / "counter"
            fake = fake_bin / "sbatch"
            fake.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "n=200; [[ -f \"$FAKE_COUNTER\" ]] && n=$(<\"$FAKE_COUNTER\")\n"
                "n=$((n+1)); printf '%s\\n' \"$n\" > \"$FAKE_COUNTER\"\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"; printf '%s\\n' \"$n\"\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "PROJECT_ROOT": str(repo),
                    "SCRATCH": str(root / "scratch"),
                    "FAKE_COUNTER": str(counter),
                    "FAKE_SBATCH_LOG": str(log),
                    "RUN_ID": "test-run",
                    "COMMIT_SHA": "test-sha",
                    "GPU_GRES": "gpu:1",
                }
            )
            completed = subprocess.run(
                ["bash", str(repo / "slurm" / "submit_pbm_transformers.sh"), str(handoff)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 9)
            self.assertTrue(all("--ntasks=1" in call for call in calls))
            gpu_calls = [call for call in calls if "run_pbm_transformer_cell.sbatch" in call]
            self.assertEqual(len(gpu_calls), 3)
            self.assertTrue(all("--gres=gpu:1" in call for call in gpu_calls))
            self.assertIn("--array=0-1%2", gpu_calls[0])
            self.assertIn("afterok:201", gpu_calls[0])
            self.assertIn("--array=0-7%2", gpu_calls[1])
            self.assertIn("afterok:203", gpu_calls[1])
            self.assertIn("--array=8-15%2", gpu_calls[2])
            self.assertIn("afterok:205", gpu_calls[2])
            self.assertIn("FINAL_AUDIT_JOB=208", completed.stdout)
            self.assertIn("FINAL_REPORT_JOB=209", completed.stdout)

    def test_submitter_rejects_more_than_one_gpu_before_sbatch(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            handoff = root / "handoff"
            handoff.mkdir()
            (handoff / "manifest.json").write_text("{}\n", encoding="utf-8")
            (handoff / "BUILD_COMPLETE_AND_AUDITED").write_text("PASS\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "PROJECT_ROOT": str(repo),
                    "SCRATCH": str(root / "scratch"),
                    "RUN_ID": "bad-gpu-run",
                    "GPU_GRES": "gpu:2",
                }
            )
            completed = subprocess.run(
                ["bash", str(repo / "slurm" / "submit_pbm_transformers.sh"), str(handoff)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("exactly one GPU", completed.stderr)

    def test_exact_wrapper_uses_same_runtime_for_smoke_and_production(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            (run_root / "tokenizer").mkdir(parents=True)
            (run_root / "PREPARED_AND_AUDITED").write_text("PASS\n", encoding="utf-8")
            (run_root / "tokenizer" / "TOKENIZER_READY").write_text("PASS\n", encoding="utf-8")
            command_log = root / "commands.log"
            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_COMMAND_LOG\"\n"
                "case \"$*\" in\n"
                "  *'pbm-transformer-manifest'*) printf '%s\\n' \"$RUN_ROOT/manifests/smoke.json\" ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PROJECT_ROOT": str(repo),
                    "RUN_ROOT": str(run_root),
                    "PYTHON_CMD": str(fake_python),
                    "FAKE_COMMAND_LOG": str(command_log),
                    "SLURM_ARRAY_TASK_ID": "0",
                    "SLURM_JOB_ID": "999",
                }
            )
            subprocess.run(
                ["bash", str(repo / "slurm" / "run_pbm_transformer_cell.sbatch"), "smoke"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = command_log.read_text(encoding="utf-8")
            self.assertIn("validate-pbm-transformer-runtime", calls)
            self.assertIn("pbm-transformer-manifest", calls)
            self.assertIn("run-pbm-transformer-cell", calls)
            self.assertIn("audit-pbm-transformer-cell", calls)


if __name__ == "__main__":
    unittest.main()
