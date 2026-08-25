from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generate_baselines_mila.full79_lstm import (
    audit_full79_lstm_run,
    audit_lstm_output,
    finalize_full79_lstm_report,
    load_cell_index,
    parse_indices,
    prepare_full79_lstm_run,
)
from generate_baselines_mila.io import iter_csv_dicts, sha256_file, write_csv_dicts, write_json
from generate_baselines_mila.manifest import BaselineManifest
from generate_baselines_mila.ngram import output_fieldnames


AGE_LABELS = ("006-023", "024-029", "030-035", "036-041", "042-047", "048-053", "054-059", "060-065")
AGE_MONTHS = (18, 25, 31, 37, 43, 49, 55, 61)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_tiny_full79_bundle(root: Path) -> Path:
    bundle = root / "bundle"
    bins_dir = bundle / "age_ngram_dicts" / "merged_early_006_023"
    bins_dir.mkdir(parents=True)
    write_json(
        bins_dir / "age_bins.json",
        {
            "strategy": "merged_early_006_023",
            "bins": [
                {"label": label, "start": month if index else 6, "end": (23 if index == 0 else month + 4)}
                for index, (label, month) in enumerate(zip(AGE_LABELS, AGE_MONTHS))
            ],
        },
    )
    manifest_rows = []
    for index in range(79):
        dataset = f"Corpus{index % 13:02d}"
        child = f"Child{index:02d}"
        age = AGE_MONTHS[index % len(AGE_MONTHS)]
        folder = bundle / "preprocessed_data" / dataset / child
        chi = folder / "chi.csv"
        caretakers = folder / "caretakers.csv"
        scoring = folder / "chi.surprisal_scoring.csv"
        file_name = f"{child}/session.cha"
        common_child = {
            "dataset": dataset,
            "child_id": child,
            "source_group": dataset,
            "session_id": "1",
            "age_months": str(age),
            "file": file_name,
            "line_no": "2",
            "utt_id": "2",
            "utterance_clean": "more milk",
        }
        write_csv(chi, [common_child])
        write_csv(
            caretakers,
            [
                {
                    **common_child,
                    "line_no": "1",
                    "utt_id": "1",
                    "utterance_clean": "do you want some milk",
                }
            ],
        )
        write_csv(
            scoring,
            [
                {
                    "dataset": dataset,
                    "child_id": child,
                    "source_group": dataset,
                    "session_id": "1",
                    "age_months": str(age),
                    "file": file_name,
                    "line_no": "2",
                    "utt_id": "2",
                    "context_k1": "do you want some milk",
                    "context_k2": "do you want some milk",
                    "context_k3": "do you want some milk",
                    "chi_utterance_clean": "more milk",
                }
            ],
        )
        manifest_rows.append(
            {
                "dataset": dataset,
                "child_id": child,
                "child_scoring_ready": "1",
                "chi_csv": str(chi),
                "caretakers_csv": str(caretakers),
                "child_scoring_csv": str(scoring),
            }
        )
    write_csv(bundle / "manifest.csv", manifest_rows)
    return bundle


def materialize_valid_cell(manifest_path: Path) -> None:
    manifest = BaselineManifest.from_path(manifest_path)
    rows = []
    for target in iter_csv_dicts(manifest.target_csv):
        rows.append(
            {
                "baseline_run_id": manifest.run_id,
                "target_word_count": "2",
                "row_uid": target["row_uid"],
                **{column: target.get(column, "") for column in manifest.carry_columns},
                "source_model": manifest.raw["source_model"],
                "sample_index": "0",
                "generated_utterance": "want milk",
                "generated_word_count": "2",
                "generation_failed": "0",
                "fallback_used": "0",
                "failure_reason": "",
                "fallback_reason": "",
            }
        )
    write_csv_dicts(manifest.output_csv, rows, fieldnames=output_fieldnames(manifest))
    write_json(
        manifest.audit_json,
        {
            "run_id": manifest.run_id,
            "row_count": len(rows),
            "output_sha256": sha256_file(manifest.output_csv),
        },
    )
    artifact = Path(manifest.raw["model_dir"]) / "age_index_00"
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "model.pt").write_bytes(b"tiny-checkpoint")
    (artifact / "training_state.pt").write_bytes(b"tiny-training-state")
    write_json(artifact / "vocab.json", {"id_to_token": ["want", "milk"]})
    write_json(artifact / "child_output_vocab.json", {"id_to_token": ["want", "milk"]})
    write_json(artifact / "train_audit.json", {"architecture": "seq2seq_lstm"})


class Full79LstmProductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bundle = make_tiny_full79_bundle(self.root)
        self.run_root = self.root / "run"
        self.audit = prepare_full79_lstm_run(
            bundle_root=self.bundle,
            run_root=self.run_root,
            epochs=2,
            batch_size=4,
            embedding_dim=8,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            max_vocab_size=100,
            smoke_train_examples=8,
            smoke_target_rows=2,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_prepare_builds_eight_k3_additive_cells(self) -> None:
        self.assertEqual(self.audit["unit_count"], 79)
        self.assertEqual(self.audit["row_count"], 79)
        self.assertEqual(self.audit["context_alignment_mismatches"], 0)
        self.assertEqual(self.audit["contexts"], [3])
        self.assertEqual(self.audit["production_cell_count"], 8)
        cells = load_cell_index(self.run_root)
        self.assertEqual([cell["age_bin"] for cell in cells], list(AGE_LABELS))
        for cell in cells:
            payload = json.loads(Path(cell["manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["architecture"], "seq2seq_lstm")
            self.assertEqual(payload["context_column"], "generation_context_k3")
            self.assertEqual(payload["source_model"], "lstm_additive_k3_same_length")
            self.assertTrue(payload["same_length"])
            self.assertIn("chi_utterance_clean", payload["carry_columns"])

    def test_cell_audit_requires_checkpoint_vocab_and_same_length_rows(self) -> None:
        cell = load_cell_index(self.run_root)[0]
        manifest_path = Path(cell["manifest"])
        materialize_valid_cell(manifest_path)
        report = audit_lstm_output(manifest_path)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["output_rows"], report["expected_rows"])

        manifest = BaselineManifest.from_path(manifest_path)
        rows = list(iter_csv_dicts(manifest.output_csv))
        rows[0]["generated_utterance"] = ""
        write_csv_dicts(manifest.output_csv, rows, fieldnames=output_fieldnames(manifest))
        failed = audit_lstm_output(manifest_path)
        self.assertEqual(failed["status"], "FAIL")
        self.assertGreater(failed["empty_generated_rows"], 0)

    def test_final_audit_writes_complete_marker_after_all_eight_cells(self) -> None:
        cells = load_cell_index(self.run_root)
        for cell in cells:
            materialize_valid_cell(Path(cell["manifest"]))
        summary = audit_full79_lstm_run(run_root=self.run_root, stage="final")
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["cell_count"], 8)
        self.assertEqual(summary["missing_rows"], 0)
        self.assertEqual(summary["duplicate_rows"], 0)
        self.assertEqual(summary["same_length_mismatches"], 0)
        self.assertEqual(summary["fallback_rows"], 0)
        self.assertTrue(Path(summary["handoff"]["handoff_csv"]).exists())
        self.assertEqual(summary["handoff"]["handoff_rows"], 79)
        self.assertTrue((self.run_root / "COMPLETE_AND_AUDITED").exists())

    def test_parse_indices_supports_staged_ranges(self) -> None:
        self.assertEqual(parse_indices("0-3"), [0, 1, 2, 3])
        self.assertEqual(parse_indices("4-7"), [4, 5, 6, 7])
        with self.assertRaises(ValueError):
            parse_indices("0-8")

    def test_final_report_records_completed_job_states_and_hashes(self) -> None:
        for cell in load_cell_index(self.run_root):
            materialize_valid_cell(Path(cell["manifest"]))
        audit_full79_lstm_run(run_root=self.run_root, stage="final")
        submission_path = self.root / "submission.json"
        job_ids = {f"stage_{index}": str(100 + index) for index in range(7)}
        write_json(
            submission_path,
            {
                "commit_sha": "abc123",
                "configuration": {"embedding_dim": 256, "hidden_dim": 512},
                "job_ids": job_ids,
            },
        )
        stdout = "".join(
            f"{job_id}|full79|COMPLETED|0:0|00:01:00|start|end|\n"
            for job_id in job_ids.values()
        )
        completed = subprocess.CompletedProcess(args=["sacct"], returncode=0, stdout=stdout, stderr="")
        with patch.dict(os.environ, {"SUBMISSION_METADATA_JSON": str(submission_path)}), patch(
            "generate_baselines_mila.full79_lstm.subprocess.run", return_value=completed
        ):
            report = finalize_full79_lstm_report(self.run_root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["commit_sha"], "abc123")
        self.assertEqual(len(report["job_states"]), 7)
        self.assertTrue((self.run_root / "FINAL_REPORT_READY").exists())


class Full79LstmSubmitDagTests(unittest.TestCase):
    def test_submitter_builds_smoke_gated_two_wave_dag(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            bundle = root / "bundle"
            fake_bin = root / "bin"
            project.mkdir()
            bundle.mkdir()
            fake_bin.mkdir()
            (bundle / "manifest.csv").write_text("dataset,child_id\n", encoding="utf-8")
            log = root / "sbatch.log"
            counter = root / "counter"
            fake_sbatch = fake_bin / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "n=100\n"
                "[[ -f \"$FAKE_COUNTER\" ]] && n=$(cat \"$FAKE_COUNTER\")\n"
                "n=$((n + 1))\n"
                "printf '%s\\n' \"$n\" > \"$FAKE_COUNTER\"\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
                "printf '%s\\n' \"$n\"\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "PROJECT_ROOT": str(project),
                    "SCRATCH": str(root / "scratch"),
                    "FAKE_COUNTER": str(counter),
                    "FAKE_SBATCH_LOG": str(log),
                    "RUN_ID": "test-run",
                    "COMMIT_SHA": "test-commit-sha",
                }
            )
            completed = subprocess.run(
                ["bash", str(repo / "slurm" / "submit_full_79_lstm.sh"), str(bundle)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 8)
            self.assertTrue(all("--ntasks=1" in call for call in calls))
            self.assertIn("afterok:101", calls[1])
            self.assertIn("afterok:102", calls[2])
            self.assertIn("--array=0-3%3", calls[2])
            self.assertIn("afterok:104", calls[4])
            self.assertIn("--array=4-7%3", calls[4])
            self.assertIn("afterok:106", calls[6])
            self.assertIn("afterok:107", calls[7])
            self.assertIn("report_full_79_lstm.sbatch", calls[7])
            self.assertIn("FINAL_AUDIT_JOB=107", completed.stdout)
            self.assertIn("FINAL_REPORT_JOB=108", completed.stdout)

    def test_exact_cell_wrapper_exercises_smoke_path_with_fake_model_command(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_root = root / "run"
            (run_root / "manifests").mkdir(parents=True)
            (run_root / "manifests" / "smoke_manifest.json").write_text("{}\n", encoding="utf-8")
            (run_root / "PREPARED_AND_AUDITED").write_text("PASS\n", encoding="utf-8")
            command_log = root / "commands.log"
            generated_marker = root / "generated"
            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$FAKE_COMMAND_LOG\"\n"
                "case \"$*\" in\n"
                "  *'audit-lstm-output'*) [[ -f \"$FAKE_GENERATED\" ]] ;;\n"
                "  *'generate-lstm'*) touch \"$FAKE_GENERATED\" ;;\n"
                "  *'audit-full79-lstm-smoke'*) touch \"$RUN_ROOT/SMOKE_PASSED\" ;;\n"
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
                    "FAKE_GENERATED": str(generated_marker),
                    "SLURM_JOB_ID": "999",
                }
            )
            subprocess.run(
                ["bash", str(repo / "slurm" / "run_full_79_lstm_cell.sbatch"), "smoke"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            calls = command_log.read_text(encoding="utf-8")
            self.assertIn("validate-manifest", calls)
            self.assertIn("generate-lstm", calls)
            self.assertIn("audit-lstm-output", calls)
            self.assertIn("audit-full79-lstm-smoke", calls)
            self.assertTrue((run_root / "SMOKE_PASSED").exists())


if __name__ == "__main__":
    unittest.main()
