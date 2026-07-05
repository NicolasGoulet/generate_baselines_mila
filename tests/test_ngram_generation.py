from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from generate_baselines_mila.manifest import BaselineManifest
from generate_baselines_mila.ngram import run_ngram_generation
from generate_baselines_mila.tokenize import tokenize_words


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class NgramGenerationTests(unittest.TestCase):
    def test_manifest_drives_same_length_generation_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "train.csv"
            target_csv = root / "target.csv"
            output_csv = root / "generated.csv.gz"
            manifest_json = root / "manifest.json"

            write_csv(
                train_csv,
                [
                    {
                        "utterance_id": "t1",
                        "age_bin": "006-023",
                        "context_text": "do you want",
                        "utterance_clean": "more milk",
                        "target_utterance_clean": "",
                    },
                    {
                        "utterance_id": "t2",
                        "age_bin": "024-029",
                        "context_text": "where did it",
                        "utterance_clean": "go there",
                        "target_utterance_clean": "",
                    },
                ],
            )
            write_csv(
                target_csv,
                [
                    {
                        "utterance_id": "u1",
                        "age_bin": "006-023",
                        "context_text": "do you want",
                        "utterance_clean": "",
                        "target_utterance_clean": "want milk",
                    },
                    {
                        "utterance_id": "u2",
                        "age_bin": "024-029",
                        "context_text": "where did it",
                        "utterance_clean": "",
                        "target_utterance_clean": "go there now",
                    },
                ],
            )
            manifest_json.write_text(
                json.dumps(
                    {
                        "run_id": "unit",
                        "train_csv": "train.csv",
                        "target_csv": "target.csv",
                        "output_csv": "generated.csv.gz",
                        "text_column": "utterance_clean",
                        "target_text_column": "target_utterance_clean",
                        "id_columns": ["utterance_id"],
                        "carry_columns": ["age_bin", "context_text"],
                        "age_bin_column": "age_bin",
                        "age_bins": ["006-023", "024-029"],
                        "context_column": "context_text",
                        "context_tail_words": 2,
                        "generators": ["random", "unigram", "bigram", "trigram"],
                        "same_length": True,
                        "samples_per_target": 2,
                        "seed": 7,
                    }
                ),
                encoding="utf-8",
            )

            manifest = BaselineManifest.from_path(manifest_json)
            audit = run_ngram_generation(manifest)

            self.assertEqual(audit["row_count"], 16)
            self.assertTrue(manifest.audit_json.exists())

            with gzip.open(output_csv, "rt", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 16)
            by_target = {row["utterance_id"]: int(row["target_word_count"]) for row in rows}
            self.assertEqual(by_target["u1"], 2)
            self.assertEqual(by_target["u2"], 3)
            for row in rows:
                self.assertEqual(
                    len(tokenize_words(row["generated_utterance"])),
                    int(row["target_word_count"]),
                )

    def test_empty_target_utterances_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_csv(
                root / "train.csv",
                [
                    {
                        "utterance_id": "t1",
                        "age_bin": "006-023",
                        "context_text": "",
                        "utterance_clean": "hello there",
                        "target_utterance_clean": "",
                    }
                ],
            )
            write_csv(
                root / "target.csv",
                [
                    {
                        "utterance_id": "u1",
                        "age_bin": "006-023",
                        "context_text": "",
                        "utterance_clean": "",
                        "target_utterance_clean": "...",
                    }
                ],
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "skip-empty",
                        "train_csv": "train.csv",
                        "target_csv": "target.csv",
                        "output_csv": "generated.csv.gz",
                        "text_column": "utterance_clean",
                        "target_text_column": "target_utterance_clean",
                        "id_columns": ["utterance_id"],
                    }
                ),
                encoding="utf-8",
            )

            audit = run_ngram_generation(BaselineManifest.from_path(manifest_path))
            self.assertEqual(audit["row_count"], 0)


if __name__ == "__main__":
    unittest.main()
