from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from generate_baselines_mila.lstm import (
    Vocab,
    _encoded_examples,
    _encoded_seq2seq_examples,
    require_torch,
    run_lstm_generation,
)
from generate_baselines_mila.manifest import BaselineManifest


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class LstmGenerationTests(unittest.TestCase):
    def test_encoded_examples_mask_context_loss(self) -> None:
        vocab = Vocab.build([["do", "you", "want", "milk"]])
        encoded = _encoded_examples([(["do", "you"], ["want", "milk"])], vocab)
        input_ids, labels = encoded[0]
        self.assertEqual(len(input_ids), len(labels))
        self.assertEqual(labels[0], -100)
        self.assertEqual(labels[1], -100)
        self.assertNotEqual(labels[2], -100)

    def test_seq2seq_examples_separate_encoder_and_decoder(self) -> None:
        vocab = Vocab.build([["do", "you", "want", "milk"]])
        encoded = _encoded_seq2seq_examples([(["do", "you"], ["want", "milk"])], vocab)
        encoder_ids, decoder_ids, labels = encoded[0]
        self.assertEqual(encoder_ids, [vocab.encode("do"), vocab.encode("you")])
        self.assertEqual(decoder_ids, [vocab.bos_id, vocab.encode("want"), vocab.encode("milk")])
        self.assertEqual(labels, [vocab.encode("want"), vocab.encode("milk"), vocab.eos_id])
        empty_context = _encoded_seq2seq_examples([([], ["milk"])], vocab)[0]
        self.assertEqual(empty_context[0], [vocab.no_context_id])

    def test_lstm_generation_requires_torch_when_not_installed(self) -> None:
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch is installed; this test is for the no-torch local environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "train.csv"
            target_csv = root / "target.csv"
            manifest_json = root / "manifest.json"
            write_csv(
                train_csv,
                [
                    {
                        "utterance_id": "t1",
                        "context_text": "do you want",
                        "utterance_clean": "more milk",
                        "target_utterance_clean": "",
                    }
                ],
            )
            write_csv(
                target_csv,
                [
                    {
                        "utterance_id": "u1",
                        "context_text": "do you want",
                        "utterance_clean": "",
                        "target_utterance_clean": "want milk",
                    }
                ],
            )
            manifest_json.write_text(
                json.dumps(
                    {
                        "run_id": "lstm-no-torch",
                        "train_csv": "train.csv",
                        "target_csv": "target.csv",
                        "output_csv": "generated.csv.gz",
                        "text_column": "utterance_clean",
                        "target_text_column": "target_utterance_clean",
                        "id_columns": ["utterance_id"],
                        "context_column": "context_text",
                        "epochs": 1,
                        "batch_size": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "requires PyTorch"):
                run_lstm_generation(BaselineManifest.from_path(manifest_json))

    def test_require_torch_error_is_actionable(self) -> None:
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch is installed")
        with self.assertRaisesRegex(RuntimeError, "Install torch"):
            require_torch()

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
    def test_tiny_lstm_smoke_generates_rows_when_torch_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_csv = root / "train.csv"
            target_csv = root / "target.csv"
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
                        "run_id": "lstm-tiny-smoke",
                        "train_csv": "train.csv",
                        "target_csv": "target.csv",
                        "output_csv": "lstm_generated.csv.gz",
                        "text_column": "utterance_clean",
                        "target_text_column": "target_utterance_clean",
                        "id_columns": ["utterance_id"],
                        "carry_columns": ["age_bin", "context_text"],
                        "age_bin_column": "age_bin",
                        "age_bins": ["006-023", "024-029"],
                        "context_column": "context_text",
                        "context_tail_words": 2,
                        "same_length": True,
                        "samples_per_target": 1,
                        "seed": 3,
                        "source_model": "lstm_smoke",
                        "architecture": "seq2seq_lstm",
                        "model_dir": str(root / "models"),
                        "device": "cpu",
                        "epochs": 1,
                        "batch_size": 2,
                        "embedding_dim": 8,
                        "hidden_dim": 16,
                        "max_vocab_size": 100,
                        "temperature": 1.0,
                        "top_k": 0,
                    }
                ),
                encoding="utf-8",
            )

            audit = run_lstm_generation(BaselineManifest.from_path(manifest_json))

            self.assertEqual(audit["row_count"], 2)
            self.assertTrue((root / "lstm_generated.csv.gz").exists())
            self.assertTrue((root / "models" / "age_index_00" / "model.pt").exists())
            self.assertTrue((root / "models" / "age_index_01" / "model.pt").exists())
            train_audit = json.loads(
                (root / "models" / "age_index_01" / "train_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(train_audit["architecture"], "seq2seq_lstm")
            self.assertGreater(train_audit["child_output_vocab_size"], 0)
            self.assertTrue((root / "models" / "age_index_01" / "training_state.pt").exists())

            run_lstm_generation(BaselineManifest.from_path(manifest_json))
            resumed = json.loads(
                (root / "models" / "age_index_01" / "train_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed["resumed_from_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
