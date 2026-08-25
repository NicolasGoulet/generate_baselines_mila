"""Prepare and audit the strict-naturalistic full-79 LSTM production run."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .big_cleaned import (
    DEFAULT_AGE_BINS,
    PROVENANCE_COLUMNS,
    _age_bin_for,
    _load_age_bins,
    _read_bundle_manifest,
    _resolve_bundle_path,
    _stable_id,
    _token_count,
)
from .io import iter_csv_dicts, open_text, sha256_file, write_csv_dicts, write_json
from .manifest import BaselineManifest
from .ngram import output_fieldnames
from .tokenize import tokenize_words

DEFAULT_CONTEXTS = (3,)
DEFAULT_MAX_CONTEXT_TOKENS = 60

GENERATION_CONTEXT_COLUMNS = tuple(f"generation_context_k{k}" for k in DEFAULT_CONTEXTS)
FULL79_LSTM_COLUMNS = PROVENANCE_COLUMNS + GENERATION_CONTEXT_COLUMNS


def _as_sort_number(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value or ""))


def _row_uid(row: dict[str, str]) -> str:
    return _stable_id(
        [
            row.get("dataset", ""),
            row.get("child_id", ""),
            row.get("session_id", ""),
            row.get("file", ""),
            row.get("line_no", ""),
            row.get("utt_id", ""),
        ]
    )


def _context_from_history(
    history: Sequence[list[str]], *, context_utterances: int, max_context_tokens: int
) -> str:
    tokens = [token for turn in history[-context_utterances:] for token in turn]
    if max_context_tokens > 0:
        tokens = tokens[-max_context_tokens:]
    return " ".join(tokens)


def _generation_contexts_for_unit(
    chi_csv: Path,
    caretakers_csv: Path,
    *,
    contexts: Sequence[int],
    max_context_tokens: int,
) -> dict[str, dict[str, str]]:
    rows: list[tuple[str, dict[str, str]]] = []
    with chi_csv.open(newline="", encoding="utf-8") as handle:
        rows.extend(("child", dict(row)) for row in csv.DictReader(handle))
    with caretakers_csv.open(newline="", encoding="utf-8") as handle:
        rows.extend(("caretaker", dict(row)) for row in csv.DictReader(handle))

    rows.sort(
        key=lambda item: (
            _as_sort_number(item[1].get("session_id", "")),
            item[1].get("file", ""),
            _as_sort_number(item[1].get("line_no", "")),
            _as_sort_number(item[1].get("utt_id", "")),
            0 if item[0] == "caretaker" else 1,
        )
    )
    history_by_session: dict[str, list[list[str]]] = defaultdict(list)
    contexts_by_uid: dict[str, dict[str, str]] = {}
    for role, row in rows:
        session_key = row.get("session_id", "")
        tokens = tokenize_words(row.get("utterance_clean", ""), lowercase=True)
        if role == "caretaker":
            if tokens:
                history_by_session[session_key].append(tokens)
            continue
        if not tokens:
            continue
        history = history_by_session.get(session_key, [])
        contexts_by_uid[_row_uid(row)] = {
            f"generation_context_k{k}": _context_from_history(
                history,
                context_utterances=k,
                max_context_tokens=max_context_tokens,
            )
            for k in contexts
        }
    return contexts_by_uid


def _iter_full79_lstm_rows(
    bundle_root: Path,
    *,
    contexts: Sequence[int],
    max_context_tokens: int,
    age_bins: list[dict[str, Any]],
    stats: dict[str, Any],
) -> Iterator[dict[str, str]]:
    for manifest_row in _read_bundle_manifest(bundle_root):
        if manifest_row.get("child_scoring_ready") != "1":
            continue
        chi_csv = _resolve_bundle_path(bundle_root, manifest_row.get("chi_csv", ""))
        caretakers_csv = _resolve_bundle_path(bundle_root, manifest_row.get("caretakers_csv", ""))
        scoring_csv = _resolve_bundle_path(bundle_root, manifest_row.get("child_scoring_csv", ""))
        for path in (chi_csv, caretakers_csv, scoring_csv):
            if not path.exists():
                raise FileNotFoundError(f"Missing full-79 LSTM input: {path}")

        generation_contexts = _generation_contexts_for_unit(
            chi_csv,
            caretakers_csv,
            contexts=contexts,
            max_context_tokens=max_context_tokens,
        )
        stats["unit_count"] += 1
        stats["datasets"].add(manifest_row.get("dataset", ""))
        with scoring_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "dataset",
                "child_id",
                "session_id",
                "age_months",
                "file",
                "line_no",
                "utt_id",
                "chi_utterance_clean",
                "context_k1",
                "context_k2",
                "context_k3",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{scoring_csv} is missing required columns: {sorted(missing)}")
            for row in reader:
                utterance = row.get("chi_utterance_clean", "")
                if _token_count(utterance) == 0:
                    stats["skipped_empty"] += 1
                    continue
                age_bin = _age_bin_for(row.get("age_months", ""), age_bins)
                if not age_bin:
                    stats["skipped_age"] += 1
                    continue
                row_uid = _row_uid(row)
                if row_uid not in generation_contexts:
                    raise ValueError(
                        f"Could not align scoring row to child/caretaker history: {scoring_csv} row_uid={row_uid}"
                    )
                scoring_context_tokens = tokenize_words(row.get("context_k3", ""), lowercase=True)
                expected_generation_context = " ".join(scoring_context_tokens[-max_context_tokens:])
                if generation_contexts[row_uid].get("generation_context_k3", "") != expected_generation_context:
                    stats["context_alignment_mismatches"] += 1
                stats["age_bin_counts"][age_bin] += 1
                yield {
                    "row_uid": row_uid,
                    "dataset": row.get("dataset", ""),
                    "child_id": row.get("child_id", ""),
                    "source_group": row.get("source_group", ""),
                    "session_id": row.get("session_id", ""),
                    "age_months": row.get("age_months", ""),
                    "age_bin": age_bin,
                    "file": row.get("file", ""),
                    "line_no": row.get("line_no", ""),
                    "utt_id": row.get("utt_id", ""),
                    "context_k1": row.get("context_k1", ""),
                    "context_k2": row.get("context_k2", ""),
                    "context_k3": row.get("context_k3", ""),
                    "chi_utterance_clean": utterance,
                    **generation_contexts[row_uid],
                }


def _write_full79_inputs(
    bundle_root: Path,
    run_root: Path,
    *,
    contexts: Sequence[int],
    max_context_tokens: int,
    age_bins: list[dict[str, Any]],
) -> dict[str, Any]:
    inputs_dir = run_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    train_csv = inputs_dir / "full79_lstm_train.csv.gz"
    target_paths = {
        str(age_bin["label"]): inputs_dir / f"target_{age_bin['label']}.csv.gz"
        for age_bin in age_bins
    }
    temporary_train = inputs_dir / ".full79_lstm_train.tmp.csv.gz"
    temporary_targets = {
        label: inputs_dir / f".target_{label}.tmp.csv.gz" for label in target_paths
    }
    seen: set[str] = set()
    duplicate_ids = 0
    row_count = 0
    stats: dict[str, Any] = {
        "unit_count": 0,
        "datasets": set(),
        "age_bin_counts": Counter(),
        "skipped_empty": 0,
        "skipped_age": 0,
        "context_alignment_mismatches": 0,
    }
    handles = []
    try:
        train_handle = open_text(temporary_train, "wt")
        handles.append(train_handle)
        train_writer = csv.DictWriter(train_handle, fieldnames=list(FULL79_LSTM_COLUMNS))
        train_writer.writeheader()
        target_writers: dict[str, csv.DictWriter] = {}
        for label, path in temporary_targets.items():
            handle = open_text(path, "wt")
            handles.append(handle)
            writer = csv.DictWriter(handle, fieldnames=list(FULL79_LSTM_COLUMNS))
            writer.writeheader()
            target_writers[label] = writer

        rows = _iter_full79_lstm_rows(
            bundle_root,
            contexts=contexts,
            max_context_tokens=max_context_tokens,
            age_bins=age_bins,
            stats=stats,
        )
        for row in rows:
            if row["row_uid"] in seen:
                duplicate_ids += 1
            seen.add(row["row_uid"])
            train_writer.writerow(row)
            target_writers[row["age_bin"]].writerow(row)
            row_count += 1
    except Exception:
        temporary_train.unlink(missing_ok=True)
        for path in temporary_targets.values():
            path.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles:
            handle.close()

    if row_count == 0:
        raise ValueError("No full-79 LSTM rows were prepared.")
    if duplicate_ids:
        raise ValueError(f"Prepared full-79 LSTM input has {duplicate_ids} duplicate row ids.")
    if stats["context_alignment_mismatches"]:
        raise ValueError(
            "Generated k3 contexts disagreed with scorer k3 contexts for "
            f"{stats['context_alignment_mismatches']} rows."
        )
    if stats["unit_count"] != 79:
        raise ValueError(f"Expected 79 child units, found {stats['unit_count']}.")
    empty_bins = [label for label in target_paths if not stats["age_bin_counts"][label]]
    if empty_bins:
        raise ValueError(f"Prepared full-79 LSTM input has empty age bins: {empty_bins}")

    temporary_train.replace(train_csv)
    for label, path in target_paths.items():
        temporary_targets[label].replace(path)
    return {
        "train_csv": train_csv,
        "target_paths": target_paths,
        "row_count": row_count,
        "duplicate_row_ids": duplicate_ids,
        "unit_count": stats["unit_count"],
        "datasets": sorted(stats["datasets"]),
        "age_bin_counts": dict(stats["age_bin_counts"]),
        "skipped_empty": stats["skipped_empty"],
        "skipped_age": stats["skipped_age"],
        "context_alignment_mismatches": stats["context_alignment_mismatches"],
    }


def _write_smoke_target(source: Path, destination: Path, *, row_limit: int) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    count = 0
    with open_text(source, "rt") as source_handle, open_text(temporary, "wt") as output_handle:
        reader = csv.DictReader(source_handle)
        writer = csv.DictWriter(output_handle, fieldnames=list(reader.fieldnames or []))
        writer.writeheader()
        for row in reader:
            if count >= row_limit:
                break
            writer.writerow(row)
            count += 1
    temporary.replace(destination)
    if count != row_limit:
        raise ValueError(f"Smoke target expected {row_limit} rows, found {count}.")
    return count


def prepare_full79_lstm_run(
    *,
    bundle_root: str | Path,
    run_root: str | Path,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    epochs: int = 20,
    batch_size: int = 256,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.2,
    max_vocab_size: int = 30000,
    smoke_train_examples: int = 1024,
    smoke_target_rows: int = 25,
    seed: int = 123,
) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    run_root = Path(run_root).resolve()
    contexts = tuple(int(value) for value in contexts)
    if contexts != DEFAULT_CONTEXTS:
        raise ValueError(f"Production contexts must be {DEFAULT_CONTEXTS}; received {contexts}.")
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Fresh full-79 LSTM run root required: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    age_bins = _load_age_bins(bundle_root)
    age_labels = [str(item["label"]) for item in age_bins]
    expected_labels = [str(item["label"]) for item in DEFAULT_AGE_BINS]
    if age_labels != expected_labels:
        raise ValueError(f"Unexpected additive age bins: {age_labels}")

    prepared = _write_full79_inputs(
        bundle_root,
        run_root,
        contexts=contexts,
        max_context_tokens=max_context_tokens,
        age_bins=age_bins,
    )
    manifests_dir = run_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    carry_columns = [column for column in FULL79_LSTM_COLUMNS if column != "row_uid"]
    cumulative_train_rows = 0
    training_cells: list[dict[str, Any]] = []
    generation_cells: list[dict[str, Any]] = []
    for context in contexts:
        for age_index, age_label in enumerate(age_labels):
            cumulative_train_rows += prepared["age_bin_counts"][age_label]
            cell_index = len(cells)
            cell_label = f"cell_{cell_index:02d}_k{context}_{age_label}"
            manifest_path = manifests_dir / f"{cell_label}.json"
            output_csv = run_root / "generated" / f"{cell_label}.csv.gz"
            model_dir = run_root / "models" / cell_label
            manifest = {
                "run_id": f"full79_lstm_additive_k{context}_{age_label}",
                "train_csv": str(prepared["train_csv"]),
                "target_csv": str(prepared["target_paths"][age_label]),
                "output_csv": str(output_csv),
                "text_column": "chi_utterance_clean",
                "target_text_column": "chi_utterance_clean",
                "id_columns": ["row_uid"],
                "carry_columns": carry_columns,
                "age_bin_column": "age_bin",
                "age_bins": age_labels,
                "context_column": f"generation_context_k{context}",
                "context_tail_words": 0,
                "same_length": True,
                "samples_per_target": 1,
                "seed": seed,
                "source_model": f"lstm_additive_k{context}_same_length",
                "training_scope": "strict_naturalistic_full79_additive_age_bins",
                "generation_length_mode": "same_as_child",
                "architecture": "seq2seq_lstm",
                "model_dir": str(model_dir),
                "device": "cuda",
                "epochs": epochs,
                "batch_size": batch_size,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": 0.001,
                "grad_clip": 1.0,
                "min_freq": 1,
                "max_vocab_size": max_vocab_size,
                "temperature": 0.9,
                "top_k": 50,
                "resume_training": True,
                "expected_target_rows": prepared["age_bin_counts"][age_label],
                "production_cell_index": cell_index,
                "generation_context_utterances": context,
                "target_age_bin": age_label,
            }
            write_json(manifest_path, manifest)
            cells.append(
                {
                    "cell_index": cell_index,
                    "context_utterances": context,
                    "age_bin": age_label,
                    "manifest": str(manifest_path),
                    "output_csv": str(output_csv),
                    "model_dir": str(model_dir),
                    "expected_target_rows": prepared["age_bin_counts"][age_label],
                }
            )
            training_cells.append(
                {
                    "cell_index": cell_index,
                    "age_bin": age_label,
                    "training_age_bins": age_labels[: age_index + 1],
                    "expected_training_rows": cumulative_train_rows,
                    "train_csv": str(prepared["train_csv"]),
                    "manifest": str(manifest_path),
                    "model_dir": str(model_dir),
                }
            )
            generation_cells.append(
                {
                    "cell_index": cell_index,
                    "age_bin": age_label,
                    "expected_target_rows": prepared["age_bin_counts"][age_label],
                    "target_csv": str(prepared["target_paths"][age_label]),
                    "output_csv": str(output_csv),
                    "manifest": str(manifest_path),
                }
            )

    smoke_age_label = age_labels[-1]
    smoke_target = run_root / "smoke" / "inputs" / f"target_{smoke_age_label}_{smoke_target_rows}.csv.gz"
    _write_smoke_target(
        prepared["target_paths"][smoke_age_label],
        smoke_target,
        row_limit=smoke_target_rows,
    )
    smoke_manifest = json.loads(Path(cells[-1]["manifest"]).read_text(encoding="utf-8"))
    smoke_manifest.update(
        {
            "run_id": "full79_lstm_production_wrapper_smoke",
            "target_csv": str(smoke_target),
            "output_csv": str(run_root / "smoke" / "generated" / "smoke.csv.gz"),
            "model_dir": str(run_root / "smoke" / "models"),
            "max_train_examples": smoke_train_examples,
            "expected_target_rows": smoke_target_rows,
            "smoke": True,
        }
    )
    smoke_manifest_path = manifests_dir / "smoke_manifest.json"
    write_json(smoke_manifest_path, smoke_manifest)
    write_json(manifests_dir / "cell_index.json", {"cells": cells})
    frozen_configuration = {
        "architecture": "seq2seq_lstm",
        "training_scope": "strict_naturalistic_full79_additive_age_bins",
        "generation_context_utterances": 3,
        "same_length": True,
        "samples_per_target": 1,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "max_context_tokens": max_context_tokens,
    }
    commit_sha = os.environ.get("COMMIT_SHA", "")
    write_json(
        manifests_dir / "training_manifest.json",
        {
            "commit_sha": commit_sha,
            "configuration": frozen_configuration,
            "train_csv": str(prepared["train_csv"]),
            "train_sha256": sha256_file(prepared["train_csv"]),
            "cells": training_cells,
        },
    )
    write_json(
        manifests_dir / "generation_manifest.json",
        {
            "commit_sha": commit_sha,
            "configuration": frozen_configuration,
            "expected_total_rows": prepared["row_count"],
            "cells": generation_cells,
        },
    )

    audit = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle_root),
        "run_root": str(run_root),
        "unit_count": prepared["unit_count"],
        "dataset_count": len(prepared["datasets"]),
        "datasets": prepared["datasets"],
        "row_count": prepared["row_count"],
        "duplicate_row_ids": prepared["duplicate_row_ids"],
        "context_alignment_mismatches": prepared["context_alignment_mismatches"],
        "age_bin_counts": prepared["age_bin_counts"],
        "contexts": list(contexts),
        "production_cell_count": len(cells),
        "architecture": "seq2seq_lstm",
        "same_length": True,
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "max_vocab_size": max_vocab_size,
        "max_context_tokens": max_context_tokens,
        "smoke_manifest": str(smoke_manifest_path),
        "smoke_train_examples": smoke_train_examples,
        "smoke_target_rows": smoke_target_rows,
        "train_sha256": sha256_file(prepared["train_csv"]),
        "commit_sha": commit_sha,
        "training_manifest": str(manifests_dir / "training_manifest.json"),
        "generation_manifest": str(manifests_dir / "generation_manifest.json"),
    }
    report_dir = run_root / "reports" / "preparation"
    write_json(report_dir / "preparation_audit.json", audit)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preparation_report.md").write_text(
        "\n".join(
            [
                "# Full-79 LSTM Preparation Report",
                "",
                "- status: `PASS`",
                f"- child units: `{audit['unit_count']}`",
                f"- datasets: `{audit['dataset_count']}`",
                f"- child rows: `{audit['row_count']}`",
                f"- additive age bins: `{len(age_labels)}`",
                f"- generation contexts: `{','.join(str(value) for value in contexts)}`",
                f"- production cells: `{len(cells)}`",
                "- architecture: `seq2seq_lstm`",
                "- generation: `same_length`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "PREPARED_AND_AUDITED").write_text("PREPARED_AND_AUDITED\n", encoding="utf-8")
    return audit


def load_cell_index(run_root: str | Path) -> list[dict[str, Any]]:
    path = Path(run_root) / "manifests" / "cell_index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 8:
        raise ValueError(f"Expected 8 production cells in {path}")
    return cells


def manifest_for_cell(run_root: str | Path, index: int) -> Path:
    cells = load_cell_index(run_root)
    if index < 0 or index >= len(cells):
        raise IndexError(f"Cell index out of range: {index}")
    return Path(str(cells[index]["manifest"]))


def audit_lstm_output(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = BaselineManifest.from_path(manifest_path)
    problems: list[str] = []
    if not manifest.output_csv.exists():
        target_rows = sum(1 for _ in iter_csv_dicts(manifest.target_csv))
        expected_rows = target_rows * manifest.samples_per_target
        return {
            "status": "FAIL",
            "manifest": str(manifest_path),
            "run_id": manifest.run_id,
            "output_csv": str(manifest.output_csv),
            "output_rows": 0,
            "expected_rows": expected_rows,
            "missing_output_rows": expected_rows,
            "duplicate_output_rows": 0,
            "empty_generated_rows": 0,
            "same_length_mismatches": 0,
            "declared_length_mismatches": 0,
            "provenance_mismatches": 0,
            "invalid_flag_rows": 0,
            "generation_failed_rows": 0,
            "fallback_rows": 0,
            "checkpoint_count": 0,
            "output_sha256": "",
            "problems": ["missing output CSV"],
        }
    output_audit: dict[str, Any] = {}
    if not manifest.audit_json.exists():
        problems.append("missing output audit JSON")
    else:
        try:
            output_audit = json.loads(manifest.audit_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"invalid output audit JSON: {exc}")

    required_output_columns = {
        "row_uid",
        "dataset",
        "child_id",
        "age_months",
        "age_bin",
        "context_k3",
        "generation_context_k3",
        "chi_utterance_clean",
        "generated_utterance",
        "generated_word_count",
        "target_word_count",
        "generation_failed",
        "fallback_used",
        "failure_reason",
        "fallback_reason",
    }
    with open_text(manifest.output_csv, "rt") as output_handle:
        output_reader = csv.DictReader(output_handle)
        missing_columns = sorted(required_output_columns - set(output_reader.fieldnames or []))
    if missing_columns:
        problems.append(f"output CSV missing required columns: {missing_columns}")

    target_rows_by_uid: dict[str, dict[str, str]] = {}
    target_lengths: dict[str, int] = {}
    for row in iter_csv_dicts(manifest.target_csv):
        row_uid = row.get("row_uid", "")
        if not row_uid:
            problems.append("target row missing row_uid")
            continue
        if row_uid in target_lengths:
            problems.append(f"duplicate target row_uid: {row_uid}")
            continue
        target_rows_by_uid[row_uid] = row
        target_lengths[row_uid] = _token_count(row.get(manifest.target_text_column, ""))

    seen: set[tuple[str, str]] = set()
    matched_seen: set[tuple[str, str]] = set()
    output_count = 0
    duplicate_output_rows = 0
    empty_generated = 0
    length_mismatches = 0
    unknown_ids = 0
    wrong_source = 0
    failed_rows = 0
    fallback_rows = 0
    invalid_flag_rows = 0
    provenance_mismatches = 0
    length_field_mismatches = 0
    expected_source = str(manifest.raw.get("source_model", "lstm"))
    for row in iter_csv_dicts(manifest.output_csv):
        output_count += 1
        row_uid = row.get("row_uid", "")
        sample_index = row.get("sample_index", "")
        key = (row_uid, sample_index)
        if key in seen:
            duplicate_output_rows += 1
            problems.append(f"duplicate output key: {key}")
        seen.add(key)
        if row_uid not in target_lengths:
            unknown_ids += 1
            continue
        matched_seen.add(key)
        expected_target = target_rows_by_uid[row_uid]
        if any(
            row.get(column, "") != expected_target.get(column, "")
            for column in (*manifest.id_columns, *manifest.carry_columns)
        ):
            provenance_mismatches += 1
        generated = row.get("generated_utterance", "")
        generated_length = _token_count(generated)
        if generated_length == 0:
            empty_generated += 1
        if generated_length != target_lengths[row_uid]:
            length_mismatches += 1
        try:
            declared_generated_length = int(row.get("generated_word_count", ""))
            declared_target_length = int(row.get("target_word_count", ""))
        except ValueError:
            length_field_mismatches += 1
        else:
            if (
                declared_generated_length != generated_length
                or declared_target_length != target_lengths[row_uid]
            ):
                length_field_mismatches += 1
        if row.get("source_model") != expected_source:
            wrong_source += 1
        failed_flag = row.get("generation_failed", "").strip().lower()
        fallback_flag = row.get("fallback_used", "").strip().lower()
        if failed_flag not in {"0", "1", "false", "true"} or fallback_flag not in {
            "0",
            "1",
            "false",
            "true",
        }:
            invalid_flag_rows += 1
        if failed_flag in {"1", "true"}:
            failed_rows += 1
        if fallback_flag in {"1", "true"}:
            fallback_rows += 1

    expected_count = len(target_lengths) * manifest.samples_per_target
    expected_keys = {
        (row_uid, str(sample_index))
        for row_uid in target_lengths
        for sample_index in range(manifest.samples_per_target)
    }
    missing_output_rows = len(expected_keys - matched_seen)
    manifest_expected = int(manifest.raw.get("expected_target_rows", len(target_lengths)))
    if len(target_lengths) != manifest_expected:
        problems.append(f"target rows {len(target_lengths)} != manifest expected {manifest_expected}")
    if output_count != expected_count:
        problems.append(f"output rows {output_count} != expected {expected_count}")
    if missing_output_rows:
        problems.append(f"missing target/sample output rows: {missing_output_rows}")
    if empty_generated:
        problems.append(f"empty generated rows: {empty_generated}")
    if length_mismatches:
        problems.append(f"same-length mismatches: {length_mismatches}")
    if length_field_mismatches:
        problems.append(f"declared word-count mismatches: {length_field_mismatches}")
    if provenance_mismatches:
        problems.append(f"identifier/context/target provenance mismatches: {provenance_mismatches}")
    if invalid_flag_rows:
        problems.append(f"invalid or empty generation/fallback flags: {invalid_flag_rows}")
    if unknown_ids:
        problems.append(f"unknown output row ids: {unknown_ids}")
    if wrong_source:
        problems.append(f"wrong source_model rows: {wrong_source}")
    output_sha256 = sha256_file(manifest.output_csv)
    if output_audit:
        if output_audit.get("run_id") != manifest.run_id:
            problems.append("output audit run_id does not match manifest")
        if int(output_audit.get("row_count", -1)) != output_count:
            problems.append("output audit row_count does not match output")
        if output_audit.get("output_sha256") != output_sha256:
            problems.append("output audit checksum does not match output")

    model_dir = Path(str(manifest.raw.get("model_dir", "")))
    checkpoints = list(model_dir.glob("**/model.pt")) if model_dir.exists() else []
    vocabs = list(model_dir.glob("**/vocab.json")) if model_dir.exists() else []
    child_vocabs = list(model_dir.glob("**/child_output_vocab.json")) if model_dir.exists() else []
    train_audits = list(model_dir.glob("**/train_audit.json")) if model_dir.exists() else []
    training_states = list(model_dir.glob("**/training_state.pt")) if model_dir.exists() else []
    if len(checkpoints) != 1:
        problems.append(f"expected one model checkpoint, found {len(checkpoints)}")
    if len(vocabs) != 1 or len(child_vocabs) != 1 or len(train_audits) != 1:
        problems.append(
            "expected one vocabulary, child-output vocabulary, and training audit "
            f"(found {len(vocabs)}, {len(child_vocabs)}, {len(train_audits)})"
        )
    if bool(manifest.raw.get("resume_training", True)) and len(training_states) != 1:
        problems.append(f"expected one resumable training state, found {len(training_states)}")

    return {
        "status": "PASS" if not problems else "FAIL",
        "manifest": str(manifest_path),
        "run_id": manifest.run_id,
        "output_csv": str(manifest.output_csv),
        "output_rows": output_count,
        "expected_rows": expected_count,
        "missing_output_rows": missing_output_rows,
        "duplicate_output_rows": duplicate_output_rows,
        "empty_generated_rows": empty_generated,
        "same_length_mismatches": length_mismatches,
        "declared_length_mismatches": length_field_mismatches,
        "provenance_mismatches": provenance_mismatches,
        "invalid_flag_rows": invalid_flag_rows,
        "generation_failed_rows": failed_rows,
        "fallback_rows": fallback_rows,
        "checkpoint_count": len(checkpoints),
        "output_sha256": output_sha256,
        "problems": problems,
    }


def _read_submission_metadata() -> dict[str, Any]:
    metadata_path = os.environ.get("SUBMISSION_METADATA_JSON", "")
    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.is_file():
        return {"path": metadata_path, "error": "submission metadata file is missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": metadata_path, "error": str(exc)}
    payload["path"] = str(path)
    return payload


def _slurm_accounting(submission: dict[str, Any]) -> dict[str, Any]:
    job_ids = [str(value) for value in submission.get("job_ids", {}).values() if str(value)]
    if not job_ids:
        return {"records": [], "error": "no job ids were available"}
    command = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        ",".join(job_ids),
        "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"records": [], "error": str(exc)}
    records: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 7:
            continue
        records.append(
            dict(
                zip(
                    ("job_id", "job_name", "state", "exit_code", "elapsed", "start", "end"),
                    fields[:7],
                )
            )
        )
    return {"records": records, "error": ""}


def _publish_final_handoff(
    run_root: Path,
    cells: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    handoff_dir = run_root / "handoff"
    handoff_csv = handoff_dir / "full79_lstm_scorer_ready.csv.gz"
    first_manifest = BaselineManifest.from_path(cells[0]["manifest"])
    fieldnames = output_fieldnames(first_manifest)
    seen: set[tuple[str, str]] = set()
    duplicate_rows = 0

    def rows() -> Iterator[dict[str, str]]:
        nonlocal duplicate_rows
        for cell in cells:
            for row in iter_csv_dicts(cell["output_csv"]):
                key = (row.get("row_uid", ""), row.get("sample_index", ""))
                if key in seen:
                    duplicate_rows += 1
                    raise ValueError(f"Cross-bin duplicate handoff row: {key}")
                seen.add(key)
                yield row

    handoff_rows = write_csv_dicts(handoff_csv, rows(), fieldnames=fieldnames)
    expected_rows = sum(int(report["expected_rows"]) for report in reports)
    if handoff_rows != expected_rows:
        raise ValueError(f"Handoff rows {handoff_rows} != expected {expected_rows}")

    per_bin_rows = []
    for cell, report in zip(cells, reports):
        per_bin_rows.append(
            {
                "cell_index": cell["cell_index"],
                "age_bin": cell["age_bin"],
                "expected_rows": report["expected_rows"],
                "output_rows": report["output_rows"],
                "missing_rows": report["missing_output_rows"],
                "duplicate_rows": report["duplicate_output_rows"],
                "length_mismatches": report["same_length_mismatches"],
                "declared_length_mismatches": report["declared_length_mismatches"],
                "provenance_mismatches": report["provenance_mismatches"],
                "invalid_flag_rows": report["invalid_flag_rows"],
                "generation_failed_rows": report["generation_failed_rows"],
                "fallback_rows": report["fallback_rows"],
                "output_csv": report["output_csv"],
                "output_sha256": report["output_sha256"],
            }
        )
    per_bin_audit = run_root / "reports" / "final" / "per_bin_audit.csv"
    write_csv_dicts(per_bin_audit, per_bin_rows, fieldnames=list(per_bin_rows[0]))

    manifest_path = handoff_dir / "full79_lstm_scorer_ready.manifest.json"
    handoff_manifest = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "handoff_csv": str(handoff_csv),
        "handoff_rows": handoff_rows,
        "handoff_sha256": sha256_file(handoff_csv),
        "training_manifest": str(run_root / "manifests" / "training_manifest.json"),
        "generation_manifest": str(run_root / "manifests" / "generation_manifest.json"),
        "per_bin_audit": str(per_bin_audit),
        "duplicate_rows": duplicate_rows,
        "missing_rows": sum(int(report["missing_output_rows"]) for report in reports),
        "length_mismatches": sum(int(report["same_length_mismatches"]) for report in reports),
        "declared_length_mismatches": sum(
            int(report["declared_length_mismatches"]) for report in reports
        ),
        "provenance_mismatches": sum(int(report["provenance_mismatches"]) for report in reports),
        "invalid_flag_rows": sum(int(report["invalid_flag_rows"]) for report in reports),
        "generation_failed_rows": sum(int(report["generation_failed_rows"]) for report in reports),
        "fallback_rows": sum(int(report["fallback_rows"]) for report in reports),
        "source_outputs": [
            {
                "cell_index": cell["cell_index"],
                "age_bin": cell["age_bin"],
                "path": report["output_csv"],
                "sha256": report["output_sha256"],
            }
            for cell, report in zip(cells, reports)
        ],
    }
    write_json(manifest_path, handoff_manifest)
    handoff_manifest["manifest_path"] = str(manifest_path)
    handoff_manifest["manifest_sha256"] = sha256_file(manifest_path)
    return handoff_manifest


def parse_indices(value: str, *, maximum: int = 8) -> list[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            indices.update(range(int(start_text), int(end_text) + 1))
        else:
            indices.add(int(part))
    ordered = sorted(indices)
    if not ordered or ordered[0] < 0 or ordered[-1] >= maximum:
        raise ValueError(f"Invalid cell indices: {value}")
    return ordered


def audit_full79_lstm_run(
    *,
    run_root: str | Path,
    stage: str,
    indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    cells = load_cell_index(run_root)
    selected = list(indices) if indices is not None else list(range(len(cells)))
    reports = [audit_lstm_output(cells[index]["manifest"]) for index in selected]
    failures = [report for report in reports if report["status"] != "PASS"]
    status = "PASS" if not failures else "FAIL"
    expected_rows = sum(int(report["expected_rows"]) for report in reports)
    output_rows = sum(int(report["output_rows"]) for report in reports)
    missing_rows = sum(int(report["missing_output_rows"]) for report in reports)
    duplicate_rows = sum(int(report["duplicate_output_rows"]) for report in reports)
    length_mismatches = sum(int(report["same_length_mismatches"]) for report in reports)
    declared_length_mismatches = sum(
        int(report["declared_length_mismatches"]) for report in reports
    )
    provenance_mismatches = sum(int(report["provenance_mismatches"]) for report in reports)
    invalid_flag_rows = sum(int(report["invalid_flag_rows"]) for report in reports)
    generation_failed_rows = sum(int(report["generation_failed_rows"]) for report in reports)
    fallback_rows = sum(int(report["fallback_rows"]) for report in reports)
    summary = {
        "status": status,
        "stage": stage,
        "run_root": str(run_root),
        "cell_indices": selected,
        "cell_count": len(selected),
        "passed_cells": len(reports) - len(failures),
        "failed_cells": len(failures),
        "expected_rows": expected_rows,
        "output_rows": output_rows,
        "missing_rows": missing_rows,
        "duplicate_rows": duplicate_rows,
        "same_length_mismatches": length_mismatches,
        "declared_length_mismatches": declared_length_mismatches,
        "provenance_mismatches": provenance_mismatches,
        "invalid_flag_rows": invalid_flag_rows,
        "generation_failed_rows": generation_failed_rows,
        "fallback_rows": fallback_rows,
        "cells": reports,
    }
    if stage == "final" and not failures and selected == list(range(8)):
        summary["handoff"] = _publish_final_handoff(run_root, cells, reports)
        summary["submission"] = _read_submission_metadata()
        summary["slurm_accounting"] = _slurm_accounting(summary["submission"])
    report_dir = run_root / "reports" / stage
    write_json(report_dir / f"{stage}_summary.json", summary)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{stage}_report.md").write_text(
        "\n".join(
            [
                f"# Full-79 LSTM {stage.replace('_', ' ').title()} Report",
                "",
                f"- status: `{status}`",
                f"- cells: `{len(selected)}`",
                f"- passed cells: `{summary['passed_cells']}`",
                f"- failed cells: `{summary['failed_cells']}`",
                f"- expected generated rows: `{expected_rows}`",
                f"- validated generated rows: `{output_rows}`",
                f"- missing rows: `{missing_rows}`",
                f"- duplicate rows: `{duplicate_rows}`",
                f"- same-length mismatches: `{length_mismatches}`",
                f"- declared word-count mismatches: `{declared_length_mismatches}`",
                f"- identifier/context/target provenance mismatches: `{provenance_mismatches}`",
                f"- invalid generation/fallback flags: `{invalid_flag_rows}`",
                f"- generation failures: `{generation_failed_rows}`",
                f"- fallback rows: `{fallback_rows}`",
                *(
                    [
                        f"- scorer-ready handoff: `{summary['handoff']['handoff_csv']}`",
                        f"- handoff SHA-256: `{summary['handoff']['handoff_sha256']}`",
                        f"- handoff manifest: `{summary['handoff']['manifest_path']}`",
                        f"- handoff manifest SHA-256: `{summary['handoff']['manifest_sha256']}`",
                        f"- commit SHA: `{summary.get('submission', {}).get('commit_sha', '')}`",
                        f"- job IDs: `{json.dumps(summary.get('submission', {}).get('job_ids', {}), sort_keys=True)}`",
                        f"- Slurm exit states: `{json.dumps(summary.get('slurm_accounting', {}).get('records', []), sort_keys=True)}`",
                        "- configuration: `embedding=256 hidden=512 layers=2 epochs=20 batch=256 k3 same-length`",
                    ]
                    if stage == "final" and "handoff" in summary
                    else []
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{stage} audit failed for {len(failures)} cells")

    marker = {
        "wave1": "WAVE1_READY",
        "wave2": "WAVE2_READY",
        "final": "COMPLETE_AND_AUDITED",
    }.get(stage)
    if marker:
        (run_root / marker).write_text(f"{marker}\n", encoding="utf-8")
    return summary


def finalize_full79_lstm_report(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    complete_marker = run_root / "COMPLETE_AND_AUDITED"
    final_summary_path = run_root / "reports" / "final" / "final_summary.json"
    if not complete_marker.is_file() or not final_summary_path.is_file():
        raise FileNotFoundError("Final report requires COMPLETE_AND_AUDITED and final_summary.json")
    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    submission = _read_submission_metadata()
    accounting = _slurm_accounting(submission)
    exact_records = {record["job_id"]: record for record in accounting.get("records", [])}
    job_states: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for label, job_id_value in submission.get("job_ids", {}).items():
        job_id = str(job_id_value)
        record = exact_records.get(job_id)
        if record is None:
            problems.append(f"missing sacct record for {label} job {job_id}")
            continue
        job_states[label] = record
        if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
            problems.append(
                f"{label} job {job_id} ended {record['state']} with exit {record['exit_code']}"
            )
    if accounting.get("error"):
        problems.append(f"sacct query failed: {accounting['error']}")

    handoff = final_summary.get("handoff", {})
    configuration = submission.get("configuration", {})
    hashes = {
        "handoff_sha256": handoff.get("handoff_sha256", ""),
        "handoff_manifest_sha256": handoff.get("manifest_sha256", ""),
        "training_manifest_sha256": sha256_file(run_root / "manifests" / "training_manifest.json"),
        "generation_manifest_sha256": sha256_file(run_root / "manifests" / "generation_manifest.json"),
        "per_bin_audit_sha256": sha256_file(run_root / "reports" / "final" / "per_bin_audit.csv"),
    }
    report = {
        "status": "PASS" if not problems and final_summary.get("status") == "PASS" else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "commit_sha": submission.get("commit_sha", ""),
        "configuration": configuration,
        "job_ids": submission.get("job_ids", {}),
        "job_states": job_states,
        "expected_rows": final_summary.get("expected_rows", 0),
        "output_rows": final_summary.get("output_rows", 0),
        "missing_rows": final_summary.get("missing_rows", 0),
        "duplicate_rows": final_summary.get("duplicate_rows", 0),
        "same_length_mismatches": final_summary.get("same_length_mismatches", 0),
        "declared_length_mismatches": final_summary.get("declared_length_mismatches", 0),
        "provenance_mismatches": final_summary.get("provenance_mismatches", 0),
        "invalid_flag_rows": final_summary.get("invalid_flag_rows", 0),
        "generation_failed_rows": final_summary.get("generation_failed_rows", 0),
        "fallback_rows": final_summary.get("fallback_rows", 0),
        "scorer_ready_handoff": handoff.get("handoff_csv", ""),
        "handoff_manifest": handoff.get("manifest_path", ""),
        "training_manifest": handoff.get("training_manifest", ""),
        "generation_manifest": handoff.get("generation_manifest", ""),
        "per_bin_audit": handoff.get("per_bin_audit", ""),
        "hashes": hashes,
        "problems": problems,
    }
    report_dir = run_root / "reports" / "final"
    report_json = report_dir / "final_report.json"
    write_json(report_json, report)
    report_md = report_dir / "final_report.md"
    markdown = [
        "# Full-79 Additive K3 LSTM Final Report",
        "",
        f"- status: `{report['status']}`",
        f"- run root: `{run_root}`",
        f"- scorer-ready handoff: `{report['scorer_ready_handoff']}`",
        f"- handoff manifest: `{report['handoff_manifest']}`",
        f"- training manifest: `{report['training_manifest']}`",
        f"- generation manifest: `{report['generation_manifest']}`",
        f"- per-bin audit: `{report['per_bin_audit']}`",
        f"- expected/output rows: `{report['expected_rows']}` / `{report['output_rows']}`",
        f"- missing/duplicate rows: `{report['missing_rows']}` / `{report['duplicate_rows']}`",
        f"- same-length/declared-length mismatches: `{report['same_length_mismatches']}` / `{report['declared_length_mismatches']}`",
        f"- provenance mismatches: `{report['provenance_mismatches']}`",
        f"- generation failures/fallbacks: `{report['generation_failed_rows']}` / `{report['fallback_rows']}`",
        f"- commit SHA: `{report['commit_sha']}`",
        f"- configuration: `{json.dumps(configuration, sort_keys=True)}`",
        f"- hashes: `{json.dumps(hashes, sort_keys=True)}`",
        "",
        "## Slurm jobs",
        "",
    ]
    for label, job_id in report["job_ids"].items():
        state = job_states.get(label, {})
        markdown.append(
            f"- {label}: `{job_id}` state=`{state.get('state', 'MISSING')}` "
            f"exit=`{state.get('exit_code', 'MISSING')}`"
        )
    if problems:
        markdown.extend(["", "## Problems", "", *[f"- {problem}" for problem in problems]])
    temporary_md = report_md.with_name(f".{report_md.name}.tmp-{os.getpid()}")
    temporary_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    os.replace(temporary_md, report_md)
    if report["status"] != "PASS":
        raise RuntimeError(f"Final report failed: {problems}")
    (run_root / "FINAL_REPORT_READY").write_text("FINAL_REPORT_READY\n", encoding="utf-8")
    return report


def audit_smoke(run_root: str | Path, *, job_id: str = "") -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    manifest = run_root / "manifests" / "smoke_manifest.json"
    report = audit_lstm_output(manifest)
    report.update(
        {
            "job_id": job_id,
            "selected_condition": "k3 / 060-065 / same-length",
            "exact_wrapper": "slurm/run_full_79_lstm_cell.sbatch",
        }
    )
    report_dir = run_root / "reports" / "smoke"
    write_json(report_dir / "smoke_summary.json", report)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "smoke_report.md").write_text(
        "\n".join(
            [
                "# Full-79 LSTM GPU Smoke Report",
                "",
                f"- status: `{report['status']}`",
                "- selected condition: `k3 / 060-065 / same-length`",
                "- exact wrapper: `slurm/run_full_79_lstm_cell.sbatch`",
                f"- target rows: `{report['expected_rows']}`",
                f"- job id: `{job_id or 'unknown'}`",
                f"- output: `{report['output_csv']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if report["status"] != "PASS":
        raise RuntimeError(f"GPU smoke failed: {report['problems']}")
    (run_root / "SMOKE_PASSED").write_text("SMOKE_PASSED\n", encoding="utf-8")
    return report
