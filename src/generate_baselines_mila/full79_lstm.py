"""Prepare and audit the strict-naturalistic full-79 LSTM production run."""

from __future__ import annotations

import csv
import json
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
from .io import iter_csv_dicts, open_text, sha256_file, write_json
from .manifest import BaselineManifest
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
    carry_columns = [column for column in FULL79_LSTM_COLUMNS if column not in {"row_uid", "chi_utterance_clean"}]
    for context in contexts:
        for age_index, age_label in enumerate(age_labels):
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
        return {"status": "FAIL", "manifest": str(manifest_path), "problems": ["missing output CSV"]}
    output_audit: dict[str, Any] = {}
    if not manifest.audit_json.exists():
        problems.append("missing output audit JSON")
    else:
        try:
            output_audit = json.loads(manifest.audit_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"invalid output audit JSON: {exc}")

    target_lengths: dict[str, int] = {}
    for row in iter_csv_dicts(manifest.target_csv):
        row_uid = row.get("row_uid", "")
        if not row_uid:
            problems.append("target row missing row_uid")
            continue
        if row_uid in target_lengths:
            problems.append(f"duplicate target row_uid: {row_uid}")
            continue
        target_lengths[row_uid] = _token_count(row.get(manifest.target_text_column, ""))

    seen: set[tuple[str, str]] = set()
    output_count = 0
    empty_generated = 0
    length_mismatches = 0
    unknown_ids = 0
    wrong_source = 0
    expected_source = str(manifest.raw.get("source_model", "lstm"))
    for row in iter_csv_dicts(manifest.output_csv):
        output_count += 1
        row_uid = row.get("row_uid", "")
        sample_index = row.get("sample_index", "")
        key = (row_uid, sample_index)
        if key in seen:
            problems.append(f"duplicate output key: {key}")
        seen.add(key)
        if row_uid not in target_lengths:
            unknown_ids += 1
            continue
        generated = row.get("generated_utterance", "")
        if _token_count(generated) == 0:
            empty_generated += 1
        if _token_count(generated) != target_lengths[row_uid]:
            length_mismatches += 1
        if row.get("source_model") != expected_source:
            wrong_source += 1

    expected_count = len(target_lengths) * manifest.samples_per_target
    manifest_expected = int(manifest.raw.get("expected_target_rows", len(target_lengths)))
    if len(target_lengths) != manifest_expected:
        problems.append(f"target rows {len(target_lengths)} != manifest expected {manifest_expected}")
    if output_count != expected_count:
        problems.append(f"output rows {output_count} != expected {expected_count}")
    if empty_generated:
        problems.append(f"empty generated rows: {empty_generated}")
    if length_mismatches:
        problems.append(f"same-length mismatches: {length_mismatches}")
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
        "empty_generated_rows": empty_generated,
        "same_length_mismatches": length_mismatches,
        "checkpoint_count": len(checkpoints),
        "output_sha256": output_sha256,
        "problems": problems,
    }


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
        "cells": reports,
    }
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
