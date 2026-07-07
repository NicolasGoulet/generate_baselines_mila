"""Prepare production manifests from the strict naturalistic bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .io import open_text, sha256_file, write_json

DEFAULT_AGE_BINS = (
    {"label": "006-023", "start": 6, "end": 23},
    {"label": "024-029", "start": 24, "end": 29},
    {"label": "030-035", "start": 30, "end": 35},
    {"label": "036-041", "start": 36, "end": 41},
    {"label": "042-047", "start": 42, "end": 47},
    {"label": "048-053", "start": 48, "end": 53},
    {"label": "054-059", "start": 54, "end": 59},
    {"label": "060-065", "start": 60, "end": 65},
)

PROVENANCE_COLUMNS = (
    "row_uid",
    "dataset",
    "child_id",
    "source_group",
    "session_id",
    "age_months",
    "age_bin",
    "file",
    "line_no",
    "utt_id",
    "context_k1",
    "context_k2",
    "context_k3",
    "chi_utterance_clean",
)


def _stable_id(parts: list[str], *, length: int = 24) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or ""))


def _read_bundle_manifest(bundle_root: Path) -> list[dict[str, str]]:
    manifest_csv = bundle_root / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing bundle manifest: {manifest_csv}")
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_age_bins(bundle_root: Path) -> list[dict[str, Any]]:
    age_bins_json = bundle_root / "age_ngram_dicts" / "merged_early_006_023" / "age_bins.json"
    if age_bins_json.exists():
        payload = json.loads(age_bins_json.read_text(encoding="utf-8"))
        return list(payload["bins"])
    return [dict(item) for item in DEFAULT_AGE_BINS]


def _age_bin_for(age_months: str, age_bins: list[dict[str, Any]]) -> str:
    try:
        month = int(float(age_months))
    except (TypeError, ValueError):
        return ""
    for age_bin in age_bins:
        if int(age_bin["start"]) <= month <= int(age_bin["end"]):
            return str(age_bin["label"])
    return ""


def _resolve_bundle_path(bundle_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = bundle_root / path
    if candidate.exists():
        return candidate
    parts = path.parts
    for anchor in ("preprocessed_data", "age_ngram_dicts"):
        if anchor in parts:
            return bundle_root / Path(*parts[parts.index(anchor) :])
    return candidate


def _iter_scoring_rows(
    bundle_root: Path,
    *,
    datasets: set[str] | None,
    age_bins: list[dict[str, Any]],
):
    for manifest_row in _read_bundle_manifest(bundle_root):
        dataset = manifest_row.get("dataset", "")
        if datasets and dataset not in datasets:
            continue
        if manifest_row.get("child_scoring_ready") != "1":
            continue
        scoring_csv = _resolve_bundle_path(bundle_root, manifest_row.get("child_scoring_csv", ""))
        if not scoring_csv.exists():
            raise FileNotFoundError(f"Missing child scoring CSV: {scoring_csv}")
        with scoring_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"chi_utterance_clean", "age_months", "context_k3"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{scoring_csv} is missing required columns: {sorted(missing)}")
            for row in reader:
                utterance = row.get("chi_utterance_clean", "")
                if _token_count(utterance) == 0:
                    continue
                age_bin = _age_bin_for(row.get("age_months", ""), age_bins)
                if not age_bin:
                    continue
                row_uid = _stable_id(
                    [
                        row.get("dataset", ""),
                        row.get("child_id", ""),
                        row.get("session_id", ""),
                        row.get("file", ""),
                        row.get("line_no", ""),
                        row.get("utt_id", ""),
                    ]
                )
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
                }


def _write_rows(path: Path, rows, fieldnames: tuple[str, ...]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    duplicate_ids = 0
    count = 0
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row_uid = row["row_uid"]
            if row_uid in seen:
                duplicate_ids += 1
            seen.add(row_uid)
            writer.writerow(row)
            count += 1
    return count, duplicate_ids


def prepare_full_79_ngram_manifest(
    *,
    bundle_root: str | Path,
    output_root: str | Path,
    run_id: str = "full79_ngram_additive",
    samples_per_target: int = 1,
    seed: int = 13,
    context_column: str = "context_k3",
    context_tail_words: int = 16,
    datasets: set[str] | None = None,
) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    output_root = Path(output_root).resolve()
    age_bins = _load_age_bins(bundle_root)
    age_bin_labels = [str(item["label"]) for item in age_bins]

    input_csv = output_root / "inputs" / "full79_child_real_rows.csv.gz"
    output_csv = output_root / "generated" / "full79_ngram_generated.csv.gz"
    manifest_json = output_root / "manifests" / "full79_ngram_manifest.json"

    rows = _iter_scoring_rows(bundle_root, datasets=datasets, age_bins=age_bins)
    row_count, duplicate_row_ids = _write_rows(input_csv, rows, PROVENANCE_COLUMNS)
    if row_count == 0:
        raise ValueError("No scorable child rows were written from the bundle.")

    manifest = {
        "run_id": run_id,
        "train_csv": str(input_csv),
        "target_csv": str(input_csv),
        "output_csv": str(output_csv),
        "text_column": "chi_utterance_clean",
        "target_text_column": "chi_utterance_clean",
        "id_columns": ["row_uid"],
        "carry_columns": [
            "dataset",
            "child_id",
            "source_group",
            "session_id",
            "age_months",
            "age_bin",
            "file",
            "line_no",
            "utt_id",
            "context_k1",
            "context_k2",
            "context_k3",
        ],
        "age_bin_column": "age_bin",
        "age_bins": age_bin_labels,
        "context_column": context_column,
        "context_tail_words": context_tail_words,
        "generators": ["random", "unigram", "bigram", "trigram"],
        "same_length": True,
        "samples_per_target": samples_per_target,
        "seed": seed,
        "training_scope": "strict_naturalistic_full79",
    }
    write_json(manifest_json, manifest)

    audit = {
        "status": "ok",
        "bundle_root": str(bundle_root),
        "output_root": str(output_root),
        "manifest_json": str(manifest_json),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "row_count": row_count,
        "duplicate_row_ids": duplicate_row_ids,
        "age_bins": age_bin_labels,
        "input_sha256": sha256_file(input_csv),
        "datasets": sorted(datasets) if datasets else "ALL",
    }
    audit_json = output_root / "manifests" / "full79_ngram_manifest.audit.json"
    write_json(audit_json, audit)
    audit["audit_json"] = str(audit_json)
    return audit
