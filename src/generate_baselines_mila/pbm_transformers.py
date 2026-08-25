"""Contracts and audits for PBM-held-out small-Transformer generation.

The runtime intentionally keeps data preparation, model generation, and
Mistral scoring separate.  This module validates the immutable training
handoff, builds exact per-architecture/per-age-cell manifests, and audits the
generated response handoff before it leaves this repository.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import sha256_file, write_csv_dicts, write_json


PBM_DATASETS = frozenset({"Brown", "Manchester", "Providence"})
AGE_BINS = (
    (6, 23, "006-023"),
    (24, 29, "024-029"),
    (30, 35, "030-035"),
    (36, 41, "036-041"),
    (42, 47, "042-047"),
    (48, 53, "048-053"),
    (54, 59, "054-059"),
    (60, 65, "060-065"),
)
SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>", "<turn>", "<child>")

# Both models use a tokenizer learned only from non-PBM training rows.  The
# LLaMA dimensions reproduce the published 58M BabyLlama student architecture,
# but this condition is trained directly and therefore is not called a
# reproduction of the distilled BabyLlama training procedure.
ARCHITECTURES: dict[str, dict[str, Any]] = {
    "babyllama_sized_llama_58m": {
        "display_name": "BabyLlama-sized LLaMA (direct training)",
        "family": "decoder_only",
        "initialization": "from_scratch",
        "teacher_distillation": False,
        "vocab_size": 16000,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 8,
        "max_position_embeddings": 256,
        "tie_word_embeddings": False,
        "nominal_parameter_count": 58343936,
    },
    "t5_58m": {
        "display_name": "parameter-matched T5-small-style encoder-decoder",
        "family": "encoder_decoder",
        "initialization": "from_scratch",
        "teacher_distillation": False,
        "vocab_size": 16000,
        "d_model": 512,
        "d_ff": 2560,
        "d_kv": 64,
        "num_layers": 6,
        "num_decoder_layers": 6,
        "num_heads": 8,
        "relative_attention_num_buckets": 32,
        "tie_word_embeddings": True,
        "nominal_parameter_count": 58540544,
    },
}

REQUIRED_EXAMPLE_FIELDS = frozenset(
    {
        "example_id",
        "dataset",
        "child_id",
        "session_id",
        "file",
        "line_no",
        "reference_line",
        "age_months",
        "target_age_bin",
        "context_text",
        "target_text",
        "target_word_count",
        "split",
    }
)

GENERATED_COLUMNS = [
    "generated_id",
    "example_id",
    "dataset",
    "child_id",
    "session_id",
    "file",
    "line_no",
    "reference_line",
    "age_months",
    "target_age_bin",
    "context_text",
    "context_k3",
    "real_target_text",
    "real_target_word_count",
    "source_model",
    "sample_index",
    "generated_utterance",
    "generated_word_count",
    "generated_token_count",
    "eos_reached",
    "max_token_censored",
    "generation_seed",
    "max_new_tokens",
    "temperature",
    "top_p",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield value


def _write_jsonl_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with gzip.GzipFile(filename=str(temporary), mode="wb", mtime=0) as raw:
            with __import__("io").TextIOWrapper(raw, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _stable_id(parts: Sequence[object]) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _expected_handoff_paths(handoff_root: Path) -> list[Path]:
    paths = [handoff_root / "examples" / "train_all_ages.jsonl.gz"]
    for _, end, label in AGE_BINS:
        base = handoff_root / "cumulative_age_models" / f"through_{end:03d}_months"
        paths.extend(base / f"{split}.jsonl.gz" for split in ("train", "validation", "development"))
        paths.append(handoff_root / "pbm_target_age_bins" / f"pbm_{label}.jsonl.gz")
    return paths


def _audit_jsonl(
    path: Path,
    *,
    expected_age_bin: str | None = None,
    expected_role: str,
) -> dict[str, Any]:
    rows = 0
    ids: set[str] = set()
    duplicate_ids = 0
    datasets: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        missing = REQUIRED_EXAMPLE_FIELDS - set(row)
        if missing:
            raise ValueError(f"{path} row {rows + 1} missing fields: {sorted(missing)}")
        example_id = str(row["example_id"])
        if example_id in ids:
            duplicate_ids += 1
        ids.add(example_id)
        dataset = str(row["dataset"])
        datasets[dataset] += 1
        splits[str(row["split"])] += 1
        if expected_age_bin and str(row["target_age_bin"]) != expected_age_bin:
            raise ValueError(
                f"{path} contains target age bin {row['target_age_bin']}, expected {expected_age_bin}"
            )
        rows += 1
    if rows == 0:
        raise ValueError(f"Empty transformer input: {path}")
    if duplicate_ids:
        raise ValueError(f"Duplicate example ids in {path}: {duplicate_ids}")
    observed = set(datasets)
    if expected_role == "train" and observed & PBM_DATASETS:
        raise ValueError(f"PBM leakage in training input {path}: {sorted(observed & PBM_DATASETS)}")
    if expected_role == "pbm" and not observed <= PBM_DATASETS:
        raise ValueError(f"Non-PBM dataset in evaluation input {path}: {sorted(observed - PBM_DATASETS)}")
    return {
        "path": str(path),
        "rows": rows,
        "datasets": dict(sorted(datasets.items())),
        "splits": dict(sorted(splits.items())),
        "sha256": sha256_file(path),
    }


def _cell_manifest(
    *,
    run_root: Path,
    architecture_id: str,
    age_bin: str,
    train_file: Path,
    validation_file: Path,
    development_file: Path,
    target_file: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_censored_fraction: float,
    smoke: bool,
) -> dict[str, Any]:
    label = "smoke" if smoke else age_bin
    root = run_root / ("smoke" if smoke else "cells") / architecture_id / label
    return {
        "schema_version": 1,
        "run_root": str(run_root.resolve()),
        "architecture_id": architecture_id,
        "architecture": ARCHITECTURES[architecture_id],
        "age_bin": age_bin,
        "train_file": str(train_file.resolve()),
        "validation_file": str(validation_file.resolve()),
        "development_file": str(development_file.resolve()),
        "target_file": str(target_file.resolve()),
        "artifact_dir": str((root / "model").resolve()),
        "output_file": str((root / "generated_responses.jsonl.gz").resolve()),
        "audit_file": str((root / "cell_audit.json").resolve()),
        "seed": seed,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "max_sequence_tokens": 256,
        "max_encoder_tokens": 192,
        "max_target_tokens": 128,
        "max_new_tokens": max_new_tokens,
        "samples_per_target": 1,
        "generation_variant": "unconstrained_length",
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_censored_fraction": max_censored_fraction,
        "tokenizer_dir": str((run_root / "tokenizer").resolve()),
        "smoke": smoke,
    }


def prepare_pbm_transformer_run(
    *,
    handoff_root: str | Path,
    run_root: str | Path,
    seed: int = 20260825,
    max_epochs: int = 10,
    patience: int = 2,
    per_device_batch_size: int = 16,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 5e-4,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_censored_fraction: float = 0.001,
    smoke_train_examples: int = 1024,
    smoke_target_rows: int = 25,
) -> dict[str, Any]:
    """Audit an upstream handoff and write immutable production cell manifests."""
    handoff_root = Path(handoff_root).resolve()
    run_root = Path(run_root).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"Fresh run root required: {run_root}")
    if max_epochs <= 0 or patience < 0:
        raise ValueError("Invalid epoch or early-stopping configuration")
    if not 0.0 <= max_censored_fraction <= 1.0:
        raise ValueError("max_censored_fraction must be in [0, 1]")
    if smoke_train_examples <= 0 or smoke_target_rows <= 0:
        raise ValueError("Smoke row counts must be positive")
    marker = handoff_root / "BUILD_COMPLETE_AND_AUDITED"
    if not marker.is_file():
        raise FileNotFoundError(f"Missing upstream completion marker: {marker}")
    upstream_manifest_path = handoff_root / "manifest.json"
    upstream = _read_json(upstream_manifest_path)
    if upstream.get("status") != "complete" or upstream.get("fatal_issues"):
        raise ValueError("Upstream handoff manifest is not complete and clean")
    if set(upstream.get("held_out_evaluation_datasets", [])) != PBM_DATASETS:
        raise ValueError("Upstream held-out evaluation set is not exactly PBM")
    if upstream.get("pbm_dataset_overlap_with_training"):
        raise ValueError("Upstream manifest reports PBM training leakage")
    hashes = upstream.get("output_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Upstream manifest has no output_sha256 map")

    run_root.mkdir(parents=True, exist_ok=True)
    input_audits: list[dict[str, Any]] = []
    for path in _expected_handoff_paths(handoff_root):
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream handoff file: {path}")
        relative = str(path.relative_to(handoff_root))
        expected_hash = hashes.get(relative)
        observed_hash = sha256_file(path)
        if expected_hash != observed_hash:
            raise ValueError(
                f"SHA-256 mismatch for {relative}: expected {expected_hash}, observed {observed_hash}"
            )

    # Audit every distinct production input, including dataset isolation and
    # exact target-bin membership.  The all-age tokenizer source is train-only.
    tokenizer_source = handoff_root / "examples" / "train_all_ages.jsonl.gz"
    input_audits.append(_audit_jsonl(tokenizer_source, expected_role="train"))
    train_audits: dict[str, dict[str, Any]] = {}
    validation_audits: dict[str, dict[str, Any]] = {}
    development_audits: dict[str, dict[str, Any]] = {}
    target_audits: dict[str, dict[str, Any]] = {}
    for _, end, label in AGE_BINS:
        base = handoff_root / "cumulative_age_models" / f"through_{end:03d}_months"
        train = base / "train.jsonl.gz"
        validation = base / "validation.jsonl.gz"
        development = base / "development.jsonl.gz"
        target = handoff_root / "pbm_target_age_bins" / f"pbm_{label}.jsonl.gz"
        train_audits[label] = _audit_jsonl(train, expected_role="train")
        validation_audits[label] = _audit_jsonl(validation, expected_role="train")
        development_audits[label] = _audit_jsonl(development, expected_role="train")
        target_audits[label] = _audit_jsonl(target, expected_age_bin=label, expected_role="pbm")
        if development_audits[label]["rows"] != train_audits[label]["rows"] + validation_audits[label]["rows"]:
            raise ValueError(f"Development rows are not train plus validation for {label}")
        input_audits.extend(
            [train_audits[label], validation_audits[label], development_audits[label], target_audits[label]]
        )

    smoke_input_dir = run_root / "inputs" / "smoke"
    last_label = AGE_BINS[-1][2]
    smoke_train_rows = []
    for row in _iter_jsonl(Path(train_audits[last_label]["path"])):
        smoke_train_rows.append(row)
        if len(smoke_train_rows) >= smoke_train_examples:
            break
    smoke_validation_rows = []
    for row in _iter_jsonl(Path(validation_audits[last_label]["path"])):
        smoke_validation_rows.append(row)
        if len(smoke_validation_rows) >= min(smoke_target_rows, smoke_train_examples):
            break
    smoke_target_data = []
    for row in _iter_jsonl(Path(target_audits[last_label]["path"])):
        smoke_target_data.append(row)
        if len(smoke_target_data) >= smoke_target_rows:
            break
    smoke_train = smoke_input_dir / "train.jsonl.gz"
    smoke_validation = smoke_input_dir / "validation.jsonl.gz"
    smoke_development = smoke_input_dir / "development.jsonl.gz"
    smoke_target = smoke_input_dir / "targets.jsonl.gz"
    _write_jsonl_gz(smoke_train, smoke_train_rows)
    _write_jsonl_gz(smoke_validation, smoke_validation_rows)
    _write_jsonl_gz(smoke_development, [*smoke_train_rows, *smoke_validation_rows])
    _write_jsonl_gz(smoke_target, smoke_target_data)

    manifest_dir = run_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    final_cells: list[dict[str, Any]] = []
    smoke_cells: list[dict[str, Any]] = []
    for architecture_id in ARCHITECTURES:
        smoke_payload = _cell_manifest(
            run_root=run_root,
            architecture_id=architecture_id,
            age_bin=last_label,
            train_file=smoke_train,
            validation_file=smoke_validation,
            development_file=smoke_development,
            target_file=smoke_target,
            seed=seed,
            max_epochs=1,
            patience=0,
            per_device_batch_size=min(per_device_batch_size, 2),
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            max_new_tokens=min(max_new_tokens, 16),
            temperature=temperature,
            top_p=top_p,
            max_censored_fraction=1.0,
            smoke=True,
        )
        smoke_payload.update(
            {
                "train_rows": len(smoke_train_rows),
                "validation_rows": len(smoke_validation_rows),
                "development_rows": len(smoke_train_rows) + len(smoke_validation_rows),
                "target_rows": len(smoke_target_data),
            }
        )
        smoke_manifest = manifest_dir / f"smoke_{architecture_id}.json"
        write_json(smoke_manifest, smoke_payload)
        smoke_cells.append(
            {
                "index": len(smoke_cells),
                "architecture_id": architecture_id,
                "age_bin": last_label,
                "manifest_path": str(smoke_manifest.resolve()),
            }
        )
        for _, end, label in AGE_BINS:
            base = handoff_root / "cumulative_age_models" / f"through_{end:03d}_months"
            payload = _cell_manifest(
                run_root=run_root,
                architecture_id=architecture_id,
                age_bin=label,
                train_file=base / "train.jsonl.gz",
                validation_file=base / "validation.jsonl.gz",
                development_file=base / "development.jsonl.gz",
                target_file=handoff_root / "pbm_target_age_bins" / f"pbm_{label}.jsonl.gz",
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
                per_device_batch_size=per_device_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_censored_fraction=max_censored_fraction,
                smoke=False,
            )
            payload.update(
                {
                    "train_rows": train_audits[label]["rows"],
                    "validation_rows": validation_audits[label]["rows"],
                    "development_rows": development_audits[label]["rows"],
                    "target_rows": target_audits[label]["rows"],
                }
            )
            manifest_path = manifest_dir / f"cell_{architecture_id}_{label}.json"
            write_json(manifest_path, payload)
            final_cells.append(
                {
                    "index": len(final_cells),
                    "architecture_id": architecture_id,
                    "age_bin": label,
                    "manifest_path": str(manifest_path.resolve()),
                    "train_rows": train_audits[label]["rows"],
                    "validation_rows": validation_audits[label]["rows"],
                    "target_rows": target_audits[label]["rows"],
                }
            )
    write_json(run_root / "cell_index.json", {"cells": final_cells})
    write_json(run_root / "smoke_cell_index.json", {"cells": smoke_cells})
    config = {
        "status": "PASS",
        "prepared_at": _utc_now(),
        "handoff_root": str(handoff_root),
        "handoff_manifest_sha256": sha256_file(upstream_manifest_path),
        "tokenizer_source": str(tokenizer_source),
        "tokenizer_source_sha256": sha256_file(tokenizer_source),
        "tokenizer_vocab_size": 16000,
        "special_tokens": list(SPECIAL_TOKENS),
        "architectures": ARCHITECTURES,
        "age_bins": [label for _, _, label in AGE_BINS],
        "training_policy": "select the epoch on train plus child-disjoint validation, then reinitialize and refit for that many epochs on development (train plus validation); PBM is evaluated once after refitting",
        "generation_policy": "one unconstrained stochastic response per PBM target; censoring at max_new_tokens is measured and gated",
        "final_cell_count": len(final_cells),
        "smoke_cell_count": len(smoke_cells),
        "input_audits": input_audits,
    }
    write_json(run_root / "preparation_audit.json", config)
    (run_root / "PREPARED_AND_AUDITED").write_text("PASS\n", encoding="utf-8")
    return config


def load_cell_index(run_root: str | Path, *, smoke: bool = False) -> list[dict[str, Any]]:
    name = "smoke_cell_index.json" if smoke else "cell_index.json"
    payload = _read_json(Path(run_root) / name)
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Invalid cell index: {Path(run_root) / name}")
    return [dict(cell) for cell in cells]


def manifest_for_cell(run_root: str | Path, index: int, *, smoke: bool = False) -> Path:
    cells = load_cell_index(run_root, smoke=smoke)
    if index < 0 or index >= len(cells):
        raise IndexError(f"Cell index {index} outside 0-{len(cells) - 1}")
    return Path(str(cells[index]["manifest_path"]))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def audit_cell_output(manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    target_path = Path(manifest["target_file"])
    output_path = Path(manifest["output_file"])
    artifact_dir = Path(manifest["artifact_dir"])
    problems: list[str] = []
    target_ids = [str(row["example_id"]) for row in _iter_jsonl(target_path)]
    expected = Counter(target_ids)
    observed: Counter[str] = Counter()
    generated_ids: set[str] = set()
    duplicate_generated_ids = 0
    empty_rows = 0
    censored_rows = 0
    output_rows = 0
    if not output_path.is_file():
        problems.append(f"Missing generated output: {output_path}")
    else:
        for row in _iter_jsonl(output_path):
            output_rows += 1
            missing = set(GENERATED_COLUMNS) - set(row)
            if missing:
                problems.append(f"Generated row missing columns: {sorted(missing)}")
                continue
            generated_id = str(row["generated_id"])
            if generated_id in generated_ids:
                duplicate_generated_ids += 1
            generated_ids.add(generated_id)
            observed[str(row["example_id"])] += 1
            if str(row["source_model"]) != manifest["architecture_id"]:
                problems.append("source_model does not match architecture_id")
            if str(row["dataset"]) not in PBM_DATASETS:
                problems.append("Generated output contains non-PBM target")
            if not str(row["generated_utterance"]).strip():
                empty_rows += 1
            if _as_bool(row["max_token_censored"]):
                censored_rows += 1
    samples = int(manifest["samples_per_target"])
    expected_counts = Counter({key: count * samples for key, count in expected.items()})
    missing_examples = sum((expected_counts - observed).values())
    extra_examples = sum((observed - expected_counts).values())
    if missing_examples:
        problems.append(f"Missing generated responses: {missing_examples}")
    if extra_examples:
        problems.append(f"Unexpected generated responses: {extra_examples}")
    if duplicate_generated_ids:
        problems.append(f"Duplicate generated ids: {duplicate_generated_ids}")
    if empty_rows:
        problems.append(f"Empty generated responses: {empty_rows}")
    censored_fraction = censored_rows / output_rows if output_rows else 0.0
    if censored_fraction > float(manifest["max_censored_fraction"]):
        problems.append(
            f"Max-token censoring fraction {censored_fraction:.6f} exceeds "
            f"{float(manifest['max_censored_fraction']):.6f}"
        )
    required_artifacts = (
        artifact_dir / "config.json",
        artifact_dir / "model.safetensors",
        artifact_dir / "training_report.json",
    )
    for path in required_artifacts:
        if not path.is_file():
            problems.append(f"Missing model artifact: {path}")
    parameter_count = None
    report_path = artifact_dir / "training_report.json"
    if report_path.is_file():
        training = _read_json(report_path)
        parameter_count = training.get("parameter_count")
        if training.get("status") != "PASS":
            problems.append("Training report is not PASS")
        if not isinstance(parameter_count, int) or not 45_000_000 <= parameter_count <= 70_000_000:
            problems.append(f"Parameter count outside frozen small-model range: {parameter_count}")
    report = {
        "status": "PASS" if not problems else "FAIL",
        "manifest_path": str(manifest_path),
        "architecture_id": manifest["architecture_id"],
        "age_bin": manifest["age_bin"],
        "smoke": bool(manifest["smoke"]),
        "expected_rows": sum(expected_counts.values()),
        "output_rows": output_rows,
        "missing_examples": missing_examples,
        "extra_examples": extra_examples,
        "duplicate_generated_ids": duplicate_generated_ids,
        "empty_generated_rows": empty_rows,
        "max_token_censored_rows": censored_rows,
        "max_token_censored_fraction": censored_fraction,
        "parameter_count": parameter_count,
        "output_sha256": sha256_file(output_path) if output_path.is_file() else None,
        "problems": sorted(set(problems)),
    }
    write_json(Path(manifest["audit_file"]), report)
    return report


def audit_smoke(run_root: str | Path, *, job_id: str = "") -> dict[str, Any]:
    run_root = Path(run_root)
    reports = [audit_cell_output(cell["manifest_path"]) for cell in load_cell_index(run_root, smoke=True)]
    problems = [problem for report in reports for problem in report["problems"]]
    status = "PASS" if not problems and len(reports) == len(ARCHITECTURES) else "FAIL"
    payload = {
        "status": status,
        "job_id": job_id,
        "exact_production_wrapper": "slurm/run_pbm_transformer_cell.sbatch",
        "architectures": [report["architecture_id"] for report in reports],
        "cell_reports": reports,
        "problems": problems,
    }
    report_dir = run_root / "reports" / "smoke"
    write_json(report_dir / "smoke_summary.json", payload)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "smoke_report.md").write_text(
        "# PBM Transformer GPU Smoke\n\n"
        f"Status: **{status}**\n\n"
        f"Architectures: {', '.join(payload['architectures'])}\n\n"
        f"Problems: {len(problems)}\n",
        encoding="utf-8",
    )
    if status == "PASS":
        (run_root / "SMOKE_PASSED").write_text("PASS\n", encoding="utf-8")
    else:
        raise RuntimeError(f"Transformer smoke audit failed: {problems}")
    return payload


def parse_indices(value: str) -> list[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid range: {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    if any(index < 0 or index >= 16 for index in result):
        raise ValueError("Production indices must be within 0-15")
    return sorted(result)


def audit_pbm_transformer_run(
    run_root: str | Path,
    *,
    stage: str,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root)
    cells = load_cell_index(run_root)
    if indices is None:
        indices = list(range(len(cells)))
    reports = [audit_cell_output(cells[index]["manifest_path"]) for index in indices]
    problems = [problem for report in reports for problem in report["problems"]]
    if problems:
        raise RuntimeError(f"PBM transformer {stage} audit failed: {problems[:10]}")
    if stage == "wave1":
        if list(indices) != list(range(8)):
            raise ValueError("wave1 must audit cells 0-7")
        (run_root / "WAVE1_READY").write_text("PASS\n", encoding="utf-8")
    elif stage == "wave2":
        if list(indices) != list(range(8, 16)):
            raise ValueError("wave2 must audit cells 8-15")
        (run_root / "WAVE2_READY").write_text("PASS\n", encoding="utf-8")
    elif stage != "final":
        raise ValueError(f"Unknown audit stage: {stage}")

    summary: dict[str, Any] = {
        "status": "PASS",
        "stage": stage,
        "cell_count": len(reports),
        "output_rows": sum(report["output_rows"] for report in reports),
        "duplicate_generated_ids": sum(report["duplicate_generated_ids"] for report in reports),
        "max_token_censored_rows": sum(report["max_token_censored_rows"] for report in reports),
        "cell_reports": reports,
        "problems": [],
    }
    if stage == "final":
        if sorted(indices) != list(range(16)):
            raise ValueError("Final audit must cover all 16 cells")
        handoff_dir = run_root / "handoff"
        handoff_csv = handoff_dir / "pbm_transformer_responses_scorer_ready.csv.gz"

        def all_rows() -> Iterator[dict[str, Any]]:
            for cell in cells:
                manifest = _read_json(Path(cell["manifest_path"]))
                yield from _iter_jsonl(Path(manifest["output_file"]))

        output_rows = write_csv_dicts(handoff_csv, all_rows(), fieldnames=GENERATED_COLUMNS)
        if output_rows != summary["output_rows"]:
            raise RuntimeError("Merged handoff row count differs from audited cell rows")
        manifest_payload = {
            "status": "PASS",
            "schema_version": 1,
            "created_at": _utc_now(),
            "architectures": ARCHITECTURES,
            "age_bins": [label for _, _, label in AGE_BINS],
            "samples_per_target": 1,
            "generation_variant": "unconstrained_length",
            "handoff_csv": str(handoff_csv),
            "handoff_rows": output_rows,
            "handoff_sha256": sha256_file(handoff_csv),
            "contexts_for_scoring": ["k0", "k3"],
            "score_target_column": "generated_utterance",
            "context_column": "context_k3",
            "cell_output_sha256": {
                f"{report['architecture_id']}::{report['age_bin']}": report["output_sha256"]
                for report in reports
            },
        }
        write_json(handoff_dir / "manifest.json", manifest_payload)
        summary.update(
            {
                "handoff_csv": str(handoff_csv),
                "handoff_manifest": str(handoff_dir / "manifest.json"),
                "handoff_sha256": manifest_payload["handoff_sha256"],
            }
        )
        write_json(run_root / "reports" / "final" / "final_audit.json", summary)
        (run_root / "COMPLETE_AND_AUDITED").write_text("PASS\n", encoding="utf-8")
    else:
        write_json(run_root / "reports" / stage / f"{stage}_audit.json", summary)
    return summary


def finalize_report(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root)
    if not (run_root / "COMPLETE_AND_AUDITED").is_file():
        raise FileNotFoundError("Final scientific audit marker is absent")
    metadata_path_text = os.environ.get("SUBMISSION_METADATA_JSON", "")
    metadata = _read_json(Path(metadata_path_text)) if metadata_path_text else {}
    job_ids = metadata.get("job_ids", {})
    job_states: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    if job_ids:
        ids = ",".join(str(value) for value in job_ids.values())
        completed = subprocess.run(
            ["sacct", "-n", "-P", "-j", ids, "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            problems.append(f"sacct failed: {completed.stderr.strip()}")
        else:
            for line in completed.stdout.splitlines():
                fields = line.split("|")
                if len(fields) < 7 or "." in fields[0]:
                    continue
                job_states[fields[0]] = {
                    "name": fields[1],
                    "state": fields[2],
                    "exit_code": fields[3],
                    "elapsed": fields[4],
                    "start": fields[5],
                    "end": fields[6],
                }
            for stage, job_id in job_ids.items():
                state = job_states.get(str(job_id))
                if not state or state["state"] != "COMPLETED" or state["exit_code"] != "0:0":
                    problems.append(f"Job {stage} ({job_id}) is not COMPLETED 0:0")
    handoff = _read_json(run_root / "handoff" / "manifest.json")
    payload = {
        "status": "PASS" if not problems else "FAIL",
        "commit_sha": metadata.get("commit_sha", ""),
        "job_states": job_states,
        "handoff": handoff,
        "problems": problems,
    }
    write_json(run_root / "reports" / "final" / "final_state_report.json", payload)
    if problems:
        raise RuntimeError(f"Final state report failed: {problems}")
    (run_root / "FINAL_REPORT_READY").write_text("PASS\n", encoding="utf-8")
    return payload
