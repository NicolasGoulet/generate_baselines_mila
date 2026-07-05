"""Manifest parsing for baseline generation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_GENERATORS = ("random", "unigram", "bigram", "trigram")


@dataclass(frozen=True)
class BaselineManifest:
    run_id: str
    train_csv: Path
    target_csv: Path
    output_csv: Path
    text_column: str
    target_text_column: str
    id_columns: tuple[str, ...]
    carry_columns: tuple[str, ...] = ()
    age_bin_column: str | None = None
    age_bins: tuple[str, ...] = ()
    context_column: str | None = None
    context_tail_words: int = 0
    generators: tuple[str, ...] = DEFAULT_GENERATORS
    same_length: bool = True
    samples_per_target: int = 1
    seed: int = 13
    lowercase: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "BaselineManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if candidate.is_absolute():
                return candidate
            return (base / candidate).resolve()

        required = [
            "run_id",
            "train_csv",
            "target_csv",
            "output_csv",
            "text_column",
            "target_text_column",
            "id_columns",
        ]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Manifest is missing required fields: {', '.join(missing)}")

        generators = tuple(payload.get("generators", DEFAULT_GENERATORS))
        unknown = set(generators) - set(DEFAULT_GENERATORS)
        if unknown:
            raise ValueError(f"Unsupported generators: {sorted(unknown)}")

        samples_per_target = int(payload.get("samples_per_target", 1))
        if samples_per_target < 1:
            raise ValueError("samples_per_target must be >= 1")

        context_tail_words = int(payload.get("context_tail_words", 0))
        if context_tail_words < 0:
            raise ValueError("context_tail_words must be >= 0")

        return cls(
            run_id=str(payload["run_id"]),
            train_csv=resolve(payload["train_csv"]),
            target_csv=resolve(payload["target_csv"]),
            output_csv=resolve(payload["output_csv"]),
            text_column=str(payload["text_column"]),
            target_text_column=str(payload["target_text_column"]),
            id_columns=tuple(payload["id_columns"]),
            carry_columns=tuple(payload.get("carry_columns", ())),
            age_bin_column=payload.get("age_bin_column"),
            age_bins=tuple(payload.get("age_bins", ())),
            context_column=payload.get("context_column"),
            context_tail_words=context_tail_words,
            generators=generators,
            same_length=bool(payload.get("same_length", True)),
            samples_per_target=samples_per_target,
            seed=int(payload.get("seed", 13)),
            lowercase=bool(payload.get("lowercase", True)),
            raw=payload,
        )

    def validate_existing_inputs(self) -> None:
        missing = [str(path) for path in (self.train_csv, self.target_csv) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))

    @property
    def audit_json(self) -> Path:
        if self.output_csv.suffix == ".gz":
            stem = self.output_csv.with_suffix("").with_suffix("")
        else:
            stem = self.output_csv.with_suffix("")
        return stem.with_name(stem.name + ".audit.json")
