"""I/O helpers with gzip support and checksum audits."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def open_text(path: str | Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def iter_csv_dicts(path: str | Path) -> Iterator[dict[str, str]]:
    with open_text(path, "rt") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def write_csv_dicts(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
