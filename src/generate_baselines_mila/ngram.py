"""Count-based random, unigram, bigram, and trigram baseline generation."""

from __future__ import annotations

import bisect
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .io import iter_csv_dicts, sha256_file, write_csv_dicts, write_json
from .manifest import BaselineManifest
from .tokenize import tokenize_words

BOS = "<bos>"


class WeightedSampler:
    def __init__(self, counts: Counter[str], rng: random.Random):
        if not counts:
            raise ValueError("Cannot sample from empty counts")
        self.items: list[str] = []
        self.cumulative: list[int] = []
        total = 0
        for item, count in sorted(counts.items()):
            if count <= 0:
                continue
            total += int(count)
            self.items.append(item)
            self.cumulative.append(total)
        if not self.items:
            raise ValueError("Cannot sample from empty positive counts")
        self.total = total
        self.rng = rng

    def sample(self) -> str:
        index = bisect.bisect_left(self.cumulative, self.rng.randint(1, self.total))
        return self.items[index]


class CountGenerator:
    def __init__(
        self,
        *,
        order: int,
        vocabulary: list[str],
        unigram_counts: Counter[str],
        ngram_counts: dict[tuple[str, ...], Counter[str]],
        rng: random.Random,
    ) -> None:
        self.order = order
        self.vocabulary = vocabulary
        self.unigram_sampler = WeightedSampler(unigram_counts, rng)
        self.ngram_counts = ngram_counts
        self.rng = rng

    def random_word(self) -> str:
        return self.rng.choice(self.vocabulary)

    def next_word(self, history: list[str]) -> str:
        if self.order == 1:
            return self.unigram_sampler.sample()

        max_context = self.order - 1
        for context_size in range(max_context, 0, -1):
            context = tuple(history[-context_size:])
            counts = self.ngram_counts.get(context)
            if counts:
                return WeightedSampler(counts, self.rng).sample()
        return self.unigram_sampler.sample()

    def generate(self, length: int, *, context_text: str = "", lowercase: bool = True) -> list[str]:
        if length <= 0:
            return []
        context_tokens = tokenize_words(context_text, lowercase=lowercase)
        history = [BOS] * max(1, self.order - 1) + context_tokens
        generated: list[str] = []
        for _ in range(length):
            word = self.next_word(history)
            generated.append(word)
            history.append(word)
        return generated


def build_count_generator(
    rows: list[dict[str, str]],
    *,
    text_column: str,
    context_column: str | None,
    context_tail_words: int,
    order: int,
    rng: random.Random,
    lowercase: bool,
) -> CountGenerator:
    unigram_counts: Counter[str] = Counter()
    ngram_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    for row in rows:
        tokens = tokenize_words(row.get(text_column, ""), lowercase=lowercase)
        if not tokens:
            continue
        unigram_counts.update(tokens)
        context_tokens: list[str] = []
        if context_column:
            context_tokens = tokenize_words(row.get(context_column, ""), lowercase=lowercase)
            if context_tail_words:
                context_tokens = context_tokens[-context_tail_words:]
        history = [BOS] * max(1, order - 1) + context_tokens
        for token in tokens:
            for context_size in range(1, max(1, order)):
                context = tuple(history[-context_size:])
                ngram_counts[context][token] += 1
            history.append(token)

    if not unigram_counts:
        raise ValueError("Training data produced no word tokens")

    return CountGenerator(
        order=order,
        vocabulary=sorted(unigram_counts),
        unigram_counts=unigram_counts,
        ngram_counts=dict(ngram_counts),
        rng=rng,
    )


def _age_index(manifest: BaselineManifest, row: dict[str, str]) -> int | None:
    if not manifest.age_bin_column or not manifest.age_bins:
        return None
    value = row.get(manifest.age_bin_column, "")
    try:
        return manifest.age_bins.index(value)
    except ValueError:
        return None


def _training_rows_for_target_bin(
    train_rows: list[dict[str, str]],
    *,
    manifest: BaselineManifest,
    target_age_index: int | None,
) -> list[dict[str, str]]:
    if target_age_index is None or not manifest.age_bin_column or not manifest.age_bins:
        return train_rows
    allowed = set(manifest.age_bins[: target_age_index + 1])
    scoped = [row for row in train_rows if row.get(manifest.age_bin_column, "") in allowed]
    return scoped or train_rows


def _build_models_for_age_bin(
    train_rows: list[dict[str, str]],
    *,
    manifest: BaselineManifest,
    target_age_index: int | None,
) -> dict[str, CountGenerator]:
    scoped_rows = _training_rows_for_target_bin(
        train_rows,
        manifest=manifest,
        target_age_index=target_age_index,
    )
    models: dict[str, CountGenerator] = {}
    for generator in manifest.generators:
        order = {"random": 1, "unigram": 1, "bigram": 2, "trigram": 3}[generator]
        seed_offset = 101 * (target_age_index if target_age_index is not None else 0)
        seed_offset += {"random": 1, "unigram": 2, "bigram": 3, "trigram": 4}[generator]
        rng = random.Random(manifest.seed + seed_offset)
        models[generator] = build_count_generator(
            scoped_rows,
            text_column=manifest.text_column,
            context_column=manifest.context_column,
            context_tail_words=manifest.context_tail_words,
            order=order,
            rng=rng,
            lowercase=manifest.lowercase,
        )
    return models


def generate_rows(manifest: BaselineManifest) -> Iterator[dict[str, Any]]:
    train_rows = list(iter_csv_dicts(manifest.train_csv))
    model_cache: dict[int | None, dict[str, CountGenerator]] = {}

    for target_row in iter_csv_dicts(manifest.target_csv):
        target_age_index = _age_index(manifest, target_row)
        if target_age_index not in model_cache:
            model_cache[target_age_index] = _build_models_for_age_bin(
                train_rows,
                manifest=manifest,
                target_age_index=target_age_index,
            )

        target_tokens = tokenize_words(
            target_row.get(manifest.target_text_column, ""),
            lowercase=manifest.lowercase,
        )
        if not target_tokens:
            continue
        length = len(target_tokens) if manifest.same_length else max(1, len(target_tokens))
        context_text = target_row.get(manifest.context_column or "", "")

        base: dict[str, Any] = {
            "baseline_run_id": manifest.run_id,
            "target_word_count": len(target_tokens),
        }
        for column in manifest.id_columns:
            base[column] = target_row.get(column, "")
        for column in manifest.carry_columns:
            base[column] = target_row.get(column, "")

        for generator_name, model in model_cache[target_age_index].items():
            for sample_index in range(manifest.samples_per_target):
                if generator_name == "random":
                    generated = [model.random_word() for _ in range(length)]
                else:
                    generated = model.generate(
                        length,
                        context_text=context_text,
                        lowercase=manifest.lowercase,
                    )
                yield {
                    **base,
                    "source_model": generator_name,
                    "sample_index": sample_index,
                    "generated_utterance": " ".join(generated),
                    "generated_word_count": len(generated),
                }


def output_fieldnames(manifest: BaselineManifest) -> list[str]:
    fields = [
        "baseline_run_id",
        *manifest.id_columns,
        *manifest.carry_columns,
        "source_model",
        "sample_index",
        "generated_utterance",
        "generated_word_count",
        "target_word_count",
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            deduped.append(field)
    return deduped


def run_ngram_generation(manifest: BaselineManifest) -> dict[str, Any]:
    manifest.validate_existing_inputs()
    row_count = write_csv_dicts(
        manifest.output_csv,
        generate_rows(manifest),
        fieldnames=output_fieldnames(manifest),
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.run_id,
        "row_count": row_count,
        "train_csv": str(manifest.train_csv),
        "target_csv": str(manifest.target_csv),
        "output_csv": str(manifest.output_csv),
        "train_sha256": sha256_file(manifest.train_csv),
        "target_sha256": sha256_file(manifest.target_csv),
        "output_sha256": sha256_file(manifest.output_csv),
        "manifest": {
            key: str(value) if key.endswith("_csv") else value
            for key, value in asdict(manifest).items()
            if key != "raw"
        },
    }
    write_json(manifest.audit_json, audit)
    return audit
