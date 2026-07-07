"""Command-line interface for baseline generation."""

from __future__ import annotations

import argparse
import json
import sys

from .big_cleaned import prepare_full_79_ngram_manifest
from .lstm import run_lstm_generation
from .manifest import BaselineManifest
from .ngram import run_ngram_generation


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    manifest = BaselineManifest.from_path(args.manifest)
    if args.check_inputs:
        manifest.validate_existing_inputs()
    print(json.dumps({"status": "ok", "run_id": manifest.run_id}, indent=2))
    return 0


def cmd_generate_ngram(args: argparse.Namespace) -> int:
    manifest = BaselineManifest.from_path(args.manifest)
    audit = run_ngram_generation(manifest)
    print(json.dumps({"status": "ok", "audit_json": str(manifest.audit_json), "row_count": audit["row_count"]}, indent=2))
    return 0


def cmd_generate_lstm(args: argparse.Namespace) -> int:
    manifest = BaselineManifest.from_path(args.manifest)
    audit = run_lstm_generation(manifest)
    print(json.dumps({"status": "ok", "audit_json": str(manifest.audit_json), "row_count": audit["row_count"]}, indent=2))
    return 0


def cmd_describe_compute_lanes(_: argparse.Namespace) -> int:
    print(
        "\n".join(
            [
                "CPU-first: random/unigram/bigram/trigram generation, audits, checksums.",
                "CPU-smoke/GPU-production: LSTM and small neural generators.",
                "Mila scoring repo: Mistral/large LLM scoring and Bayes neural likelihoods.",
            ]
        )
    )
    return 0


def _parse_dataset_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    datasets = {item.strip() for item in value.split(",") if item.strip()}
    return datasets or None


def cmd_prepare_full79_ngram_manifest(args: argparse.Namespace) -> int:
    audit = prepare_full_79_ngram_manifest(
        bundle_root=args.bundle_root,
        output_root=args.output_root,
        run_id=args.run_id,
        samples_per_target=args.samples_per_target,
        seed=args.seed,
        context_column=args.context_column,
        context_tail_words=args.context_tail_words,
        datasets=_parse_dataset_filter(args.datasets),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate-baselines-mila")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--check-inputs", action="store_true")
    validate.set_defaults(func=cmd_validate_manifest)

    ngram = subparsers.add_parser("generate-ngram")
    ngram.add_argument("--manifest", required=True)
    ngram.set_defaults(func=cmd_generate_ngram)

    lstm = subparsers.add_parser("generate-lstm")
    lstm.add_argument("--manifest", required=True)
    lstm.set_defaults(func=cmd_generate_lstm)

    lanes = subparsers.add_parser("describe-compute-lanes")
    lanes.set_defaults(func=cmd_describe_compute_lanes)

    prep_full79 = subparsers.add_parser("prepare-full79-ngram-manifest")
    prep_full79.add_argument("--bundle-root", required=True)
    prep_full79.add_argument("--output-root", required=True)
    prep_full79.add_argument("--run-id", default="full79_ngram_additive")
    prep_full79.add_argument("--samples-per-target", type=int, default=1)
    prep_full79.add_argument("--seed", type=int, default=13)
    prep_full79.add_argument("--context-column", default="context_k3")
    prep_full79.add_argument("--context-tail-words", type=int, default=16)
    prep_full79.add_argument("--datasets", help="Optional comma-separated dataset filter.")
    prep_full79.set_defaults(func=cmd_prepare_full79_ngram_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
