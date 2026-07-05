"""Command-line interface for baseline generation."""

from __future__ import annotations

import argparse
import json
import sys

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
    print(
        json.dumps(
            {
                "status": "not_implemented",
                "run_id": manifest.run_id,
                "message": (
                    "The LSTM cluster command is scaffolded but has not been "
                    "ported into this lightweight repo yet. Use this command "
                    "only after adding real training/generation code and tests."
                ),
            },
            indent=2,
        )
    )
    return 2


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
