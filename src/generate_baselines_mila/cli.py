"""Command-line interface for baseline generation."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .big_cleaned import prepare_full_79_ngram_manifest
from .full79_lstm import (
    DEFAULT_CONTEXTS,
    audit_full79_lstm_run,
    audit_lstm_output,
    audit_smoke,
    finalize_full79_lstm_report,
    manifest_for_cell,
    parse_indices,
    prepare_full79_lstm_run,
)
from .lstm import run_lstm_generation
from .manifest import BaselineManifest
from .ngram import run_ngram_generation
from .pbm_transformers import (
    audit_pbm_transformer_run,
    audit_smoke as audit_pbm_transformer_smoke,
    audit_cell_output as audit_pbm_transformer_cell,
    finalize_report as finalize_pbm_transformer_report,
    manifest_for_cell as pbm_transformer_manifest_for_cell,
    parse_indices as parse_pbm_transformer_indices,
    prepare_pbm_transformer_run,
)


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


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def cmd_prepare_full79_lstm(args: argparse.Namespace) -> int:
    audit = prepare_full79_lstm_run(
        bundle_root=args.bundle_root,
        run_root=args.run_root,
        contexts=_parse_int_list(args.contexts),
        max_context_tokens=args.max_context_tokens,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_vocab_size=args.max_vocab_size,
        smoke_train_examples=args.smoke_train_examples,
        smoke_target_rows=args.smoke_target_rows,
        seed=args.seed,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def cmd_full79_lstm_manifest(args: argparse.Namespace) -> int:
    print(manifest_for_cell(args.run_root, args.index))
    return 0


def cmd_audit_lstm_output(args: argparse.Namespace) -> int:
    report = audit_lstm_output(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def cmd_audit_full79_lstm(args: argparse.Namespace) -> int:
    indices = parse_indices(args.indices) if args.indices else None
    summary = audit_full79_lstm_run(run_root=args.run_root, stage=args.stage, indices=indices)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_audit_full79_lstm_smoke(args: argparse.Namespace) -> int:
    report = audit_smoke(args.run_root, job_id=args.job_id or os.environ.get("SLURM_JOB_ID", ""))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_finalize_full79_lstm_report(args: argparse.Namespace) -> int:
    report = finalize_full79_lstm_report(args.run_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_prepare_pbm_transformers(args: argparse.Namespace) -> int:
    report = prepare_pbm_transformer_run(
        handoff_root=args.handoff_root,
        run_root=args.run_root,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_censored_fraction=args.max_censored_fraction,
        smoke_train_examples=args.smoke_train_examples,
        smoke_target_rows=args.smoke_target_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_train_pbm_tokenizer(args: argparse.Namespace) -> int:
    from .transformer_runtime import train_shared_tokenizer

    report = train_shared_tokenizer(
        source_file=args.source_file,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_validate_pbm_transformer_runtime(args: argparse.Namespace) -> int:
    from .transformer_runtime import validate_runtime

    report = validate_runtime(require_cuda=not args.allow_cpu)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_pbm_transformer_manifest(args: argparse.Namespace) -> int:
    print(pbm_transformer_manifest_for_cell(args.run_root, args.index, smoke=args.smoke))
    return 0


def cmd_run_pbm_transformer_cell(args: argparse.Namespace) -> int:
    from .transformer_runtime import run_cell

    report = run_cell(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_audit_pbm_transformer_cell(args: argparse.Namespace) -> int:
    report = audit_pbm_transformer_cell(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def cmd_audit_pbm_transformer_smoke(args: argparse.Namespace) -> int:
    report = audit_pbm_transformer_smoke(args.run_root, job_id=args.job_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_audit_pbm_transformers(args: argparse.Namespace) -> int:
    indices = parse_pbm_transformer_indices(args.indices) if args.indices else None
    report = audit_pbm_transformer_run(args.run_root, stage=args.stage, indices=indices)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_finalize_pbm_transformer_report(args: argparse.Namespace) -> int:
    report = finalize_pbm_transformer_report(args.run_root)
    print(json.dumps(report, indent=2, sort_keys=True))
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

    prep_lstm = subparsers.add_parser("prepare-full79-lstm")
    prep_lstm.add_argument("--bundle-root", required=True)
    prep_lstm.add_argument("--run-root", required=True)
    prep_lstm.add_argument("--contexts", default=",".join(str(value) for value in DEFAULT_CONTEXTS))
    prep_lstm.add_argument("--max-context-tokens", type=int, default=60)
    prep_lstm.add_argument("--epochs", type=int, default=20)
    prep_lstm.add_argument("--batch-size", type=int, default=256)
    prep_lstm.add_argument("--embedding-dim", type=int, default=256)
    prep_lstm.add_argument("--hidden-dim", type=int, default=512)
    prep_lstm.add_argument("--num-layers", type=int, default=2)
    prep_lstm.add_argument("--dropout", type=float, default=0.2)
    prep_lstm.add_argument("--max-vocab-size", type=int, default=30000)
    prep_lstm.add_argument("--smoke-train-examples", type=int, default=1024)
    prep_lstm.add_argument("--smoke-target-rows", type=int, default=25)
    prep_lstm.add_argument("--seed", type=int, default=123)
    prep_lstm.set_defaults(func=cmd_prepare_full79_lstm)

    show_lstm = subparsers.add_parser("full79-lstm-manifest")
    show_lstm.add_argument("--run-root", required=True)
    show_lstm.add_argument("--index", type=int, required=True)
    show_lstm.set_defaults(func=cmd_full79_lstm_manifest)

    audit_lstm = subparsers.add_parser("audit-lstm-output")
    audit_lstm.add_argument("--manifest", required=True)
    audit_lstm.set_defaults(func=cmd_audit_lstm_output)

    audit_full_lstm = subparsers.add_parser("audit-full79-lstm")
    audit_full_lstm.add_argument("--run-root", required=True)
    audit_full_lstm.add_argument("--stage", required=True, choices=("wave1", "wave2", "final"))
    audit_full_lstm.add_argument("--indices", help="Comma-separated indices and inclusive ranges.")
    audit_full_lstm.set_defaults(func=cmd_audit_full79_lstm)

    audit_smoke_lstm = subparsers.add_parser("audit-full79-lstm-smoke")
    audit_smoke_lstm.add_argument("--run-root", required=True)
    audit_smoke_lstm.add_argument("--job-id", default="")
    audit_smoke_lstm.set_defaults(func=cmd_audit_full79_lstm_smoke)

    final_report_lstm = subparsers.add_parser("finalize-full79-lstm-report")
    final_report_lstm.add_argument("--run-root", required=True)
    final_report_lstm.set_defaults(func=cmd_finalize_full79_lstm_report)

    prep_transformers = subparsers.add_parser("prepare-pbm-transformers")
    prep_transformers.add_argument("--handoff-root", required=True)
    prep_transformers.add_argument("--run-root", required=True)
    prep_transformers.add_argument("--seed", type=int, default=20260825)
    prep_transformers.add_argument("--max-epochs", type=int, default=10)
    prep_transformers.add_argument("--patience", type=int, default=2)
    prep_transformers.add_argument("--per-device-batch-size", type=int, default=16)
    prep_transformers.add_argument("--gradient-accumulation-steps", type=int, default=16)
    prep_transformers.add_argument("--learning-rate", type=float, default=5e-4)
    prep_transformers.add_argument("--max-new-tokens", type=int, default=128)
    prep_transformers.add_argument("--temperature", type=float, default=1.0)
    prep_transformers.add_argument("--top-p", type=float, default=1.0)
    prep_transformers.add_argument("--max-censored-fraction", type=float, default=0.001)
    prep_transformers.add_argument("--smoke-train-examples", type=int, default=1024)
    prep_transformers.add_argument("--smoke-target-rows", type=int, default=25)
    prep_transformers.set_defaults(func=cmd_prepare_pbm_transformers)

    tokenizer = subparsers.add_parser("train-pbm-transformer-tokenizer")
    tokenizer.add_argument("--source-file", required=True)
    tokenizer.add_argument("--output-dir", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=16000)
    tokenizer.set_defaults(func=cmd_train_pbm_tokenizer)

    runtime = subparsers.add_parser("validate-pbm-transformer-runtime")
    runtime.add_argument("--allow-cpu", action="store_true")
    runtime.set_defaults(func=cmd_validate_pbm_transformer_runtime)

    show_transformer = subparsers.add_parser("pbm-transformer-manifest")
    show_transformer.add_argument("--run-root", required=True)
    show_transformer.add_argument("--index", type=int, required=True)
    show_transformer.add_argument("--smoke", action="store_true")
    show_transformer.set_defaults(func=cmd_pbm_transformer_manifest)

    run_transformer = subparsers.add_parser("run-pbm-transformer-cell")
    run_transformer.add_argument("--manifest", required=True)
    run_transformer.set_defaults(func=cmd_run_pbm_transformer_cell)

    audit_transformer_cell = subparsers.add_parser("audit-pbm-transformer-cell")
    audit_transformer_cell.add_argument("--manifest", required=True)
    audit_transformer_cell.set_defaults(func=cmd_audit_pbm_transformer_cell)

    audit_transformer_smoke = subparsers.add_parser("audit-pbm-transformer-smoke")
    audit_transformer_smoke.add_argument("--run-root", required=True)
    audit_transformer_smoke.add_argument("--job-id", default="")
    audit_transformer_smoke.set_defaults(func=cmd_audit_pbm_transformer_smoke)

    audit_transformers = subparsers.add_parser("audit-pbm-transformers")
    audit_transformers.add_argument("--run-root", required=True)
    audit_transformers.add_argument("--stage", required=True, choices=("wave1", "wave2", "final"))
    audit_transformers.add_argument("--indices")
    audit_transformers.set_defaults(func=cmd_audit_pbm_transformers)

    final_transformers = subparsers.add_parser("finalize-pbm-transformer-report")
    final_transformers.add_argument("--run-root", required=True)
    final_transformers.set_defaults(func=cmd_finalize_pbm_transformer_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
