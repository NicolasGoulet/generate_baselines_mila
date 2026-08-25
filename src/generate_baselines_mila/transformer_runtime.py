"""Lazy-imported PyTorch/Transformers runtime for the PBM generator cells."""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import shutil
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .io import sha256_file, write_json
from .pbm_transformers import GENERATED_COLUMNS, SPECIAL_TOKENS, _read_json, _stable_id


def _imports() -> tuple[Any, ...]:
    try:
        import torch
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        from transformers import (
            LlamaConfig,
            LlamaForCausalLM,
            T5Config,
            T5ForConditionalGeneration,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - exercised on Mila
        raise RuntimeError(
            "PBM transformer runtime requires the project transformer extra: "
            "pip install -e '.[transformers]'"
        ) from exc
    return (
        torch,
        Tokenizer,
        ByteLevelDecoder,
        BPE,
        ByteLevel,
        BpeTrainer,
        LlamaConfig,
        LlamaForCausalLM,
        T5Config,
        T5ForConditionalGeneration,
        get_linear_schedule_with_warmup,
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _atomic_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    try:
        with gzip.GzipFile(filename=str(temporary), mode="wb", mtime=0) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def train_shared_tokenizer(
    *,
    source_file: str | Path,
    output_dir: str | Path,
    vocab_size: int = 16000,
) -> dict[str, Any]:
    """Train one byte-level BPE tokenizer on non-PBM train rows only."""
    (
        _,
        Tokenizer,
        ByteLevelDecoder,
        BPE,
        ByteLevel,
        BpeTrainer,
        *_,
    ) = _imports()
    source_file = Path(source_file).resolve()
    output_dir = Path(output_dir).resolve()
    if vocab_size != 16000:
        raise ValueError("The frozen architecture comparison requires a 16,000-token vocabulary")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    audit_path = output_dir / "tokenizer_audit.json"
    if tokenizer_path.is_file() and audit_path.is_file():
        audit = _read_json(audit_path)
        if (
            audit.get("status") == "PASS"
            and audit.get("source_sha256") == sha256_file(source_file)
            and audit.get("tokenizer_sha256") == sha256_file(tokenizer_path)
        ):
            return audit

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=True,
    )

    def corpus() -> Iterator[str]:
        for row in _iter_jsonl(source_file):
            yield str(row["context_text"])
            yield str(row["target_text"])

    tokenizer.train_from_iterator(corpus(), trainer=trainer, length=None)
    if tokenizer.get_vocab_size() != vocab_size:
        raise RuntimeError(
            f"Tokenizer vocabulary is {tokenizer.get_vocab_size()}, expected exactly {vocab_size}"
        )
    temporary = tokenizer_path.with_name(f".{tokenizer_path.name}.tmp-{os.getpid()}")
    tokenizer.save(str(temporary))
    os.replace(temporary, tokenizer_path)
    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(value is None for value in ids.values()) or len(set(ids.values())) != len(ids):
        raise RuntimeError(f"Tokenizer special-token contract failed: {ids}")
    audit = {
        "status": "PASS",
        "source_file": str(source_file),
        "source_sha256": sha256_file(source_file),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": ids,
        "tokenizer_file": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "pbm_seen": False,
    }
    write_json(audit_path, audit)
    (output_dir / "TOKENIZER_READY").write_text("PASS\n", encoding="utf-8")
    return audit


def validate_runtime(*, require_cuda: bool = True) -> dict[str, Any]:
    torch, *rest = _imports()
    import tokenizers
    import transformers

    cuda = bool(torch.cuda.is_available())
    if require_cuda and not cuda:
        raise RuntimeError("CUDA is unavailable inside the allocated transformer job")
    return {
        "status": "PASS",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
    }


def _load_tokenizer(path: Path) -> Any:
    _, Tokenizer, *_ = _imports()
    tokenizer_path = path / "tokenizer.json"
    audit = _read_json(path / "tokenizer_audit.json")
    if audit.get("status") != "PASS" or audit.get("tokenizer_sha256") != sha256_file(tokenizer_path):
        raise RuntimeError("Shared tokenizer is absent or fails its hash contract")
    return Tokenizer.from_file(str(tokenizer_path))


def _make_model(manifest: dict[str, Any], *, model_dir: Path | None = None) -> Any:
    (
        _,
        _,
        _,
        _,
        _,
        _,
        LlamaConfig,
        LlamaForCausalLM,
        T5Config,
        T5ForConditionalGeneration,
        _,
    ) = _imports()
    architecture = manifest["architecture"]
    ids = _read_json(Path(manifest["tokenizer_dir"]) / "tokenizer_audit.json")["special_token_ids"]
    if model_dir is not None and (model_dir / "config.json").is_file():
        if architecture["family"] == "decoder_only":
            return LlamaForCausalLM.from_pretrained(model_dir, local_files_only=True)
        return T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    if architecture["family"] == "decoder_only":
        config = LlamaConfig(
            vocab_size=architecture["vocab_size"],
            hidden_size=architecture["hidden_size"],
            intermediate_size=architecture["intermediate_size"],
            num_hidden_layers=architecture["num_hidden_layers"],
            num_attention_heads=architecture["num_attention_heads"],
            max_position_embeddings=architecture["max_position_embeddings"],
            hidden_act="silu",
            rms_norm_eps=1e-6,
            tie_word_embeddings=architecture["tie_word_embeddings"],
            pad_token_id=ids["<pad>"],
            bos_token_id=ids["<s>"],
            eos_token_id=ids["</s>"],
        )
        return LlamaForCausalLM(config)
    config = T5Config(
        vocab_size=architecture["vocab_size"],
        d_model=architecture["d_model"],
        d_ff=architecture["d_ff"],
        d_kv=architecture["d_kv"],
        num_layers=architecture["num_layers"],
        num_decoder_layers=architecture["num_decoder_layers"],
        num_heads=architecture["num_heads"],
        relative_attention_num_buckets=architecture["relative_attention_num_buckets"],
        dropout_rate=0.1,
        feed_forward_proj="relu",
        tie_word_embeddings=architecture["tie_word_embeddings"],
        pad_token_id=ids["<pad>"],
        eos_token_id=ids["</s>"],
        decoder_start_token_id=ids["<pad>"],
    )
    return T5ForConditionalGeneration(config)


def _shuffle_buffer(rows: Iterable[dict[str, Any]], *, seed: int, size: int = 10000) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    buffer: list[dict[str, Any]] = []
    for row in rows:
        if len(buffer) < size:
            buffer.append(row)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer


def _chunks(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_training_batch(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    manifest: dict[str, Any],
    torch: Any,
) -> tuple[dict[str, Any], dict[str, int]]:
    ids = _read_json(Path(manifest["tokenizer_dir"]) / "tokenizer_audit.json")["special_token_ids"]
    pad, bos, eos, child = ids["<pad>"], ids["<s>"], ids["</s>"], ids["<child>"]
    family = manifest["architecture"]["family"]
    examples: list[tuple[list[int], list[int]]] = []
    diagnostics = {"context_truncated": 0, "target_truncated": 0}
    for row in rows:
        context_ids = tokenizer.encode(str(row["context_text"]), add_special_tokens=False).ids
        target_ids = tokenizer.encode(str(row["target_text"]), add_special_tokens=False).ids
        if len(target_ids) > int(manifest["max_target_tokens"]):
            target_ids = target_ids[: int(manifest["max_target_tokens"])]
            diagnostics["target_truncated"] += 1
        if family == "decoder_only":
            room = int(manifest["max_sequence_tokens"]) - len(target_ids) - 3
            if len(context_ids) > max(0, room):
                context_ids = context_ids[-max(0, room) :] if room > 0 else []
                diagnostics["context_truncated"] += 1
            prompt = [bos, *context_ids, child]
            sequence = [*prompt, *target_ids, eos]
            labels = [-100] * len(prompt) + [*target_ids, eos]
            examples.append((sequence, labels))
        else:
            room = int(manifest["max_encoder_tokens"]) - 2
            if len(context_ids) > max(0, room):
                context_ids = context_ids[-max(0, room) :] if room > 0 else []
                diagnostics["context_truncated"] += 1
            examples.append(([*context_ids, child, eos], [*target_ids, eos]))
    max_input = max(len(item[0]) for item in examples)
    max_labels = max(len(item[1]) for item in examples)
    input_ids = [item[0] + [pad] * (max_input - len(item[0])) for item in examples]
    attention = [[1] * len(item[0]) + [0] * (max_input - len(item[0])) for item in examples]
    labels = [item[1] + [-100] * (max_labels - len(item[1])) for item in examples]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }, diagnostics


def _evaluate(model: Any, path: Path, *, tokenizer: Any, manifest: dict[str, Any], torch: Any, device: Any) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for rows in _chunks(_iter_jsonl(path), int(manifest["per_device_batch_size"])):
            batch, _ = _encode_training_batch(rows, tokenizer=tokenizer, manifest=manifest, torch=torch)
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            total_loss += float(loss.detach().cpu()) * len(rows)
            total_rows += len(rows)
    if total_rows == 0:
        raise RuntimeError(f"Empty validation file: {path}")
    return total_loss / total_rows


def _train_one_epoch(
    model: Any,
    *,
    path: Path,
    expected_rows: int,
    epoch: int,
    seed: int,
    tokenizer: Any,
    manifest: dict[str, Any],
    torch: Any,
    device: Any,
    optimizer: Any,
    scheduler: Any,
    use_bf16: bool,
) -> tuple[float, dict[str, int]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_rows = 0
    accumulation = int(manifest["gradient_accumulation_steps"])
    batches_per_epoch = math.ceil(expected_rows / int(manifest["per_device_batch_size"]))
    truncation = {"context_truncated": 0, "target_truncated": 0}
    shuffled = _shuffle_buffer(_iter_jsonl(path), seed=seed + epoch)
    for batch_number, rows in enumerate(
        _chunks(shuffled, int(manifest["per_device_batch_size"])), start=1
    ):
        batch, diagnostics = _encode_training_batch(
            rows, tokenizer=tokenizer, manifest=manifest, torch=torch
        )
        for key in truncation:
            truncation[key] += diagnostics[key]
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss = model(**batch).loss
            scaled = loss / accumulation
        scaled.backward()
        if batch_number % accumulation == 0 or batch_number == batches_per_epoch:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * len(rows)
        total_rows += len(rows)
    if total_rows != expected_rows:
        raise RuntimeError(f"Training row count changed for {path}: {total_rows} != {expected_rows}")
    return total_loss / total_rows, truncation


def _write_resume(
    *,
    model: Any,
    resume_dir: Path,
    state: dict[str, Any],
    torch: Any,
) -> None:
    resume_model = resume_dir / "model"
    if resume_model.exists():
        shutil.rmtree(resume_model)
    resume_model.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(resume_model, safe_serialization=True)
    resume_dir.mkdir(parents=True, exist_ok=True)
    state_path = resume_dir / "training_state.pt"
    temporary_state = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
    torch.save(state, temporary_state)
    os.replace(temporary_state, state_path)


def train_cell(manifest_path: str | Path) -> dict[str, Any]:
    """Select an epoch on validation, then refit from scratch on all development rows."""
    manifest = _read_json(Path(manifest_path))
    torch, *imports = _imports()
    get_linear_schedule_with_warmup = imports[-1]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PBM Transformer training")
    seed = int(manifest["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older torch
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    tokenizer = _load_tokenizer(Path(manifest["tokenizer_dir"]))
    artifact_dir = Path(manifest["artifact_dir"])
    report_path = artifact_dir / "training_report.json"
    if report_path.is_file() and (artifact_dir / "model.safetensors").is_file():
        report = _read_json(report_path)
        if report.get("status") == "PASS":
            return report
    selection_dir = artifact_dir.parent / "selection_best"
    selection_resume = artifact_dir.parent / "selection_resume"
    selection_state_path = selection_resume / "training_state.pt"
    model = _make_model(
        manifest,
        model_dir=(selection_resume / "model") if selection_state_path.is_file() else None,
    )
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 45_000_000 <= parameter_count <= 70_000_000:
        raise RuntimeError(f"Model has {parameter_count:,} parameters, outside frozen 45M-70M range")
    nominal = int(manifest["architecture"]["nominal_parameter_count"])
    if abs(parameter_count - nominal) / nominal > 0.01:
        raise RuntimeError(
            f"Model has {parameter_count:,} parameters, more than 1% from nominal {nominal:,}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(manifest["learning_rate"]),
        weight_decay=float(manifest["weight_decay"]),
    )
    batches_per_epoch = math.ceil(int(manifest["train_rows"]) / int(manifest["per_device_batch_size"]))
    update_steps_per_epoch = math.ceil(batches_per_epoch / int(manifest["gradient_accumulation_steps"]))
    total_steps = max(1, update_steps_per_epoch * int(manifest["max_epochs"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(manifest["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    selection_truncation = {"context_truncated": 0, "target_truncated": 0}
    if selection_state_path.is_file():
        state = torch.load(selection_state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["next_epoch"])
        best_loss = float(state["best_validation_loss"])
        best_epoch = int(state["best_epoch"])
        no_improvement = int(state["no_improvement"])
        history = list(state["history"])
        selection_truncation = dict(state.get("truncation", selection_truncation))
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    for epoch in range(start_epoch, int(manifest["max_epochs"])):
        training_loss, epoch_truncation = _train_one_epoch(
            model,
            path=Path(manifest["train_file"]),
            expected_rows=int(manifest["train_rows"]),
            epoch=epoch,
            seed=seed,
            tokenizer=tokenizer,
            manifest=manifest,
            torch=torch,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_bf16=use_bf16,
        )
        for key in selection_truncation:
            selection_truncation[key] += epoch_truncation[key]
        validation_loss = _evaluate(
            model,
            Path(manifest["validation_file"]),
            tokenizer=tokenizer,
            manifest=manifest,
            torch=torch,
            device=device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            no_improvement = 0
            selection_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(selection_dir, safe_serialization=True)
        else:
            no_improvement += 1
        _write_resume(
            model=model,
            resume_dir=selection_resume,
            torch=torch,
            state={
                "next_epoch": epoch + 1,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "no_improvement": no_improvement,
                "history": history,
                "truncation": selection_truncation,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
        )
        patience = int(manifest["early_stopping_patience"])
        if patience > 0 and no_improvement >= patience:
            break
    if best_epoch <= 0 or not (selection_dir / "model.safetensors").is_file():
        raise RuntimeError("Selection ended without a best model checkpoint")

    # Reinitialize instead of continuing the selection checkpoint.  The final
    # model sees every non-PBM development row for exactly the selected number
    # of epochs; it never sees PBM text, even through the tokenizer.
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    refit_resume = artifact_dir.parent / "refit_resume"
    refit_state_path = refit_resume / "training_state.pt"
    if not refit_state_path.is_file():
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    refit_model = _make_model(
        manifest,
        model_dir=(refit_resume / "model") if refit_state_path.is_file() else None,
    )
    refit_model.to(device)
    refit_optimizer = torch.optim.AdamW(
        refit_model.parameters(),
        lr=float(manifest["learning_rate"]),
        weight_decay=float(manifest["weight_decay"]),
    )
    development_rows = int(manifest["development_rows"])
    refit_batches = math.ceil(development_rows / int(manifest["per_device_batch_size"]))
    refit_updates = math.ceil(refit_batches / int(manifest["gradient_accumulation_steps"]))
    refit_total_steps = max(1, refit_updates * best_epoch)
    refit_scheduler = get_linear_schedule_with_warmup(
        refit_optimizer,
        num_warmup_steps=int(refit_total_steps * float(manifest["warmup_ratio"])),
        num_training_steps=refit_total_steps,
    )
    refit_start_epoch = 0
    refit_history: list[dict[str, Any]] = []
    refit_truncation = {"context_truncated": 0, "target_truncated": 0}
    if refit_state_path.is_file():
        refit_state = torch.load(refit_state_path, map_location="cpu", weights_only=False)
        refit_optimizer.load_state_dict(refit_state["optimizer"])
        refit_scheduler.load_state_dict(refit_state["scheduler"])
        refit_start_epoch = int(refit_state["next_epoch"])
        refit_history = list(refit_state["history"])
        refit_truncation = dict(refit_state["truncation"])
    for epoch in range(refit_start_epoch, best_epoch):
        refit_loss, epoch_truncation = _train_one_epoch(
            refit_model,
            path=Path(manifest["development_file"]),
            expected_rows=development_rows,
            epoch=epoch,
            seed=seed,
            tokenizer=tokenizer,
            manifest=manifest,
            torch=torch,
            device=device,
            optimizer=refit_optimizer,
            scheduler=refit_scheduler,
            use_bf16=use_bf16,
        )
        for key in refit_truncation:
            refit_truncation[key] += epoch_truncation[key]
        refit_history.append({"epoch": epoch + 1, "training_loss": refit_loss})
        _write_resume(
            model=refit_model,
            resume_dir=refit_resume,
            torch=torch,
            state={
                "next_epoch": epoch + 1,
                "history": refit_history,
                "truncation": refit_truncation,
                "optimizer": refit_optimizer.state_dict(),
                "scheduler": refit_scheduler.state_dict(),
            },
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    refit_model.save_pretrained(artifact_dir, safe_serialization=True)
    report = {
        "status": "PASS",
        "architecture_id": manifest["architecture_id"],
        "architecture_family": manifest["architecture"]["family"],
        "initialization": "from_scratch",
        "teacher_distillation": False,
        "parameter_count": parameter_count,
        "train_rows": manifest["train_rows"],
        "validation_rows": manifest["validation_rows"],
        "development_rows": manifest["development_rows"],
        "selection_best_epoch": best_epoch,
        "selection_best_validation_loss": best_loss,
        "selection_epochs_completed": len(history),
        "selection_history": history,
        "refit_epochs_completed": len(refit_history),
        "refit_history": refit_history,
        "final_model_training_scope": "development=train+validation; PBM excluded",
        "bf16": use_bf16,
        "selection_truncation_diagnostics": selection_truncation,
        "refit_truncation_diagnostics": refit_truncation,
        "tokenizer_sha256": sha256_file(Path(manifest["tokenizer_dir"]) / "tokenizer.json"),
        "model_sha256": sha256_file(artifact_dir / "model.safetensors"),
    }
    write_json(report_path, report)
    return report


def _generation_prompt(
    row: dict[str, Any], *, tokenizer: Any, manifest: dict[str, Any], special_ids: dict[str, int]
) -> list[int]:
    context = tokenizer.encode(str(row["context_text"]), add_special_tokens=False).ids
    family = manifest["architecture"]["family"]
    limit = int(manifest["max_sequence_tokens"] if family == "decoder_only" else manifest["max_encoder_tokens"])
    reserved = 2 if family == "encoder_decoder" else 2
    context = context[-max(0, limit - reserved) :]
    if family == "decoder_only":
        return [special_ids["<s>"], *context, special_ids["<child>"]]
    return [*context, special_ids["<child>"], special_ids["</s>"]]


def generate_cell(manifest_path: str | Path) -> dict[str, Any]:
    manifest = _read_json(Path(manifest_path))
    torch, *_ = _imports()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PBM Transformer generation")
    artifact_dir = Path(manifest["artifact_dir"])
    if not (artifact_dir / "training_report.json").is_file():
        raise FileNotFoundError("Training report is absent; refusing generation")
    device = torch.device("cuda")
    tokenizer = _load_tokenizer(Path(manifest["tokenizer_dir"]))
    special = _read_json(Path(manifest["tokenizer_dir"]) / "tokenizer_audit.json")["special_token_ids"]
    model = _make_model(manifest, model_dir=artifact_dir)
    model.to(device)
    model.eval()
    torch.manual_seed(int(manifest["seed"]))
    torch.cuda.manual_seed_all(int(manifest["seed"]))
    pad = special["<pad>"]
    eos = special["</s>"]
    family = manifest["architecture"]["family"]
    suppressed = [special[token] for token in ("<pad>", "<s>", "<unk>", "<turn>", "<child>")]

    def rows() -> Iterator[dict[str, Any]]:
        for target_batch in _chunks(
            _iter_jsonl(Path(manifest["target_file"])), int(manifest["per_device_batch_size"])
        ):
            prompts = [
                _generation_prompt(row, tokenizer=tokenizer, manifest=manifest, special_ids=special)
                for row in target_batch
            ]
            max_length = max(len(prompt) for prompt in prompts)
            if family == "decoder_only":
                padded = [[pad] * (max_length - len(prompt)) + prompt for prompt in prompts]
                attention = [[0] * (max_length - len(prompt)) + [1] * len(prompt) for prompt in prompts]
            else:
                padded = [prompt + [pad] * (max_length - len(prompt)) for prompt in prompts]
                attention = [[1] * len(prompt) + [0] * (max_length - len(prompt)) for prompt in prompts]
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(attention, dtype=torch.long, device=device)
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=bool(manifest["do_sample"]),
                    temperature=float(manifest["temperature"]),
                    top_p=float(manifest["top_p"]),
                    min_new_tokens=1,
                    max_new_tokens=int(manifest["max_new_tokens"]),
                    eos_token_id=eos,
                    pad_token_id=pad,
                    suppress_tokens=suppressed,
                    use_cache=True,
                )
            generated = output[:, max_length:] if family == "decoder_only" else output[:, 1:]
            for target, token_tensor in zip(target_batch, generated):
                token_ids = [int(value) for value in token_tensor.detach().cpu().tolist()]
                eos_reached = eos in token_ids
                if eos_reached:
                    token_ids = token_ids[: token_ids.index(eos)]
                token_ids = [value for value in token_ids if value != pad]
                text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                yield {
                    "generated_id": _stable_id(
                        [manifest["architecture_id"], target["example_id"], 0, manifest["seed"]]
                    ),
                    "example_id": target["example_id"],
                    "dataset": target["dataset"],
                    "child_id": target["child_id"],
                    "session_id": target["session_id"],
                    "file": target["file"],
                    "line_no": target["line_no"],
                    "reference_line": target["reference_line"],
                    "age_months": target["age_months"],
                    "target_age_bin": target["target_age_bin"],
                    "context_text": target["context_text"],
                    "context_k3": target["context_text"],
                    "real_target_text": target["target_text"],
                    "real_target_word_count": target["target_word_count"],
                    "source_model": manifest["architecture_id"],
                    "sample_index": 0,
                    "generated_utterance": text,
                    "generated_word_count": len(text.split()),
                    "generated_token_count": len(token_ids),
                    "eos_reached": eos_reached,
                    "max_token_censored": not eos_reached,
                    "generation_seed": manifest["seed"],
                    "max_new_tokens": manifest["max_new_tokens"],
                    "temperature": manifest["temperature"],
                    "top_p": manifest["top_p"],
                }

    output = Path(manifest["output_file"])
    count = _atomic_jsonl_gz(output, rows())
    return {"status": "PASS", "output_file": str(output), "rows": count, "sha256": sha256_file(output)}


def run_cell(manifest_path: str | Path) -> dict[str, Any]:
    """Exact production entry point used for both smoke and final cells."""
    from .pbm_transformers import audit_cell_output

    training = train_cell(manifest_path)
    generation = generate_cell(manifest_path)
    audit = audit_cell_output(manifest_path)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Cell audit failed: {audit['problems']}")
    return {"status": "PASS", "training": training, "generation": generation, "audit": audit}
