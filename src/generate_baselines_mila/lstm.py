"""Optional PyTorch LSTM baseline generation.

This module is intentionally generic: it consumes the same manifest contract as
the CPU n-gram generator and trains a small word-level prefix LSTM. Caretaker
context tokens are used as a conditioning prefix; loss is computed only over
the child utterance and EOS target tokens.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import iter_csv_dicts, sha256_file, write_csv_dicts, write_json
from .manifest import BaselineManifest
from .ngram import _age_index, _training_rows_for_target_bin, output_fieldnames
from .tokenize import tokenize_words

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIAL = (PAD, UNK, BOS, EOS)
IGNORE_INDEX = -100


def require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "LSTM generation requires PyTorch. Install torch on the Mila "
            "environment or use the CPU n-gram generators."
        ) from exc
    return torch, nn, DataLoader, Dataset


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(
        cls,
        token_sequences: Iterable[list[str]],
        *,
        min_freq: int = 1,
        max_vocab_size: int | None = None,
    ) -> "Vocab":
        counts: Counter[str] = Counter()
        for tokens in token_sequences:
            counts.update(tokens)
        lexical = [
            token
            for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_freq and token not in SPECIAL
        ]
        if max_vocab_size is not None:
            lexical = lexical[: max(0, int(max_vocab_size) - len(SPECIAL))]
        id_to_token = list(SPECIAL) + lexical
        return cls({token: index for index, token in enumerate(id_to_token)}, id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, self.unk_id)

    def decode(self, token_id: int) -> str:
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        return UNK

    def to_json(self) -> dict[str, Any]:
        return {"id_to_token": self.id_to_token}


class LSTMGenerator:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        torch, nn, _, _ = require_torch()
        super_class = nn.Module

        class _Model(super_class):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.output = nn.Linear(hidden_dim, vocab_size)

            def forward(self, input_ids):
                embedded = self.embedding(input_ids)
                hidden, _ = self.lstm(embedded)
                return self.output(hidden)

        self.model = _Model()
        self.torch = torch


def _context_tokens(row: dict[str, str], manifest: BaselineManifest) -> list[str]:
    if not manifest.context_column:
        return []
    tokens = tokenize_words(row.get(manifest.context_column, ""), lowercase=manifest.lowercase)
    if manifest.context_tail_words:
        return tokens[-manifest.context_tail_words :]
    return tokens


def _training_examples(rows: list[dict[str, str]], manifest: BaselineManifest) -> list[tuple[list[str], list[str]]]:
    examples: list[tuple[list[str], list[str]]] = []
    for row in rows:
        child_tokens = tokenize_words(row.get(manifest.text_column, ""), lowercase=manifest.lowercase)
        if child_tokens:
            examples.append((_context_tokens(row, manifest), child_tokens))
    return examples


def _encoded_examples(examples: list[tuple[list[str], list[str]]], vocab: Vocab) -> list[tuple[list[int], list[int]]]:
    encoded: list[tuple[list[int], list[int]]] = []
    for context_tokens, child_tokens in examples:
        sequence = context_tokens + [BOS] + child_tokens + [EOS]
        input_ids = [vocab.encode(token) for token in sequence[:-1]]
        labels = [vocab.encode(token) for token in sequence[1:]]
        for index in range(len(context_tokens)):
            labels[index] = IGNORE_INDEX
        encoded.append((input_ids, labels))
    return encoded


def _select_device(device: str):
    torch, _, _, _ = require_torch()
    requested = device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Manifest requested device='cuda', but CUDA is not available.")
    return torch.device(requested)


def _collate(batch, *, pad_id: int):
    torch, _, _, _ = require_torch()
    max_len = max(len(input_ids) for input_ids, _ in batch)
    input_batch = []
    label_batch = []
    for input_ids, labels in batch:
        pad = max_len - len(input_ids)
        input_batch.append(input_ids + [pad_id] * pad)
        label_batch.append(labels + [IGNORE_INDEX] * pad)
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(label_batch, dtype=torch.long)


def _train_model(
    examples: list[tuple[list[str], list[str]]],
    *,
    manifest: BaselineManifest,
) -> tuple[Any, Vocab, dict[str, Any]]:
    torch, nn, DataLoader, Dataset = require_torch()
    raw = manifest.raw
    max_train_examples = raw.get("max_train_examples")
    if max_train_examples is not None:
        examples = examples[: int(max_train_examples)]
    if not examples:
        raise ValueError("No nonempty training examples available for LSTM training.")

    token_sequences = [context + child for context, child in examples]
    vocab = Vocab.build(
        token_sequences,
        min_freq=int(raw.get("min_freq", 1)),
        max_vocab_size=raw.get("max_vocab_size"),
    )
    encoded = _encoded_examples(examples, vocab)

    class _Dataset(Dataset):
        def __len__(self):
            return len(encoded)

        def __getitem__(self, index):
            return encoded[index]

    seed = int(raw.get("seed", manifest.seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _select_device(str(raw.get("device", "auto")))
    wrapper = LSTMGenerator(
        vocab_size=len(vocab.id_to_token),
        embedding_dim=int(raw.get("embedding_dim", 128)),
        hidden_dim=int(raw.get("hidden_dim", 256)),
        num_layers=int(raw.get("num_layers", 1)),
        dropout=float(raw.get("dropout", 0.0)),
    )
    model = wrapper.model.to(device)

    loader = DataLoader(
        _Dataset(),
        batch_size=int(raw.get("batch_size", 32)),
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, pad_id=vocab.pad_id),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(raw.get("learning_rate", 0.001)))
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    epochs = int(raw.get("epochs", 3))
    grad_clip = float(raw.get("grad_clip", 1.0))
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            token_count = int((labels != IGNORE_INDEX).sum().item())
            total_loss += float(loss.item()) * max(1, token_count)
            total_tokens += token_count
        mean_loss = total_loss / max(1, total_tokens)
        history.append({"epoch": float(epoch + 1), "mean_loss": mean_loss, "perplexity": math.exp(min(20, mean_loss))})

    audit = {
        "train_examples": len(examples),
        "vocab_size": len(vocab.id_to_token),
        "device": str(device),
        "epochs": epochs,
        "history": history,
    }
    return model, vocab, audit


def _sample_next(logits, *, vocab: Vocab, temperature: float, top_k: int, allow_eos: bool):
    torch, _, _, _ = require_torch()
    logits = logits.clone()
    for token in (PAD, UNK, BOS):
        logits[vocab.token_to_id[token]] = -float("inf")
    if not allow_eos:
        logits[vocab.eos_id] = -float("inf")
    logits = logits / max(temperature, 1e-6)
    if top_k > 0 and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _generate_tokens(
    model,
    vocab: Vocab,
    context_tokens: list[str],
    *,
    target_len: int,
    manifest: BaselineManifest,
) -> list[str]:
    torch, _, _, _ = require_torch()
    raw = manifest.raw
    device = next(model.parameters()).device
    model.eval()
    prefix = [vocab.encode(token) for token in context_tokens] + [vocab.bos_id]
    generated: list[int] = []
    length_mode = str(raw.get("generation_length_mode", "same_as_child"))
    max_generated_tokens = int(raw.get("max_generated_tokens", max(1, target_len)))
    min_generated_tokens = int(raw.get("min_generated_tokens", 1))
    fixed_len = target_len if manifest.same_length and length_mode == "same_as_child" else max_generated_tokens
    temperature = float(raw.get("temperature", 1.0))
    top_k = int(raw.get("top_k", 0))

    with torch.no_grad():
        for _ in range(fixed_len):
            input_ids = torch.tensor([prefix + generated], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1]
            allow_eos = length_mode != "same_as_child" and len(generated) >= min_generated_tokens
            next_id = _sample_next(
                logits,
                vocab=vocab,
                temperature=temperature,
                top_k=top_k,
                allow_eos=allow_eos,
            )
            if next_id == vocab.eos_id and allow_eos:
                break
            generated.append(next_id)
            if length_mode != "same_as_child" and len(generated) >= max_generated_tokens:
                break
    return [vocab.decode(token_id) for token_id in generated if vocab.decode(token_id) not in SPECIAL]


def _artifact_dir(manifest: BaselineManifest, age_key: int | None) -> Path:
    root = Path(str(manifest.raw.get("model_dir", manifest.output_csv.parent / "models")))
    label = "global" if age_key is None else f"age_index_{age_key:02d}"
    return root / label


def _save_artifacts(model, vocab: Vocab, manifest: BaselineManifest, age_key: int | None, audit: dict[str, Any]) -> None:
    torch, _, _, _ = require_torch()
    artifact_dir = _artifact_dir(manifest, age_key)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "vocab.json").write_text(json.dumps(vocab.to_json(), indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "train_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_lstm_rows(manifest: BaselineManifest) -> Iterator[dict[str, Any]]:
    train_rows = list(iter_csv_dicts(manifest.train_csv))
    target_rows = list(iter_csv_dicts(manifest.target_csv))
    model_cache: dict[int | None, tuple[Any, Vocab, dict[str, Any]]] = {}
    source_model = str(manifest.raw.get("source_model", "lstm"))

    for target_row in target_rows:
        target_age_index = _age_index(manifest, target_row)
        if target_age_index not in model_cache:
            scoped_rows = _training_rows_for_target_bin(
                train_rows,
                manifest=manifest,
                target_age_index=target_age_index,
            )
            examples = _training_examples(scoped_rows, manifest)
            model, vocab, train_audit = _train_model(examples, manifest=manifest)
            _save_artifacts(model, vocab, manifest, target_age_index, train_audit)
            model_cache[target_age_index] = (model, vocab, train_audit)

        target_tokens = tokenize_words(target_row.get(manifest.target_text_column, ""), lowercase=manifest.lowercase)
        if not target_tokens:
            continue
        model, vocab, _ = model_cache[target_age_index]
        context_tokens = _context_tokens(target_row, manifest)

        base: dict[str, Any] = {
            "baseline_run_id": manifest.run_id,
            "target_word_count": len(target_tokens),
        }
        for column in manifest.id_columns:
            base[column] = target_row.get(column, "")
        for column in manifest.carry_columns:
            base[column] = target_row.get(column, "")
        for sample_index in range(manifest.samples_per_target):
            generated = _generate_tokens(
                model,
                vocab,
                context_tokens,
                target_len=len(target_tokens),
                manifest=manifest,
            )
            yield {
                **base,
                "source_model": source_model,
                "sample_index": sample_index,
                "generated_utterance": " ".join(generated),
                "generated_word_count": len(generated),
            }


def run_lstm_generation(manifest: BaselineManifest) -> dict[str, Any]:
    manifest.validate_existing_inputs()
    row_count = write_csv_dicts(
        manifest.output_csv,
        generate_lstm_rows(manifest),
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
        "generator": "lstm",
    }
    write_json(manifest.audit_json, audit)
    return audit
