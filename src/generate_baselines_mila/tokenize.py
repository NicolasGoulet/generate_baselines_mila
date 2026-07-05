"""Small tokenizer used by count-based baseline generators."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def tokenize_words(text: str, *, lowercase: bool = True) -> list[str]:
    """Return simple word tokens from cleaned CHILDES-style text."""

    if text is None:
        return []
    value = str(text)
    if lowercase:
        value = value.lower()
    return WORD_RE.findall(value)
