"""
Shared text preprocessing for all data sources.

Provides a tokenizer registry, a stopword registry, a language-aware text
cleaner, and a unified preprocess() entry point used by every corpus builder.

Imports of heavy tokenizer libraries (jieba, nltk) are lazy — loading this
module does not pull them in unless their tokenizer is actually called.
"""

from __future__ import annotations

import re
from typing import Callable


# ---------------------------------------------------------------------------
# Tokenizer registry
# ---------------------------------------------------------------------------

def tokenize_whitespace(text: str) -> list[str]:
    """Split on any whitespace. Empty string → empty list."""
    return text.split()


_jieba_module = None


def tokenize_jieba(text: str) -> list[str]:
    """Chinese segmentation via jieba (HMM on)."""
    global _jieba_module
    if _jieba_module is None:
        import jieba as _j
        _jieba_module = _j
    return [w for w in _jieba_module.lcut(text, HMM=True) if w and not w.isspace()]


_nltk_word_tokenize = None


def tokenize_nltk_en(text: str) -> list[str]:
    """English word tokenization via NLTK Penn Treebank."""
    global _nltk_word_tokenize
    if _nltk_word_tokenize is None:
        from nltk.tokenize import word_tokenize as _wt
        _nltk_word_tokenize = _wt
    return _nltk_word_tokenize(text)


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "whitespace": tokenize_whitespace,
    "jieba": tokenize_jieba,
    "nltk_en": tokenize_nltk_en,
}


def tokenize(text: str, tokenizer: str) -> list[str]:
    """Dispatch to the named tokenizer. Raises KeyError on unknown name."""
    fn = TOKENIZERS[tokenizer]
    return fn(text)
