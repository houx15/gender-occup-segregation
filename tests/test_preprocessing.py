"""Tests for shared preprocessing module."""

import pytest

from scripts.common.preprocessing import TOKENIZERS, tokenize


def test_whitespace_tokenizer():
    assert tokenize("hello world", "whitespace") == ["hello", "world"]


def test_whitespace_empty():
    assert tokenize("", "whitespace") == []


def test_jieba_tokenizer_basic():
    pytest.importorskip("jieba")
    tokens = tokenize("我爱北京天安门", "jieba")
    assert len(tokens) >= 2  # jieba will segment this into multiple words
    assert "".join(tokens) == "我爱北京天安门"


def test_nltk_en_tokenizer_basic():
    pytest.importorskip("nltk")
    tokens = tokenize("Hello, world! It's me.", "nltk_en")
    assert "hello" in [t.lower() for t in tokens]
    assert "world" in [t.lower() for t in tokens]


def test_unknown_tokenizer_raises():
    with pytest.raises(KeyError):
        tokenize("anything", "klingon")


def test_registry_contains_expected_keys():
    assert set(TOKENIZERS.keys()) >= {"whitespace", "jieba", "nltk_en"}
