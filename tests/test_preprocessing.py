"""Tests for shared preprocessing module."""

import pytest

from scripts.common.preprocessing import (
    STOPWORDS,
    TOKENIZERS,
    clean_text,
    get_stopwords,
    tokenize,
)


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


def test_stopwords_zh_default_contains_common_particles():
    sw = get_stopwords("zh_default")
    assert "的" in sw
    assert "了" in sw
    assert "在" in sw


def test_stopwords_zh_weibo_includes_zh_default():
    default = get_stopwords("zh_default")
    weibo = get_stopwords("zh_weibo")
    assert default.issubset(weibo)


def test_stopwords_zh_newspaper_includes_journalism_terms():
    sw = get_stopwords("zh_newspaper")
    assert "本报" in sw
    assert "记者" in sw


def test_stopwords_en_default_contains_common_words():
    pytest.importorskip("nltk")
    sw = get_stopwords("en_default")
    assert "the" in sw
    assert "and" in sw


def test_stopwords_unknown_key_raises():
    with pytest.raises(KeyError):
        get_stopwords("martian_default")


def test_clean_text_zh_removes_urls():
    out = clean_text("看这个 http://example.com/foo 链接", "zh")
    assert "http" not in out
    assert "example" not in out


def test_clean_text_zh_keeps_chinese_only():
    out = clean_text("Hello世界 abc123 你好", "zh")
    assert "Hello" not in out
    assert "abc" not in out
    assert "世界" in out
    assert "你好" in out


def test_clean_text_en_lowercases_when_asked():
    out = clean_text("Hello WORLD", "en", lowercase=True)
    assert out == "hello world"


def test_clean_text_en_preserves_case_without_lowercase():
    out = clean_text("Hello WORLD", "en")
    assert out == "Hello WORLD"


def test_clean_text_en_strips_urls():
    out = clean_text("click http://example.com/x here", "en", lowercase=True)
    assert "http" not in out
    assert "click" in out
    assert "here" in out


def test_clean_text_strips_zero_width():
    out = clean_text("hi\u200bthere", "en", lowercase=True)
    assert "\u200b" not in out


def test_clean_text_en_mentions_and_parens():
    out = clean_text(
        "hi @user (aside) end", "en", lowercase=True,
        strip_mentions=True, strip_parens=True,
    )
    assert "@user" not in out
    assert "aside" not in out
    assert "hi" in out
    assert "end" in out


def test_clean_text_unsupported_language_raises():
    with pytest.raises(ValueError, match="Unsupported language"):
        clean_text("hello", "fr")
