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


# ---------------------------------------------------------------------------
# Stopword registry
# ---------------------------------------------------------------------------

_ZH_DEFAULT = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "他", "她", "它", "们", "为", "而", "以", "与", "及", "或",
    " ", "\t", "\n", "\r",
})

_ZH_WEIBO_EXTRA = frozenset({
    "帅哥", "美女", "闺蜜", "闺女", "老公", "老婆", "男友", "女友",
    "转发", "微博", "哈哈", "嘿嘿", "呵呵", "回复", "评论",
})

_ZH_NEWSPAPER_EXTRA = frozenset({
    "本报", "记者", "报道", "日前", "近日", "今天", "昨天", "今年",
    "讯", "电", "新华社", "中新社",
})


_nltk_stopwords_en = None


def _load_en_stopwords() -> frozenset[str]:
    global _nltk_stopwords_en
    if _nltk_stopwords_en is None:
        from nltk.corpus import stopwords as _sw
        _nltk_stopwords_en = frozenset(_sw.words("english"))
    return _nltk_stopwords_en


STOPWORDS: dict[str, "frozenset[str] | Callable[[], frozenset[str]]"] = {
    "zh_default":   _ZH_DEFAULT,
    "zh_weibo":     _ZH_DEFAULT | _ZH_WEIBO_EXTRA,
    "zh_newspaper": _ZH_DEFAULT | _ZH_NEWSPAPER_EXTRA,
    "en_default":   _load_en_stopwords,  # lazy
}


def get_stopwords(key: str) -> frozenset[str]:
    """Return the frozenset for the given stopword key. Loads lazily if needed."""
    val = STOPWORDS[key]
    if callable(val):
        resolved = val()
        STOPWORDS[key] = resolved  # memoize
        return resolved
    return val


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\S+")
_PAREN_RE = re.compile(r"[(\uff08][^)\uff09]*[)\uff09]")  # () and （）
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_BRACKETED_RE = re.compile(r"\[.*?\]")

KEEP_PATTERNS: dict[str, re.Pattern] = {
    "zh": re.compile(r"[\u4e00-\u9fff]+"),
    "en": re.compile(r"[a-zA-Z']+"),
}


def clean_text(
    text: str,
    language: str,
    *,
    lowercase: bool = False,
    strip_urls: bool = True,
    strip_mentions: bool = False,
    strip_parens: bool = False,
    strip_bracketed: bool = True,
    strip_zero_width: bool = True,
) -> str:
    """
    Clean raw text for downstream tokenization.

    Language 'zh' returns concatenated Chinese runs. Language 'en' returns a
    whitespace-joined string of lowercased letter tokens (if lowercase=True).
    """
    if not text or not isinstance(text, str):
        return ""

    if strip_urls:
        text = _URL_RE.sub("", text)
    if strip_bracketed:
        text = _BRACKETED_RE.sub("", text)
    if strip_mentions:
        text = _MENTION_RE.sub("", text)
    if strip_parens:
        text = _PAREN_RE.sub("", text)
    if strip_zero_width:
        text = _ZERO_WIDTH_RE.sub("", text)
    text = text.replace("…", "")

    if language == "zh":
        return "".join(KEEP_PATTERNS["zh"].findall(text)).strip()

    if language == "en":
        if lowercase:
            text = text.lower()
        tokens = KEEP_PATTERNS["en"].findall(text)
        return " ".join(tokens)

    raise ValueError(f"Unsupported language for clean_text: {language!r}")
