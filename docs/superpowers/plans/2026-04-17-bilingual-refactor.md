# Bilingual Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Chinese-only gender-norms embedding pipeline into a bilingual (Chinese + English) pipeline. Phase 1 MVP adds Google Ngram English and free COHA n-grams, both as longitudinal analyses. Chinese pipeline must keep working at every step.

**Architecture:** Introduce a required top-level `language: zh|en` config key. Consolidate Chinese-specific text preprocessing (jieba, stopwords, CJK regex) into one shared module (`scripts/common/preprocessing.py`) with a tokenizer/stopword registry. Per-source builders stay flat but delegate text processing to the shared module. Wordlists move under language subdirs. Training, analysis, and embedding math remain language-agnostic (they already are). Visualization font handling becomes conditional on language.

**Tech Stack:** Python 3.10+, YAML config, gensim Word2Vec, jieba, NLTK (new), pandas, matplotlib, fire, pytest, Slurm.

**Design spec:** `docs/superpowers/specs/2026-04-17-bilingual-refactor-design.md` (read first).

---

## File Structure

### New files

- `scripts/common/preprocessing.py` — tokenizer registry, stopword registry, `clean_text`, `preprocess`
- `scripts/data_prep/build_corpora_ngram_en.py` — English Google 5-gram corpus builder
- `scripts/data_prep/download_coha.py` — COHA n-gram downloader
- `scripts/data_prep/build_corpora_coha.py` — COHA n-gram corpus builder
- `config/profiles/ngram_en_server.yml` — English Ngram + prestige
- `config/profiles/ngram_en_weat.yml` — English Ngram + WEAT
- `config/profiles/coha_server.yml` — COHA + WEAT
- `slurm/download_coha.slurm`, `slurm/build_corpus_coha.slurm`, `slurm/full_pipeline_en.slurm`
- `tests/test_preprocessing.py` — unit tests for preprocessing module
- `tests/test_config_loader.py` — unit tests for config validation
- `wordlists/en/prestige/occupations.txt`, `gender_words.json`, `prestige_axes.json`
- `wordlists/en/weat_formal/gender_words.json`, `domestic_work_words.json`, `leadership_words.json`, `stem_words.json`

### Moved (via `git mv`)

- `wordlists/prestige/` → `wordlists/zh/prestige/`
- `wordlists/weat_formal/` → `wordlists/zh/weat_formal/`
- `wordlists/weat_informal/` → `wordlists/zh/weat_informal/`
- `wordlists/gender_words_zh.json`, `gender_words_zh_backup.json`, `occup_category.json`, `occup_category_zh.json`, `occupations_zh.txt`, `prestige_axes_zh.json` — moved under `wordlists/zh/prestige/` (and renamed to drop `_zh` suffix where present)

### Modified files

- `scripts/common/config_loader.py` — language validation, compat matrix, defaults table, wordlist dir resolution
- `scripts/data_prep/download_ngrams.py` — URL parameterized by `ngram.language`
- `scripts/data_prep/build_corpora_rmrb.py` — delegates to `preprocessing.preprocess`
- `scripts/data_prep/build_corpora_weibo.py` — delegates to `preprocessing.preprocess`
- `scripts/data_prep/build_corpora_newspaper.py` — delegates to `preprocessing.preprocess`
- `scripts/visualize.py` — extract `_configure_fonts(language)`, add `LABELS` dict, guard Chinese-only code paths
- `scripts/analyze_correlation.py` — same font extraction, language guards
- `config/config.example.yml` — new keys documented
- `config/profiles/*.yml` (7 files) — add `language: zh`, update `wordlists.dir`
- `run_pipeline.sh` — extend Stage 1 and Stage 2 dispatch for new sources
- `requirements.txt` — add `nltk>=3.8`
- `setup_server.sh` — install NLTK data
- `README.md` — new "English pipeline" section
- `tests/test_build_corpora.py` — add English fixtures
- `tests/test_analyze_embeddings.py` — add English wordlist fixture

### Deleted

- `wordlists/test.py` — stray file

---

## Conventions used in this plan

- **Commit message prefix** follows the existing repo style (imperative verb, no Conventional Commits). Each commit ends with the two-line Co-Authored-By trailer.
- **Testing command:** `pytest -xvs tests/<file>::<test_name>` for single tests, `pytest -x tests/` for the full suite.
- **Commit trailer block** (reuse in every commit):
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```

---

## Task 1: Add `language` validation + compatibility matrix to `config_loader.py`

**Files:**
- Modify: `scripts/common/config_loader.py`
- Test: `tests/test_config_loader.py` (new)

- [ ] **Step 1: Write failing test for language validation**

Create `tests/test_config_loader.py`:

```python
"""Tests for the bilingual config loader."""

import pytest
import yaml
from pathlib import Path

from scripts.common.config_loader import load_config


def _write_config(tmp_path: Path, overrides: dict) -> Path:
    base = {
        "language": "zh",
        "data_source": "ngram",
        "paths": {
            "base_dir": str(tmp_path),
            "corpora_dir": "data/corpora",
            "models_dir": "data/models",
            "results_dir": "data/results",
            "log_dir": "logs",
            "raw_ngram_dir": "data/raw_ngrams",
        },
        "time_slices": {
            "start_year": 1940,
            "end_year": 2015,
            "window_size": 10,
            "step_size": 5,
        },
    }
    # Deep-merge overrides
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k].update(v)
        else:
            base[k] = v
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def test_missing_language_raises(tmp_path):
    path = _write_config(tmp_path, {"language": None})
    # Remove key entirely
    data = yaml.safe_load(path.read_text())
    data.pop("language")
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="language"):
        load_config(str(path))


def test_invalid_language_raises(tmp_path):
    path = _write_config(tmp_path, {"language": "fr"})
    with pytest.raises(ValueError, match="language"):
        load_config(str(path))


def test_incompatible_language_data_source_raises(tmp_path):
    # renminribao is zh-only
    path = _write_config(tmp_path, {"language": "en", "data_source": "renminribao"})
    with pytest.raises(ValueError, match="not compatible"):
        load_config(str(path))


def test_en_ngram_accepted(tmp_path):
    path = _write_config(tmp_path, {"language": "en"})
    config = load_config(str(path))
    assert config["language"] == "en"
    assert config["data_source"] == "ngram"


def test_en_coha_accepted(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "language": "en",
            "data_source": "coha",
            "coha": {
                "ngram_order": 4,
                "source_archive_urls": ["http://example.com/coha.zip"],
                "decade_min": 1810,
                "decade_max": 2000,
            },
        },
    )
    # coha doesn't require time_slices; remove it
    data = yaml.safe_load(path.read_text())
    data.pop("time_slices", None)
    # coha doesn't require raw_ngram_dir; add raw_coha_dir
    data["paths"].pop("raw_ngram_dir", None)
    data["paths"]["raw_coha_dir"] = "data/raw_coha"
    data["paths"]["coha_decompressed_dir"] = "data/raw_coha_decompressed"
    path.write_text(yaml.safe_dump(data))

    config = load_config(str(path))
    assert config["data_source"] == "coha"
    assert config["language"] == "en"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest -xvs tests/test_config_loader.py::test_missing_language_raises
```

Expected: FAIL — loader doesn't check `language` yet.

- [ ] **Step 3: Implement language validation + compat matrix in `config_loader.py`**

Edit `scripts/common/config_loader.py`. At the top, add:

```python
VALID_LANGUAGES = {"zh", "en"}

DATA_SOURCE_LANGUAGE_COMPAT = {
    "ngram":       {"zh", "en"},
    "renminribao": {"zh"},
    "weibo":       {"zh"},
    "newspaper":   {"zh"},
    "coha":        {"en"},
}
```

Update `DATA_SOURCE_DEFAULTS` to include `coha`:

```python
DATA_SOURCE_DEFAULTS = {
    "ngram": "longitudinal",
    "renminribao": "longitudinal",
    "weibo": "provincial",
    "newspaper": "provincial",
    "coha": "longitudinal",
}
```

Replace the body of `_validate_config()` with:

```python
def _validate_config(config: dict) -> None:
    """Validate required fields based on data_source type."""
    language = config.get("language")
    if not language:
        raise ValueError("Missing required top-level key: language ('zh' or 'en')")
    if language not in VALID_LANGUAGES:
        raise ValueError(
            f"Invalid language: {language!r}. Must be one of: {sorted(VALID_LANGUAGES)}"
        )

    data_source = config.get("data_source")
    if data_source not in DATA_SOURCE_DEFAULTS:
        raise ValueError(
            f"Invalid data_source: {data_source!r}. "
            f"Must be one of: {list(DATA_SOURCE_DEFAULTS.keys())}"
        )

    compat = DATA_SOURCE_LANGUAGE_COMPAT.get(data_source, set())
    if language not in compat:
        raise ValueError(
            f"data_source={data_source!r} is not compatible with language={language!r}. "
            f"Allowed languages for {data_source!r}: {sorted(compat)}"
        )

    analysis_mode = config.get("analysis_mode")
    if analysis_mode and analysis_mode not in VALID_ANALYSIS_MODES:
        raise ValueError(
            f"Invalid analysis_mode: {analysis_mode}. "
            f"Must be one of: {list(VALID_ANALYSIS_MODES)}"
        )

    paths = config.get("paths", {})
    required_paths = ["base_dir", "corpora_dir", "models_dir", "results_dir", "log_dir"]
    for key in required_paths:
        if key not in paths:
            raise ValueError(f"Missing required path: paths.{key}")

    if data_source == "ngram":
        if "raw_ngram_dir" not in paths:
            raise ValueError("ngram data_source requires paths.raw_ngram_dir")
    if data_source == "coha":
        if "raw_coha_dir" not in paths:
            raise ValueError("coha data_source requires paths.raw_coha_dir")
        if "coha" not in config:
            raise ValueError("coha data_source requires a top-level 'coha' config block")
    if data_source in ("ngram", "renminribao"):
        if "time_slices" not in config:
            raise ValueError(f"{data_source} data_source requires time_slices config")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest -xvs tests/test_config_loader.py
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/common/config_loader.py tests/test_config_loader.py
git commit -m "$(cat <<'EOF'
Add language validation and compat matrix to config loader

Introduces required top-level `language` key ('zh' | 'en') and the
(data_source, language) compatibility matrix. No callers use the new
key yet; existing profiles will be migrated in a later commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Extend `_set_defaults` and `get_wordlist_dir` for language-aware defaults

**Files:**
- Modify: `scripts/common/config_loader.py`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write failing tests for defaults and wordlist dir**

Append to `tests/test_config_loader.py`:

```python
def test_zh_ngram_defaults(tmp_path):
    path = _write_config(tmp_path, {"language": "zh"})
    config = load_config(str(path))
    assert config["corpus"]["tokenizer"] == "whitespace"
    assert config["corpus"]["lowercase"] is False
    assert config["corpus"].get("stopwords") in (None, "")


def test_zh_renminribao_defaults(tmp_path):
    path = _write_config(tmp_path, {"language": "zh", "data_source": "renminribao"})
    config = load_config(str(path))
    assert config["corpus"]["tokenizer"] == "jieba"
    assert config["corpus"]["stopwords"] == "zh_default"
    assert config["corpus"]["lowercase"] is False


def test_en_ngram_defaults(tmp_path):
    path = _write_config(tmp_path, {"language": "en"})
    config = load_config(str(path))
    assert config["corpus"]["tokenizer"] == "whitespace"
    assert config["corpus"]["lowercase"] is True


def test_explicit_stopwords_override_default(tmp_path):
    path = _write_config(
        tmp_path,
        {"language": "zh", "data_source": "weibo", "corpus": {"stopwords": "zh_default"}},
    )
    config = load_config(str(path))
    assert config["corpus"]["stopwords"] == "zh_default"  # override, not zh_weibo


def test_wordlist_dir_under_language_subdir(tmp_path):
    from scripts.common.config_loader import get_wordlist_dir

    path = _write_config(tmp_path, {"language": "zh"})
    config = load_config(str(path))
    wl = get_wordlist_dir(config)
    # For ngram + prestige default mode, expect wordlists/zh/prestige
    assert str(wl).endswith("wordlists/zh/prestige")


def test_wordlist_dir_explicit_wins(tmp_path, monkeypatch):
    from scripts.common.config_loader import get_wordlist_dir

    path = _write_config(
        tmp_path,
        {"language": "en", "wordlists": {"dir": "wordlists/en/weat_formal"}},
    )
    # Create dir so loader resolves it
    (tmp_path / "wordlists" / "en" / "weat_formal").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    config = load_config(str(path))
    wl = get_wordlist_dir(config)
    assert str(wl).endswith("wordlists/en/weat_formal")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest -xvs tests/test_config_loader.py::test_zh_renminribao_defaults
```

Expected: FAIL — `stopwords` key missing.

- [ ] **Step 3: Implement language-aware defaults**

Replace `_set_defaults()` in `scripts/common/config_loader.py`:

```python
# Defaults by (language, data_source) — order: (tokenizer, stopwords, lowercase)
_CORPUS_DEFAULTS = {
    ("zh", "ngram"):       ("whitespace", None,             False),
    ("zh", "renminribao"): ("jieba",      "zh_default",     False),
    ("zh", "weibo"):       ("jieba",      "zh_weibo",       False),
    ("zh", "newspaper"):   ("jieba",      "zh_newspaper",   False),
    ("en", "ngram"):       ("whitespace", None,             True),
    ("en", "coha"):        ("whitespace", None,             True),
}


def _set_defaults(config: dict) -> None:
    """Set default values based on (language, data_source)."""
    language = config["language"]
    data_source = config["data_source"]

    if "analysis_unit" not in config:
        config["analysis_unit"] = DATA_SOURCE_DEFAULTS[data_source]

    if "analysis_mode" not in config:
        if data_source in ("ngram", "renminribao", "coha"):
            config["analysis_mode"] = "prestige" if language == "zh" and data_source != "coha" else "weat"
            # Simpler: ngram/renminribao default to prestige; coha defaults to weat
            if data_source == "coha":
                config["analysis_mode"] = "weat"
            elif data_source in ("ngram", "renminribao"):
                config["analysis_mode"] = "prestige"
        else:
            config["analysis_mode"] = "weat"

    corpus = config.setdefault("corpus", {})
    tok_default, sw_default, lc_default = _CORPUS_DEFAULTS[(language, data_source)]
    corpus.setdefault("tokenizer", tok_default)
    if sw_default is not None:
        corpus.setdefault("stopwords", sw_default)
    corpus.setdefault("lowercase", lc_default)
```

Replace `get_wordlist_dir()`:

```python
def get_wordlist_dir(config: dict) -> Path:
    """Get the resolved wordlist directory path (language-aware)."""
    wl_dir = config.get("wordlists", {}).get("dir")
    if wl_dir:
        return Path(wl_dir)

    # Fallback based on (language, analysis_mode, data_source)
    repo_root = Path.cwd()
    language = config["language"]
    analysis_mode = config.get("analysis_mode", "prestige")

    if analysis_mode == "prestige":
        return repo_root / "wordlists" / language / "prestige"

    # WEAT
    data_source = config["data_source"]
    if data_source == "weibo":
        return repo_root / "wordlists" / language / "weat_informal"
    return repo_root / "wordlists" / language / "weat_formal"
```

- [ ] **Step 4: Run tests**

```bash
pytest -xvs tests/test_config_loader.py
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/common/config_loader.py tests/test_config_loader.py
git commit -m "$(cat <<'EOF'
Add language-aware defaults and wordlist dir resolution

Corpus tokenizer/stopwords/lowercase now defaulted from a
(language, data_source) table. get_wordlist_dir resolves under
wordlists/{language}/{mode}/. Loader behavior unchanged when callers
pass explicit values.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create `scripts/common/preprocessing.py` skeleton with tokenizer registry

**Files:**
- Create: `scripts/common/preprocessing.py`
- Test: `tests/test_preprocessing.py` (new)

- [ ] **Step 1: Write failing tests for tokenizer registry**

Create `tests/test_preprocessing.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest -xvs tests/test_preprocessing.py::test_whitespace_tokenizer
```

Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Create `scripts/common/preprocessing.py` with tokenizer registry**

```python
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
```

- [ ] **Step 4: Install nltk if missing, download required data**

```bash
pip install "nltk>=3.8"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

- [ ] **Step 5: Run tests**

```bash
pytest -xvs tests/test_preprocessing.py
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/common/preprocessing.py tests/test_preprocessing.py
git commit -m "$(cat <<'EOF'
Add tokenizer registry to shared preprocessing module

Introduces TOKENIZERS dict with whitespace, jieba, and nltk_en
tokenizers. Heavy imports (jieba, nltk) are lazy so Chinese-only
environments don't need NLTK and vice versa.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add stopword registry and `clean_text` to preprocessing module

**Files:**
- Modify: `scripts/common/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for stopwords and `clean_text`**

Append to `tests/test_preprocessing.py`:

```python
from scripts.common.preprocessing import (
    STOPWORDS,
    clean_text,
    get_stopwords,
)


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
    # Only Chinese chars survive
    assert "Hello" not in out
    assert "abc" not in out
    assert "世界" in out
    assert "你好" in out


def test_clean_text_en_lowercases_when_asked():
    out = clean_text("Hello WORLD", "en", lowercase=True)
    assert out == "hello world"


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest -xvs tests/test_preprocessing.py::test_stopwords_zh_default_contains_common_particles
```

Expected: FAIL — `STOPWORDS` not defined.

- [ ] **Step 3: Append to `scripts/common/preprocessing.py`**

Add after the tokenizer registry:

```python
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
    "en": re.compile(r"[a-z']+"),
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
```

- [ ] **Step 4: Download NLTK stopwords corpus**

```bash
python -c "import nltk; nltk.download('stopwords')"
```

- [ ] **Step 5: Run tests**

```bash
pytest -xvs tests/test_preprocessing.py
```

Expected: 17 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/common/preprocessing.py tests/test_preprocessing.py
git commit -m "$(cat <<'EOF'
Add stopword registry and clean_text to preprocessing module

Stopword sets for zh_default / zh_weibo / zh_newspaper / en_default,
mirroring the inline sets scattered across current Chinese builders.
clean_text is language-aware: 'zh' keeps only CJK runs, 'en' keeps
letter tokens and can lowercase.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add `preprocess()` one-shot pipeline

**Files:**
- Modify: `scripts/common/preprocessing.py`
- Modify: `tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests for `preprocess`**

Append to `tests/test_preprocessing.py`:

```python
from scripts.common.preprocessing import preprocess


def test_preprocess_zh_rmrb_sample():
    pytest.importorskip("jieba")
    text = "本报讯 新华社北京电 我爱北京天安门 http://x.com"
    out = preprocess(
        text,
        language="zh",
        tokenizer="jieba",
        stopwords_key="zh_newspaper",
        lowercase=False,
        min_words=2,
    )
    assert out is not None
    assert "本报" not in out  # stopword
    assert "北京" in out
    assert all(len(w) > 0 for w in out)


def test_preprocess_min_words_filters_short_docs():
    pytest.importorskip("jieba")
    out = preprocess(
        "你",
        language="zh",
        tokenizer="jieba",
        stopwords_key="zh_default",
        lowercase=False,
        min_words=5,
    )
    assert out is None


def test_preprocess_en_cleans_and_lowercases():
    tokens = preprocess(
        "Hello WORLD http://x.com",
        language="en",
        tokenizer="whitespace",
        stopwords_key="en_default",
        lowercase=True,
        min_words=1,
    )
    assert tokens is not None
    assert "hello" in tokens
    assert "world" in tokens
    assert not any(t.startswith("http") for t in tokens)


def test_preprocess_no_stopword_filter_when_key_none():
    tokens = preprocess(
        "hello world",
        language="en",
        tokenizer="whitespace",
        stopwords_key=None,
        lowercase=True,
        min_words=1,
    )
    assert tokens == ["hello", "world"]


def test_preprocess_passes_cleaner_opts():
    tokens = preprocess(
        "hi (aside) end",
        language="en",
        tokenizer="whitespace",
        stopwords_key=None,
        lowercase=True,
        min_words=1,
        cleaner_opts={"strip_parens": True},
    )
    assert tokens is not None
    assert "aside" not in tokens
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest -xvs tests/test_preprocessing.py::test_preprocess_en_cleans_and_lowercases
```

Expected: FAIL — `preprocess` not defined.

- [ ] **Step 3: Append `preprocess()` to `scripts/common/preprocessing.py`**

```python
# ---------------------------------------------------------------------------
# One-shot preprocess pipeline
# ---------------------------------------------------------------------------

def preprocess(
    text: str,
    *,
    language: str,
    tokenizer: str,
    stopwords_key: str | None,
    lowercase: bool,
    min_words: int,
    cleaner_opts: dict | None = None,
) -> list[str] | None:
    """
    Clean → tokenize → filter stopwords → length-check.

    Returns the filtered token list, or None when the document should be
    dropped (too short, or empty after cleaning).
    """
    opts = dict(cleaner_opts or {})
    opts["lowercase"] = lowercase
    cleaned = clean_text(text, language, **opts)
    if not cleaned:
        return None

    tokens = tokenize(cleaned, tokenizer)
    if not tokens:
        return None

    if stopwords_key:
        sw = get_stopwords(stopwords_key)
        tokens = [t for t in tokens if t and t.strip() and t not in sw]

    if len(tokens) < min_words:
        return None

    return tokens
```

- [ ] **Step 4: Run tests**

```bash
pytest -xvs tests/test_preprocessing.py
```

Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/common/preprocessing.py tests/test_preprocessing.py
git commit -m "$(cat <<'EOF'
Add preprocess() one-shot pipeline

Single entry point every corpus builder will use: clean → tokenize →
filter stopwords → length-check. Returns None when a document should
be dropped.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Move wordlists under `wordlists/zh/` and drop `_zh` suffixes

**Files:**
- Move: `wordlists/prestige/*` → `wordlists/zh/prestige/*`
- Move: `wordlists/weat_formal/*` → `wordlists/zh/weat_formal/*`
- Move: `wordlists/weat_informal/*` → `wordlists/zh/weat_informal/*`
- Move: `wordlists/*_zh.{json,txt}` → `wordlists/zh/prestige/*` (with renames)
- Move: `wordlists/occup_category*.json` → `wordlists/zh/prestige/`
- Move: `wordlists/gender_words_zh_backup.json` → `wordlists/zh/prestige/`
- Delete: `wordlists/test.py`

- [ ] **Step 1: Inspect current wordlist layout**

```bash
ls wordlists/
ls wordlists/prestige/ wordlists/weat_formal/ wordlists/weat_informal/
```

Expected: 3 subdirs (`prestige`, `weat_formal`, `weat_informal`) + top-level loose files.

- [ ] **Step 2: Create target dirs and move mode subdirs**

```bash
mkdir -p wordlists/zh
git mv wordlists/prestige wordlists/zh/prestige
git mv wordlists/weat_formal wordlists/zh/weat_formal
git mv wordlists/weat_informal wordlists/zh/weat_informal
```

- [ ] **Step 3: Move top-level Chinese files into `wordlists/zh/prestige/` and drop `_zh` suffixes**

```bash
git mv wordlists/occupations_zh.txt wordlists/zh/prestige/occupations.txt
git mv wordlists/gender_words_zh.json wordlists/zh/prestige/gender_words.json
git mv wordlists/prestige_axes_zh.json wordlists/zh/prestige/prestige_axes.json
git mv wordlists/occup_category.json wordlists/zh/prestige/occup_category.json
git mv wordlists/occup_category_zh.json wordlists/zh/prestige/occup_category_zh.json
git mv wordlists/gender_words_zh_backup.json wordlists/zh/prestige/gender_words_zh_backup.json
```

Note: If any of these top-level files are duplicates of files already inside `wordlists/zh/prestige/` (because prestige subdir already contained `occupations_zh.txt` etc.), compare them first:

```bash
diff wordlists/zh/prestige/occupations.txt wordlists/zh/prestige/occupations_zh.txt 2>/dev/null
```

If identical, keep the dropped-suffix one and `git rm` the `_zh` version. If not, investigate — the prestige subdir was likely the canonical copy; remove the loose top-level copy.

- [ ] **Step 4: Rename files inside `wordlists/zh/prestige/` to drop `_zh` suffix**

```bash
# If these still exist after Step 3, rename
test -f wordlists/zh/prestige/occupations_zh.txt && git mv wordlists/zh/prestige/occupations_zh.txt wordlists/zh/prestige/occupations.txt
test -f wordlists/zh/prestige/gender_words_zh.json && git mv wordlists/zh/prestige/gender_words_zh.json wordlists/zh/prestige/gender_words.json
test -f wordlists/zh/prestige/prestige_axes_zh.json && git mv wordlists/zh/prestige/prestige_axes_zh.json wordlists/zh/prestige/prestige_axes.json
```

- [ ] **Step 5: Delete stray `test.py`**

```bash
git rm wordlists/test.py
```

- [ ] **Step 6: Verify final layout**

```bash
find wordlists -type f | sort
```

Expected (Chinese side, pre-English):
```
wordlists/zh/prestige/gender_words.json
wordlists/zh/prestige/gender_words_zh_backup.json
wordlists/zh/prestige/occup_category.json
wordlists/zh/prestige/occup_category_zh.json
wordlists/zh/prestige/occupations.txt
wordlists/zh/prestige/prestige_axes.json
wordlists/zh/weat_formal/domestic_work_words.json
wordlists/zh/weat_formal/gender_words.json
wordlists/zh/weat_formal/leadership_words.json
wordlists/zh/weat_formal/stem_words.json
wordlists/zh/weat_informal/domestic_work_words.json
wordlists/zh/weat_informal/gender_words.json
wordlists/zh/weat_informal/leadership_words.json
wordlists/zh/weat_informal/stem_words.json
```

- [ ] **Step 7: Commit**

```bash
git add -A wordlists/
git commit -m "$(cat <<'EOF'
Reorganize wordlists under wordlists/zh/ language subdir

Move prestige/ weat_formal/ weat_informal/ under wordlists/zh/ and drop
_zh suffixes from individual files. Prepares for bilingual layout with
wordlists/en/ added in a later commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Migrate 7 Chinese profiles to add `language: zh` and update wordlist paths

**Files:**
- Modify: `config/profiles/ngram_server.yml`
- Modify: `config/profiles/ngram_weat.yml`
- Modify: `config/profiles/renminribao_server.yml`
- Modify: `config/profiles/renminribao_weat.yml`
- Modify: `config/profiles/weibo_server.yml`
- Modify: `config/profiles/newspaper_server.yml`
- Modify: `config/profiles/provincial_newspaper_server.yml`
- Modify: `config/config.example.yml`

- [ ] **Step 1: Check each profile's current `wordlists.dir` and file refs**

```bash
grep -nE '^(wordlists|  dir|  occupations_file|  gender_words_file|  prestige_axes_file)' config/profiles/*.yml config/config.example.yml
```

Note which profiles use `wordlists/prestige`, `wordlists/weat_formal`, or `wordlists/weat_informal`.

- [ ] **Step 2: For each profile, add `language: zh` at the top and update `wordlists.dir`**

For each of the 7 profiles + `config.example.yml`, apply these edits:

1. Insert `language: "zh"` as the first non-comment line after the header.
2. Change `wordlists.dir` value:
   - `wordlists/prestige` → `wordlists/zh/prestige`
   - `wordlists/weat_formal` → `wordlists/zh/weat_formal`
   - `wordlists/weat_informal` → `wordlists/zh/weat_informal`
3. In the `wordlists:` block, change filenames to drop `_zh`:
   - `occupations_file: "occupations_zh.txt"` → `occupations_file: "occupations.txt"`
   - `gender_words_file: "gender_words_zh.json"` → `gender_words_file: "gender_words.json"`
   - `prestige_axes_file: "prestige_axes_zh.json"` → `prestige_axes_file: "prestige_axes.json"`

Example diff on `config/profiles/ngram_server.yml`:

```diff
 # ngram + prestige profile (network scratch)
+language: "zh"
 data_source: "ngram"
 analysis_mode: "prestige"
 ...
 wordlists:
-  dir: "wordlists/prestige"
-  occupations_file: "occupations_zh.txt"
-  gender_words_file: "gender_words_zh.json"
-  prestige_axes_file: "prestige_axes_zh.json"
+  dir: "wordlists/zh/prestige"
+  occupations_file: "occupations.txt"
+  gender_words_file: "gender_words.json"
+  prestige_axes_file: "prestige_axes.json"
```

- [ ] **Step 3: Update `config/config.example.yml` comment block**

In `config/config.example.yml`, update:

1. Add at the very top (after the header comments):
   ```yaml
   # Language: "zh" or "en" (REQUIRED)
   language: "zh"
   ```
2. Update the wordlists block comment:
   ```yaml
   wordlists:
     dir: "wordlists/zh/prestige"  # e.g., wordlists/{zh,en}/{prestige,weat_formal,weat_informal}
     occupations_file: "occupations.txt"
     gender_words_file: "gender_words.json"
     prestige_axes_file: "prestige_axes.json"
     weat_gender_file: "gender_words.json"
     weat_domestic_work_file: "domestic_work_words.json"
     weat_leadership_file: "leadership_words.json"
     weat_stem_file: "stem_words.json"
   ```

- [ ] **Step 4: Verify every profile loads without errors**

```bash
for cfg in config/profiles/*.yml; do
  echo "=== $cfg ==="
  python -c "from scripts.common.config_loader import load_config; c = load_config('$cfg'); print('OK', c['language'], c['data_source'])"
done
```

Expected: 7 `OK zh <source>` lines.

- [ ] **Step 5: Run full existing test suite to confirm Chinese pipeline still passes**

```bash
pytest -x tests/
```

Expected: all existing tests pass (Chinese tests + new config/preprocessing tests).

- [ ] **Step 6: Commit**

```bash
git add config/
git commit -m "$(cat <<'EOF'
Migrate Chinese profiles to explicit language key and zh wordlists

Every existing profile now declares language: zh and points at
wordlists/zh/{mode}. Wordlist file references drop the _zh suffix.
Breaking change: configs without a language key are now rejected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Refactor `build_corpora_rmrb.py` to use shared `preprocess()`

**Files:**
- Modify: `scripts/data_prep/build_corpora_rmrb.py`

- [ ] **Step 1: Identify what the file currently does**

Re-read `scripts/data_prep/build_corpora_rmrb.py`. Key inline definitions to replace:
- `STOPWORDS` (~lines 29–34) — move to preprocessing module (already done in Task 4).
- `CHINESE_RE` (line 36) — replaced by `preprocessing.clean_text`.
- `clean_text()` (~lines 51–58) — replaced by `preprocessing.clean_text(text, "zh")`.
- `segment_text()` (~lines 61–68) — replaced by `preprocessing.preprocess()`.

- [ ] **Step 2: Replace the preprocessing block with a call to `preprocess()`**

In `scripts/data_prep/build_corpora_rmrb.py`:

1. Delete local `STOPWORDS` and `CHINESE_RE` definitions.
2. Delete local `clean_text()` and `segment_text()` functions.
3. Remove `import jieba`.
4. Add imports:
   ```python
   from scripts.common.preprocessing import preprocess
   ```
5. Find the call site(s) of the old `segment_text()`. Replace:
   ```python
   # OLD:
   cleaned = clean_text(line)
   segmented = segment_text(cleaned, min_words=5)
   if segmented:
       out.write(segmented + "\n")
   ```
   with:
   ```python
   # NEW:
   min_words = config.get("corpus", {}).get("min_words", 5)
   tokens = preprocess(
       line,
       language=config["language"],
       tokenizer=config["corpus"]["tokenizer"],
       stopwords_key=config["corpus"].get("stopwords"),
       lowercase=config["corpus"].get("lowercase", False),
       min_words=min_words,
   )
   if tokens is not None:
       out.write(" ".join(tokens) + "\n")
   ```
6. Any helper that previously took `(text)` and returned a segmented string must now take `(text, config)` and call `preprocess`. Thread `config` through.

- [ ] **Step 3: Run the existing build-corpora test**

```bash
pytest -xvs tests/test_build_corpora.py
```

Expected: existing tests still pass (they may need `config["language"]` in fixtures — if so, update the fixture to set `language: "zh"`).

- [ ] **Step 4: Smoke-test on a tiny hand-crafted rmrb-like fixture**

Create `tests/fixtures/rmrb_sample/1940s/1945/报刊/人民日报/rmrb_1945_05.txt` with ~20 lines of fake Chinese text. Write a quick one-off script or pytest that exercises `build_corpora` on it, verifies output corpus files are created and non-empty.

If no such fixture test exists, add one:

```python
# tests/test_build_corpora.py (extend)
def test_rmrb_builder_produces_corpus(tmp_path):
    pytest.importorskip("jieba")
    raw = tmp_path / "raw" / "1940s" / "1945" / "报刊" / "人民日报"
    raw.mkdir(parents=True)
    (raw / "rmrb_1945_05.txt").write_text(
        "新华社北京电 我爱北京天安门 http://example.com 工人阶级伟大\n" * 20,
        encoding="gb18030",
    )
    # Build minimal config
    # ... (wire up load_config with tmp_path paths; call build_corpora)
    # Assert corpora_dir has a 1940_1949 or similar slice with non-empty corpus files.
```

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/build_corpora_rmrb.py tests/test_build_corpora.py
git commit -m "$(cat <<'EOF'
Route rmrb builder through shared preprocess()

Delete inline STOPWORDS, CHINESE_RE, clean_text, segment_text from
rmrb builder; call scripts.common.preprocessing.preprocess instead.
Tokenizer, stopwords, and lowercase flags come from config defaults.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Refactor `build_corpora_weibo.py` to use shared `preprocess()`

**Files:**
- Modify: `scripts/data_prep/build_corpora_weibo.py`

- [ ] **Step 1: Identify weibo-specific preprocessing flags**

Weibo differs from rmrb in two places:
- Content has `@mentions` and smart-quote `"..."` — mentions must be stripped (`strip_mentions=True`).
- Stopwords default is `zh_weibo` (already in config).

- [ ] **Step 2: Edit `scripts/data_prep/build_corpora_weibo.py`**

1. Delete local `STOPWORDS`, `CHINESE_RE`, `clean_weibo_content`, `preprocess_batch` preprocessing internals (keep the parquet-reading + batching frame).
2. Remove `import jieba` and the `jieba_fast` try/except fallback.
3. Add `from scripts.common.preprocessing import preprocess`.
4. In the batch-processing loop, replace the old clean→segment flow with:
   ```python
   tokens = preprocess(
       raw_text,
       language=config["language"],
       tokenizer=config["corpus"]["tokenizer"],
       stopwords_key=config["corpus"].get("stopwords"),
       lowercase=config["corpus"].get("lowercase", False),
       min_words=config.get("corpus", {}).get("min_words", 5),
       cleaner_opts={"strip_mentions": True, "strip_parens": True},
   )
   if tokens is None:
       continue
   line = " ".join(tokens)
   ```
5. Keep `PROVINCE_CODE_TO_NAME`, `PROVINCE_NAME_TO_CODE`, `PROVINCE_GROUPS`, and `RollingFile` unchanged.

- [ ] **Step 3: Run tests**

```bash
pytest -x tests/
```

Expected: all pass.

- [ ] **Step 4: Smoke test on a tiny weibo-like parquet fixture**

If `tests/test_build_corpora.py` already has a weibo fixture, ensure it passes; otherwise add one that constructs a small pandas DataFrame, writes parquet, and calls the builder on `tmp_path`.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/build_corpora_weibo.py tests/test_build_corpora.py
git commit -m "$(cat <<'EOF'
Route weibo builder through shared preprocess()

Delete inline preprocessing; delegate to preprocessing.preprocess with
strip_mentions=True and strip_parens=True via cleaner_opts. Keeps the
province mapping and RollingFile logic intact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Refactor `build_corpora_newspaper.py` to use shared `preprocess()`

**Files:**
- Modify: `scripts/data_prep/build_corpora_newspaper.py`

- [ ] **Step 1: Edit `scripts/data_prep/build_corpora_newspaper.py`**

1. Delete local `STOPWORDS`, `clean_text`, `segment_text`.
2. Remove `import jieba`.
3. Add `from scripts.common.preprocessing import preprocess`.
4. Replace the old clean→segment block in the doc-processing loop:
   ```python
   tokens = preprocess(
       text,
       language=config["language"],
       tokenizer=config["corpus"]["tokenizer"],
       stopwords_key=config["corpus"].get("stopwords"),
       lowercase=config["corpus"].get("lowercase", False),
       min_words=config.get("corpus", {}).get("min_words", 5),
       cleaner_opts={"strip_parens": True},
   )
   if tokens is None:
       continue
   line = " ".join(tokens)
   ```
5. Keep newspaper→province mapping, `ProvinceCorpusWriter`, and JSONL reading.

- [ ] **Step 2: Run tests**

```bash
pytest -x tests/
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add scripts/data_prep/build_corpora_newspaper.py
git commit -m "$(cat <<'EOF'
Route newspaper builder through shared preprocess()

Delete inline preprocessing; delegate to preprocessing.preprocess with
strip_parens=True via cleaner_opts. Journalism stopwords default
(zh_newspaper) picked up from config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Refactor `visualize.py` — extract font config, add labels, language guards

**Files:**
- Modify: `scripts/visualize.py`

- [ ] **Step 1: Extract font registration into `_configure_fonts(language)`**

At the top of `scripts/visualize.py`, replace the import-time CJK font setup (lines ~27–42) with:

```python
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt


_DEFAULT_CJK_FONT_PATH = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"


def _configure_fonts(config: dict) -> None:
    """Register language-appropriate fonts with matplotlib."""
    language = config["language"]
    if language != "zh":
        return  # matplotlib defaults are fine for English

    cjk_path = config.get("fonts", {}).get("cjk_path", _DEFAULT_CJK_FONT_PATH)
    try:
        _fm.fontManager.addfont(cjk_path)
        family = _fm.FontProperties(fname=cjk_path).get_name()
        plt.rcParams["font.sans-serif"] = [family] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except FileNotFoundError:
        # Non-fatal; rely on system default and let CJK glyphs render as-available.
        pass
```

Remove the old top-level side effect. Callers will invoke `_configure_fonts(config)` from the `main()` entry point immediately after `load_config()`.

- [ ] **Step 2: Add `LABELS` dict near the top of the module**

```python
LABELS = {
    "zh": {
        "year": "年份",
        "start_year": "起始年份",
        "province": "省份",
        "state": "州",
        "gender_norm": "性别规范指数",
        "cohens_d": "Cohen's d 效应量",
        "cohens_d_abs": "|Cohen's d|",
        "prestige": "声望",
        "evaluation": "评价",
        "potency": "力量",
        "activity": "活动",
        "gender_axis": "性别轴投影",
        "work_family": "工作-家庭",
        "leadership": "领导力",
        "stem": "STEM",
        "male": "男性",
        "female": "女性",
        "occupation": "职业",
        "correlation": "皮尔逊相关系数",
        "slice": "时间窗",
        "value": "值",
    },
    "en": {
        "year": "Year",
        "start_year": "Start year",
        "province": "State",
        "state": "State",
        "gender_norm": "Gender norm index",
        "cohens_d": "Cohen's d",
        "cohens_d_abs": "|Cohen's d|",
        "prestige": "Prestige",
        "evaluation": "Evaluation",
        "potency": "Potency",
        "activity": "Activity",
        "gender_axis": "Gender-axis projection",
        "work_family": "Work–Family",
        "leadership": "Leadership",
        "stem": "STEM",
        "male": "Male",
        "female": "Female",
        "occupation": "Occupation",
        "correlation": "Pearson r",
        "slice": "Time window",
        "value": "Value",
    },
}


def L(config: dict, key: str) -> str:
    """Look up a user-facing label in the current language. Unknown keys fall back to the key itself."""
    return LABELS.get(config["language"], {}).get(key, key)
```

- [ ] **Step 3: Replace hardcoded Chinese strings in plot functions with `L(config, <key>)`**

Scan `visualize.py` for Chinese string literals used as axis labels, titles, or legend entries, and replace them with `L(config, "<key>")` calls, using the key names introduced above. For the small subset that has no existing Chinese equivalent (e.g., the English "Work–Family" vs "工作-家庭"), add the key to both sub-dicts in `LABELS`.

Example:
```python
# Before:
ax.set_xlabel("年份")
ax.set_ylabel("性别规范指数")

# After:
ax.set_xlabel(L(config, "year"))
ax.set_ylabel(L(config, "gender_norm"))
```

- [ ] **Step 4: Guard Chinese-only code paths**

Near the top of every plot function that merges in CFPS/CGSS survey data (search for `ENGLISH_TO_CHINESE_PROVINCE` or survey file loads), add:

```python
if config["language"] != "zh":
    logger.info(f"Skipping {plot_name}: survey comparison is zh-only")
    return
```

- [ ] **Step 5: Update `main()` entry point to call `_configure_fonts(config)`**

```python
def main(config="config/config.yml", mode=None):
    config_data = load_config(config)
    _configure_fonts(config_data)
    # ... existing dispatch logic
```

- [ ] **Step 6: Run visualize in a dry-pass for both languages**

Spot-check by generating figures on a previously-computed results directory (Chinese), confirming CJK characters still render correctly, then with a fake English config and a fake results CSV — confirm no font-registration error and English labels appear.

```bash
# Chinese sanity check:
python -m scripts.visualize main --config=config/profiles/ngram_weat.yml

# Minimal English check (create a tiny weat_results.csv in a tmp results dir):
# ... manual verification
```

- [ ] **Step 7: Commit**

```bash
git add scripts/visualize.py
git commit -m "$(cat <<'EOF'
Make visualize.py language-aware

Extract CJK font registration into _configure_fonts(config) called
from main; add LABELS dict with zh and en translations for every
user-facing string; guard CFPS/CGSS survey-comparison plots behind
language == 'zh'. English runs with matplotlib defaults and English
labels.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Refactor `analyze_correlation.py` — same font extraction + language guard

**Files:**
- Modify: `scripts/analyze_correlation.py`

- [ ] **Step 1: Extract font registration (same pattern as Task 11)**

Replace the top-level font registration block with an import from visualize (to keep the logic in one place):

```python
from scripts.visualize import _configure_fonts, L
```

Then in the entry point:
```python
def main(config="config/config.yml"):
    config_data = load_config(config)
    _configure_fonts(config_data)
    # ...
```

- [ ] **Step 2: Guard `PROVINCE_NAME_MAPPING` usage**

Find where `PROVINCE_NAME_MAPPING` is applied to normalize short↔long Chinese province names. Wrap the normalization step with:

```python
if config["language"] == "zh":
    df["province"] = df["province"].map(lambda p: PROVINCE_NAME_MAPPING.get(p, p))
# else: English configs ship clean names, no normalization needed
```

- [ ] **Step 3: Replace Chinese-hardcoded plot strings with `L(config, ...)` calls**

Same pattern as Task 11.

- [ ] **Step 4: Test with existing Chinese provincial+WEAT results**

```bash
python -m scripts.analyze_correlation --config=config/profiles/weibo_server.yml
```

Expected: runs to completion; output CSVs and figures regenerated identically to pre-refactor.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_correlation.py
git commit -m "$(cat <<'EOF'
Make analyze_correlation.py language-aware

Share _configure_fonts and L with visualize.py; guard
PROVINCE_NAME_MAPPING under language == 'zh'. English configs will
merge correlation inputs on year without name normalization.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Parameterize `download_ngrams.py` by `ngram.language`

**Files:**
- Modify: `scripts/data_prep/download_ngrams.py`

- [ ] **Step 1: Change base URL construction**

Replace hardcoded `base_url = "http://storage.googleapis.com/books/ngrams/books/20200217/chi_sim"` with:

```python
NGRAM_BASE_TEMPLATE = "http://storage.googleapis.com/books/ngrams/books/20200217/{lang}"
NGRAM_LANGUAGE_CODES = {"zh": "chi_sim", "en": "eng"}


def _resolve_ngram_language(config: dict) -> str:
    """Return the Google ngram language code for the current config."""
    ngram_cfg = config.get("ngram", {})
    if "language" in ngram_cfg:
        return ngram_cfg["language"]
    # Fall back to top-level language → standard code
    return NGRAM_LANGUAGE_CODES[config["language"]]
```

In `main()`, build:
```python
lang_code = _resolve_ngram_language(config_data)
base_url = NGRAM_BASE_TEMPLATE.format(lang=lang_code)
```

- [ ] **Step 2: Replace hardcoded shard count with index-page scraping**

Replace `generate_download_urls()` with:

```python
import re as _re

def generate_download_urls(base_url: str, logger: logging.Logger) -> List[str]:
    """
    Fetch the base_url index page and extract shard filenames.

    Pattern (Chinese as of 2020): 5-00000-of-00105.gz ... 5-00104-of-00105.gz
    Pattern (English as of 2020): 5-00000-of-NNNNN.gz ... (shard count differs)
    """
    logger.info(f"Fetching index page: {base_url}/")
    response = requests.get(f"{base_url}/", timeout=30)
    response.raise_for_status()
    html = response.text

    # Extract all 5-*-of-*.gz filenames from the directory listing
    shard_names = sorted(set(_re.findall(r"5-\d{5}-of-\d{5}\.gz", html)))
    if not shard_names:
        raise RuntimeError(f"No 5-gram shard files found at {base_url}/")

    download_urls = [f"{base_url}/totalcounts-5"]
    download_urls.extend(f"{base_url}/{name}" for name in shard_names)

    logger.info(f"Discovered {len(shard_names)} shard files at {base_url}")
    return download_urls
```

- [ ] **Step 3: Use `load_config()` from the common module**

Replace the local `load_config()` in this script with `from scripts.common.config_loader import load_config` so that downloads also validate the `language` field.

- [ ] **Step 4: Run Chinese download stub**

```bash
python -m scripts.data_prep.download_ngrams --config=config/profiles/ngram_server.yml --skip_decompress --max_workers=1
```

Expected: logs show 106 URLs (1 totalcounts + 105 shards) scraped for `chi_sim`.

(Skip the actual bytes-download in testing if slow; just verify URL discovery and `--max_workers=1` behavior.)

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/download_ngrams.py
git commit -m "$(cat <<'EOF'
Parameterize ngram downloader by language

Base URL built from ngram.language (chi_sim | eng), with top-level
language as fallback. Shard list scraped from the index page instead
of hardcoded to 105 so English Ngram's different shard count works
out of the box.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Seed English wordlists under `wordlists/en/`

**Files:**
- Create: `wordlists/en/prestige/occupations.txt`
- Create: `wordlists/en/prestige/gender_words.json`
- Create: `wordlists/en/prestige/prestige_axes.json`
- Create: `wordlists/en/weat_formal/gender_words.json`
- Create: `wordlists/en/weat_formal/domestic_work_words.json`
- Create: `wordlists/en/weat_formal/leadership_words.json`
- Create: `wordlists/en/weat_formal/stem_words.json`

- [ ] **Step 1: Create directories**

```bash
mkdir -p wordlists/en/prestige wordlists/en/weat_formal
```

- [ ] **Step 2: Create `wordlists/en/weat_formal/gender_words.json`** (Caliskan 2017 gender stimuli)

```json
{
  "male": ["male", "man", "boy", "brother", "he", "him", "his", "son", "father", "uncle", "grandfather", "husband", "nephew", "gentleman", "king", "prince", "sir"],
  "female": ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter", "mother", "aunt", "grandmother", "wife", "niece", "lady", "queen", "princess", "madam"]
}
```

- [ ] **Step 3: Create `wordlists/en/weat_formal/domestic_work_words.json`** (Caliskan 2017 career/family)

```json
{
  "family": ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives", "household", "kitchen"],
  "work": ["executive", "management", "professional", "corporation", "salary", "office", "business", "career", "company", "promotion"]
}
```

- [ ] **Step 4: Create `wordlists/en/weat_formal/leadership_words.json`**

```json
{
  "leadership": ["leader", "manager", "executive", "director", "supervisor", "chief", "president", "commander", "boss", "head"],
  "non_leadership": ["assistant", "clerk", "helper", "follower", "subordinate", "trainee", "intern", "apprentice", "aide", "attendant"]
}
```

- [ ] **Step 5: Create `wordlists/en/weat_formal/stem_words.json`** (Caliskan 2017 STEM/arts)

```json
{
  "stem": ["science", "technology", "physics", "chemistry", "engineering", "mathematics", "computation", "algorithm", "biology", "astronomy", "geology", "statistics"],
  "non_stem": ["arts", "literature", "poetry", "dance", "symphony", "drama", "sculpture", "painting", "novel", "philosophy", "history", "linguistics"]
}
```

- [ ] **Step 6: Create `wordlists/en/prestige/gender_words.json`** (Bolukbasi 2016 definitional pairs)

```json
{
  "male": ["man", "boy", "he", "him", "his", "father", "son", "husband", "brother", "uncle", "nephew", "grandfather", "king", "prince", "sir", "gentleman", "male", "himself", "mr", "dad"],
  "female": ["woman", "girl", "she", "her", "hers", "mother", "daughter", "wife", "sister", "aunt", "niece", "grandmother", "queen", "princess", "madam", "lady", "female", "herself", "mrs", "mom"]
}
```

- [ ] **Step 7: Create `wordlists/en/prestige/prestige_axes.json`** (Osgood EPA + Nakao-Treas prestige)

```json
{
  "evaluation": {
    "positive": ["good", "pleasant", "beautiful", "wonderful", "nice", "kind", "happy", "honest", "friendly", "warm"],
    "negative": ["bad", "unpleasant", "ugly", "terrible", "awful", "mean", "sad", "dishonest", "hostile", "cold"]
  },
  "potency": {
    "positive": ["strong", "powerful", "large", "heavy", "hard", "solid", "dominant", "tough", "robust", "firm"],
    "negative": ["weak", "powerless", "small", "light", "soft", "fragile", "submissive", "tender", "delicate", "loose"]
  },
  "activity": {
    "positive": ["active", "fast", "noisy", "young", "lively", "energetic", "dynamic", "vigorous", "alert", "busy"],
    "negative": ["passive", "slow", "quiet", "old", "sluggish", "lethargic", "static", "weary", "drowsy", "idle"]
  },
  "general_prestige": {
    "positive": ["prestigious", "respected", "esteemed", "admired", "distinguished", "eminent", "renowned", "elite", "honored", "influential"],
    "negative": ["lowly", "scorned", "despised", "disrespected", "common", "ordinary", "menial", "unskilled", "humble", "marginal"]
  }
}
```

- [ ] **Step 8: Create `wordlists/en/prestige/occupations.txt`** (O*NET sample, ~60 terms)

```text
accountant
actor
architect
artist
attorney
author
baker
banker
biologist
carpenter
chef
chemist
clerk
dentist
designer
doctor
economist
engineer
farmer
firefighter
florist
geologist
hairdresser
historian
housekeeper
janitor
journalist
judge
lawyer
librarian
lifeguard
mechanic
musician
nurse
nutritionist
painter
pharmacist
photographer
physician
physicist
pilot
plumber
poet
police
politician
professor
programmer
psychologist
receptionist
reporter
researcher
scientist
secretary
sociologist
soldier
surgeon
teacher
technician
therapist
translator
veterinarian
waiter
writer
```

(Note: this is a minimal seed list. Expand to ~200 O*NET terms in a follow-up tuning pass.)

- [ ] **Step 9: Validate JSON files**

```bash
for f in wordlists/en/prestige/*.json wordlists/en/weat_formal/*.json; do
  python -c "import json; json.load(open('$f')); print('OK: $f')"
done
```

Expected: 6 `OK` lines.

- [ ] **Step 10: Commit**

```bash
git add wordlists/en/
git commit -m "$(cat <<'EOF'
Seed English wordlists from published sources

Caliskan 2017 stimuli for WEAT (gender, work/family, STEM), Garg 2018
leadership replication list, Bolukbasi 2016 gender pairs, Osgood 1957
EPA dimensions and Nakao-Treas 1994 general prestige, and a 60-term
O*NET occupation seed. These are starting points; expect to tune.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Create `build_corpora_ngram_en.py`

**Files:**
- Create: `scripts/data_prep/build_corpora_ngram_en.py`
- Modify: `tests/test_build_corpora.py`

- [ ] **Step 1: Write a failing test for the English ngram parser**

Append to `tests/test_build_corpora.py`:

```python
def test_build_corpora_ngram_en_parses_english_5gram_line(tmp_path):
    from scripts.data_prep.build_corpora_ngram_en import parse_ngram_line, clean_ngram

    line = "the quick brown fox jumps\t2000,42,3\t2001,50,4\n"
    entries = parse_ngram_line(line)
    assert len(entries) == 2
    ngram_text, year, count = entries[0]
    assert year == 2000
    assert count == 42
    assert "the" in ngram_text or "quick" in ngram_text
```

```python
def test_build_corpora_ngram_en_clean_drops_short_cleaned(tmp_path):
    from scripts.data_prep.build_corpora_ngram_en import clean_ngram
    # Only one letter-token surviving → should return None
    assert clean_ngram("!@#$ ^&*(  ") is None
```

```python
def test_build_corpora_ngram_en_lowercases(tmp_path):
    from scripts.data_prep.build_corpora_ngram_en import clean_ngram
    out = clean_ngram("Hello WORLD FROM Mars")
    assert out is not None
    assert out.islower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest -xvs tests/test_build_corpora.py::test_build_corpora_ngram_en_parses_english_5gram_line
```

Expected: FAIL — module not found.

- [ ] **Step 3: Create `scripts/data_prep/build_corpora_ngram_en.py`**

```python
#!/usr/bin/env python3
"""
Build time-sliced corpora from English Google 5-gram data.

Usage:
    python -m scripts.data_prep.build_corpora_ngram_en --config=config/config.yml
    python -m scripts.data_prep.build_corpora_ngram_en --config=config/config.yml --slice=1940_1949
"""

import os
import re
import gzip
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


ENGLISH_TOKEN_RE = re.compile(r"[a-z']+")


def decompress_file(gz_path: Path, output_path: Path, logger) -> Tuple[bool, str]:
    """Decompress a gzip file."""
    filename = gz_path.name
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Skipping decompression of {filename} (already exists)")
        return True, f"Already decompressed: {filename}"
    try:
        logger.info(f"Decompressing {filename}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        output_size = output_path.stat().st_size
        logger.info(f"Completed decompressing {filename} ({output_size:,} bytes)")
        return True, f"Decompressed: {filename}"
    except Exception as e:
        logger.error(f"Failed to decompress {filename}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False, f"Failed: {filename} - {e}"


def generate_time_slices(start_year, end_year, window_size, step_size):
    slices = []
    current_start = start_year
    while current_start <= end_year:
        current_end = min(current_start + window_size - 1, end_year)
        slices.append((current_start, current_end))
        current_start += step_size
        if current_start > end_year:
            break
    return slices


def clean_ngram(ngram: str) -> Optional[str]:
    """Clean an English n-gram: lowercase, keep only letter tokens (apostrophes OK)."""
    lowered = ngram.lower()
    tokens = ENGLISH_TOKEN_RE.findall(lowered)
    if len(tokens) <= 1:
        return None
    return " ".join(tokens)


def parse_ngram_line(line: str) -> List[Tuple[str, int, int]]:
    """Parse a Google Ngram v3 line: text<TAB>year,count,volume<TAB>..."""
    parts = line.strip().split("\t")
    if len(parts) < 2:
        return []
    ngram = clean_ngram(parts[0])
    if not ngram:
        return []
    result = []
    for yc in parts[1:]:
        try:
            year, count1, count2 = yc.split(",")
            result.append((ngram, int(year), int(count1)))
        except Exception:
            continue
    return result


def process_ngram_file(file_path, time_slices, config, logger):
    """Process a single ngram file and write to time-slice corpus files."""
    min_count = config["corpus"]["min_count_threshold"]
    corpora_dir = Path(config["paths"]["corpora_dir"])
    os.makedirs(corpora_dir, exist_ok=True)

    logger.info(f"Processing {file_path.name}...")
    lines_processed = 0
    lines_included = defaultdict(int)
    file_index = file_path.name.split("-")[1]
    write_buffer = defaultdict(set)
    largest_buffer = 10000

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines_processed += 1
            entries = parse_ngram_line(line)
            if not entries:
                continue
            for ngram_text, year, match_count in entries:
                if match_count < min_count:
                    continue
                for start_year, end_year in time_slices:
                    if start_year <= year <= end_year:
                        slice_name = f"{start_year}_{end_year}"
                        write_buffer[slice_name].add(ngram_text)
                        if len(write_buffer[slice_name]) > largest_buffer:
                            os.makedirs(corpora_dir / slice_name, exist_ok=True)
                            with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", "a", encoding="utf-8") as out:
                                out.write("\n".join(list(write_buffer[slice_name])) + "\n")
                            write_buffer[slice_name] = set()
                        lines_included[slice_name] += 1
            if lines_processed % 1_000_000 == 0:
                logger.info(f"  Processed {lines_processed:,} lines from {file_path.name}")

    for slice_name, buffer in write_buffer.items():
        if buffer:
            os.makedirs(corpora_dir / slice_name, exist_ok=True)
            with open(corpora_dir / slice_name / f"corpus_{file_index}.txt", "a", encoding="utf-8") as out:
                out.write("\n".join(list(buffer)) + "\n")

    logger.info(f"Completed {file_path.name}: {lines_processed:,} lines processed")
    for slice_name, count in lines_included.items():
        logger.info(f"  {slice_name}: {count:,} n-grams included")


def build_corpora(config, logger, specific_slice=None, file_name=None):
    ts_config = config["time_slices"]
    time_slices = generate_time_slices(
        ts_config["start_year"], ts_config["end_year"],
        ts_config["window_size"], ts_config["step_size"],
    )
    logger.info(f"Generated {len(time_slices)} time slices")

    if specific_slice:
        start, end = map(int, specific_slice.split("_"))
        time_slices = [(start, end)]

    decompressed_dir = Path(config["paths"]["decompressed_dir"])
    raw_ngram_dir = Path(config["paths"]["raw_ngram_dir"])
    decompress = True

    if file_name:
        ngram_zips = [decompressed_dir / file_name]
        decompress = False
    else:
        ngram_zips = sorted(raw_ngram_dir.glob("5-*.gz"))

    logger.info(f"Found {len(ngram_zips)} n-gram files to process")

    for single_zip in ngram_zips:
        if decompress:
            ngram_file = decompressed_dir / single_zip.stem
            decompress_file(single_zip, ngram_file, logger)
        else:
            ngram_file = single_zip
        process_ngram_file(ngram_file, time_slices, config, logger)
        if decompress:
            os.remove(ngram_file)


def main(file_name=None, config="config/config.yml", slice=None):
    """Build time-sliced corpora from English Google 5-gram data."""
    config_data = load_config(config)
    if config_data["language"] != "en" or config_data["data_source"] != "ngram":
        raise ValueError(
            "build_corpora_ngram_en requires language='en' and data_source='ngram'"
        )
    logger = setup_logging(Path(config_data["paths"]["log_dir"]), "build_corpora_ngram_en.log")

    logger.info("=" * 80)
    logger.info("Starting English ngram corpus building")
    logger.info("=" * 80)

    build_corpora(config_data, logger, specific_slice=slice, file_name=file_name)

    logger.info("=" * 80)
    logger.info("English ngram corpus building completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run tests**

```bash
pytest -xvs tests/test_build_corpora.py -k ngram_en
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/data_prep/build_corpora_ngram_en.py tests/test_build_corpora.py
git commit -m "$(cat <<'EOF'
Add English Google Ngram corpus builder

Structural parallel to build_corpora_ngram.py. Uses lowercase + [a-z']
token regex instead of CJK filtering. Time-slicing logic reused.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Create English Ngram config profiles

**Files:**
- Create: `config/profiles/ngram_en_server.yml`
- Create: `config/profiles/ngram_en_weat.yml`

- [ ] **Step 1: Create `config/profiles/ngram_en_server.yml`** (Ngram EN + prestige)

```yaml
# English Google 5-gram + prestige (longitudinal)
language: "en"
data_source: "ngram"
analysis_mode: "prestige"

paths:
  base_dir: "/scratch/network/yh6580/gender-occup"
  raw_ngram_dir: "/scratch/network/yh6580/gender-occup/data/raw_ngrams_en"
  decompressed_dir: "/scratch/network/yh6580/gender-occup/data/raw_ngrams_en_decompressed"
  corpora_dir: "/scratch/network/yh6580/gender-occup/data/corpora_en_ngram"
  models_dir: "/scratch/network/yh6580/gender-occup/data/models_en_ngram"
  results_dir: "/scratch/network/yh6580/gender-occup/data/results_en_ngram"
  log_dir: "/scratch/network/yh6580/gender-occup/logs"
  figures_dir: "/scratch/network/yh6580/gender-occup/figures"

ngram:
  language: "eng"
  n: 5
  min_year: 1800
  max_year: 2019
  delimiter: "\t"
  year_column: 1
  match_count_column: 2
  volume_count_column: 3

time_slices:
  window_size: 10
  step_size: 5
  start_year: 1900
  end_year: 2015

embedding:
  vector_size: 300
  window: 4
  min_count: 50
  sg: 1
  negative: 15
  workers: 16
  epochs: 5
  seed: 42
  model_name_template: "eng_5gram_{unit_name}.model"

corpus:
  use_counts: false
  min_count_threshold: 1
  tokenizer: "whitespace"
  lowercase: true
  min_words: 2

wordlists:
  dir: "wordlists/en/prestige"
  occupations_file: "occupations.txt"
  gender_words_file: "gender_words.json"
  prestige_axes_file: "prestige_axes.json"

analysis:
  occupation_strategy: "whole_token"
  min_coverage: 0.5
```

- [ ] **Step 2: Create `config/profiles/ngram_en_weat.yml`** (Ngram EN + WEAT)

Copy `ngram_en_server.yml` to `ngram_en_weat.yml`, then change:

```yaml
analysis_mode: "weat"

# Results go to a separate dir to avoid clobbering prestige outputs
paths:
  ...
  results_dir: "/scratch/network/yh6580/gender-occup/data/results_en_ngram_weat"

wordlists:
  dir: "wordlists/en/weat_formal"
  weat_gender_file: "gender_words.json"
  weat_domestic_work_file: "domestic_work_words.json"
  weat_leadership_file: "leadership_words.json"
  weat_stem_file: "stem_words.json"
```

(Drop the `occupations_file` / `gender_words_file` / `prestige_axes_file` keys; they're prestige-only.)

- [ ] **Step 3: Validate both profiles load**

```bash
python -c "from scripts.common.config_loader import load_config; load_config('config/profiles/ngram_en_server.yml'); load_config('config/profiles/ngram_en_weat.yml'); print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add config/profiles/ngram_en_server.yml config/profiles/ngram_en_weat.yml
git commit -m "$(cat <<'EOF'
Add English Ngram profiles (prestige and WEAT)

Two longitudinal English Ngram configs: one for prestige mode, one
for WEAT mode. Paths point at the Princeton scratch layout; adjust
per server.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Create `download_coha.py`

**Files:**
- Create: `scripts/data_prep/download_coha.py`

- [ ] **Step 1: Write the file**

```python
#!/usr/bin/env python3
"""
Download and decompress COHA n-gram archives.

Expects a list of ZIP archive URLs in config['coha']['source_archive_urls'].
Each URL is typically a decade-level archive that the user obtained via the
corpusdata.org email-gated signup.

Usage:
    python -m scripts.data_prep.download_coha --config=config/config.yml
    python -m scripts.data_prep.download_coha --config=config/config.yml --max_workers=4
"""

import logging
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests
import fire

from scripts.common.config_loader import load_config


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("download_coha")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_dir / "download_coha.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(sh)
    return logger


def download_one(url: str, out_path: Path, logger: logging.Logger) -> Tuple[bool, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info(f"Skipping {out_path.name} (already exists)")
        return True, "skipped"
    try:
        logger.info(f"Downloading {url} -> {out_path.name}")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded {out_path.name} ({out_path.stat().st_size:,} bytes)")
        return True, "downloaded"
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if out_path.exists():
            out_path.unlink()
        return False, str(e)


def decompress_one(zip_path: Path, out_dir: Path, logger: logging.Logger) -> Tuple[bool, str]:
    try:
        logger.info(f"Decompressing {zip_path.name} -> {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        logger.info(f"Decompressed {zip_path.name}")
        return True, "decompressed"
    except Exception as e:
        logger.error(f"Failed to decompress {zip_path.name}: {e}")
        return False, str(e)


def main(config: str = "config/config.yml", max_workers: int = 4, skip_decompress: bool = False):
    """Download COHA n-gram archives to raw_coha_dir and decompress to coha_decompressed_dir."""
    cfg = load_config(config)
    if cfg["data_source"] != "coha":
        raise ValueError("download_coha requires data_source='coha' in config")

    urls: List[str] = cfg.get("coha", {}).get("source_archive_urls", [])
    if not urls:
        raise ValueError(
            "config.coha.source_archive_urls is empty. "
            "Paste the download URLs from your corpusdata.org signup email."
        )

    raw_dir = Path(cfg["paths"]["raw_coha_dir"])
    decomp_dir = Path(cfg["paths"]["coha_decompressed_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    logger = _setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info(f"Starting COHA download ({len(urls)} URLs)")
    logger.info("=" * 80)

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download
    download_tasks = []
    for url in urls:
        filename = url.rstrip("/").split("/")[-1]
        out_path = raw_dir / filename
        download_tasks.append((url, out_path))

    results = {"ok": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(download_one, u, p, logger): (u, p) for u, p in download_tasks}
        for f in as_completed(futs):
            ok, _ = f.result()
            results["ok" if ok else "fail"] += 1
    logger.info(f"Downloads: {results}")

    # Decompress
    if skip_decompress:
        logger.info("Skipping decompression as requested")
        return
    decomp_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(raw_dir.glob("*.zip"))
    dresults = {"ok": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(decompress_one, z, decomp_dir, logger): z for z in zips}
        for f in as_completed(futs):
            ok, _ = f.result()
            dresults["ok" if ok else "fail"] += 1
    logger.info(f"Decompressions: {dresults}")

    logger.info("=" * 80)
    logger.info("COHA download completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 2: Syntax-check**

```bash
python -c "import ast; ast.parse(open('scripts/data_prep/download_coha.py').read())"
```

Expected: no output (ast.parse succeeded).

- [ ] **Step 3: Commit**

```bash
git add scripts/data_prep/download_coha.py
git commit -m "$(cat <<'EOF'
Add COHA downloader

Reads archive URLs from config.coha.source_archive_urls (paste from
corpusdata.org signup email). Parallel download, parallel ZIP
decompression. No scraping; caller supplies URLs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Create `build_corpora_coha.py`

**Files:**
- Create: `scripts/data_prep/build_corpora_coha.py`
- Modify: `tests/test_build_corpora.py`

COHA free n-grams are distributed as TSV files, one per n-gram order, typically named by decade. Each line is `w1<TAB>w2<TAB>...<TAB>wN<TAB>freq`. Decade is embedded in filename.

- [ ] **Step 1: Write failing test**

Append to `tests/test_build_corpora.py`:

```python
def test_build_corpora_coha_parses_4gram_line(tmp_path):
    from scripts.data_prep.build_corpora_coha import parse_coha_line

    # COHA 4-gram format: word1<TAB>word2<TAB>word3<TAB>word4<TAB>freq
    entries = parse_coha_line("the quick brown fox\t42", n=4)
    assert entries is not None
    text, freq = entries
    assert freq == 42
    assert text == "the quick brown fox"


def test_build_corpora_coha_decade_from_filename():
    from scripts.data_prep.build_corpora_coha import decade_from_filename
    # Typical COHA filename pattern: w4_1940.txt or coha_4grams_1940s.txt
    assert decade_from_filename(Path("w4_1940.txt")) == "1940s"
    assert decade_from_filename(Path("coha_4grams_1950s.txt")) == "1950s"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest -xvs tests/test_build_corpora.py::test_build_corpora_coha_parses_4gram_line
```

Expected: FAIL — module not found.

- [ ] **Step 3: Create `scripts/data_prep/build_corpora_coha.py`**

```python
#!/usr/bin/env python3
"""
Build decade-partitioned corpora from COHA free n-gram data.

Expected input: TSV files under config['paths']['coha_decompressed_dir'],
one or more files per decade, each line:
    word1<TAB>word2<TAB>...<TAB>wordN<TAB>freq

Filenames are expected to embed the decade (e.g., "w4_1940.txt",
"coha_4grams_1950s.txt"). Any filename containing a 4-digit year or a
"<year>s" decade marker is accepted.

Output:
    {corpora_dir}/{decade}s/corpus_{shard_idx}.txt
    (e.g., {corpora_dir}/1940s/corpus_000.txt)

Each output line is a single n-gram with tokens joined by single spaces.
train_embeddings.py treats each line as a mini-document, matching the
existing Chinese ngram flow.

Usage:
    python -m scripts.data_prep.build_corpora_coha --config=config/config.yml
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


ENGLISH_TOKEN_RE = re.compile(r"[a-z']+")
DECADE_RE = re.compile(r"(1[89]\d{2}|20\d{2})s?")


def decade_from_filename(path: Path) -> Optional[str]:
    """Extract a '1940s'-style decade label from a COHA filename."""
    m = DECADE_RE.search(path.stem)
    if not m:
        return None
    year = int(m.group(1))
    return f"{(year // 10) * 10}s"


def parse_coha_line(line: str, n: int) -> Optional[Tuple[str, int]]:
    """
    Parse a single TSV line: word1<TAB>...<TAB>wordN<TAB>freq.

    Returns (joined_ngram, freq) or None if the line is malformed or
    fewer than 2 tokens survive cleaning.
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) < n + 1:
        return None
    words = [w.strip().lower() for w in parts[:n]]
    try:
        freq = int(parts[n])
    except (ValueError, IndexError):
        return None

    # Keep only alphabetic letter tokens (with apostrophes)
    cleaned = []
    for w in words:
        matches = ENGLISH_TOKEN_RE.findall(w)
        if matches:
            cleaned.append("".join(matches))
    if len(cleaned) < 2:
        return None
    return " ".join(cleaned), freq


def process_coha_file(src: Path, config: dict, logger) -> int:
    """Read a single COHA n-gram file and emit lines to the right decade dir."""
    n = config["coha"]["ngram_order"]
    min_count = config["corpus"]["min_count_threshold"]
    corpora_dir = Path(config["paths"]["corpora_dir"])

    decade = decade_from_filename(src)
    if decade is None:
        logger.warning(f"Skipping {src.name}: no decade in filename")
        return 0

    out_dir = corpora_dir / decade
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = sum(1 for _ in out_dir.glob("corpus_*.txt"))
    out_path = out_dir / f"corpus_{shard_idx:03d}.txt"

    count = 0
    buffer = []
    BUFFER_MAX = 10_000

    with open(src, "r", encoding="utf-8", errors="ignore") as f, \
         open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            parsed = parse_coha_line(line, n=n)
            if parsed is None:
                continue
            text, freq = parsed
            if freq < min_count:
                continue
            buffer.append(text)
            count += 1
            if len(buffer) >= BUFFER_MAX:
                out.write("\n".join(buffer) + "\n")
                buffer.clear()
        if buffer:
            out.write("\n".join(buffer) + "\n")

    logger.info(f"{src.name} → {out_path.name}: {count:,} n-grams")
    return count


def main(config: str = "config/config.yml"):
    """Build decade-partitioned COHA corpora."""
    cfg = load_config(config)
    if cfg["language"] != "en" or cfg["data_source"] != "coha":
        raise ValueError(
            "build_corpora_coha requires language='en' and data_source='coha'"
        )

    logger = setup_logging(Path(cfg["paths"]["log_dir"]), "build_corpora_coha.log")
    logger.info("=" * 80)
    logger.info("Starting COHA corpus building")
    logger.info("=" * 80)

    decomp_dir = Path(cfg["paths"]["coha_decompressed_dir"])
    files = sorted(decomp_dir.glob("**/*.txt"))
    if not files:
        raise RuntimeError(f"No .txt files under {decomp_dir}")

    total = 0
    for f in files:
        total += process_coha_file(f, cfg, logger)

    logger.info(f"Total n-grams emitted: {total:,}")
    logger.info("=" * 80)
    logger.info("COHA corpus building completed")
    logger.info("=" * 80)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run tests**

```bash
pytest -xvs tests/test_build_corpora.py -k coha
```

Expected: 2 passed.

- [ ] **Step 5: Add a fixture-based smoke test for COHA end-to-end**

Append to `tests/test_build_corpora.py`:

```python
def test_build_corpora_coha_writes_decade_corpora(tmp_path, monkeypatch):
    import yaml
    from scripts.data_prep.build_corpora_coha import main as coha_main

    # Prepare fake COHA 4-gram file
    decomp = tmp_path / "coha_decompressed"
    decomp.mkdir()
    sample = decomp / "w4_1940.txt"
    sample.write_text(
        "the quick brown fox\t42\nlazy dog over the\t7\n" * 10,
        encoding="utf-8",
    )

    cfg = {
        "language": "en",
        "data_source": "coha",
        "paths": {
            "base_dir": str(tmp_path),
            "raw_coha_dir": str(tmp_path / "raw_coha"),
            "coha_decompressed_dir": str(decomp),
            "corpora_dir": str(tmp_path / "corpora"),
            "models_dir": str(tmp_path / "models"),
            "results_dir": str(tmp_path / "results"),
            "log_dir": str(tmp_path / "logs"),
        },
        "coha": {
            "ngram_order": 4,
            "source_archive_urls": ["http://example.com/dummy"],
            "decade_min": 1940,
            "decade_max": 2000,
        },
        "corpus": {"min_count_threshold": 1},
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    coha_main(config=str(cfg_path))

    corpus_files = list((tmp_path / "corpora" / "1940s").glob("corpus_*.txt"))
    assert corpus_files, "expected at least one corpus file under 1940s"
    content = corpus_files[0].read_text()
    assert "the quick brown fox" in content
```

```bash
pytest -xvs tests/test_build_corpora.py::test_build_corpora_coha_writes_decade_corpora
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/data_prep/build_corpora_coha.py tests/test_build_corpora.py
git commit -m "$(cat <<'EOF'
Add COHA n-gram corpus builder

Parses COHA 4-gram TSVs, decade-buckets by filename, emits mini-doc
lines into corpora_dir/{decade}s/. Output layout matches the Chinese
Ngram builder so train_embeddings.py needs zero changes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: Create COHA config profile

**Files:**
- Create: `config/profiles/coha_server.yml`

- [ ] **Step 1: Write the file**

```yaml
# COHA free n-grams + WEAT (longitudinal, decade-bucketed)
language: "en"
data_source: "coha"
analysis_mode: "weat"

paths:
  base_dir: "/scratch/network/yh6580/gender-occup"
  raw_coha_dir: "/scratch/network/yh6580/gender-occup/data/raw_coha"
  coha_decompressed_dir: "/scratch/network/yh6580/gender-occup/data/raw_coha_decompressed"
  corpora_dir: "/scratch/network/yh6580/gender-occup/data/corpora_coha"
  models_dir: "/scratch/network/yh6580/gender-occup/data/models_coha"
  results_dir: "/scratch/network/yh6580/gender-occup/data/results_coha"
  log_dir: "/scratch/network/yh6580/gender-occup/logs"
  figures_dir: "/scratch/network/yh6580/gender-occup/figures"

coha:
  ngram_order: 4
  source_archive_urls:
    # Paste URLs from your corpusdata.org signup email, one per decade or archive.
    # - "https://www.corpusdata.org/coha/sample/4grams-1940s.zip"
    # - "https://www.corpusdata.org/coha/sample/4grams-1950s.zip"
  decade_min: 1810
  decade_max: 2000

embedding:
  vector_size: 300
  window: 3      # Smaller because 4-grams → max useful context = 3
  min_count: 20
  sg: 1
  negative: 15
  workers: 16
  epochs: 5
  seed: 42
  model_name_template: "coha_4gram_{unit_name}.model"

corpus:
  use_counts: false
  min_count_threshold: 1
  tokenizer: "whitespace"
  lowercase: true
  min_words: 2

wordlists:
  dir: "wordlists/en/weat_formal"
  weat_gender_file: "gender_words.json"
  weat_domestic_work_file: "domestic_work_words.json"
  weat_leadership_file: "leadership_words.json"
  weat_stem_file: "stem_words.json"

analysis:
  min_coverage: 0.5
```

- [ ] **Step 2: Validate**

```bash
python -c "from scripts.common.config_loader import load_config; load_config('config/profiles/coha_server.yml'); print('OK')"
```

Expected: `OK` (assuming the TODO URLs are left commented out — loader doesn't require non-empty URLs unless you run `download_coha`).

If the validator rejects empty urls at config-load time, relax the rule: validation only requires the key's presence; empty list is tolerated until download time when `download_coha.main` raises a clear error.

- [ ] **Step 3: Commit**

```bash
git add config/profiles/coha_server.yml
git commit -m "$(cat <<'EOF'
Add COHA config profile (WEAT)

Longitudinal COHA profile wired at decade granularity. URLs are
commented out so the user can paste them in once they have them from
corpusdata.org.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Extend `run_pipeline.sh` dispatch for English sources

**Files:**
- Modify: `run_pipeline.sh`

- [ ] **Step 1: Read current `run_pipeline.sh` to understand the dispatch**

```bash
cat run_pipeline.sh
```

Identify the sections that dispatch on `$DATA_SOURCE`.

- [ ] **Step 2: Extend Stage 1 (download) to handle COHA and EN ngram**

In the Stage 1 block, replace:
```bash
if [ "$DATA_SOURCE" = "ngram" ]; then
    python -m scripts.data_prep.download_ngrams --config=$CONFIG ...
fi
```

with:
```bash
case "$DATA_SOURCE" in
    ngram)
        python -m scripts.data_prep.download_ngrams --config="$CONFIG" ...
        ;;
    coha)
        python -m scripts.data_prep.download_coha --config="$CONFIG" ...
        ;;
    *)
        # other sources have no download stage
        ;;
esac
```

- [ ] **Step 3: Extend Stage 2 (corpus build) dispatch**

Replace the existing dispatch with:
```bash
LANGUAGE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['language'])")

case "$DATA_SOURCE" in
    ngram)
        if [ "$LANGUAGE" = "en" ]; then
            python -m scripts.data_prep.build_corpora_ngram_en --config="$CONFIG"
        else
            python -m scripts.data_prep.build_corpora_ngram --config="$CONFIG"
        fi
        ;;
    coha)
        python -m scripts.data_prep.build_corpora_coha --config="$CONFIG"
        ;;
    renminribao)
        python -m scripts.data_prep.build_corpora_rmrb --config="$CONFIG"
        ;;
    weibo)
        python -m scripts.data_prep.build_corpora_weibo --config="$CONFIG"
        ;;
    newspaper)
        python -m scripts.data_prep.build_corpora_newspaper --config="$CONFIG"
        ;;
    *)
        echo "Unknown data_source: $DATA_SOURCE" >&2
        exit 1
        ;;
esac
```

- [ ] **Step 4: Dry-run on existing Chinese profile**

```bash
bash run_pipeline.sh --config=config/profiles/ngram_server.yml --force-corpus --dry-run 2>&1 | head -30
```

(If there's no `--dry-run` flag, just inspect the printed dispatch lines and kill early. The goal is to confirm Stage 2 still picks the Chinese ngram builder.)

- [ ] **Step 5: Commit**

```bash
git add run_pipeline.sh
git commit -m "$(cat <<'EOF'
Extend run_pipeline.sh dispatch for English sources

Stage 1 downloads COHA when data_source=coha. Stage 2 picks
build_corpora_ngram_en when language=en and data_source=ngram, and
build_corpora_coha for COHA. Existing Chinese dispatch unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Add NLTK dependency and setup step

**Files:**
- Modify: `requirements.txt`
- Modify: `setup_server.sh`

- [ ] **Step 1: Add NLTK to `requirements.txt`**

Append to `requirements.txt`:
```
nltk>=3.8
```

- [ ] **Step 2: Add NLTK data download to `setup_server.sh`**

Append to `setup_server.sh`:
```bash
# Download NLTK corpora required by the English tokenizer / stopwords
python -m nltk.downloader -d "${NLTK_DATA:-$HOME/nltk_data}" punkt punkt_tab stopwords
```

- [ ] **Step 3: Run setup locally to confirm it works**

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt setup_server.sh
git commit -m "$(cat <<'EOF'
Add NLTK dependency and data download step

nltk>=3.8 added to requirements.txt. setup_server.sh downloads punkt
/ punkt_tab / stopwords corpora needed by the English tokenizer path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: Create Slurm templates for English sources

**Files:**
- Create: `slurm/download_coha.slurm`
- Create: `slurm/build_corpus_coha.slurm`
- Create: `slurm/full_pipeline_en.slurm`

- [ ] **Step 1: Create `slurm/download_coha.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=download_coha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

CONFIG="${1:-config/profiles/coha_server.yml}"
echo "Downloading COHA with config=$CONFIG"
python -m scripts.data_prep.download_coha --config="$CONFIG" --max_workers=4
```

- [ ] **Step 2: Create `slurm/build_corpus_coha.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=coha_corpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

CONFIG="${1:-config/profiles/coha_server.yml}"
echo "Building COHA corpus with config=$CONFIG"
python -m scripts.data_prep.build_corpora_coha --config="$CONFIG"
```

- [ ] **Step 3: Create `slurm/full_pipeline_en.slurm`**

```bash
#!/bin/bash
#SBATCH --job-name=pipeline_en
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

CONFIG="${1:-config/profiles/ngram_en_weat.yml}"
echo "Running full English pipeline with config=$CONFIG"
bash run_pipeline.sh --config "$CONFIG"
```

- [ ] **Step 4: Make them executable**

```bash
chmod +x slurm/download_coha.slurm slurm/build_corpus_coha.slurm slurm/full_pipeline_en.slurm
```

- [ ] **Step 5: Commit**

```bash
git add slurm/download_coha.slurm slurm/build_corpus_coha.slurm slurm/full_pipeline_en.slurm
git commit -m "$(cat <<'EOF'
Add Slurm templates for English pipeline

download_coha.slurm, build_corpus_coha.slurm, and full_pipeline_en.slurm
mirror the existing Chinese templates. Adjust account / partition as
needed per cluster.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: Update README with English pipeline section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a new "English pipeline" section at the end of the README**

Append this section to `README.md`:

```markdown
## English pipeline (MVP)

The pipeline supports `language: en` with two data sources in MVP:

- **Google Ngram English** — longitudinal, analog of the Chinese `ngram` source.
- **COHA free n-grams** — longitudinal, decade-bucketed.

### Quick start — English Google Ngram

```bash
# 1. Edit paths in config/profiles/ngram_en_weat.yml for your server.
# 2. Download shards (takes ~2 hours):
sbatch slurm/download_ngrams.slurm config/profiles/ngram_en_weat.yml
# 3. Build corpora:
sbatch slurm/build_corpus.slurm config/profiles/ngram_en_weat.yml
# 4. Train + analyze + visualize:
sbatch slurm/full_pipeline_en.slurm config/profiles/ngram_en_weat.yml
```

### Quick start — COHA

1. Request the free COHA n-gram archives at https://www.corpusdata.org/coha.asp (email-gated).
2. Paste the download URLs into `config/profiles/coha_server.yml` under `coha.source_archive_urls`.
3. Run:
   ```bash
   sbatch slurm/download_coha.slurm config/profiles/coha_server.yml
   sbatch slurm/build_corpus_coha.slurm config/profiles/coha_server.yml
   sbatch slurm/full_pipeline_en.slurm config/profiles/coha_server.yml
   ```

### English wordlists

Seeded under `wordlists/en/prestige/` and `wordlists/en/weat_formal/` from:
- Caliskan et al. 2017 (WEAT stimuli: gender / career-family / STEM-arts)
- Garg et al. 2018 (leadership replication)
- Bolukbasi et al. 2016 (gender pairs for prestige-mode axis)
- Osgood 1957 EPA + Nakao-Treas 1994 (prestige axes)
- O*NET (occupation list, 60-term seed; expand as needed)

Tune these for your research question; drop in replacements that match your citation.

### Language config

Every profile must declare `language: "zh"` or `language: "en"` at the top. Omitting it is a validation error.

### Phase 2 (not yet implemented)

State-level COHA analysis via full-text access and publication-source mapping; US survey correlation (GSS / ANES); additional English sources.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
Document English pipeline in README

New section covers Ngram EN and COHA quick-start, wordlist sources,
the language config key, and the Phase 2 roadmap.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 24: End-to-end smoke test on synthetic English fixture

**Files:**
- Create: `tests/fixtures/en_ngram_sample/5-00000-of-00001.gz` (tiny fake shard)
- Modify: `tests/test_build_corpora.py`

- [ ] **Step 1: Craft a tiny English 5-gram fixture**

```python
# One-off helper to produce the fixture. Run once:
import gzip
from pathlib import Path
sample_lines = [
    "the quick brown fox jumps\t2000,42,3\t2001,50,4",
    "she is a good doctor\t2000,100,10\t2001,95,9",
    "he is a good engineer\t2000,80,8\t2001,85,9",
] * 50
path = Path("tests/fixtures/en_ngram_sample/5-00000-of-00001.gz")
path.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(path, "wt", encoding="utf-8") as f:
    f.write("\n".join(sample_lines) + "\n")
```

Run this once and commit the `.gz`.

- [ ] **Step 2: Add an end-to-end smoke test**

Append to `tests/test_build_corpora.py`:

```python
def test_build_corpora_ngram_en_end_to_end(tmp_path):
    import shutil
    from scripts.data_prep.build_corpora_ngram_en import build_corpora
    from scripts.common.logging_utils import setup_logging

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    decomp_dir = tmp_path / "decomp"
    decomp_dir.mkdir()
    shutil.copy("tests/fixtures/en_ngram_sample/5-00000-of-00001.gz", raw_dir)

    config = {
        "language": "en",
        "data_source": "ngram",
        "paths": {
            "raw_ngram_dir": str(raw_dir),
            "decompressed_dir": str(decomp_dir),
            "corpora_dir": str(tmp_path / "corpora"),
            "log_dir": str(tmp_path / "logs"),
        },
        "time_slices": {"start_year": 1995, "end_year": 2005, "window_size": 10, "step_size": 5},
        "corpus": {"min_count_threshold": 1, "tokenizer": "whitespace", "lowercase": True},
    }
    logger = setup_logging(tmp_path / "logs", "test.log")
    build_corpora(config, logger)

    corpus_files = list((tmp_path / "corpora" / "1995_2004").glob("corpus_*.txt"))
    assert corpus_files, "expected English corpus output for the 1995_2004 slice"
    content = corpus_files[0].read_text()
    assert "quick brown fox" in content.lower() or "good doctor" in content.lower()
```

- [ ] **Step 3: Run the smoke test**

```bash
pytest -xvs tests/test_build_corpora.py::test_build_corpora_ngram_en_end_to_end
```

Expected: pass.

- [ ] **Step 4: Run the full test suite end-to-end**

```bash
pytest -x tests/
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "$(cat <<'EOF'
Add end-to-end smoke test for English Ngram pipeline

Tiny synthetic 5-gram fixture + test that runs the full build_corpora
flow on it and verifies output files exist and contain expected
content.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 25: Final pre-ship check

**Files:**
- None (verification only)

- [ ] **Step 1: Run the complete test suite**

```bash
pytest -x tests/
```

Expected: all tests pass. Capture the final test count.

- [ ] **Step 2: Verify every existing Chinese profile still loads**

```bash
for cfg in config/profiles/*.yml; do
  echo "=== $cfg ==="
  python -c "from scripts.common.config_loader import load_config; load_config('$cfg')" && echo OK || echo FAIL
done
```

Expected: 10 `OK` lines (7 original zh + 2 en ngram + 1 coha).

- [ ] **Step 3: Confirm the Chinese pipeline still produces identical output on an existing corpus**

If you have a pre-refactor Chinese corpus, train a model on it and spot-check that vocabulary size and a handful of nearest-neighbor queries match the pre-refactor baseline. (Optional but strongly recommended.)

- [ ] **Step 4: Verify git log**

```bash
git log --oneline -25
```

Expected: 24 commits matching the task numbers above, plus the spec commit.

- [ ] **Step 5: No commit for this task** — it's a verification pass. If any step fails, return to the relevant Task and fix.

---

## Self-review notes

- **Spec coverage:**
  - §2 decisions table → Tasks 1 (language validation), 2 (defaults table), 6 (wordlist move), 14 (English wordlists), 16 (profiles), 17–18 (COHA).
  - §3 config schema → Tasks 1, 2, 7, 16, 19.
  - §4 preprocessing module → Tasks 3, 4, 5.
  - §5 data pipeline (download_ngrams parameterize, build_corpora_ngram_en, download_coha, build_corpora_coha) → Tasks 13, 15, 17, 18.
  - §6 wordlist layout (zh moves, en seed) → Tasks 6, 14.
  - §7 visualize / correlation → Tasks 11, 12.
  - §8 Slurm → Task 22.
  - §9 NLTK dependency → Task 21.
  - §10 tests → Tasks 1 (config tests), 3/4/5 (preprocessing tests), 15/18 (parser tests), 24 (smoke).
  - §11 migration order → plan task order matches.
  - §12 Phase 2 → documented in README (Task 23); no tasks.

- **Placeholder scan:** `...` inside bash ellipses refer to existing flags in `run_pipeline.sh` (Task 20). Not a plan placeholder; the existing script already has the right flags. All code blocks are concrete.

- **Type consistency:** `preprocess()` signature is identical in every call site (Tasks 8–10). `tokenize()` dispatch name is consistent. `_configure_fonts(config)` signature unchanged between Task 11 and Task 12 (Task 12 imports it).

- **Noted caveats:**
  - Task 19 Step 2 warns the loader may need a rule-relaxation if it rejects empty `source_archive_urls`. Implementer should confirm behavior and adjust only if necessary.
  - Task 14's occupation list is a 60-term seed. Not a placeholder — it's genuinely a seed list, and the README (Task 23) tells the user to tune.
  - Task 24's fixture creation is a manual one-shot step (not fully automated in a test). Fine — fixture files are checked in.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-17-bilingual-refactor.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
