"""
Unified configuration loader for gender-occupation segregation analysis.

Loads YAML config, validates required fields based on data_source type,
and resolves relative paths against base_dir.
"""

from pathlib import Path
from typing import Optional

import yaml


VALID_LANGUAGES = {"zh", "en"}

DATA_SOURCE_LANGUAGE_COMPAT = {
    "ngram":       {"zh", "en"},
    "renminribao": {"zh"},
    "weibo":       {"zh"},
    "newspaper":   {"zh"},
    "coha":        {"en"},
}

# Valid data sources and their default analysis units
DATA_SOURCE_DEFAULTS = {
    "ngram": "longitudinal",
    "renminribao": "longitudinal",
    "weibo": "provincial",
    "newspaper": "provincial",
    "coha": "longitudinal",
}

VALID_ANALYSIS_MODES = {"prestige", "weat", "garg"}

# Embedding formats that bypass our download/corpus/train pipeline. Profiles
# using these formats only need models_dir/results_dir/log_dir; the corpus
# and raw-data paths are not required.
PRETRAINED_EMBEDDING_FORMATS = {"histwords", "word2vec_bin", "glove_text"}


def load_config(config_path: str) -> dict:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _validate_config(config)
    _resolve_paths(config)
    _set_defaults(config)

    return config


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
    # When using a pretrained-embedding format, we bypass corpus building and
    # training entirely — only models_dir/results_dir/log_dir matter.
    embedding_format = config.get("embedding", {}).get("format", "gensim_kv")
    is_pretrained = embedding_format in PRETRAINED_EMBEDDING_FORMATS
    if is_pretrained:
        required_paths = ["base_dir", "models_dir", "results_dir", "log_dir"]
    else:
        required_paths = ["base_dir", "corpora_dir", "models_dir", "results_dir", "log_dir"]
    for key in required_paths:
        if key not in paths:
            raise ValueError(f"Missing required path: paths.{key}")

    if not is_pretrained:
        if data_source == "ngram":
            if "raw_ngram_dir" not in paths:
                raise ValueError("ngram data_source requires paths.raw_ngram_dir")
        if data_source == "coha":
            if "raw_coha_dir" not in paths:
                raise ValueError("coha data_source requires paths.raw_coha_dir")
            if "coha_decompressed_dir" not in paths:
                raise ValueError("coha data_source requires paths.coha_decompressed_dir")
            if "coha" not in config:
                raise ValueError("coha data_source requires a top-level 'coha' config block")
        if data_source in ("ngram", "renminribao"):
            if "time_slices" not in config:
                raise ValueError(f"{data_source} data_source requires time_slices config")


def _resolve_paths(config: dict) -> None:
    """Resolve relative paths against base_dir."""
    base_dir = Path(config["paths"]["base_dir"])
    paths = config["paths"]

    for key, value in paths.items():
        if key == "base_dir":
            continue
        path = Path(value)
        if not path.is_absolute():
            paths[key] = str(base_dir / path)

    # Resolve wordlist paths — relative to repo root (cwd), NOT base_dir,
    # since wordlists are shipped with the repo, not stored in the data directory.
    wordlists = config.get("wordlists", {})
    wl_dir = wordlists.get("dir")
    if wl_dir:
        wl_path = Path(wl_dir)
        if not wl_path.is_absolute():
            # Try repo root (cwd) first, fall back to base_dir
            if (Path.cwd() / wl_path).exists():
                wordlists["dir"] = str(Path.cwd() / wl_path)
            elif (base_dir / wl_path).exists():
                wordlists["dir"] = str(base_dir / wl_path)
            # else leave as-is (will error later with a clear message)


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
        if data_source in ("ngram", "renminribao"):
            config["analysis_mode"] = "prestige"
        else:
            config["analysis_mode"] = "weat"

    corpus = config.setdefault("corpus", {})
    tok_default, sw_default, lc_default = _CORPUS_DEFAULTS[(language, data_source)]
    corpus.setdefault("tokenizer", tok_default)
    if sw_default is not None:
        corpus.setdefault("stopwords", sw_default)
    corpus.setdefault("lowercase", lc_default)


def get_analysis_unit(config: dict) -> str:
    """Get the analysis unit ('longitudinal' or 'provincial')."""
    return config.get("analysis_unit", DATA_SOURCE_DEFAULTS[config["data_source"]])


def _parse_model_template(config: dict):
    """
    Parse model_name_template into (prefix, suffix).

    Handles all placeholder variants: {unit_name}, {slice_name}, {province}.
    """
    import re
    template = config.get("embedding", {}).get("model_name_template", "{unit_name}.model")
    # Split on any of the supported placeholders
    parts = re.split(r"\{(?:unit_name|slice_name|province)\}", template)
    prefix = parts[0] if len(parts) > 0 else ""
    suffix = parts[1] if len(parts) > 1 else ".model"
    return prefix, suffix


def get_model_name(unit_name: str, config: dict) -> str:
    """
    Get model filename for a given unit (time slice or province).

    Args:
        unit_name: Name of the unit (e.g., "1940_1949" or "北京")
        config: Configuration dictionary

    Returns:
        Model filename string
    """
    template = config.get("embedding", {}).get("model_name_template", "{unit_name}.model")
    return template.format(unit_name=unit_name, slice_name=unit_name, province=unit_name)


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

    if analysis_mode == "garg":
        return repo_root / "wordlists" / language / "garg"

    # WEAT
    data_source = config["data_source"]
    if data_source == "weibo":
        return repo_root / "wordlists" / language / "weat_informal"
    return repo_root / "wordlists" / language / "weat_formal"
