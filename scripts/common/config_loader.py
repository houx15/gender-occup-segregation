"""
Unified configuration loader for gender-occupation segregation analysis.

Loads YAML config, validates required fields based on data_source type,
and resolves relative paths against base_dir.
"""

from pathlib import Path
from typing import Optional

import yaml


# Valid data sources and their default analysis units
DATA_SOURCE_DEFAULTS = {
    "ngram": "longitudinal",
    "renminribao": "longitudinal",
    "weibo": "provincial",
    "newspaper": "provincial",
}

VALID_ANALYSIS_MODES = {"prestige", "weat"}


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
    data_source = config.get("data_source")
    if data_source not in DATA_SOURCE_DEFAULTS:
        raise ValueError(
            f"Invalid data_source: {data_source}. "
            f"Must be one of: {list(DATA_SOURCE_DEFAULTS.keys())}"
        )

    analysis_mode = config.get("analysis_mode")
    if analysis_mode and analysis_mode not in VALID_ANALYSIS_MODES:
        raise ValueError(
            f"Invalid analysis_mode: {analysis_mode}. "
            f"Must be one of: {list(VALID_ANALYSIS_MODES)}"
        )

    # Validate required paths
    paths = config.get("paths", {})
    required_paths = ["base_dir", "corpora_dir", "models_dir", "results_dir", "log_dir"]
    for key in required_paths:
        if key not in paths:
            raise ValueError(f"Missing required path: paths.{key}")

    # Source-specific validation
    if data_source == "ngram":
        if "raw_ngram_dir" not in paths:
            raise ValueError("ngram data_source requires paths.raw_ngram_dir")
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

    # Resolve wordlist paths
    wordlists = config.get("wordlists", {})
    wl_dir = wordlists.get("dir")
    if wl_dir:
        wl_path = Path(wl_dir)
        if not wl_path.is_absolute():
            wordlists["dir"] = str(base_dir / wl_path)


def _set_defaults(config: dict) -> None:
    """Set default values based on data_source."""
    data_source = config["data_source"]

    # Default analysis unit
    if "analysis_unit" not in config:
        config["analysis_unit"] = DATA_SOURCE_DEFAULTS[data_source]

    # Default analysis mode
    if "analysis_mode" not in config:
        if data_source in ("ngram", "renminribao"):
            config["analysis_mode"] = "prestige"
        else:
            config["analysis_mode"] = "weat"

    # Default tokenizer
    corpus = config.setdefault("corpus", {})
    if "tokenizer" not in corpus:
        if data_source == "ngram":
            corpus["tokenizer"] = "whitespace"
        else:
            corpus["tokenizer"] = "jieba"


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
    """Get the resolved wordlist directory path."""
    wl_dir = config.get("wordlists", {}).get("dir")
    if wl_dir:
        return Path(wl_dir)
    # Fallback based on analysis_mode
    base_dir = Path(config["paths"]["base_dir"])
    analysis_mode = config.get("analysis_mode", "prestige")
    if analysis_mode == "prestige":
        return base_dir / "wordlists" / "prestige"
    elif analysis_mode == "weat":
        # Default to formal for newspaper/renminribao, informal for weibo
        data_source = config["data_source"]
        if data_source == "weibo":
            return base_dir / "wordlists" / "weat_informal"
        else:
            return base_dir / "wordlists" / "weat_formal"
    return base_dir / "wordlists"
