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
    data = yaml.safe_load(path.read_text())
    data.pop("time_slices", None)
    data["paths"].pop("raw_ngram_dir", None)
    data["paths"]["raw_coha_dir"] = "data/raw_coha"
    data["paths"]["coha_decompressed_dir"] = "data/raw_coha_decompressed"
    path.write_text(yaml.safe_dump(data))
    config = load_config(str(path))
    assert config["data_source"] == "coha"
    assert config["language"] == "en"
