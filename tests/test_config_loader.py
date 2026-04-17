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
