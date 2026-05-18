"""
Tests for scripts.analyze_weat — specifically the changes added to make the
script runnable across all three WEAT-on-Garg-datasets configs:
  - embedding.format dispatch (gensim_kv default vs histwords)
  - analysis.decade_range clip

Fixture strategy mirrors tests/test_analyze_garg.py: install a fake gensim
module (host gensim is broken via scipy.linalg.triu removal), monkeypatch
the loader, drive the script through ``analyze_weat.main`` end-to-end.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pytest
import yaml


# -----------------------------------------------------------------------------
# Stub KeyedVectors — same surface as tests/test_analyze_garg.py
# -----------------------------------------------------------------------------

class StubKV:
    def __init__(self, vectors: Dict[str, Iterable[float]]):
        self._vectors = {w: np.asarray(v, dtype=float) for w, v in vectors.items()}
        self.key_to_index = {w: i for i, w in enumerate(self._vectors)}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._vectors[key]

    def __contains__(self, key: str) -> bool:
        return key in self._vectors


def _install_fake_gensim() -> None:
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake_gensim = types.ModuleType("gensim")
    fake_gensim._fake = True
    fake_models = types.ModuleType("gensim.models")
    fake_models.KeyedVectors = StubKV
    fake_gensim.models = fake_models
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_MALE = {"he": [1.0, 0.0], "man": [1.0, 0.0]}
_FEMALE = {"she": [-1.0, 0.0], "woman": [-1.0, 0.0]}
# Three attribute dimensions × 2 words each — minimal viable WEAT vocab.
_CONCEPTS = {
    "family": [("home", [-0.8, 0.1]), ("kitchen", [-0.7, 0.2])],
    "work": [("office", [0.7, 0.1]), ("salary", [0.8, -0.1])],
    "leadership": [("manager", [0.6, 0.2]), ("chief", [0.7, 0.0])],
    "non_leadership": [("assistant", [-0.5, 0.2]), ("intern", [-0.4, 0.0])],
    "stem": [("science", [0.3, 0.5]), ("physics", [0.4, 0.6])],
    "non_stem": [("arts", [-0.3, 0.5]), ("poetry", [-0.4, 0.6])],
}


def _all_concept_vecs() -> Dict[str, list]:
    out = {}
    for words in _CONCEPTS.values():
        for w, v in words:
            out[w] = v
    return out


def _write_wordlists(wl_dir: Path) -> None:
    wl_dir.mkdir(parents=True, exist_ok=True)
    (wl_dir / "gender_words.json").write_text(
        json.dumps({"male": list(_MALE), "female": list(_FEMALE)}),
        encoding="utf-8",
    )
    (wl_dir / "domestic_work_words.json").write_text(
        json.dumps({
            "family": [w for w, _ in _CONCEPTS["family"]],
            "work": [w for w, _ in _CONCEPTS["work"]],
        }),
        encoding="utf-8",
    )
    (wl_dir / "leadership_words.json").write_text(
        json.dumps({
            "leadership": [w for w, _ in _CONCEPTS["leadership"]],
            "non_leadership": [w for w, _ in _CONCEPTS["non_leadership"]],
        }),
        encoding="utf-8",
    )
    (wl_dir / "stem_words.json").write_text(
        json.dumps({
            "stem": [w for w, _ in _CONCEPTS["stem"]],
            "non_stem": [w for w, _ in _CONCEPTS["non_stem"]],
        }),
        encoding="utf-8",
    )


def _write_config(
    tmp_path: Path,
    *,
    model_template: str = "coha_{unit_name}.kv",
    embedding_format: str | None = None,
    decade_range: list | None = None,
) -> Path:
    base = tmp_path / "proj"
    (base / "data" / "models").mkdir(parents=True)
    (base / "data" / "results").mkdir(parents=True)
    (base / "logs").mkdir(parents=True)
    wl_dir = tmp_path / "wordlists" / "en" / "weat_formal"
    _write_wordlists(wl_dir)

    embedding_block: dict = {"model_name_template": model_template}
    if embedding_format is not None:
        embedding_block["format"] = embedding_format

    cfg = {
        "language": "en",
        "data_source": "coha",
        "analysis_mode": "weat",
        "paths": {
            "base_dir": str(base),
            # corpora_dir + raw_coha_dir are required by config_loader for
            # non-pretrained COHA configs; the directories aren't actually read.
            "raw_coha_dir": str(base / "data" / "raw_coha"),
            "coha_decompressed_dir": str(base / "data" / "coha_dec"),
            "corpora_dir": str(base / "data" / "corpora"),
            "models_dir": str(base / "data" / "models"),
            "results_dir": str(base / "data" / "results"),
            "log_dir": str(base / "logs"),
        },
        "embedding": embedding_block,
        "wordlists": {
            "dir": str(wl_dir),
            "weat_gender_file": "gender_words.json",
            "weat_domestic_work_file": "domestic_work_words.json",
            "weat_leadership_file": "leadership_words.json",
            "weat_stem_file": "stem_words.json",
        },
    }
    # Validator requires a top-level `coha` block when data_source=coha;
    # contents aren't read by analyze_weat.
    cfg["coha"] = {"source_archive_urls": []}
    if decade_range is not None:
        cfg["analysis"] = {"decade_range": list(decade_range)}
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _make_kv(extra: Dict[str, Iterable[float]] | None = None) -> StubKV:
    vecs = {**_MALE, **_FEMALE, **_all_concept_vecs()}
    if extra:
        vecs.update(extra)
    return StubKV(vecs)


def _make_loader(unit_to_kv: Dict[str, StubKV]):
    def _load(model_path):
        name = Path(str(model_path)).name
        for unit, kv in unit_to_kv.items():
            if unit in name:
                return kv
        raise KeyError(f"No StubKV registered for {name}")
    return _load


def _touch_models(models_dir: Path, units: List[str], template: str = "coha_{unit_name}.kv") -> None:
    for u in units:
        (models_dir / template.format(unit_name=u)).touch()


# =============================================================================
# Tests
# =============================================================================

def test_default_gensim_kv_discovery_unchanged(tmp_path, monkeypatch):
    """Regression: default (no embedding.format) still finds template-named
    .kv files and produces weat_results.csv with one row per (unit, dimension).
    """
    cfg = _write_config(tmp_path)
    kvs = {"1990s": _make_kv(), "2000s": _make_kv()}
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_weat as aw
    monkeypatch.setattr(aw, "load_model", _make_loader(kvs))

    aw.main(config=str(cfg), skip_oov=True)

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    weat_df = pd.read_csv(results_dir / "weat_results.csv")
    # 2 units × 3 dimensions
    assert len(weat_df) == 6
    assert set(weat_df["unit"]) == {"1990s", "2000s"}
    assert set(weat_df["dimension"]) == {"work_family", "leadership", "stem"}


def test_histwords_format_routes_through_loader(tmp_path, monkeypatch):
    """embedding.format=histwords: discover_units must find *-w.npy pairs and
    load_model_for_unit must dispatch to load_histwords_decade.
    """
    cfg = _write_config(tmp_path, embedding_format="histwords")
    models_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"])

    # Lay out fake -w.npy + -vocab.pkl pairs.
    import pickle
    for year in (1990, 2000):
        np.save(models_dir / f"{year}-w.npy", np.zeros((1, 1), dtype=np.float32))
        with open(models_dir / f"{year}-vocab.pkl", "wb") as f:
            pickle.dump(["dummy"], f)

    kvs = {"1990s": _make_kv(), "2000s": _make_kv()}

    def fake_load_histwords(npy_path, vocab_path=None):
        stem = Path(npy_path).name[: -len("-w.npy")]
        return kvs[f"{stem}s"]

    import scripts.common.embedding_loaders as el
    monkeypatch.setattr(el, "load_histwords_decade", fake_load_histwords)

    import scripts.analyze_weat as aw
    aw.main(config=str(cfg), skip_oov=True)

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    weat_df = pd.read_csv(results_dir / "weat_results.csv")
    assert set(weat_df["unit"]) == {"1990s", "2000s"}
    assert len(weat_df) == 6  # 2 units × 3 dims


def test_decade_range_filter_clips_units(tmp_path, monkeypatch):
    """decade_range=[1990, 1990] keeps only 1990s and drops 1810s + 2000s.
    Mirrors the Garg pipeline's HistWords decade-range bugfix.
    """
    cfg = _write_config(tmp_path, decade_range=[1990, 1990])
    models_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"])
    kvs = {"1810s": _make_kv(), "1990s": _make_kv(), "2000s": _make_kv()}
    _touch_models(models_dir, list(kvs))

    import scripts.analyze_weat as aw
    monkeypatch.setattr(aw, "load_model", _make_loader(kvs))

    aw.main(config=str(cfg), skip_oov=True)

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    weat_df = pd.read_csv(results_dir / "weat_results.csv")
    assert set(weat_df["unit"]) == {"1990s"}
    assert len(weat_df) == 3  # 1 unit × 3 dimensions
