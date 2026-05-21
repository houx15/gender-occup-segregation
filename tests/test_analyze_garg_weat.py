"""
Tests for scripts.analyze_garg_weat — per-category RND analyzer.

Fixture strategy mirrors test_analyze_garg.py: install a fake gensim
module (host gensim is broken), monkeypatch the loader on analyze_garg
(which analyze_garg_weat imports from), drive analyze_garg_weat.main
end-to-end through tmp_path-scoped config + on-disk model placeholders.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pytest
import yaml


# -----------------------------------------------------------------------------
# Stub KeyedVectors
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
# Reusable vectors — after L2-normalize, centroid_male = (1,0), centroid_female = (-1,0)
# -----------------------------------------------------------------------------

_MALE = {"he": [1.0, 0.0], "man": [1.0, 0.0]}
_FEMALE = {"she": [-1.0, 0.0], "woman": [-1.0, 0.0]}


def _two_unit_kvs() -> Dict[str, StubKV]:
    """Two decades, three categories with two occupations each."""
    occs_1990 = {
        # leadership: male-leaning
        "president": [0.6, 0.0],
        "manager": [0.5, 0.0],
        # family: female-leaning
        "cooking": [-0.5, 0.0],
        "cleaning": [-0.6, 0.0],
        # science: mildly male-leaning
        "scientist": [0.3, 0.0],
        "engineer": [0.4, 0.0],
    }
    occs_2000 = {
        "president": [0.5, 0.0],
        "manager": [0.4, 0.0],
        "cooking": [-0.3, 0.0],
        "cleaning": [-0.4, 0.0],
        "scientist": [0.2, 0.0],
        "engineer": [0.3, 0.0],
    }
    return {
        "1990s": StubKV({**_MALE, **_FEMALE, **occs_1990}),
        "2000s": StubKV({**_MALE, **_FEMALE, **occs_2000}),
    }


def _make_loader(unit_to_kv: Dict[str, StubKV]):
    def _load(model_path):
        name = Path(str(model_path)).name
        for unit, kv in unit_to_kv.items():
            if unit in name:
                return kv
        raise KeyError(f"No StubKV registered for {name}")
    return _load


def _write_config(
    tmp_path: Path,
    *,
    categories: Dict[str, List[str]],
    decade_range: Optional[List[int]] = None,
    model_template: str = "coha_{unit_name}.kv",
) -> Path:
    base = tmp_path / "proj"
    (base / "data" / "models").mkdir(parents=True)
    (base / "data" / "results").mkdir(parents=True)
    (base / "logs").mkdir(parents=True)
    wl_dir = tmp_path / "wordlists" / "en" / "garg_weat"
    wl_dir.mkdir(parents=True)
    (wl_dir / "gender_words.json").write_text(
        json.dumps({"male": list(_MALE), "female": list(_FEMALE)}),
        encoding="utf-8",
    )
    cats_block: Dict[str, str] = {}
    for cat, words in categories.items():
        fname = f"candidates_{cat}.txt"
        (wl_dir / fname).write_text("\n".join(words) + "\n", encoding="utf-8")
        cats_block[cat] = fname

    cfg = {
        "language": "en",
        "data_source": "coha",
        "analysis_mode": "garg_weat",
        "paths": {
            "base_dir": str(base),
            "raw_coha_dir": str(base / "data" / "raw_coha"),
            "coha_decompressed_dir": str(base / "data" / "coha_dec"),
            "corpora_dir": str(base / "data" / "corpora"),
            "models_dir": str(base / "data" / "models"),
            "results_dir": str(base / "data" / "results"),
            "log_dir": str(base / "logs"),
        },
        "coha": {"source_archive_urls": []},
        "embedding": {"model_name_template": model_template},
        "wordlists": {
            "dir": str(wl_dir),
            "gender_words_file": "gender_words.json",
            "categories": cats_block,
        },
    }
    if decade_range is not None:
        cfg["analysis"] = {"decade_range": list(decade_range)}
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _touch_models(models_dir: Path, units: List[str], template: str = "coha_{unit_name}.kv") -> None:
    for u in units:
        (models_dir / template.format(unit_name=u)).touch()


_CATEGORIES = {
    "leadership": ["president", "manager"],
    "family": ["cooking", "cleaning"],
    "science": ["scientist", "engineer"],
}


# =============================================================================
# Tests
# =============================================================================

def test_long_table_schema_and_rowcount(tmp_path, monkeypatch):
    """Long parquet has columns (unit_name, category, occupation, rnd, in_vocab)
    and the expected (2 units × 3 categories × 2 occupations) = 12 rows."""
    cfg = _write_config(tmp_path, categories=_CATEGORIES)
    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    import scripts.analyze_garg_weat as agw
    agw.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    long_df = pd.read_parquet(results_dir / "garg_weat_rnd_long.parquet")
    assert set(long_df.columns) >= {"unit_name", "category", "occupation", "rnd", "in_vocab"}
    assert len(long_df) == 12
    assert set(long_df["unit_name"]) == {"1990s", "2000s"}
    assert set(long_df["category"]) == {"leadership", "family", "science"}


def test_summary_per_unit_per_category(tmp_path, monkeypatch):
    """Summary has one row per (unit, category) = 2 × 3 = 6 rows, with
    mean_rnd populated for every row when all occupations are in vocab."""
    cfg = _write_config(tmp_path, categories=_CATEGORIES)
    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    import scripts.analyze_garg_weat as agw
    agw.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    summary = pd.read_parquet(results_dir / "garg_weat_summary_by_category.parquet")
    assert set(summary.columns) >= {
        "unit_name", "category", "mean_rnd", "ci_low", "ci_high",
        "n_occupations", "n_consistent",
    }
    assert len(summary) == 6
    # Each category has 2 occupations in vocab across both decades
    assert (summary["n_consistent"] == 2).all()
    assert summary["mean_rnd"].notna().all()


def test_sign_convention_leadership_male_family_female(tmp_path, monkeypatch):
    """RND = ||v − c_male|| − ||v − c_female||, so positive = female-leaning.
    Leadership occupations are placed near c_male (positive x) so their
    mean_rnd should be NEGATIVE; family is near c_female so POSITIVE.
    """
    cfg = _write_config(tmp_path, categories=_CATEGORIES)
    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    import scripts.analyze_garg_weat as agw
    agw.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    summary = pd.read_parquet(results_dir / "garg_weat_summary_by_category.parquet")
    leadership = summary[summary["category"] == "leadership"]
    family = summary[summary["category"] == "family"]
    # Leadership words at +x are closer to c_male=(1,0) → smaller ||v − c_male|| →
    # negative RND.
    assert (leadership["mean_rnd"] < 0).all()
    assert (family["mean_rnd"] > 0).all()


def test_decade_range_filter_restricts_units(tmp_path, monkeypatch):
    """decade_range=[1990, 1990] keeps only the 1990s unit."""
    cfg = _write_config(
        tmp_path, categories=_CATEGORIES, decade_range=[1990, 1990],
    )
    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    import scripts.analyze_garg_weat as agw
    agw.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    summary = pd.read_parquet(results_dir / "garg_weat_summary_by_category.parquet")
    assert set(summary["unit_name"]) == {"1990s"}
    assert len(summary) == 3  # 1 unit × 3 categories


def test_unit_skipped_when_gender_centroids_unobtainable(tmp_path, monkeypatch, caplog):
    """If a unit's vocab contains no gender words, the unit is skipped and
    the analyzer continues to the next."""
    cfg = _write_config(tmp_path, categories=_CATEGORIES)
    # 1990s has gender words; 2000s only has occupations (no he/she/man/woman)
    occs_2000_only = {
        "president": [0.5, 0.0],
        "manager": [0.4, 0.0],
        "cooking": [-0.3, 0.0],
        "cleaning": [-0.4, 0.0],
        "scientist": [0.2, 0.0],
        "engineer": [0.3, 0.0],
    }
    kv_1990 = StubKV({
        **_MALE, **_FEMALE,
        "president": [0.6, 0.0], "manager": [0.5, 0.0],
        "cooking": [-0.5, 0.0], "cleaning": [-0.6, 0.0],
        "scientist": [0.3, 0.0], "engineer": [0.4, 0.0],
    })
    kv_2000 = StubKV(occs_2000_only)
    kvs = {"1990s": kv_1990, "2000s": kv_2000}
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    import scripts.analyze_garg_weat as agw
    agw.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    summary = pd.read_parquet(results_dir / "garg_weat_summary_by_category.parquet")
    # Only 1990s should produce results
    assert set(summary["unit_name"]) == {"1990s"}


def test_consistent_set_per_category_drops_unit_oov(tmp_path, monkeypatch):
    """An occupation in vocab in one unit but not the other should be excluded
    from BOTH decades' mean for its category (per-category consistent set)."""
    categories = {
        "leadership": ["president", "manager", "rareleader"],
        "family": ["cooking", "cleaning"],
        "science": ["scientist", "engineer"],
    }
    cfg = _write_config(tmp_path, categories=categories)
    # rareleader appears in 1990s only
    kv_1990 = StubKV({
        **_MALE, **_FEMALE,
        "president": [0.6, 0.0], "manager": [0.5, 0.0],
        "rareleader": [0.7, 0.0],
        "cooking": [-0.5, 0.0], "cleaning": [-0.6, 0.0],
        "scientist": [0.3, 0.0], "engineer": [0.4, 0.0],
    })
    kv_2000 = StubKV({
        **_MALE, **_FEMALE,
        "president": [0.5, 0.0], "manager": [0.4, 0.0],
        # rareleader missing
        "cooking": [-0.3, 0.0], "cleaning": [-0.4, 0.0],
        "scientist": [0.2, 0.0], "engineer": [0.3, 0.0],
    })
    kvs = {"1990s": kv_1990, "2000s": kv_2000}
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    import scripts.analyze_garg_weat as agw
    agw.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    summary = pd.read_parquet(results_dir / "garg_weat_summary_by_category.parquet")
    leadership_rows = summary[summary["category"] == "leadership"]
    # consistent set = {president, manager} (rareleader dropped) → n_consistent==2
    assert (leadership_rows["n_consistent"] == 2).all()
    # Other categories unchanged
    family_rows = summary[summary["category"] == "family"]
    assert (family_rows["n_consistent"] == 2).all()
