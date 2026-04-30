"""
Tests for scripts.analyze_garg.

Fixture strategy: Option B — monkeypatch ``scripts.analyze_garg.load_model``
to return an in-memory stub. The host environment has a broken gensim
install (scipy.linalg.triu removed), so building real KeyedVectors files
is not viable here. The stub mimics only the surface used by
``compute_centroid`` / ``check_oov`` / vector indexing:
  - ``key_to_index`` (dict, used for ``word in model`` via ``in model.key_to_index``)
  - ``__getitem__`` (returns the np.ndarray for the word)
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pytest
import yaml


# -----------------------------------------------------------------------------
# Stub KeyedVectors
# -----------------------------------------------------------------------------

class StubKV:
    """Minimal KeyedVectors stand-in: dict of word -> np.ndarray."""

    def __init__(self, vectors: Dict[str, Iterable[float]]):
        self._vectors = {w: np.asarray(v, dtype=float) for w, v in vectors.items()}
        # The real gensim KeyedVectors exposes .key_to_index for membership.
        self.key_to_index = {w: i for i, w in enumerate(self._vectors)}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._vectors[key]

    def __contains__(self, key: str) -> bool:  # convenience, not strictly needed
        return key in self._vectors


# -----------------------------------------------------------------------------
# Install a fake gensim before importing the analyzer.
#
# The host env's gensim is broken (scipy.linalg.triu was removed). Real gensim
# is unnecessary for this test suite — we stub load_model anyway. Install a
# minimal fake so ``from gensim.models import KeyedVectors`` succeeds.
# -----------------------------------------------------------------------------

def _install_fake_gensim() -> None:
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        # Real gensim already loaded successfully — leave it alone.
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake_gensim = types.ModuleType("gensim")
    fake_gensim._fake = True
    fake_models = types.ModuleType("gensim.models")
    fake_models.KeyedVectors = StubKV  # any class works as a type alias here
    fake_gensim.models = fake_models
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()


# -----------------------------------------------------------------------------
# Helpers: build a fake config + wordlists + models on disk
# -----------------------------------------------------------------------------

def _write_config_and_wordlists(
    tmp_path: Path,
    occupations: List[str],
    gender_words: Dict[str, List[str]],
    model_template: str = "coha_{unit_name}.kv",
) -> Path:
    """Write a minimal valid config plus wordlists. Returns the config path."""
    base_dir = tmp_path / "proj"
    (base_dir / "data" / "models").mkdir(parents=True)
    (base_dir / "data" / "results").mkdir(parents=True)
    (base_dir / "logs").mkdir(parents=True)

    wl_dir = tmp_path / "wordlists" / "en" / "garg"
    wl_dir.mkdir(parents=True)
    (wl_dir / "occupations.txt").write_text(
        "\n".join(occupations) + "\n", encoding="utf-8"
    )
    (wl_dir / "gender_words.json").write_text(
        json.dumps(gender_words), encoding="utf-8"
    )

    config = {
        "language": "en",
        "data_source": "coha",
        "analysis_mode": "garg",
        "paths": {
            "base_dir": str(base_dir),
            "raw_coha_dir": str(base_dir / "data" / "raw_coha"),
            "coha_decompressed_dir": str(base_dir / "data" / "coha_dec"),
            "corpora_dir": str(base_dir / "data" / "corpora"),
            "models_dir": str(base_dir / "data" / "models"),
            "results_dir": str(base_dir / "data" / "results"),
            "log_dir": str(base_dir / "logs"),
        },
        "coha": {"n": 5, "source_archive_urls": []},
        "embedding": {"model_name_template": model_template},
        "wordlists": {
            "dir": str(wl_dir),
            "occupations_file": "occupations.txt",
            "gender_words_file": "gender_words.json",
        },
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _make_loader(unit_to_kv: Dict[str, StubKV]):
    """
    Build a stub for ``scripts.analyze_garg.load_model`` that resolves a
    model file path back to its unit name and returns the matching StubKV.
    """
    def _load(model_path):
        # filename like 'coha_1990s.kv' -> '1990s'
        name = Path(str(model_path)).name
        # Strip prefix up to the first underscore (we know template format here)
        for unit, kv in unit_to_kv.items():
            if unit in name:
                return kv
        raise KeyError(f"No StubKV registered for {name}")
    return _load


def _touch_models(models_dir: Path, unit_names: List[str], template: str = "coha_{unit_name}.kv") -> None:
    for u in unit_names:
        (models_dir / template.format(unit_name=u)).touch()


# -----------------------------------------------------------------------------
# Reusable vector fixtures
# -----------------------------------------------------------------------------

# Two male, two female anchor words. After analyze_garg's L2-normalize step
# every fetched vector is rescaled to unit length, so we pick already-unit
# vectors to make the post-normalize centroids hand-computable:
#   centroid_male = (1, 0)   centroid_female = (-1, 0)
_MALE_VECS = {
    "he":  [1.0, 0.0],
    "man": [1.0, 0.0],
}
_FEMALE_VECS = {
    "she":   [-1.0, 0.0],
    "woman": [-1.0, 0.0],
}
# centroids -> male=(1,0), female=(-1,0)


def _two_unit_kvs() -> Dict[str, StubKV]:
    """Two units sharing the same gender vectors but different occupation vectors."""
    occ_1990 = {
        "doctor":  [0.5, 0.0],   # closer to male
        "nurse":   [-0.5, 0.0],  # closer to female
        "teacher": [0.0, 0.5],   # neutral on x-axis
    }
    occ_2000 = {
        "doctor":  [0.2, 0.0],
        "nurse":   [-0.7, 0.0],
        "teacher": [0.1, 0.2],
    }
    return {
        "1990s": StubKV({**_MALE_VECS, **_FEMALE_VECS, **occ_1990}),
        "2000s": StubKV({**_MALE_VECS, **_FEMALE_VECS, **occ_2000}),
    }


# =============================================================================
# Tests
# =============================================================================

def test_long_table_schema_and_rowcount(tmp_path, monkeypatch):
    """WI-3 case 1: long table has correct schema + 2 units * 3 occupations rows."""
    occupations = ["doctor", "nurse", "teacher"]
    gender_words = {"male": list(_MALE_VECS), "female": list(_FEMALE_VECS)}
    cfg = _write_config_and_wordlists(tmp_path, occupations, gender_words)

    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    ag.main(config=str(cfg))

    long_df = pd.read_parquet(
        Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
        / "garg_relative_norm_by_decade.parquet"
    )
    assert set(long_df.columns) == {"unit_name", "occupation", "rnd", "in_vocab"}
    assert len(long_df) == 6
    assert long_df["in_vocab"].all()


def test_summary_table_schema_and_rowcount(tmp_path, monkeypatch):
    """WI-3 case 2: summary table schema + 2 rows + n_occupations==3."""
    occupations = ["doctor", "nurse", "teacher"]
    gender_words = {"male": list(_MALE_VECS), "female": list(_FEMALE_VECS)}
    cfg = _write_config_and_wordlists(tmp_path, occupations, gender_words)

    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    ag.main(config=str(cfg))

    summary_df = pd.read_parquet(
        Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
        / "garg_average_bias_by_decade.parquet"
    )
    assert set(summary_df.columns) == {
        "unit_name", "mean_rnd", "ci_low", "ci_high", "n_occupations"
    }
    assert len(summary_df) == 2
    assert (summary_df["n_occupations"] == 3).all()


def test_known_value_rnd_for_doctor(tmp_path, monkeypatch):
    """WI-3 case 3: hand-computable RND for one occupation (Garg sign convention).

    Post-L2-normalize:
      centroid_male = (1, 0), centroid_female = (-1, 0)
      doctor (0.5, 0) -> (1, 0)
        ||v - c_male|| = 0,  ||v - c_female|| = 2  -> RND = 0 - 2 = -2  (male-leaning)
      nurse (-0.5, 0) -> (-1, 0)
        ||v - c_male|| = 2,  ||v - c_female|| = 0  -> RND = 2 - 0 = +2  (female-leaning)
    """
    occupations = ["doctor", "nurse", "teacher"]
    gender_words = {"male": list(_MALE_VECS), "female": list(_FEMALE_VECS)}
    cfg = _write_config_and_wordlists(tmp_path, occupations, gender_words)

    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    ag.main(config=str(cfg))

    long_df = pd.read_parquet(
        Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
        / "garg_relative_norm_by_decade.parquet"
    )
    row = long_df[(long_df["unit_name"] == "1990s") & (long_df["occupation"] == "doctor")]
    assert len(row) == 1
    assert row.iloc[0]["rnd"] == pytest.approx(-2.0)
    # Symmetric: nurse should be +2 (female-leaning).
    nurse_row = long_df[(long_df["unit_name"] == "1990s") & (long_df["occupation"] == "nurse")]
    assert nurse_row.iloc[0]["rnd"] == pytest.approx(2.0)


def test_oov_occupation_excluded_from_mean_present_in_long(tmp_path, monkeypatch):
    """WI-3 case 4: 4 occupations, 1 OOV.

    Long table: 4 rows for the unit with in_vocab=False on the missing one.
    Summary: n_occupations == 3, mean computed only over the 3 in-vocab.
    """
    occupations = ["doctor", "nurse", "teacher", "ghostwriter"]
    gender_words = {"male": list(_MALE_VECS), "female": list(_FEMALE_VECS)}
    cfg = _write_config_and_wordlists(tmp_path, occupations, gender_words)

    # Only one unit, "ghostwriter" missing from its vocab
    kv = StubKV({
        **_MALE_VECS, **_FEMALE_VECS,
        "doctor":  [0.5, 0.0],
        "nurse":   [-0.5, 0.0],
        "teacher": [0.0, 0.5],
    })
    kvs = {"1990s": kv}
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    ag.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    long_df = pd.read_parquet(results_dir / "garg_relative_norm_by_decade.parquet")
    summary_df = pd.read_parquet(results_dir / "garg_average_bias_by_decade.parquet")

    assert len(long_df) == 4
    ghost = long_df[long_df["occupation"] == "ghostwriter"].iloc[0]
    assert ghost["in_vocab"] is np.False_ or ghost["in_vocab"] is False or ghost["in_vocab"] == False  # noqa: E712
    assert pd.isna(ghost["rnd"])

    assert len(summary_df) == 1
    assert summary_df.iloc[0]["n_occupations"] == 3
    # Post-L2-normalize: doctor=-2, nurse=+2, teacher=0 (teacher unit (0,1):
    # ||(-1,1)|| - ||(1,1)|| = sqrt(2) - sqrt(2) = 0). Mean = 0.
    expected_mean = (-2.0 + 2.0 + 0.0) / 3.0
    assert summary_df.iloc[0]["mean_rnd"] == pytest.approx(expected_mean)


def test_coverage_warning_logged_when_below_50pct(tmp_path, monkeypatch, caplog):
    """WI-3 case 5: warning logged when only 1 of 4 occupations is in vocab."""
    occupations = ["doctor", "nurse", "teacher", "engineer"]
    gender_words = {"male": list(_MALE_VECS), "female": list(_FEMALE_VECS)}
    cfg = _write_config_and_wordlists(tmp_path, occupations, gender_words)

    # Only "doctor" is in vocab.
    kv = StubKV({
        **_MALE_VECS, **_FEMALE_VECS,
        "doctor": [0.5, 0.0],
    })
    kvs = {"1990s": kv}
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    with caplog.at_level(logging.WARNING, logger="analyze_garg"):
        ag.main(config=str(cfg))

    warning_messages = [
        rec.message for rec in caplog.records
        if rec.levelno == logging.WARNING
    ]
    assert any("coverage" in m.lower() for m in warning_messages), (
        f"Expected a WARNING mentioning 'coverage', got: {warning_messages}"
    )


def test_unit_skipped_when_both_gender_centroids_unobtainable(tmp_path, monkeypatch, caplog):
    """WI-3 case 6: all gender words OOV => unit skipped, no rows written for it.

    Implementation choice: skip entirely (no row in either parquet) for the
    skipped unit. The other unit still produces its rows.
    """
    occupations = ["doctor", "nurse", "teacher"]
    gender_words = {"male": list(_MALE_VECS), "female": list(_FEMALE_VECS)}
    cfg = _write_config_and_wordlists(tmp_path, occupations, gender_words)

    good_kv = StubKV({
        **_MALE_VECS, **_FEMALE_VECS,
        "doctor": [0.5, 0.0], "nurse": [-0.5, 0.0], "teacher": [0.0, 0.5],
    })
    # bad_kv: occupations exist but no gender words
    bad_kv = StubKV({
        "doctor": [0.5, 0.0], "nurse": [-0.5, 0.0], "teacher": [0.0, 0.5],
    })
    kvs = {"1990s": good_kv, "2000s": bad_kv}
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    with caplog.at_level(logging.WARNING, logger="analyze_garg"):
        ag.main(config=str(cfg))

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    long_df = pd.read_parquet(results_dir / "garg_relative_norm_by_decade.parquet")
    summary_df = pd.read_parquet(results_dir / "garg_average_bias_by_decade.parquet")

    assert set(long_df["unit_name"].unique()) == {"1990s"}
    assert set(summary_df["unit_name"].unique()) == {"1990s"}
    assert any(
        "2000s" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), "Expected a WARNING mentioning the skipped unit '2000s'"


def test_unit_cli_filter(tmp_path, monkeypatch):
    """WI-3 case 7: --unit=1990s filters to just that unit."""
    occupations = ["doctor", "nurse", "teacher"]
    gender_words = {"male": list(_MALE_VECS), "female": list(_FEMALE_VECS)}
    cfg = _write_config_and_wordlists(tmp_path, occupations, gender_words)

    kvs = _two_unit_kvs()
    _touch_models(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]),
                  list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _make_loader(kvs))

    ag.main(config=str(cfg), unit="1990s")

    results_dir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    long_df = pd.read_parquet(results_dir / "garg_relative_norm_by_decade.parquet")
    summary_df = pd.read_parquet(results_dir / "garg_average_bias_by_decade.parquet")

    assert set(long_df["unit_name"].unique()) == {"1990s"}
    assert set(summary_df["unit_name"].unique()) == {"1990s"}
    assert len(summary_df) == 1
