"""Integration tests for the analyze_category_bias orchestrator."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class StubKV:
    def __init__(self, vectors):
        self._v = {w: np.asarray(v, dtype=float) for w, v in vectors.items()}
        self.key_to_index = {w: i for i, w in enumerate(self._v)}

    def __getitem__(self, k):
        return self._v[k]

    def __contains__(self, k):
        return k in self._v


def _install_fake_gensim():
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake = types.ModuleType("gensim")
    fake._fake = True
    models = types.ModuleType("gensim.models")
    models.KeyedVectors = StubKV
    fake.models = models
    sys.modules["gensim"] = fake
    sys.modules["gensim.models"] = models


_install_fake_gensim()

_MALE = {"he": [1.0, 0.0], "man": [1.0, 0.0]}
_FEMALE = {"she": [-1.0, 0.0], "woman": [-1.0, 0.0]}
_CATEGORIES = {"leadership": ["president", "manager"], "family": ["cooking", "cleaning"]}


def _kvs():
    occs_a = {"president": [0.6, 0.0], "manager": [0.5, 0.0],
              "cooking": [-0.5, 0.0], "cleaning": [-0.6, 0.0]}
    occs_b = {"president": [0.5, 0.0], "manager": [0.4, 0.0],
              "cooking": [-0.3, 0.0], "cleaning": [-0.4, 0.0]}
    return {"1990s": StubKV({**_MALE, **_FEMALE, **occs_a}),
            "2000s": StubKV({**_MALE, **_FEMALE, **occs_b})}


def _loader(unit_to_kv):
    def _load(model_path):
        name = Path(str(model_path)).name
        for unit, kv in unit_to_kv.items():
            if unit in name:
                return kv
        raise KeyError(name)
    return _load


def _write_config(tmp_path, metrics):
    base = tmp_path / "proj"
    for sub in ("data/models", "data/results"):
        (base / sub).mkdir(parents=True)
    (base / "logs").mkdir(parents=True)
    wl = tmp_path / "wordlists" / "en" / "garg_weat"
    wl.mkdir(parents=True)
    (wl / "gender_words.json").write_text(
        json.dumps({"male": list(_MALE), "female": list(_FEMALE)}), encoding="utf-8")
    cats_block = {}
    for cat, words in _CATEGORIES.items():
        (wl / f"candidates_{cat}.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
        cats_block[cat] = f"candidates_{cat}.txt"
    cfg = {
        "language": "en", "data_source": "coha", "analysis_mode": "garg_weat",
        "paths": {
            "base_dir": str(base),
            "raw_coha_dir": str(base / "data/raw_coha"),
            "coha_decompressed_dir": str(base / "data/coha_dec"),
            "corpora_dir": str(base / "data/corpora"),
            "models_dir": str(base / "data/models"),
            "results_dir": str(base / "data/results"),
            "log_dir": str(base / "logs"),
        },
        "coha": {"source_archive_urls": []},
        "embedding": {"model_name_template": "coha_{unit_name}.kv"},
        "wordlists": {"dir": str(wl), "gender_words_file": "gender_words.json",
                      "categories": cats_block},
        "analysis": {"metrics": metrics},
    }
    p = tmp_path / "config.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _touch(models_dir, units):
    for u in units:
        (models_dir / f"coha_{u}.kv").touch()


def test_both_metrics_single_pass(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, ["rnd", "cohens_d"])
    kvs = _kvs()
    rdir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    _touch(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]), list(kvs))

    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _loader(kvs))

    import scripts.analyze_category_bias as acb
    acb.main(config=str(cfg))

    assert (rdir / "garg_weat_rnd_long.parquet").exists()
    assert (rdir / "garg_weat_summary_by_category.parquet").exists()
    assert (rdir / "cohens_d_singlelist_long.parquet").exists()
    assert (rdir / "cohens_d_singlelist_summary_by_category.parquet").exists()

    rnd_long = pd.read_parquet(rdir / "garg_weat_rnd_long.parquet")
    assert "rnd" in rnd_long.columns
    proj_long = pd.read_parquet(rdir / "cohens_d_singlelist_long.parquet")
    assert "projection" in proj_long.columns

    for f in ("garg_weat_summary_by_category.parquet",
              "cohens_d_singlelist_summary_by_category.parquet"):
        s = pd.read_parquet(rdir / f)
        assert {"prop_male", "prop_ci_low", "prop_sub_low"} <= set(s.columns)


def test_rnd_only_skips_cohens_d(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path, ["rnd"])
    kvs = _kvs()
    rdir = Path(yaml.safe_load(cfg.read_text())["paths"]["results_dir"])
    _touch(Path(yaml.safe_load(cfg.read_text())["paths"]["models_dir"]), list(kvs))
    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _loader(kvs))
    import scripts.analyze_category_bias as acb
    acb.main(config=str(cfg))
    assert (rdir / "garg_weat_summary_by_category.parquet").exists()
    assert not (rdir / "cohens_d_singlelist_summary_by_category.parquet").exists()


def test_missing_metrics_key_raises(tmp_path, monkeypatch):
    import yaml as _yaml
    cfg = _write_config(tmp_path, ["rnd"])
    data = _yaml.safe_load(cfg.read_text())
    del data["analysis"]["metrics"]
    cfg.write_text(_yaml.safe_dump(data), encoding="utf-8")
    kvs = _kvs()
    _touch(Path(data["paths"]["models_dir"]), list(kvs))
    import scripts.analyze_garg as ag
    monkeypatch.setattr(ag, "load_model", _loader(kvs))
    import scripts.analyze_category_bias as acb
    import pytest
    with pytest.raises((ValueError, KeyError)):
        acb.main(config=str(cfg))


# ---------------------------------------------------------------------------
# analysis.decade_range clip
# ---------------------------------------------------------------------------

def _units(models):
    return [u for _, u in models]


def test_decade_range_drops_rolling_window_slices():
    """The Chinese longitudinal arms label units '1940_1949', not '1940s'.
    decade_range must clip on the slice START year — this is what lets
    [1950, 2020] drop 1940_1949 and 1945_1954 from the zh profiles."""
    import logging
    import scripts.analyze_category_bias as acb

    # What build_corpora.generate_time_slices(1940, 2020, window=10, step=5)
    # emits: starts every 5 years through 2020, last window clipped.
    slices = [f"{y}_{min(y + 9, 2020)}" for y in range(1940, 2021, 5)]
    models = [(f"/m/renminribao_{s}.model", s) for s in slices]
    assert len(models) == 17

    kept = _units(acb._filter_models(models, None, [1950, 2020],
                                     logging.getLogger("t")))
    assert "1940_1949" not in kept
    assert "1945_1954" not in kept
    assert len(kept) == 15
    assert kept[0] == "1950_1959" and kept[-1] == "2020_2020"


def test_decade_range_still_clips_decade_labels():
    """COHA / HistWords '1990s' labels keep their existing behavior."""
    import logging
    import scripts.analyze_category_bias as acb

    models = [("/m/a", "1810s"), ("/m/b", "1990s"), ("/m/c", "2000s")]
    kept = _units(acb._filter_models(models, None, [1990, 1990],
                                     logging.getLogger("t")))
    assert kept == ["1990s"]


def test_decade_range_keeps_yearless_provincial_units():
    """Provincial units carry no year; a window must not silently empty a
    provincial run."""
    import logging
    import scripts.analyze_category_bias as acb

    models = [("/m/bj", "北京"), ("/m/sh", "上海_2020"), ("/m/x", "1990s")]
    kept = _units(acb._filter_models(models, None, [1950, 2020],
                                     logging.getLogger("t")))
    assert kept == ["北京", "上海_2020", "1990s"]
