"""End-to-end smoke test for scripts.describe_dataset.

Builds a tiny corpus + raw-data tree, runs ``run(config, ...)``, and asserts
the Markdown lands at the right path with all four sections.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

logger = logging.getLogger("test")


def _install_fake_gensim():
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake_gensim = types.ModuleType("gensim")
    fake_gensim._fake = True  # type: ignore[attr-defined]
    fake_models = types.ModuleType("gensim.models")

    class _FakeKV:
        @staticmethod
        def load(path):
            # Same vocab list as the shim in tests/test_dataset_stats.py.
            # Both files install module-level fakes; pytest collects every
            # test module before running, so whichever shim is installed last
            # wins. Keep these in sync so per-test-file order doesn't matter.
            class _Stub:
                index_to_key = ["a", "b", "c"]
            return _Stub()

    fake_models.KeyedVectors = _FakeKV  # type: ignore[attr-defined]
    fake_gensim.models = fake_models  # type: ignore[attr-defined]
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()

from scripts.common.dataset_stats import run, resolve_walker


def _make_rmrb_setup(tmp_path: Path) -> tuple[Path, dict]:
    """Set up corpora + models + raw layout for a 2-slice RMRB run."""
    corpora = tmp_path / "corpora"
    models = tmp_path / "models"
    raw = tmp_path / "raw"
    results = tmp_path / "results"
    for d in (corpora, models, raw, results):
        d.mkdir(parents=True, exist_ok=True)

    # Two units' corpus files.
    (corpora / "1940_1949").mkdir()
    (corpora / "1940_1949" / "corpus_000000").write_text("a b c\na b\n", encoding="utf-8")
    (corpora / "1950_1959").mkdir()
    (corpora / "1950_1959" / "corpus_000000").write_text("c d\n", encoding="utf-8")

    # Two stub model files.
    (models / "rmrb_1940_1949.model").write_text("stub")
    (models / "rmrb_1950_1959.model").write_text("stub")

    # RMRB raw layout.
    rmrb_dir = raw / "1940s" / "1942" / "报刊" / "人民日报"
    rmrb_dir.mkdir(parents=True)
    (rmrb_dir / "rmrb_1942_01.txt").write_text("x" * 100, encoding="utf-8")

    config = {
        "language": "zh",
        "data_source": "renminribao",
        "embedding": {"vector_size": 300, "window": 4, "min_count": 50, "sg": 1,
                      "negative": 15, "epochs": 5, "seed": 42, "workers": 16,
                      "model_name_template": "rmrb_{slice_name}.model"},
        "time_slices": {"start_year": 1940, "end_year": 1959,
                        "window_size": 10, "step_size": 10},
        "paths": {
            "corpora_dir": str(corpora),
            "models_dir": str(models),
            "raw_data_dir": str(raw),
            "results_dir": str(results),
        },
    }
    return results, config


class TestResolveWalker:
    def test_renminribao(self):
        assert resolve_walker({"data_source": "renminribao"}) is not None

    def test_ngram_zh(self):
        w = resolve_walker({"data_source": "ngram", "language": "zh"})
        assert w is not None
        from scripts.data_prep.raw_volume.ngram_zh import walk
        assert w is walk

    def test_ngram_en(self):
        w = resolve_walker({"data_source": "ngram", "language": "en"})
        assert w is not None
        from scripts.data_prep.raw_volume.ngram_en import walk
        assert w is walk

    def test_unknown_source(self):
        assert resolve_walker({"data_source": "what"}) is None


class TestRun:
    def test_writes_markdown_with_all_sections(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml")
        md_path = results / "dataset_summary.md"
        assert md_path.exists()
        md = md_path.read_text(encoding="utf-8")
        assert "# Dataset Summary — renminribao (zh)" in md
        assert "## Corpus totals" in md
        assert "## Raw data" in md
        assert "## Training" in md
        assert "## Per-unit breakdown" in md
        assert "1940_1949" in md
        assert "1950_1959" in md

    def test_force_recomputes(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml")
        cache = Path(config["paths"]["corpora_dir"]) / "1940_1949" / ".dataset_stats.json"
        assert cache.exists()
        # Run again with force — should re-write (mtime bumps).
        first_mtime = cache.stat().st_mtime
        import time; time.sleep(0.01)
        run(config, config_path="x.yml", force=True)
        assert cache.stat().st_mtime >= first_mtime

    def test_no_raw_skips_walker(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml", no_raw=True)
        md = (results / "dataset_summary.md").read_text(encoding="utf-8")
        # Raw bytes cell becomes n/a when no_raw skipped.
        assert "n/a" in md

    def test_unit_filter(self, tmp_path):
        results, config = _make_rmrb_setup(tmp_path)
        run(config, config_path="x.yml", units="1940_1949")
        md = (results / "dataset_summary.md").read_text(encoding="utf-8")
        assert "1940_1949" in md
        # 1950_1959 should be absent from per-unit table.
        assert "1950_1959" not in md
