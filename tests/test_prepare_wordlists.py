"""
Tests for scripts.prepare_wordlists pure helpers: per-file dedup and
coverage-threshold prune. The model-probing path needs embeddings and is
exercised on the cluster, not here.
"""

from __future__ import annotations

import sys
import types


def _install_fake_gensim() -> None:
    """Host gensim is broken (scipy.linalg.triu removed). prepare_wordlists
    imports analyze_garg -> embedding_utils which imports gensim at module
    top-level; stub it so the pure-function import succeeds."""
    try:
        from gensim.models import KeyedVectors  # noqa: F401
        return
    except Exception:
        pass
    fake_gensim = types.ModuleType("gensim")
    fake_models = types.ModuleType("gensim.models")
    fake_models.KeyedVectors = object
    fake_gensim.models = fake_models
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()

from scripts.prepare_wordlists import (  # noqa: E402
    category_from_filename,
    dedup_within_files,
    prune_by_coverage,
)


def test_category_from_filename():
    assert category_from_filename("candidates_leadership.txt") == "leadership"
    assert category_from_filename("cleaned_family.txt") == "family"
    assert category_from_filename("science.txt") == "science"


def test_dedup_keeps_first_occurrence_within_file():
    categories = {
        "leadership": ["leader", "manager", "leader", "chief", "manager"],
    }
    deduped, dropped = dedup_within_files(categories)
    assert deduped["leadership"] == ["leader", "manager", "chief"]
    dropped_words = sorted(r["word"] for r in dropped)
    assert dropped_words == ["leader", "manager"]
    assert all(r["reason"] == "duplicate_in_file" for r in dropped)


def test_dedup_preserves_cross_category_duplicates():
    """A word in two categories stays in both — dedup is per-file only."""
    categories = {
        "leadership": ["logical", "decisive"],
        "science": ["logical", "rational"],
    }
    deduped, dropped = dedup_within_files(categories)
    assert "logical" in deduped["leadership"]
    assert "logical" in deduped["science"]
    assert dropped == []


def test_prune_by_coverage_threshold_boundary():
    """9/11 = 0.818 passes a 0.8 bar; 8/11 = 0.727 fails."""
    categories = {"science": ["keepme", "dropme"]}
    coverage = {
        ("science", "keepme"): 9 / 11,
        ("science", "dropme"): 8 / 11,
    }
    kept, pruned = prune_by_coverage(categories, coverage, threshold=0.8)
    assert kept["science"] == ["keepme"]
    assert [r["word"] for r in pruned] == ["dropme"]
    assert pruned[0]["reason"] == "below_threshold"


def test_prune_missing_coverage_treated_as_zero():
    categories = {"family": ["never_in_vocab"]}
    kept, pruned = prune_by_coverage(categories, coverage={}, threshold=0.8)
    assert kept["family"] == []
    assert pruned[0]["coverage"] == 0.0


def test_prune_exact_threshold_kept():
    """coverage == threshold is inclusive (>= bar)."""
    categories = {"leadership": ["edge"]}
    kept, _ = prune_by_coverage(
        categories, {("leadership", "edge"): 0.8}, threshold=0.8
    )
    assert kept["leadership"] == ["edge"]
