"""Tests for scripts.common.embedding_loaders.

The histwords loader uses real numpy + a real KeyedVectors, but it does
NOT need a working SGNS trainer — gensim 4.x KeyedVectors in this env is
sufficient for vector-store ops. If gensim is broken on the host, mark
the suite skip.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

# discover/parse functions don't need gensim; load_histwords_decade does.
# Mark only the gensim-dependent tests as skip when gensim is unavailable
# OR when test_analyze_garg's fake-gensim shim has clobbered sys.modules
# (the StubKV class in that shim doesn't satisfy our loader's API).
_GENSIM_OK = True
_GENSIM_REASON = ""
try:
    from gensim.models import KeyedVectors
    if not hasattr(KeyedVectors, "add_vectors"):
        _GENSIM_OK = False
        _GENSIM_REASON = "gensim KeyedVectors lacks add_vectors (likely test stub)"
except Exception as e:
    _GENSIM_OK = False
    _GENSIM_REASON = f"gensim KeyedVectors unavailable: {e}"

requires_gensim = pytest.mark.skipif(not _GENSIM_OK, reason=_GENSIM_REASON)

from scripts.common.embedding_loaders import (
    discover_histwords_decades,
    histwords_unit_name_from_npy,
    load_histwords_decade,
)


def _write_histwords_pair(directory: Path, decade: int, words, vectors):
    """Write {decade}-w.npy + {decade}-vocab.pkl in HistWords layout."""
    np.save(directory / f"{decade}-w.npy", np.asarray(vectors, dtype=np.float32))
    with open(directory / f"{decade}-vocab.pkl", "wb") as f:
        pickle.dump(list(words), f)


@requires_gensim
def test_load_histwords_decade_returns_keyedvectors_with_vocab(tmp_path):
    words = ["he", "she", "doctor"]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.7, 0.0],
    ]
    _write_histwords_pair(tmp_path, 1990, words, vectors)

    kv = load_histwords_decade(tmp_path / "1990-w.npy")

    assert "he" in kv.key_to_index
    assert "doctor" in kv.key_to_index
    assert kv.vector_size == 3
    np.testing.assert_allclose(kv["he"], [1.0, 0.0, 0.0], atol=1e-6)


@requires_gensim
def test_load_histwords_decade_explicit_vocab_path(tmp_path):
    """Explicit vocab_path overrides the auto-derived sibling."""
    np.save(tmp_path / "vec.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    with open(tmp_path / "v.pkl", "wb") as f:
        pickle.dump(["only"], f)

    kv = load_histwords_decade(tmp_path / "vec.npy", vocab_path=tmp_path / "v.pkl")
    assert "only" in kv.key_to_index


@requires_gensim
def test_load_histwords_decade_raises_on_missing_vocab(tmp_path):
    np.save(tmp_path / "1990-w.npy", np.array([[1.0]], dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="vocab"):
        load_histwords_decade(tmp_path / "1990-w.npy")


@requires_gensim
def test_load_histwords_decade_raises_on_length_mismatch(tmp_path):
    np.save(tmp_path / "1990-w.npy", np.zeros((3, 5), dtype=np.float32))
    with open(tmp_path / "1990-vocab.pkl", "wb") as f:
        pickle.dump(["a", "b"], f)  # 2 words, but 3 vector rows
    with pytest.raises(ValueError, match="vocab length"):
        load_histwords_decade(tmp_path / "1990-w.npy")


def test_histwords_unit_name_parses_yyyy_w_npy():
    assert histwords_unit_name_from_npy(Path("1990-w.npy")) == "1990s"
    assert histwords_unit_name_from_npy(Path("1810-w.npy")) == "1810s"
    assert histwords_unit_name_from_npy(Path("foo.txt")) is None
    assert histwords_unit_name_from_npy(Path("notyear-w.npy")) is None


def test_discover_histwords_decades_skips_orphans(tmp_path):
    """A -w.npy without a sibling -vocab.pkl is silently skipped (we can't
    use it). Pairs are returned sorted by decade name."""
    # Three valid pairs
    for year in (1990, 1810, 1900):
        _write_histwords_pair(tmp_path, year, ["a"], [[0.0]])
    # One orphan: only -w.npy, no vocab
    np.save(tmp_path / "2000-w.npy", np.zeros((1, 1), dtype=np.float32))

    pairs = list(discover_histwords_decades(tmp_path))
    unit_names = [u for _, u in pairs]
    assert unit_names == ["1810s", "1900s", "1990s"]
    assert "2000s" not in unit_names
