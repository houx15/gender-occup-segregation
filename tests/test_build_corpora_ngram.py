"""Tests for the Chinese Google Ngram corpus builder — weight_mode dispatch."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scripts.data_prep.build_corpora_ngram import process_ngram_file

logger = logging.getLogger("test")


def _make_ngram_file(dir_path: Path, name: str, lines: list[str]) -> Path:
    """Write a plain-text v3 ngram file (process_ngram_file reads decompressed text)."""
    p = dir_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _line(ngram_text: str, year: int, match_count: int, volume_count: int = 1) -> str:
    """One v3 ngram line: ngram\\tyear,match,volume."""
    return f"{ngram_text}\t{year},{match_count},{volume_count}"


def _config(tmp_path: Path, **overrides) -> dict:
    """Minimal config dict for process_ngram_file."""
    corpus = {"min_count_threshold": 1}
    corpus.update(overrides.pop("corpus", {}))
    return {
        "corpus": corpus,
        "paths": {"corpora_dir": str(tmp_path / "corpora")},
        **overrides,
    }


def _read_corpus(corpora_dir: Path, slice_name: str) -> list[str]:
    """Read all corpus_*.txt lines for a slice (sorted by filename), return list of lines."""
    slice_dir = corpora_dir / slice_name
    if not slice_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(slice_dir.glob("corpus_*.txt")):
        out.extend(line for line in p.read_text(encoding="utf-8").splitlines() if line)
    return out


# ngram_text needs ≥ 2 Chinese tokens to survive clean_ngram (line 69 of build_corpora_ngram.py)
NGRAM = "中国 经济 发展 政策 改革"
NGRAM_CLEAN = "中国 经济 发展 政策 改革"  # already pure-Chinese, clean_ngram preserves


class TestPresenceModeDefault:
    def test_no_weight_mode_key_preserves_existing_behavior(self, tmp_path):
        # Two identical-ngram entries in different years that both fall in 1940_1949.
        # Presence mode = set dedup → ONE corpus line, regardless of count.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
            _line(NGRAM, 1948, match_count=200),
        ])
        cfg = _config(tmp_path)  # no weight_mode → defaults to "presence"
        time_slices = [(1940, 1949)]
        process_ngram_file(f, time_slices, cfg, logger)
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN]


class TestInvalidWeightMode:
    def test_raises_value_error(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=1),
        ])
        cfg = _config(tmp_path, corpus={"weight_mode": "bogus"})
        with pytest.raises(ValueError, match="weight_mode"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger)


class TestPerYearCapped:
    def test_pass_through_when_year_total_below_cap(self, tmp_path):
        # year_total = 5e7 (below cap 1e8) → scale = 1.0 → emit exactly match_count copies.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=7),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 50_000_000})
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN] * 7

    def test_scale_down_when_year_total_above_cap_is_deterministic(self, tmp_path):
        # year_total = 1e9, cap = 1e8 → scale = 0.1. match_count = 100 → expected = 10 (integer; no Bernoulli).
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=100),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 1_000_000_000})
        lines = _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
        assert lines == [NGRAM_CLEAN] * 10

    def test_bernoulli_fractional_part_is_unbiased(self, tmp_path):
        # match_count=15, scale=0.1 → expected = 1.5 → n_emit ∈ {1, 2} per trial.
        # Across 200 seeds, empirical mean should land within 3σ of 1.5.
        # Var(Bernoulli(0.5)) = 0.25; SE over 200 trials ≈ sqrt(0.25/200) ≈ 0.0354. 3σ ≈ 0.106.
        n_emits = []
        for seed in range(200):
            sub = tmp_path / f"trial_{seed}"
            sub.mkdir()
            f = _make_ngram_file(sub, "5-00000-of-00105", [
                _line(NGRAM, 1942, match_count=15),
            ])
            cfg = _config(sub, corpus={
                "weight_mode": "per_year_capped",
                "per_year_token_cap": 100_000_000,
                "rng_seed": seed,
            })
            process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 1_000_000_000})
            n_emits.append(len(_read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949")))
        mean = sum(n_emits) / len(n_emits)
        assert 1.39 < mean < 1.61, f"Bernoulli looks biased: mean={mean:.3f} over 200 trials"
        # Also confirm both outcomes occur — guards against a stuck RNG.
        assert set(n_emits) == {1, 2}

    def test_missing_year_in_totalcounts_raises_key_error(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=10),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        with pytest.raises(KeyError, match="1942"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1943: 1_000_000})

    def test_per_year_capped_without_year_total_raises_value_error(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=10),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        with pytest.raises(ValueError, match="year_total"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger)  # year_total omitted

    def test_presence_mode_does_not_require_year_total(self, tmp_path):
        # Regression: presence mode should keep working with no year_total argument.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
        ])
        cfg = _config(tmp_path)  # default → presence
        process_ngram_file(f, [(1940, 1949)], cfg, logger)  # no year_total kwarg
        assert _read_corpus(Path(cfg["paths"]["corpora_dir"]), "1940_1949") == [NGRAM_CLEAN]
