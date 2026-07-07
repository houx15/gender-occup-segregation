"""Tests for the Chinese Google Ngram corpus builder — weight_mode dispatch."""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import pytest

from scripts.data_prep.build_corpora_ngram import (
    accumulate_year_total,
    build_corpora,
    corpus_signature,
    process_ngram_file,
    resolve_specific_slice,
)
from scripts.data_prep.ngram_totalcounts import load_totalcounts

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


def _read_compact(corpora_dir: Path, slice_name: str) -> list[tuple[str, int]]:
    """Parse a per_year_capped COMPACT corpus into (ngram, count) pairs.

    Capped mode stores one line per unique ngram per slice as ``ngram<TAB>count``
    (count = Σ_year n_emit) instead of ``count`` repeated physical lines."""
    out: list[tuple[str, int]] = []
    for line in _read_corpus(corpora_dir, slice_name):
        ngram, _, count = line.partition("\t")
        out.append((ngram, int(count)))
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


TIME_SLICES = [
    (1940, 1949), (1945, 1954), (1950, 1959), (1955, 1964),
    (1960, 1969), (1965, 1974), (1970, 1979), (1975, 1984),
    (1980, 1989), (1985, 1994), (1990, 1999), (1995, 2004),
    (2000, 2009), (2005, 2014), (2010, 2019), (2015, 2020),
    (2020, 2020),
]


class TestResolveSpecificSlice:
    """Fire coerces ``--slice=1940_1949`` → int 19401949 (PEP 515 underscores
    in int literals). The SLURM driver works around this by passing start
    years (int 1940). resolve_specific_slice maps either form to the
    canonical (start, end) pair from the profile's time_slices."""

    def test_int_start_year_resolves_to_canonical_slice(self):
        assert resolve_specific_slice(1940, TIME_SLICES) == [(1940, 1949)]

    def test_string_year_range_resolves_to_canonical_slice(self):
        assert resolve_specific_slice("1940_1949", TIME_SLICES) == [(1940, 1949)]

    def test_string_start_year_only_also_works(self):
        assert resolve_specific_slice("1940", TIME_SLICES) == [(1940, 1949)]

    def test_clamped_end_slice_2020_resolves(self):
        # Last slice is (2020, 2020) — clamped at end_year, not full window.
        assert resolve_specific_slice(2020, TIME_SLICES) == [(2020, 2020)]

    def test_unknown_start_year_raises_value_error_with_valid_years(self):
        with pytest.raises(ValueError, match="Valid start years"):
            resolve_specific_slice(1941, TIME_SLICES)

    def test_pep515_coerced_full_int_raises_value_error(self):
        # Regression: --slice=1940_1949 coerced by Fire to int 19401949
        # would silently look like start_year=19401949 — must raise.
        with pytest.raises(ValueError, match="Valid start years"):
            resolve_specific_slice(19401949, TIME_SLICES)

    def test_non_int_non_str_raises_type_error(self):
        with pytest.raises(TypeError, match="must be int or str"):
            resolve_specific_slice(3.14, TIME_SLICES)


class TestPerYearCapped:
    def test_pass_through_when_year_total_below_cap(self, tmp_path):
        # year_total = 5e7 (below cap 1e8) → scale = 1.0 → count = match_count.
        # Stored COMPACT: one line "ngram<TAB>7", not 7 physical lines.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=7),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 50_000_000})
        assert _read_compact(Path(cfg["paths"]["corpora_dir"]), "1940_1949") == [(NGRAM_CLEAN, 7)]

    def test_scale_down_when_year_total_above_cap_is_deterministic(self, tmp_path):
        # year_total = 1e9, cap = 1e8 → scale = 0.1. match_count = 100 → count = 10 (integer; no Bernoulli).
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=100),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 1_000_000_000})
        assert _read_compact(Path(cfg["paths"]["corpora_dir"]), "1940_1949") == [(NGRAM_CLEAN, 10)]

    def test_capped_sums_n_emit_across_years_within_slice(self, tmp_path):
        # One v3 line carries multiple year cells for the same ngram; two of them
        # fall inside (1940,1949), both pass-through (year_total < cap → scale 1.0).
        # Compact stores ONE line with the summed count (5 + 3 = 8).
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            f"{NGRAM}\t1942,5,1\t1948,3,1",
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        process_ngram_file(f, [(1940, 1949)], cfg, logger,
                           year_total={1942: 50_000_000, 1948: 50_000_000})
        assert _read_compact(Path(cfg["paths"]["corpora_dir"]), "1940_1949") == [(NGRAM_CLEAN, 8)]

    def test_bernoulli_fractional_part_is_unbiased(self, tmp_path):
        # match_count=15, scale=0.1 → expected = 1.5 → count ∈ {1, 2} per trial.
        # Across 200 seeds, empirical mean should land within 3σ of 1.5.
        # Var(Bernoulli(0.5)) = 0.25; SE over 200 trials ≈ sqrt(0.25/200) ≈ 0.0354. 3σ ≈ 0.106.
        counts = []
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
            compact = _read_compact(Path(cfg["paths"]["corpora_dir"]), "1940_1949")
            assert len(compact) == 1                # always a single compact line
            counts.append(compact[0][1])
        mean = sum(counts) / len(counts)
        assert 1.39 < mean < 1.61, f"Bernoulli looks biased: mean={mean:.3f} over 200 trials"
        # Also confirm both outcomes occur — guards against a stuck RNG.
        assert set(counts) == {1, 2}

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

    def test_year_total_zero_raises_value_error_with_year_context(self, tmp_path):
        # Zero-year would otherwise blow up as ZeroDivisionError with no context.
        # We want a clean ValueError that names the year so operators can find
        # the bad entry in totalcounts-5.
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=10),
        ])
        cfg = _config(tmp_path, corpus={
            "weight_mode": "per_year_capped",
            "per_year_token_cap": 100_000_000,
            "rng_seed": 42,
        })
        with pytest.raises(ValueError, match="1942"):
            process_ngram_file(f, [(1940, 1949)], cfg, logger, year_total={1942: 0})

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


def _build_config(tmp_path: Path, **corpus) -> dict:
    """Full config for build_corpora (needs time_slices + decompressed/raw paths)."""
    c = {"min_count_threshold": 1}
    c.update(corpus)
    return {
        "time_slices": {
            "start_year": 1940, "end_year": 1949,
            "window_size": 10, "step_size": 10,  # → single slice (1940, 1949)
        },
        "corpus": c,
        "paths": {
            "corpora_dir": str(tmp_path / "corpora"),
            "decompressed_dir": str(tmp_path / "decomp"),
            "raw_ngram_dir": str(tmp_path / "raw"),
        },
    }


def _stage_shard(decomp_dir: Path, name: str, lines: list[str]) -> str:
    """Drop a plain-text shard into decompressed_dir; return its file_name."""
    decomp_dir.mkdir(parents=True, exist_ok=True)
    _make_ngram_file(decomp_dir, name, lines)
    return name


def _make_gz_shard(raw_dir: Path, name: str, lines: list[str]) -> Path:
    """Write a gzipped ngram shard into raw_ngram_dir (what build_corpora globs)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    p = raw_dir / name
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


class TestDataDerivedTotalcounts:
    """The denominator must come from the data (Σ kept match_count per year),
    not the shipped totalcounts-5 which under-counts and let the cap blow past
    10·cap by ~6.7×."""

    def test_accumulate_sums_match_count_per_year_over_kept_ngrams(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
            _line(NGRAM, 1942, match_count=3),     # same year → sums
            _line(NGRAM, 1948, match_count=200),
            "abc def\t1942,9999,1",                # non-Chinese → clean_ngram drops it
        ])
        yt: dict = {}
        accumulate_year_total(f, min_count=1, year_total=yt)
        assert yt == {1942: 8, 1948: 200}

    def test_accumulate_applies_min_count(self, tmp_path):
        f = _make_ngram_file(tmp_path, "5-00000-of-00105", [
            _line(NGRAM, 1942, match_count=5),
            _line(NGRAM, 1943, match_count=1),
        ])
        yt: dict = {}
        accumulate_year_total(f, min_count=5, year_total=yt)
        assert yt == {1942: 5}                      # 1943 mc=1 < 5 dropped

    def test_build_corpora_derives_totalcounts_and_cap_bites(self, tmp_path):
        # Regression for the 6.7× over-emission bug. With the data-derived
        # denominator, a year emits min(Σ_kept, cap), not the full match_count.
        cfg = _build_config(
            tmp_path, weight_mode="per_year_capped",
            per_year_token_cap=10, rng_seed=42,
        )
        raw = Path(cfg["paths"]["raw_ngram_dir"])
        _make_gz_shard(raw, "5-00000-of-00105.gz", [
            _line(NGRAM, 1942, match_count=100),
        ])
        build_corpora(cfg, logger)                  # no file_name → globs raw shards
        corpora_dir = Path(cfg["paths"]["corpora_dir"])
        # year_total[1942]=100, scale=10/100=0.1, expected=10 → exactly the cap.
        # Stored compact: one line "ngram<TAB>10".
        assert _read_compact(corpora_dir, "1940_1949") == [(NGRAM_CLEAN, 10)]
        # Derived totalcounts cached with the TRUE per-year sum (100, not ~15).
        derived = raw / "totalcounts-5.derived"
        assert derived.exists()
        assert load_totalcounts(derived)[1942] == 100


class TestShardResume:
    """build_corpora must reuse finished shards and rebuild unfinished ones,
    instead of blindly appending (the duplication bug)."""

    SHARD = "5-00000-of-00105"
    SLICE = "1940_1949"
    INDEX = "00000"

    def test_finished_shard_is_reused_not_duplicated_on_rerun(self, tmp_path):
        cfg = _build_config(tmp_path)
        decomp = Path(cfg["paths"]["decompressed_dir"])
        _stage_shard(decomp, self.SHARD, [_line(NGRAM, 1942, match_count=5)])

        # First build → one line + a completion marker.
        build_corpora(cfg, logger, file_name=self.SHARD)
        corpora_dir = Path(cfg["paths"]["corpora_dir"])
        assert _read_corpus(corpora_dir, self.SLICE) == [NGRAM_CLEAN]
        marker = corpora_dir / self.SLICE / f".corpus_{self.INDEX}.done"
        assert marker.exists()

        # Second build of the same shard → reused, NOT appended (no duplication).
        build_corpora(cfg, logger, file_name=self.SHARD)
        assert _read_corpus(corpora_dir, self.SLICE) == [NGRAM_CLEAN]

    def test_unverified_partial_corpus_is_deleted_and_regenerated(self, tmp_path):
        cfg = _build_config(tmp_path)
        corpora_dir = Path(cfg["paths"]["corpora_dir"])
        # Simulate a crashed run: a corpus file exists but no marker.
        (corpora_dir / self.SLICE).mkdir(parents=True)
        (corpora_dir / self.SLICE / f"corpus_{self.INDEX}.txt").write_text(
            "STALE 残留 数据\n", encoding="utf-8"
        )
        decomp = Path(cfg["paths"]["decompressed_dir"])
        _stage_shard(decomp, self.SHARD, [_line(NGRAM, 1942, match_count=5)])

        build_corpora(cfg, logger, file_name=self.SHARD)
        # Stale line gone, not appended after it.
        assert _read_corpus(corpora_dir, self.SLICE) == [NGRAM_CLEAN]

    def test_marker_from_different_config_is_not_trusted(self, tmp_path):
        # A marker written under presence mode must NOT cause reuse after
        # switching to per_year_capped — the corpus content would differ.
        presence_cfg = _build_config(tmp_path)
        raw = Path(presence_cfg["paths"]["raw_ngram_dir"])
        _make_gz_shard(raw, "5-00000-of-00105.gz", [_line(NGRAM, 1942, match_count=3)])
        build_corpora(presence_cfg, logger)            # glob mode
        corpora_dir = Path(presence_cfg["paths"]["corpora_dir"])
        assert _read_corpus(corpora_dir, self.SLICE) == [NGRAM_CLEAN]  # dedup → 1 line

        # Switch to per_year_capped. Derived year_total[1942]=3, cap huge →
        # scale 1.0 → emit match_count (3) copies.
        capped_cfg = _build_config(
            tmp_path,
            weight_mode="per_year_capped",
            per_year_token_cap=100_000_000,
            rng_seed=42,
        )
        build_corpora(capped_cfg, logger)
        # If the stale presence marker had been trusted, we'd still see the bare
        # presence line. Capped rebuild → compact "ngram<TAB>3".
        assert _read_compact(corpora_dir, self.SLICE) == [(NGRAM_CLEAN, 3)]

    def test_done_slice_reused_while_other_slice_rebuilt_same_shard(self, tmp_path):
        # Two slices; one already finished for this shard (valid marker + corpus),
        # the other pending. The done slice must be left untouched (reused), the
        # pending one built from the shard.
        cfg = _build_config(tmp_path)
        cfg["time_slices"]["end_year"] = 1959  # → (1940,1949) and (1950,1959)
        corpora_dir = Path(cfg["paths"]["corpora_dir"])
        sig = corpus_signature(cfg)

        # Pre-existing GOOD build for 1940_1949: corpus + trusted marker.
        done_dir = corpora_dir / "1940_1949"
        done_dir.mkdir(parents=True)
        (done_dir / f"corpus_{self.INDEX}.txt").write_text(
            "保留 内容 不变\n", encoding="utf-8"
        )
        (done_dir / f".corpus_{self.INDEX}.done").write_text(sig, encoding="utf-8")

        decomp = Path(cfg["paths"]["decompressed_dir"])
        _stage_shard(decomp, self.SHARD, [
            _line(NGRAM, 1942, match_count=5),   # falls in the DONE slice
            _line(NGRAM, 1955, match_count=5),   # falls in the PENDING slice
        ])

        build_corpora(cfg, logger, file_name=self.SHARD)

        # Done slice untouched (its 1942 line from the shard was never processed).
        assert _read_corpus(corpora_dir, "1940_1949") == ["保留 内容 不变"]
        # Pending slice built from the shard's 1955 entry.
        assert _read_corpus(corpora_dir, "1950_1959") == [NGRAM_CLEAN]


class TestCorpusSignatureFormat:
    """The signature must encode the on-disk FORMAT so a marker written by the
    old EXPANDED capped build (repeated bare-ngram lines) is not trusted by the
    new COMPACT build — otherwise an 85G expanded slice would be silently reused
    instead of rebuilt as ngram<TAB>count."""

    def test_capped_signature_tags_compact_format(self, tmp_path):
        cfg = _build_config(
            tmp_path, weight_mode="per_year_capped",
            per_year_token_cap=100_000_000, rng_seed=42,
        )
        assert "fmt=compact" in corpus_signature(cfg)

    def test_presence_signature_has_no_format_tag(self, tmp_path):
        # Presence corpora were never expanded; their format is unchanged.
        cfg = _build_config(tmp_path)  # presence
        assert "fmt=" not in corpus_signature(cfg)
