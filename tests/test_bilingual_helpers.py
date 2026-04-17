"""
Tests for bilingual-refactor helper functions not covered elsewhere:

  - download_coha.download_one / decompress_one (skip-if-exists, real decompression)
  - download_ngrams._resolve_ngram_language
  - visualize.L and visualize._parse_province_year
  - analyze_correlation.normalize_province and compute_corr
"""

import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# download_coha: download_one skip-if-exists
# ---------------------------------------------------------------------------

class TestDownloadOne:
    """Unit tests for download_coha.download_one."""

    def test_skip_when_file_exists_and_nonempty(self, tmp_path):
        from scripts.data_prep.download_coha import download_one

        existing = tmp_path / "archive.zip"
        existing.write_bytes(b"some zip data")

        logger = logging.getLogger("test_download_one")
        ok, status = download_one("http://invalid.invalid/archive.zip", existing, logger)

        assert ok is True
        assert status == "skipped"

    def test_attempts_download_when_file_absent(self, tmp_path, monkeypatch):
        """When the file doesn't exist, download_one tries requests.get and fails on
        an invalid URL — we just need to verify it does NOT skip silently."""
        import requests
        from scripts.data_prep.download_coha import download_one

        out_path = tmp_path / "new.zip"
        assert not out_path.exists()

        logger = logging.getLogger("test_download_one_absent")

        # Stub out requests.get to raise immediately
        def fake_get(*args, **kwargs):
            raise requests.RequestException("network error")

        monkeypatch.setattr("scripts.data_prep.download_coha.requests.get", fake_get)
        ok, status = download_one("http://example.com/new.zip", out_path, logger)

        assert ok is False
        assert "network" in status.lower() or "error" in status.lower()


# ---------------------------------------------------------------------------
# download_coha: decompress_one skip-if-exists and real decompression
# ---------------------------------------------------------------------------

class TestDecompressOne:
    """Unit tests for download_coha.decompress_one."""

    def test_skip_when_stem_dir_exists_and_nonempty(self, tmp_path):
        from scripts.data_prep.download_coha import decompress_one

        zip_path = tmp_path / "archive.zip"
        zip_path.write_bytes(b"dummy")

        # Create the stem directory with content
        stem_dir = tmp_path / "archive"
        stem_dir.mkdir()
        (stem_dir / "data.txt").write_text("content")

        logger = logging.getLogger("test_decompress_skip")
        ok, status = decompress_one(zip_path, tmp_path, logger)

        assert ok is True
        assert status == "skipped"

    def test_real_zip_decompresses_successfully(self, tmp_path):
        from scripts.data_prep.download_coha import decompress_one

        zip_path = tmp_path / "coha1940.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("w4_1940.txt", "the quick brown fox\t50\n")

        out_dir = tmp_path / "out"
        logger = logging.getLogger("test_decompress_real")
        ok, status = decompress_one(zip_path, out_dir, logger)

        assert ok is True
        assert status == "decompressed"
        # extractall writes directly into out_dir
        assert (out_dir / "w4_1940.txt").exists()
        assert (out_dir / "w4_1940.txt").read_text(encoding="utf-8").startswith(
            "the quick brown fox"
        )

    def test_bad_zip_returns_failure(self, tmp_path):
        from scripts.data_prep.download_coha import decompress_one

        bad_zip = tmp_path / "broken.zip"
        bad_zip.write_bytes(b"not a zip at all")

        out_dir = tmp_path / "out"
        logger = logging.getLogger("test_decompress_bad")
        ok, status = decompress_one(bad_zip, out_dir, logger)

        assert ok is False


# ---------------------------------------------------------------------------
# download_ngrams: _resolve_ngram_language
# ---------------------------------------------------------------------------

class TestResolveNgramLanguage:
    """Tests for the language resolution utility in download_ngrams."""

    def test_zh_maps_to_chi_sim(self):
        from scripts.data_prep.download_ngrams import _resolve_ngram_language

        assert _resolve_ngram_language({"language": "zh"}) == "chi_sim"

    def test_en_maps_to_eng(self):
        from scripts.data_prep.download_ngrams import _resolve_ngram_language

        assert _resolve_ngram_language({"language": "en"}) == "eng"

    def test_explicit_ngram_language_overrides_config_language(self):
        from scripts.data_prep.download_ngrams import _resolve_ngram_language

        cfg = {"language": "zh", "ngram": {"language": "eng_fiction"}}
        assert _resolve_ngram_language(cfg) == "eng_fiction"

    def test_empty_ngram_block_falls_back_to_config_language(self):
        from scripts.data_prep.download_ngrams import _resolve_ngram_language

        cfg = {"language": "en", "ngram": {}}
        assert _resolve_ngram_language(cfg) == "eng"


# ---------------------------------------------------------------------------
# visualize: L() label lookup
# ---------------------------------------------------------------------------

class TestLabelLookup:
    """Tests for visualize.L()."""

    def test_zh_year_label(self):
        from scripts.visualize import L

        assert L({"language": "zh"}, "year") == "年份"

    def test_en_year_label(self):
        from scripts.visualize import L

        assert L({"language": "en"}, "year") == "Year"

    def test_zh_gender_norm_label(self):
        from scripts.visualize import L

        assert L({"language": "zh"}, "gender_norm") == "性别规范指数"

    def test_en_gender_norm_label(self):
        from scripts.visualize import L

        assert L({"language": "en"}, "gender_norm") == "Gender norm index"

    def test_unknown_key_falls_back_to_key_itself(self):
        from scripts.visualize import L

        assert L({"language": "zh"}, "nonexistent_key") == "nonexistent_key"
        assert L({"language": "en"}, "another_missing") == "another_missing"

    def test_unknown_language_falls_back_to_key(self):
        from scripts.visualize import L

        # Language 'fr' has no LABELS entry; should return the key itself
        assert L({"language": "fr"}, "year") == "year"

    def test_all_zh_labels_defined(self):
        """Every key in the zh LABELS dict should be non-empty."""
        from scripts.visualize import LABELS

        for key, val in LABELS["zh"].items():
            assert val, f"zh label for {key!r} is empty"

    def test_zh_and_en_have_same_keys(self):
        """Both language dicts should cover the same set of label keys."""
        from scripts.visualize import LABELS

        assert set(LABELS["zh"].keys()) == set(LABELS["en"].keys())


# ---------------------------------------------------------------------------
# visualize: _parse_province_year
# ---------------------------------------------------------------------------

class TestParseProvinceYear:
    """Tests for visualize._parse_province_year."""

    def test_valid_province_year(self):
        from scripts.visualize import _parse_province_year

        prov, year = _parse_province_year("北京_2020")
        assert prov == "北京"
        assert year == 2020

    def test_year_at_lower_boundary(self):
        from scripts.visualize import _parse_province_year

        prov, year = _parse_province_year("广东_1990")
        assert prov == "广东"
        assert year == 1990

    def test_year_below_lower_boundary_returns_none(self):
        from scripts.visualize import _parse_province_year

        prov, year = _parse_province_year("广东_1989")
        assert prov is None
        assert year is None

    def test_time_slice_format_returns_none(self):
        """'1940_1949' looks like a time slice, not province_year (1949 in range but
        '1940' is not a valid province name — the year check passes but this is still
        a valid tuple; just verify parse logic returns something consistent)."""
        from scripts.visualize import _parse_province_year

        # year 1949 is in range [1990,2030], so parse fails on outer check
        prov, year = _parse_province_year("1940_1949")
        # 1949 < 1990 → should return (None, None)
        assert year is None

    def test_no_underscore_returns_none(self):
        from scripts.visualize import _parse_province_year

        prov, year = _parse_province_year("nounderscorehere")
        assert prov is None and year is None

    def test_non_integer_year_returns_none(self):
        from scripts.visualize import _parse_province_year

        prov, year = _parse_province_year("北京_notayear")
        assert prov is None and year is None


# ---------------------------------------------------------------------------
# analyze_correlation: normalize_province
# ---------------------------------------------------------------------------

class TestNormalizeProvince:
    """Tests for analyze_correlation.normalize_province."""

    def test_short_name_returns_itself(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("北京") == "北京"
        assert normalize_province("广东") == "广东"

    def test_full_name_strips_suffix(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("北京市") == "北京"
        assert normalize_province("广东省") == "广东"

    def test_autonomous_region_strips_suffix(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("广西壮族自治区") == "广西"
        assert normalize_province("内蒙古自治区") == "内蒙古"
        assert normalize_province("新疆维吾尔自治区") == "新疆"

    def test_zhongguo_prefix_stripped(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("中国广东省") == "广东"
        assert normalize_province("中国北京市") == "北京"

    def test_none_input_returns_none(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province(None) is None

    def test_whitespace_only_returns_none(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("   ") is None
        assert normalize_province("") is None

    def test_nan_string_returns_none(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("nan") is None
        assert normalize_province("NaN") is None

    def test_unknown_province_returns_none(self):
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("Nonexistent Province XYZ") is None

    def test_reverse_lookup_from_full_name(self):
        """Full shapefile name (e.g. 北京市) should return short name."""
        from scripts.analyze_correlation import normalize_province

        assert normalize_province("黑龙江省") == "黑龙江"
        assert normalize_province("上海市") == "上海"


# ---------------------------------------------------------------------------
# analyze_correlation: compute_corr
# ---------------------------------------------------------------------------

class TestComputeCorr:
    """Tests for analyze_correlation.compute_corr."""

    def test_perfect_positive_correlation(self):
        from scripts.analyze_correlation import compute_corr

        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        r, p = compute_corr(x, y)
        assert abs(r - 1.0) < 1e-9
        assert p < 0.001

    def test_perfect_negative_correlation(self):
        from scripts.analyze_correlation import compute_corr

        x = pd.Series([1.0, 2.0, 3.0, 4.0])
        y = pd.Series([4.0, 3.0, 2.0, 1.0])
        r, p = compute_corr(x, y)
        assert abs(r - (-1.0)) < 1e-9

    def test_too_few_points_returns_nan(self):
        """With fewer than 3 valid pairs, returns (nan, nan)."""
        from scripts.analyze_correlation import compute_corr

        x = pd.Series([1.0, 2.0])
        y = pd.Series([2.0, 4.0])
        r, p = compute_corr(x, y)
        assert np.isnan(r)
        assert np.isnan(p)

    def test_empty_series_returns_nan(self):
        from scripts.analyze_correlation import compute_corr

        r, p = compute_corr(pd.Series([], dtype=float), pd.Series([], dtype=float))
        assert np.isnan(r)

    def test_nan_values_dropped_before_correlation(self):
        """NaN entries in either series are dropped; correlation computed on remainder."""
        from scripts.analyze_correlation import compute_corr

        x = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        r, p = compute_corr(x, y)
        # After dropping index 1, we have 4 perfect pairs
        assert abs(r - 1.0) < 1e-9
