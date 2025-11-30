#!/usr/bin/env python3
"""
Unit tests for build_corpora.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from build_corpora import generate_time_slices, parse_ngram_line


class TestBuildCorpora(unittest.TestCase):
    """Test cases for corpus building functionality."""

    def test_generate_time_slices(self):
        """Test time slice generation."""
        # Test basic case
        slices = generate_time_slices(1940, 2015, 10, 5)
        self.assertGreater(len(slices), 0)

        # First slice should start at 1940
        self.assertEqual(slices[0][0], 1940)

        # Check window size
        self.assertEqual(slices[0][1] - slices[0][0] + 1, 10)

        # Check step size
        if len(slices) > 1:
            self.assertEqual(slices[1][0] - slices[0][0], 5)

    def test_generate_time_slices_exact(self):
        """Test specific time slice outputs."""
        slices = generate_time_slices(1940, 1960, 10, 5)

        # Should have: 1940-1949, 1945-1954, 1950-1959, 1955-1960
        expected = [
            (1940, 1949),
            (1945, 1954),
            (1950, 1959),
            (1955, 1960)
        ]

        self.assertEqual(slices, expected)

    def test_parse_ngram_line_valid(self):
        """Test parsing a valid n-gram line."""
        line = "你 好 世 界 啊\t1950\t100\t10"
        ngram, year, count = parse_ngram_line(line, "\t", 1, 2)

        self.assertEqual(ngram, "你 好 世 界 啊")
        self.assertEqual(year, 1950)
        self.assertEqual(count, 100)

    def test_parse_ngram_line_invalid(self):
        """Test parsing invalid lines."""
        # Malformed line
        line = "invalid line without proper format"
        ngram, year, count = parse_ngram_line(line, "\t", 1, 2)

        self.assertIsNone(ngram)
        self.assertIsNone(year)
        self.assertIsNone(count)

    def test_parse_ngram_line_missing_fields(self):
        """Test parsing line with missing fields."""
        line = "你 好\t1950"  # Missing count field
        ngram, year, count = parse_ngram_line(line, "\t", 1, 2)

        self.assertIsNone(ngram)


class TestCorpusIntegration(unittest.TestCase):
    """Integration tests for corpus building."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_corpus_file_creation(self):
        """Test that corpus files are created with correct format."""
        # Create a small test n-gram file
        ngram_file = self.test_path / "test_ngrams.txt"
        with open(ngram_file, 'w', encoding='utf-8') as f:
            f.write("这 是 一 个 测试\t1945\t10\t5\n")
            f.write("另 一 个 测试 句\t1948\t5\t3\n")
            f.write("超 出 范围 的 句\t1960\t3\t2\n")

        # Parse and filter for 1940-1949 slice
        corpus_file = self.test_path / "corpus.txt"

        with open(corpus_file, 'w', encoding='utf-8') as out_f:
            with open(ngram_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    ngram, year, count = parse_ngram_line(line, "\t", 1, 2)
                    if ngram and year and 1940 <= year <= 1949:
                        out_f.write(ngram + '\n')

        # Verify output
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].strip(), "这 是 一 个 测试")
        self.assertEqual(lines[1].strip(), "另 一 个 测试 句")


if __name__ == '__main__':
    unittest.main()
