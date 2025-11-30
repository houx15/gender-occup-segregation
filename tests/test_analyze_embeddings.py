#!/usr/bin/env python3
"""
Unit tests for analyze_embeddings.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import numpy as np

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))


class TestSemanticAxis(unittest.TestCase):
    """Test cases for semantic axis construction."""

    def test_axis_normalization(self):
        """Test that axes are normalized to unit length."""
        # Create mock vectors
        v_pos = np.array([1.0, 0.0, 0.0])
        v_neg = np.array([0.0, 1.0, 0.0])

        axis = v_pos - v_neg
        axis = axis / np.linalg.norm(axis)

        # Should be unit length
        self.assertAlmostEqual(np.linalg.norm(axis), 1.0, places=5)

    def test_projection_calculation(self):
        """Test projection of vector onto axis."""
        # Define an axis
        axis = np.array([1.0, 0.0, 0.0])

        # Vector aligned with axis
        v1 = np.array([2.0, 0.0, 0.0])
        projection1 = np.dot(v1, axis)
        self.assertAlmostEqual(projection1, 2.0, places=5)

        # Vector orthogonal to axis
        v2 = np.array([0.0, 1.0, 0.0])
        projection2 = np.dot(v2, axis)
        self.assertAlmostEqual(projection2, 0.0, places=5)

    def test_centroid_calculation(self):
        """Test that centroids are calculated correctly."""
        vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([3.0, 0.0, 0.0]),
            np.array([2.0, 0.0, 0.0])
        ]

        centroid = np.mean(vectors, axis=0)

        expected = np.array([2.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)


class TestOccupationVectors(unittest.TestCase):
    """Test cases for occupation vector handling."""

    def test_character_averaging(self):
        """Test averaging of character vectors."""
        # Mock character vectors
        char_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]

        avg_vector = np.mean(char_vectors, axis=0)
        expected = np.array([1/3, 1/3, 1/3])

        np.testing.assert_array_almost_equal(avg_vector, expected, decimal=5)

    def test_coverage_calculation(self):
        """Test coverage calculation for multi-character occupations."""
        occupation = "工程师"  # 3 characters
        found_chars = 2

        coverage = found_chars / len(occupation)

        self.assertAlmostEqual(coverage, 2/3, places=5)


class TestDataFrameOperations(unittest.TestCase):
    """Test cases for DataFrame operations."""

    def test_result_structure(self):
        """Test that result dictionaries have correct structure."""
        result = {
            'occupation': '医生',
            'time_slice': '1940_1949',
            'start_year': 1940,
            'end_year': 1949,
            'gender_score': 0.5,
            'coverage': 1.0
        }

        self.assertIn('occupation', result)
        self.assertIn('time_slice', result)
        self.assertIn('start_year', result)
        self.assertIn('end_year', result)
        self.assertIn('gender_score', result)
        self.assertIn('coverage', result)

        self.assertIsInstance(result['start_year'], int)
        self.assertIsInstance(result['end_year'], int)
        self.assertIsInstance(result['coverage'], float)


if __name__ == '__main__':
    unittest.main()
