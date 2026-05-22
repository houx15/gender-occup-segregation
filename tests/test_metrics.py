"""Tests for scripts.common.metrics (Garg 2018 RND + bootstrap CI)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.common.metrics import (
    bootstrap_ci, l2_normalize, relative_norm_distance, proportion_below,
)


class TestRelativeNormDistance:
    def test_hand_computed(self):
        # Garg sign convention: ||v_w - c_male|| - ||v_w - c_female||
        v_w = np.array([1.0, 0.0, 0.0])
        c_male = np.array([2.0, 0.0, 0.0])
        c_female = np.array([0.0, 1.0, 0.0])
        # ||v_w - c_male|| - ||v_w - c_female|| = 1 - sqrt(2)
        assert relative_norm_distance(v_w, c_male, c_female) == pytest.approx(
            1.0 - np.sqrt(2)
        )

    def test_sign_male_leaning(self):
        # v_w sits right on c_male; Garg convention → negative (male-leaning)
        v_w = np.array([1.0, 0.0])
        c_male = np.array([1.0, 0.0])
        c_female = np.array([-1.0, 0.0])
        assert relative_norm_distance(v_w, c_male, c_female) < 0

    def test_sign_female_leaning(self):
        # v_w sits right on c_female; Garg convention → positive (female-leaning)
        v_w = np.array([-1.0, 0.0])
        c_male = np.array([1.0, 0.0])
        c_female = np.array([-1.0, 0.0])
        assert relative_norm_distance(v_w, c_male, c_female) > 0

    def test_symmetry_swap_flips_sign(self):
        rng = np.random.default_rng(0)
        v_w = rng.standard_normal(50)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        forward = relative_norm_distance(v_w, a, b)
        reversed_ = relative_norm_distance(v_w, b, a)
        assert forward == pytest.approx(-reversed_)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            relative_norm_distance(np.array([]), np.array([]), np.array([]))

    def test_shape_mismatch_raises(self):
        v_w = np.array([1.0, 0.0, 0.0])  # 3-D
        c_male = np.array([1.0, 0.0, 0.0, 0.0])  # 4-D
        c_female = np.array([0.0, 1.0, 0.0])  # 3-D
        with pytest.raises(ValueError):
            relative_norm_distance(v_w, c_male, c_female)


class TestL2Normalize:
    def test_unit_length_after_normalize(self):
        v = np.array([3.0, 4.0])  # norm = 5
        u = l2_normalize(v)
        assert np.linalg.norm(u) == pytest.approx(1.0)
        assert u == pytest.approx(np.array([0.6, 0.8]))

    def test_zero_vector_passes_through(self):
        v = np.zeros(5)
        u = l2_normalize(v)
        # Zero vectors return unchanged (no division by zero)
        assert np.array_equal(u, v)

    def test_already_unit_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        u = l2_normalize(v)
        assert u == pytest.approx(v)


class TestBootstrapCI:
    def test_determinism(self):
        rng = np.random.default_rng(123)
        values = rng.standard_normal(200)
        ci_a = bootstrap_ci(values, n_iter=500, ci=0.95, seed=42)
        ci_b = bootstrap_ci(values, n_iter=500, ci=0.95, seed=42)
        assert ci_a == ci_b

    def test_lo_le_hi(self):
        rng = np.random.default_rng(7)
        values = rng.standard_normal(100)
        lo, hi = bootstrap_ci(values, n_iter=500, ci=0.95, seed=42)
        assert lo <= hi

    def test_tighter_with_more_iterations(self):
        rng = np.random.default_rng(0)
        values = np.ones(100) + 0.01 * rng.standard_normal(100)
        lo_low, hi_low = bootstrap_ci(values, n_iter=100, ci=0.95, seed=42)
        lo_high, hi_high = bootstrap_ci(values, n_iter=10000, ci=0.95, seed=42)
        width_low = hi_low - lo_low
        width_high = hi_high - lo_high
        # More iterations should not be substantially wider than fewer.
        assert width_high <= width_low * 1.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]), n_iter=100, ci=0.95, seed=42)


class TestProportionBelow:
    def test_counts_strictly_below_threshold(self):
        vals = np.array([-2.0, -0.5, 0.0, 1.0])
        # < 0 → two of four
        assert proportion_below(vals, 0.0) == pytest.approx(0.5)

    def test_all_below(self):
        assert proportion_below(np.array([-1.0, -2.0]), 0.0) == pytest.approx(1.0)

    def test_none_below(self):
        assert proportion_below(np.array([1.0, 2.0]), 0.0) == pytest.approx(0.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            proportion_below(np.array([]), 0.0)


class TestBootstrapCIStatistic:
    def test_default_statistic_is_mean(self):
        rng = np.random.default_rng(5)
        values = rng.standard_normal(100)
        a = bootstrap_ci(values, n_iter=300, ci=0.9, seed=11)
        b = bootstrap_ci(values, n_iter=300, ci=0.9, seed=11, statistic=np.mean)
        assert a == b

    def test_proportion_statistic_in_unit_interval(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal(200)  # ~half below 0
        lo, hi = bootstrap_ci(
            values, n_iter=500, ci=0.95, seed=42,
            statistic=lambda x: proportion_below(x, 0.0),
        )
        assert 0.0 <= lo <= hi <= 1.0
        assert 0.3 < (lo + hi) / 2 < 0.7
