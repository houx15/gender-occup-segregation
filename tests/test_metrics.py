"""Tests for scripts.common.metrics (Garg 2018 RND + bootstrap CI)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.common.metrics import bootstrap_ci, relative_norm_distance


class TestRelativeNormDistance:
    def test_hand_computed(self):
        v_w = np.array([1.0, 0.0, 0.0])
        c_male = np.array([2.0, 0.0, 0.0])
        c_female = np.array([0.0, 1.0, 0.0])
        # ||v_w - c_female|| - ||v_w - c_male|| = sqrt(2) - 1
        assert relative_norm_distance(v_w, c_male, c_female) == pytest.approx(
            np.sqrt(2) - 1.0
        )

    def test_sign_male_leaning(self):
        # v_w sits right on c_male; positive (closer to male)
        v_w = np.array([1.0, 0.0])
        c_male = np.array([1.0, 0.0])
        c_female = np.array([-1.0, 0.0])
        assert relative_norm_distance(v_w, c_male, c_female) > 0

    def test_sign_female_leaning(self):
        # v_w sits right on c_female; negative (closer to female)
        v_w = np.array([-1.0, 0.0])
        c_male = np.array([1.0, 0.0])
        c_female = np.array([-1.0, 0.0])
        assert relative_norm_distance(v_w, c_male, c_female) < 0

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
