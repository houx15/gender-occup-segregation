import logging

import numpy as np
import pandas as pd
import pytest

from scripts.common.fixed_effects import (
    build_fe_summary, coverage_word_sets, fit_unit_effects, word_coverage_table,
)


def _long(panel, category="occ"):
    """panel: {(unit, word): value or None (None = out of vocab)}."""
    return pd.DataFrame([
        {"unit_name": u, "category": category, "occupation": w,
         "value": np.nan if v is None else v, "in_vocab": v is not None}
        for (u, w), v in panel.items()
    ])


def test_balanced_panel_recovers_unit_means():
    y = {("a", "x"): 1.0, ("a", "y"): 3.0, ("b", "x"): 0.0, ("b", "y"): 4.0}
    units, words = ["a", "b"], ["x", "y"]
    u = np.array([units.index(k[0]) for k in y]); w = np.array([words.index(k[1]) for k in y])
    a = fit_unit_effects(u, w, np.array(list(y.values())), len(units), len(words))
    assert np.allclose(a, [2.0, 2.0])


def test_unbalanced_panel_recovers_additive_unit_effects():
    # value = a_u + b_w exactly, mean(b) = 0. Unit 'b' lacks the high-b word,
    # so its naive mean is biased down; the FE estimate is not.
    a_true = {"a": 0.0, "b": 0.0}
    b_true = {"x": 1.0, "y": -0.5, "z": -0.5}
    panel = {(u, w): a_true[u] + b_true[w] for u in a_true for w in b_true}
    del panel[("b", "x")]
    units, words = ["a", "b"], ["x", "y", "z"]
    keys = list(panel)
    a = fit_unit_effects(np.array([units.index(k[0]) for k in keys]),
                         np.array([words.index(k[1]) for k in keys]),
                         np.array([panel[k] for k in keys]), 2, 3)
    assert np.allclose(a, [0.0, 0.0], atol=1e-8)
    assert np.mean([panel[("b", "y")], panel[("b", "z")]]) == pytest.approx(-0.5)


def test_coverage_word_sets_and_table():
    long_df = _long({("a", "x"): 1.0, ("b", "x"): 1.0, ("c", "x"): 1.0,
                     ("a", "y"): 1.0, ("b", "y"): None, ("c", "y"): None})
    sets = coverage_word_sets(long_df, ["a", "b", "c"], min_coverage=0.5)
    assert sets == {"occ": ["x"]}
    table = word_coverage_table(long_df, ["a", "b", "c"], sets).set_index("occupation")
    assert table.loc["x", "coverage"] == 1.0 and bool(table.loc["x", "used"])
    assert table.loc["y", "coverage"] == pytest.approx(1 / 3) and not table.loc["y", "used"]


def test_build_fe_summary_shape_and_bands():
    rng = np.random.default_rng(0)
    units = [f"s{i}_2005" for i in range(6)]
    words = [f"w{j}" for j in range(12)]
    a_true = dict(zip(units, np.linspace(-0.05, 0.05, len(units))))
    b_true = dict(zip(words, rng.normal(0, 0.1, len(words))))
    panel = {(u, w): a_true[u] + b_true[w] + rng.normal(0, 0.005)
             for u in units for w in words if rng.random() < 0.8}
    long_df = _long(panel)
    sets = coverage_word_sets(long_df, units, min_coverage=0.5)
    out = build_fe_summary(long_df, units, sets, logging.getLogger("t"),
                           boot_n_iter=200, boot_ci=0.68, sub_fraction=0.8,
                           sub_rounds=50, sub_ci=0.95, seed=1, legacy_rnd_aliases=True)
    assert len(out) == len(units)
    for col in ("mean_value", "mean_ci_low", "mean_ci_high", "mean_sub_low", "prop_male",
                "prop_ci_low", "n_occupations", "n_consistent", "mean_rnd", "ci_low"):
        assert col in out.columns, col
    assert (out["mean_ci_low"] <= out["mean_value"]).all()
    assert (out["mean_value"] <= out["mean_ci_high"]).all()
    # unit effects recovered up to the common constant (mean of b over words)
    est = out.set_index("unit_name")["mean_value"]
    true = pd.Series(a_true) + np.mean(list(b_true.values()))
    assert np.allclose(est[units], true[units], atol=0.01)
    assert (out["n_consistent"] == len(sets["occ"])).all()
