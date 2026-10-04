import numpy as np
import pandas as pd
import pytest

from scripts.check_occupation_census import (
    census_correlations, occupation_period_scores, state_correlations, state_period_table,
)


def test_occupation_scores_use_only_used_words():
    long_df = pd.DataFrame([
        {"unit_name": "ohio_2005", "category": "occupation", "occupation": "nurse", "rnd": 0.02, "in_vocab": True},
        {"unit_name": "utah_2005", "category": "occupation", "occupation": "nurse", "rnd": 0.04, "in_vocab": True},
        {"unit_name": "ohio_2005", "category": "occupation", "occupation": "firefighter", "rnd": -0.05, "in_vocab": True},
        {"unit_name": "ohio_2005", "category": "occupation", "occupation": "weaver", "rnd": 0.5, "in_vocab": True},
        {"unit_name": "utah_2005", "category": "occupation", "occupation": "firefighter", "rnd": np.nan, "in_vocab": False},
        {"unit_name": "ohio_2005", "category": "family_sphere", "occupation": "home", "rnd": 0.1, "in_vocab": True},
    ])
    out = occupation_period_scores(long_df, {"nurse", "firefighter"}).set_index("occupation")
    assert set(out.index) == {"nurse", "firefighter"}
    assert out.loc["nurse", "rnd"] == pytest.approx(0.03)
    assert out.loc["nurse", "n_units"] == 2
    assert out.loc["nurse", "period"] == 2005


def test_census_correlations_per_period():
    t = pd.DataFrame({"period": [2005] * 4, "rnd": [0.01, 0.02, 0.03, 0.04],
                      "female_share": [0.1, 0.3, 0.5, 0.9]})
    c = census_correlations(t).set_index("period")
    assert c.loc[2005, "n_occupations"] == 4
    assert c.loc[2005, "pearson_r"] > 0.9
    assert c.loc[2005, "spearman_r"] == pytest.approx(1.0)


def _state_inputs():
    summary = pd.DataFrame([
        {"unit_name": "ohio_2005", "category": "occupation", "mean_value": -0.02},
        {"unit_name": "ohio_2010", "category": "occupation", "mean_value": -0.01},
        {"unit_name": "utah_2005", "category": "occupation", "mean_value": -0.03},
        {"unit_name": "ohio_2005", "category": "family_sphere", "mean_value": 9.0},
    ])
    shares = pd.DataFrame([
        {"word": "nurse", "state": "Ohio", "period": 2005, "female_share": 0.9},
        {"word": "engineer", "state": "Ohio", "period": 2005, "female_share": 0.1},
        {"word": "teller", "state": "Ohio", "period": 2005, "female_share": 0.8},   # not used
        {"word": "nurse", "state": "Ohio", "period": 2010, "female_share": 0.92},
        {"word": "engineer", "state": "Ohio", "period": 2010, "female_share": 0.14},
        {"word": "nurse", "state": "Utah", "period": 2005, "female_share": 0.88},
        {"word": "engineer", "state": "Utah", "period": 2005, "female_share": 0.06},
    ])
    labor = pd.DataFrame([
        {"state": "Ohio", "period": 2005, "duncan": 0.5, "female_emp_share": 0.47},
        {"state": "Ohio", "period": 2010, "duncan": 0.48, "female_emp_share": 0.48},
        {"state": "Utah", "period": 2005, "duncan": 0.55, "female_emp_share": 0.44},
    ])
    return summary, shares, labor


def test_state_period_table_matches_units_and_averages_used_words():
    summary, shares, labor = _state_inputs()
    t = state_period_table(summary, shares, labor, {"nurse", "engineer"}).set_index("unit_name")
    assert set(t.index) == {"ohio_2005", "ohio_2010", "utah_2005"}
    assert t.loc["ohio_2005", "our_score"] == pytest.approx(-0.02)
    assert t.loc["ohio_2005", "matched_female_share"] == pytest.approx(0.5)  # teller excluded
    assert t.loc["utah_2005", "duncan"] == pytest.approx(0.55)
    assert t.loc["ohio_2010", "period"] == 2010


def test_state_correlations_pooled_period_and_change():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(10):
        base = rng.normal()
        for p, shift in ((2005, 0.0), (2020, 0.5)):
            rows.append({"unit_name": f"s{i}_{p}", "state": f"s{i}", "period": p,
                         "our_score": base + shift + rng.normal(0, 0.01),
                         "matched_female_share": base + shift})
    c = state_correlations(pd.DataFrame(rows), ["matched_female_share"])
    scopes = set(c["scope"])
    assert {"pooled", "period 2005", "period 2020", "change 2005->2020"} <= scopes
    pooled = c[c["scope"] == "pooled"].iloc[0]
    assert pooled["n"] == 20 and pooled["pearson_r"] > 0.99
