import numpy as np
import pandas as pd
import pytest

from scripts.check_occupation_census import (
    census_correlations, occupation_period_scores, period_female_share,
)


def _garg():
    return pd.DataFrame([
        {"Census year": 2005, "Occupation": "nurse", "Female": 0.90},
        {"Census year": 2007, "Occupation": "nurse", "Female": 0.92},
        {"Census year": 2015, "Occupation": "nurse", "Female": 0.89},
        {"Census year": 2005, "Occupation": "fireperson", "Female": 0.04},
        {"Census year": 2015, "Occupation": "fireperson", "Female": 0.06},
    ])


def test_period_share_averages_census_years_in_the_period():
    s = period_female_share(_garg(), period=2005, width=5).set_index("garg_word")
    assert s.loc["nurse", "female_share"] == pytest.approx(0.91)
    assert s.loc["nurse", "share_years"] == "2005-2007"


def test_period_share_falls_back_to_latest_earlier_year():
    # 2020-24 has no census rows in Garg's file -> use the latest year (2015), flagged.
    s = period_female_share(_garg(), period=2020, width=5).set_index("garg_word")
    assert s.loc["fireperson", "female_share"] == pytest.approx(0.06)
    assert s.loc["fireperson", "share_years"] == "2015 (latest)"


def test_occupation_scores_join_grounding_and_use_only_used_words():
    long_df = pd.DataFrame([
        {"unit_name": "ohio_2005", "category": "occupation", "occupation": "nurse", "rnd": 0.02, "in_vocab": True},
        {"unit_name": "utah_2005", "category": "occupation", "occupation": "nurse", "rnd": 0.04, "in_vocab": True},
        {"unit_name": "ohio_2005", "category": "occupation", "occupation": "firefighter", "rnd": -0.05, "in_vocab": True},
        {"unit_name": "ohio_2005", "category": "occupation", "occupation": "weaver", "rnd": 0.5, "in_vocab": True},
        {"unit_name": "utah_2005", "category": "occupation", "occupation": "firefighter", "rnd": np.nan, "in_vocab": False},
    ])
    grounding = pd.DataFrame({"word": ["nurse", "firefighter", "weaver"],
                              "garg_word": ["nurse", "fireperson", "weaver"]})
    used = {"nurse", "firefighter"}
    out = occupation_period_scores(long_df, grounding, used).set_index("occupation")
    assert set(out.index) == {"nurse", "firefighter"}
    assert out.loc["nurse", "rnd"] == pytest.approx(0.03)
    assert out.loc["nurse", "n_units"] == 2
    assert out.loc["firefighter", "garg_word"] == "fireperson"
    assert out.loc["nurse", "period"] == 2005


def test_census_correlations_per_period():
    t = pd.DataFrame({
        "period": [2005] * 4,
        "rnd": [0.01, 0.02, 0.03, 0.04],
        "female_share": [0.1, 0.3, 0.5, 0.9],
    })
    c = census_correlations(t).set_index("period")
    assert c.loc[2005, "n_occupations"] == 4
    assert c.loc[2005, "pearson_r"] > 0.9
    assert c.loc[2005, "spearman_r"] == pytest.approx(1.0)
