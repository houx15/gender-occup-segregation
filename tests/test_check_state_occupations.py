import numpy as np
import pandas as pd
import pytest

from scripts.check_state_occupations import (
    by_occupation, by_occupation_and_time, by_time, merge_cells,
)


def _cells(seed=0, n_states=6, n_occ=12, periods=(2005, 2015)):
    """state x occupation x window cells with RND following the state share."""
    rng = np.random.default_rng(seed)
    rows = []
    base = rng.uniform(0.05, 0.95, n_occ)
    for s in range(n_states):
        for p in periods:
            for o in range(n_occ):
                share = np.clip(base[o] + 0.05 * s + (0.1 if p == periods[-1] else 0), 0, 1)
                rows.append({"state": f"s{s}", "period": p, "occupation": f"o{o}",
                             "rnd": 0.1 * share + rng.normal(0, 0.001),
                             "female_share": share, "national_share": base[o],
                             "weighted_n": 5000.0})
    return pd.DataFrame(rows)


def test_merge_cells_joins_rnd_with_state_and_national_shares():
    long_df = pd.DataFrame({"unit_name": ["new_york_2005", "new_york_2005"],
                            "category": ["occupation"] * 2, "occupation": ["nurse", "chef"],
                            "rnd": [0.04, -0.01], "in_vocab": [True, False]})
    state = pd.DataFrame({"word": ["nurse"], "state": ["New York"], "period": [2005],
                          "female_share": [0.9], "weighted_n": [1e5]})
    nat = pd.DataFrame({"word": ["nurse"], "period": [2005], "female_share": [0.88]})
    c = merge_cells(long_df, state, nat, {"nurse", "chef"})
    assert len(c) == 1
    row = c.iloc[0]
    assert (row["state"], row["period"], row["occupation"]) == ("new_york", 2005, "nurse")
    assert row["female_share"] == pytest.approx(0.9) and row["national_share"] == pytest.approx(0.88)


def test_by_occupation_one_r_per_state_window():
    out = by_occupation(_cells(), min_occupations=5)
    assert len(out) == 12                                  # 6 states x 2 windows
    assert (out["r_state_share"] > 0.9).all()
    assert {"r_national_share", "n_occupations"} <= set(out.columns)


def test_by_occupation_and_time_one_r_per_state():
    out = by_occupation_and_time(_cells(), min_points=5)
    assert len(out) == 6 and (out["r"] > 0.9).all()


def test_by_time_change_pairs():
    pairs, summary = by_time(_cells())
    assert len(pairs) == 6 * 12
    assert (pairs["d_rnd"] > 0).mean() > 0.8   # shares rose (except those capped at 1)
    assert {"scope", "n", "pearson_r"} <= set(summary.columns)
