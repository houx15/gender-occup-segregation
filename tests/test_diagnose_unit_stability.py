import numpy as np
import pandas as pd

from scripts.diagnose_unit_stability import adjacent_stability, summarize_stability


def _summary(values):
    """values: {(state, period): mean_value} for one category 'lead'."""
    rows = []
    for (state, period), v in values.items():
        rows.append({"unit_name": f"{state}_{period}", "category": "lead",
                     "mean_value": v, "mean_ci_low": v - 0.01, "mean_ci_high": v + 0.01,
                     "n_occupations": 5, "n_consistent": 5})
    return pd.DataFrame(rows)


def test_adjacent_stability_correlates_states_across_periods():
    vals = {}
    for i, s in enumerate(["ohio", "new_york", "texas", "utah"]):
        vals[(s, 2000)] = i * 0.1
        vals[(s, 2005)] = i * 0.1 + 0.05   # same ranking -> corr 1
        vals[(s, 2010)] = -i * 0.1          # reversed ranking -> corr -1
    pairs = adjacent_stability(_summary(vals))
    got = {(r.period_a, r.period_b): r for r in pairs.itertuples()}
    assert np.isclose(got[(2000, 2005)].r, 1.0)
    assert np.isclose(got[(2005, 2010)].r, -1.0)
    assert got[(2000, 2005)].n_states == 4


def test_adjacent_stability_uses_only_states_present_in_both_periods():
    vals = {("ohio", 2000): 0.1, ("texas", 2000): 0.2, ("utah", 2000): 0.3,
            ("ohio", 2005): 0.1, ("texas", 2005): 0.2, ("utah", 2005): 0.3,
            ("alaska", 2005): 9.0}  # only in one period: ignored
    pairs = adjacent_stability(_summary(vals))
    assert pairs.iloc[0].n_states == 3
    assert np.isclose(pairs.iloc[0].r, 1.0)


def test_summarize_reports_words_and_mean_stability():
    # states a, b, c at 0.0, 0.1, 0.2: CI is +/-0.01, so only state a straddles 0
    vals = {(s, p): i * 0.1 for i, s in enumerate(["a", "b", "c"]) for p in (2000, 2005)}
    summary = _summary(vals)
    out = summarize_stability(summary, adjacent_stability(summary))
    row = out.set_index("category").loc["lead"]
    assert row.n_words == 5
    assert row.n_units == 6
    assert np.isclose(row.mean_adjacent_corr, 1.0)
    assert np.isclose(row.share_sig, 4 / 6)
