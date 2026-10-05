import numpy as np
import pandas as pd
import pytest

from scripts.plot_garg_fig2 import garg_series, pct_difference


def test_pct_difference_is_women_minus_men_in_percent():
    assert pct_difference(pd.Series([0.5, 0.75, 0.2])).tolist() == pytest.approx([0, 50, -60])


def test_garg_series_uses_occupations_present_in_every_window_and_bootstraps():
    rows = []
    for p, shift in ((2005, 0.0), (2015, 0.01)):
        for o, share in (("nurse", 0.9), ("engineer", 0.15), ("teacher", 0.75)):
            rows.append({"period": p, "occupation": o, "rnd": 0.05 * share + shift,
                         "female_share": share + shift})
    rows.append({"period": 2005, "occupation": "chef", "rnd": 9.0, "female_share": 0.2})  # 1 window
    s = garg_series(pd.DataFrame(rows), n_boot=200, seed=0).set_index("period")
    assert s.loc[2005, "n_occupations"] == 3                      # chef excluded
    assert s.loc[2005, "avg_bias"] == pytest.approx(0.05 * (0.9 + 0.15 + 0.75) / 3)
    assert s.loc[2015, "avg_pct_diff"] > s.loc[2005, "avg_pct_diff"]
    assert (s["bias_se"] > 0).all() and (s["pct_diff_se"] > 0).all()
