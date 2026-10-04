import numpy as np
import pandas as pd
import pytest

from scripts import visualize as v


def _summary(rows):
    """rows: (unit_name, category, mean_rnd, half_width)."""
    return pd.DataFrame([
        {"unit_name": u, "category": c, "mean_rnd": m,
         "mean_ci_low": m - h, "mean_ci_high": m + h}
        for u, c, m, h in rows
    ])


def test_trend_frame_orients_values_and_keeps_ci_ordered():
    s = _summary([("ohio_2005", "family", 0.2, 0.05),
                  ("ohio_2005", "leadership", 0.1, 0.02)])
    f = v.us_trend_frame(s, {"family": -1})
    fam = f[f.category == "family"].iloc[0]
    assert (fam.state, fam.period) == ("Ohio", 2005)
    assert np.isclose(fam.value, -0.2)
    assert np.isclose(fam.lo, -0.25) and np.isclose(fam.hi, -0.15)
    lead = f[f.category == "leadership"].iloc[0]
    assert np.isclose(lead.value, 0.1)


def test_national_trend_uses_balanced_panel_from_start():
    s = _summary([
        ("ohio_2000", "lead", 9.0, 0.01),          # before start: ignored
        ("ohio_2005", "lead", 0.1, 0.01), ("ohio_2010", "lead", 0.3, 0.01),
        ("utah_2005", "lead", 0.3, 0.01), ("utah_2010", "lead", 0.5, 0.01),
        ("texas_2010", "lead", 5.0, 0.01),         # missing 2005: not balanced
    ])
    nat = v.us_national_trend(v.us_trend_frame(s, {}), start=2005)
    got = nat.set_index("period")
    assert list(got.index) == [2005, 2010]
    assert np.isclose(got.loc[2005, "mean"], 0.2)
    assert np.isclose(got.loc[2010, "mean"], 0.4)
    assert (got["n_states"] == 2).all()
    assert (got["ci_low"] < got["mean"]).all() and (got["mean"] < got["ci_high"]).all()


def test_state_change_combines_unit_uncertainty():
    # 68% bootstrap bands (half-width = 1 SE): SE_diff = sqrt(0.03^2 + 0.04^2) = 0.05
    s = _summary([("ohio_2005", "lead", 0.10, 0.03), ("ohio_2020", "lead", 0.30, 0.04),
                  ("utah_2005", "lead", 0.10, 0.03), ("utah_2020", "lead", 0.12, 0.04),
                  ("texas_2020", "lead", 0.5, 0.01)])  # no start value: dropped
    ch = v.us_state_change(v.us_trend_frame(s, {}), start=2005, end=2020, unit_ci=0.68)
    got = ch.set_index("state")
    assert set(got.index) == {"Ohio", "Utah"}
    assert np.isclose(got.loc["Ohio", "change"], 0.20)
    z68 = 0.994457883
    se = np.hypot(0.03, 0.04) / z68
    assert np.isclose(got.loc["Ohio", "ci_high"] - got.loc["Ohio", "change"], 1.959964 * se, rtol=1e-4)
    assert bool(got.loc["Ohio", "significant"]) is True
    assert bool(got.loc["Utah", "significant"]) is False


def test_state_change_requires_both_periods_present():
    s = _summary([("ohio_2005", "lead", 0.1, 0.01)])
    with pytest.raises(ValueError, match="2020"):
        v.us_state_change(v.us_trend_frame(s, {}), start=2005, end=2020, unit_ci=0.68)


def test_period_label_uses_bin_width():
    assert v.us_period_label(2005, 5) == "2005–09"
    assert v.us_period_label(2020, 5) == "2020–24"
    assert v.us_period_label(2010, None) == "2010"
