import pandas as pd
import pytest

from scripts.compare_arms import agreement_overview, trend_overview


def test_agreement_overview_averages_within_period_and_keeps_change():
    c = pd.DataFrame([
        {"ours": "ours_occupation", "survey": "iat_sex_balanced", "scope": "period 2005", "agreement_r": 0.2},
        {"ours": "ours_occupation", "survey": "iat_sex_balanced", "scope": "period 2010", "agreement_r": 0.4},
        {"ours": "ours_occupation", "survey": "iat_sex_balanced", "scope": "change 2005->2010", "agreement_r": -0.1},
        {"ours": "ours_occupation", "survey": "iat_sex_balanced", "scope": "pooled", "agreement_r": 0.5},
    ])
    o = agreement_overview(c, "arm").set_index(["ours", "survey"])
    row = o.loc[("ours_occupation", "iat_sex_balanced")]
    assert row["within_period_mean"] == pytest.approx(0.3)
    assert row["change"] == pytest.approx(-0.1)
    assert row["pooled"] == pytest.approx(0.5)
    assert row["arm"] == "arm"


def test_trend_overview_first_last_and_direction():
    t = pd.DataFrame({"period": [2005, 2010, 2015], "n_states": [46] * 3,
                      "ours_occupation": [-0.012, -0.009, -0.004]})
    o = trend_overview(t, "arm").set_index("measure").loc["ours_occupation"]
    assert o["first"] == pytest.approx(-0.012) and o["last"] == pytest.approx(-0.004)
    assert o["monotone"] is True or o["monotone"] == True  # noqa: E712
