import numpy as np
import pandas as pd
import pytest

from scripts.check_state_benchmarks import (
    benchmark_table, correlate, national_trend, our_scores, with_unit_name,
)


def test_with_unit_name_accepts_names_and_usps_codes():
    d = pd.DataFrame({"state": ["Ohio", "DC", "NY", "new_york"], "period": [2005, 2010, 2015, 2020]})
    assert list(with_unit_name(d)["unit_name"]) == [
        "ohio_2005", "district_of_columbia_2010", "new_york_2015", "new_york_2020"]


def test_our_scores_wide_by_category():
    s = pd.DataFrame({"unit_name": ["ohio_2005", "ohio_2005", "utah_2005"],
                      "category": ["occupation", "family_sphere", "occupation"],
                      "mean_value": [-0.01, -0.02, -0.03]})
    w = our_scores(s).set_index("unit_name")
    assert w.loc["ohio_2005", "ours_occupation"] == pytest.approx(-0.01)
    assert np.isnan(w.loc["utah_2005", "ours_family_sphere"])


def _table():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        base = rng.normal()
        for p in (2005, 2010):
            rows.append({"unit_name": f"s{i}_{p}", "state": f"s{i}", "period": p,
                         "ours_occupation": base + 0.1 * p, "duncan": base + 0.1 * p,
                         "iat_sex_balanced": rng.normal()})
    return pd.DataFrame(rows)


def test_correlate_scopes_and_pairs():
    c = correlate(_table(), ["ours_occupation"], ["duncan", "iat_sex_balanced"])
    pooled = c[(c.scope == "pooled") & (c.survey == "duncan")].iloc[0]
    assert pooled["n"] == 24 and pooled["pearson_r"] > 0.99
    assert {"pooled", "period 2005", "period 2010", "change 2005->2010"} <= set(c.scope)


def test_national_trend_balanced_states():
    t = _table()
    t = t[~((t.state == "s0") & (t.period == 2010))]   # s0 missing a period -> dropped
    n = national_trend(t, ["ours_occupation", "duncan"]).set_index("period")
    assert (n["n_states"] == 11).all()


def test_benchmark_table_joins_sources_on_unit_name():
    ours = pd.DataFrame({"unit_name": ["ohio_2005"], "ours_occupation": [-0.01]})
    fam = pd.DataFrame({"state": ["Ohio"], "period": [2005], "STATEFIP": [39],
                        "married_women_nilf": [0.2]})
    att = pd.DataFrame({"state": ["OH"], "period": [2005], "iat_sex_balanced": [0.35]})
    t = benchmark_table(ours, [fam, att]).set_index("unit_name")
    assert t.loc["ohio_2005", "married_women_nilf"] == pytest.approx(0.2)
    assert t.loc["ohio_2005", "iat_sex_balanced"] == pytest.approx(0.35)
    assert t.loc["ohio_2005", "period"] == 2005


def test_agreement_r_flips_to_common_traditional_direction():
    from scripts.check_state_benchmarks import TRADITIONAL_SIGN, add_agreement
    c = pd.DataFrame({"ours": ["ours_occupation", "ours_occupation", "ours_household"],
                      "survey": ["iat_sex_balanced", "matched_female_share", "iat_sex_balanced"],
                      "pearson_r": [-0.4, 0.3, 0.2]})
    a = add_agreement(c)
    # occupation (higher = less traditional) vs IAT (higher = more traditional): -0.4 agrees
    assert list(a["agreement_r"]) == pytest.approx([0.4, 0.3, 0.2])
    assert TRADITIONAL_SIGN["duncan"] == 1 and TRADITIONAL_SIGN["ours_occupation"] == -1


def test_family_index_is_mean_of_oriented_z_scores():
    from scripts.check_state_benchmarks import FAMILY, add_family_index
    t = pd.DataFrame({c: [0.0, 1.0, 2.0] for c in FAMILY})
    t["wife_earnings_share"] = [2.0, 1.0, 0.0]   # less traditional when higher -> flipped
    t["wife_earns_more"] = [2.0, 1.0, 0.0]
    out = add_family_index(t)
    z = np.sqrt(1.5)  # z-scores of 0, 1, 2 (population SD)
    assert list(out["family_index_acs"]) == pytest.approx([-z, 0.0, z])


def test_correlate_change_uses_windows_where_both_measures_exist():
    t = _table()                              # periods 2005, 2010
    early = t[t.period == 2005].assign(period=1995, iat_sex_balanced=np.nan)
    t = pd.concat([early, t], ignore_index=True)
    c = correlate(t, ["ours_occupation"], ["duncan", "iat_sex_balanced"])
    scopes = set(c[c.survey == "iat_sex_balanced"].scope)
    assert "change 2005->2010" in scopes      # IAT has no 1995 values
    assert "change 1995->2010" in set(c[c.survey == "duncan"].scope)
