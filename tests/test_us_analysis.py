import numpy as np
import pandas as pd
import pytest

from scripts.us_analysis.common import CORRELATES, SURVEY_LABEL, VALIDATION
from scripts.us_analysis import part1, part2


def _panel(seed=0, n_states=14, periods=(1995, 2000, 2005, 2010, 2015)):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_states):
        base = rng.normal()
        for p in periods:
            trad = base - 0.1 * (p - 2005) / 5 + rng.normal(0, 0.3)
            row = {"unit_name": f"s{i}_{p}", "state": f"s{i}", "period": p,
                   "ours_occupation": -0.01 * trad,
                   "ours_household": 0.005 * trad + rng.normal(0, 0.01),
                   "se_ours_occupation": 0.003,
                   "se_ours_household": 0.006, "tokens": float(rng.integers(2e5, 2e7))}
            for c in SURVEY_LABEL:
                row[c] = trad + rng.normal(0, 0.5)
            row["matched_female_share"] = -row["matched_female_share"]   # less traditional
            rows.append(row)
    return pd.DataFrame(rows)


def _cells(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for o in range(20):
        share = rng.uniform(0, 1)
        for s in range(5):
            rows.append({"state": f"s{s}", "period": 2005, "occupation": f"o{o}",
                         "rnd": 0.08 * (share - 0.5) + rng.normal(0, 0.005),
                         "female_share": share})
    return pd.DataFrame(rows)


def test_part1_validation_correlates_reliability(tmp_path):
    panel = _panel()
    text = part1.run_part1(panel, _cells(), tmp_path)
    for f in ("1_1_occupation_validity.pdf", "1_2_temporal.pdf", "1_3_geography.pdf",
              "1_4_validation_occupation.pdf", "1_4_validation_household.pdf"):
        assert (tmp_path / "main" / "figures" / f).exists(), f
    v = pd.read_csv(tmp_path / "main" / "tables" / "1_4_validation.csv")
    # every direct benchmark, three dimensions each
    assert set(v[v.domain == "occupation"]["survey"]) == set(VALIDATION["occupation"])
    assert set(v[v.domain == "household"]["survey"]) == set(VALIDATION["household"])
    assert set(v["spec"]) == {"pooled", "between", "within"}
    pooled = v[(v.spec == "pooled") & (v.survey == "matched_female_share")].iloc[0]
    assert pooled["beta"] > 0.5                # planted agreement (oriented)
    c = pd.read_csv(tmp_path / "main" / "tables" / "1_9_correlates.csv")
    assert set(c["survey"]) == set(CORRELATES) and c["q_bh"].between(0, 1).all()
    assert (tmp_path / "main" / "tables" / "1_9_correlates_table.tex").exists()
    assert (tmp_path / "main" / "tables" / "1_9_correlates_no_dc.csv").exists()
    for m in ("iat_sex_balanced", "women_share_housework"):
        assert (tmp_path / f"validation-{m}" / "figures" / "1_7_state_slopes.pdf").exists()
    hier = pd.read_csv(tmp_path / "validation-iat_sex_balanced" / "tables" / "1_7_hierarchical.csv")
    assert set(hier["domain"]) == {"occupation", "household"}   # IAT validates both domains
    assert "1.1 Occupation-level validity" in text and "Correlates" in text


def test_journal_table_formats_coefficients():
    from scripts.us_analysis.part1 import journal_table
    res = pd.DataFrame([{"domain": "occupation", "survey": "iat_sex_balanced", "spec": sp,
                         "beta": 0.41, "se": 0.08, "p": p, "n": n, "states": 51}
                        for sp, p, n in (("pooled", 1e-6, 199), ("between", 0.02, 51), ("within", 0.9, 199))])
    md, tex = journal_table(res, ["iat_sex_balanced"], "T", "note")
    assert "0.41*** (0.08)" in md and "0.41* (0.08)" in md and "0.41 (0.08)" in md
    assert "$^{***}$" in tex and "\\begin{tabular}" in tex


def test_part2_heatmaps_and_change(tmp_path):
    panel = _panel()
    out = tmp_path / "main"
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True)
    part2.heatmaps(panel, out)
    md = part2.change_ranking(panel, out)
    assert (out / "figures" / "2_2_heatmap_occupation_region.pdf").exists()
    assert (out / "figures" / "2_3_change_household.pdf").exists()
    ch = pd.read_csv(out / "tables" / "2_3_change_occupation_2000_2015.csv")
    assert (ch["change"] > 0).mean() > 0.5     # planted trend toward less traditional
    assert "Change ranking" in md


def _panel_with_context(seed=1):
    rng = np.random.default_rng(seed)
    p = _panel(seed)
    for c in ("real_income_pc", "ba_share", "metro_share", "unemployment_rate", "manufacturing_share",
              "service_share", "women_lfp", "gender_wage_gap", "female_share_managers",
              "female_share_professionals", "pfl_share", "gop_two_party_share"):
        p[c] = rng.uniform(0.1, 0.9, len(p))
    p["real_income_pc"] = rng.uniform(2e4, 6e4, len(p))
    p["unit_name"] = p["state"] + "_" + p["period"].astype(str)
    return p


def test_part2b_2c_3(tmp_path):
    from scripts.us_analysis import part2b, part2c, part3
    panel = _panel_with_context()
    # Part I first so Part III can read the I.7 slopes
    part1.run_part1(panel, _cells(), tmp_path)
    t2b = part2b.run_part2b(panel, "", tmp_path)
    assert "Block fit" in t2b
    assert (tmp_path / "main" / "figures" / "2b_predictors_occupation.pdf").exists()
    t2c = part2c.run_part2c(panel, "", tmp_path)
    assert (tmp_path / "main" / "tables" / "2c_gaps.csv").exists() and "Model fit" in t2c
    terms = pd.DataFrame([{"state": f"s{s}", "period": p, "category": c,
                           "term": t, "rnd": np.random.default_rng(s).normal(0.01, 0.01)}
                          for s in range(5) for p in (2005, 2015)
                          for c, t in (("other", "home"), ("household", "laundry"),
                                       ("household", "cooking"), ("household", "kitchen"))])
    cells = pd.concat([_cells(0).assign(period=2005), _cells(0).assign(period=2015)])
    t3 = part3.run_part3(panel, cells, terms, tmp_path)
    cases = pd.read_csv(tmp_path / "main" / "tables" / "3_case_selection.csv")
    assert {"most stable", "largest move toward less traditional",
            "largest text-survey discrepancy"} <= set(cases["criterion"])
    assert (tmp_path / "main" / "figures" / "3_3_state_cases_household.pdf").exists()
    # term cases come from the household-work list only
    assert set(cases[cases["level"].str.startswith("term")]["level"]) == {"term (household)"}
    prof = pd.read_csv(tmp_path / "main" / "tables" / "3_3_state_profiles.csv")
    assert {"text_occupation_first", "women_lfp_last", "gop_two_party_share_first"} <= set(prof.columns)
    assert "3.3 State profiles" in t3


def test_policy_timing_counts_clean_before_after(tmp_path):
    from scripts.us_analysis.policy_timing import policy_timing
    (tmp_path / "tables").mkdir()
    pfl = tmp_path / "pfl.csv"
    pfl.write_text("state,benefits_start_year\ns0,2010\ns1,2024\n")
    panel = _panel()                       # windows 1995..2015, states s0..s13
    md = policy_timing(panel, str(pfl), tmp_path)
    t = pd.read_csv(tmp_path / "tables" / "2_6_policy_timing.csv").set_index("state")
    assert bool(t.loc["s0", "usable_before_after"]) is True   # 1995-04 before, 2010-19 after
    assert bool(t.loc["s1", "usable_before_after"]) is False  # never fully after
    assert "too few treated states" in md


def test_combined_figures(tmp_path):
    from scripts.us_analysis import part2b, figures
    panel = _panel_with_context()
    part1.run_part1(panel, _cells(), tmp_path)
    main = tmp_path / "main"
    part2.heatmaps(panel, main)
    part2.change_ranking(panel, main)
    part2b.run_part2b(panel, "", tmp_path)
    figures.run_figures(panel, tmp_path)
    for f in ("figure1_validation.pdf", "figure2_validation_occupation.pdf",
              "figure2_validation_household.pdf", "figure3_reliability.pdf",
              "figure5_dynamics_occupation.pdf", "figure6_explanatory.pdf"):
        assert (tmp_path / "figures_combined" / f).exists(), f


def test_household_direction():
    """Household words closer to female words (RND > 0) = more traditional =
    lower oriented score; a higher women's housework share likewise scores lower."""
    from scripts.us_analysis.common import survey_egal, text_egal
    d = pd.DataFrame({"ours_household": [0.02, -0.02], "women_share_housework": [0.8, 0.6],
                      "family_index_acs": [1.0, -1.0]})
    assert text_egal(d, "ours_household").iloc[0] < text_egal(d, "ours_household").iloc[1]
    assert survey_egal(d, "women_share_housework").iloc[0] < survey_egal(d, "women_share_housework").iloc[1]
    assert survey_egal(d, "family_index_acs").iloc[0] < survey_egal(d, "family_index_acs").iloc[1]


def test_balanced_trend_removes_composition_effect():
    """A late-entering, less traditional state creates a spurious national trend;
    the balanced panel (states observed in every window) removes it."""
    from scripts.us_analysis.balanced import complete_units, trend_table, word_trajectories
    periods = [2000, 2005, 2010, 2015]
    rows = [{"state": "a", "period": p, "ours_occupation": -0.02, "ours_household": 0.01}
            for p in periods]
    rows += [{"state": "b", "period": p, "ours_occupation": 0.02, "ours_household": -0.01}
             for p in (2010, 2015)]   # enters late; flat within state
    panel = pd.DataFrame(rows)
    tab = trend_table(panel, periods)
    occ = tab[tab["text"] == "ours_occupation"].set_index(["sample", "period"])["mean"]
    assert occ[("all states", 2015)] > occ[("all states", 2000)]          # spurious rise
    assert occ[("balanced", 2015)] == pytest.approx(occ[("balanced", 2000)])  # flat
    assert set(complete_units(panel, "state", "ours_occupation", periods)["state"]) == {"a"}
    cells = pd.DataFrame([{"state": s, "period": p, "occupation": "nurse", "rnd": r}
                          for s, ps, r in (("a", periods, 0.01), ("b", (2010, 2015), 0.05))
                          for p in ps])
    t = word_trajectories(cells, "occupation", periods)
    assert t["balanced"].loc["nurse", 2015] == pytest.approx(0.01)
    assert t["all states"].loc["nurse", 2015] == pytest.approx(0.03)


def test_plot_state_slopes_pools_when_states_do_not_differ():
    import matplotlib.pyplot as plt
    from scripts.us_analysis.common import plot_state_slopes
    h = pd.DataFrame({"state": ["a", "b"], "slope": [0.30, 0.31], "lo": [0.05, 0.06], "hi": [0.55, 0.56]})
    fig, ax = plt.subplots()
    assert plot_state_slopes(ax, h, 0.3, 0.13, slope_sd=0.01)          # band, no bars
    assert not plot_state_slopes(ax, h, 0.3, 0.13, slope_sd=0.30)      # per-state bars
    plt.close(fig)
