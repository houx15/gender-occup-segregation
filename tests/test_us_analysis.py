import numpy as np
import pandas as pd
import pytest

from scripts.us_analysis.common import MAIN, ROBUSTNESS, SURVEY_LABEL
from scripts.us_analysis import part1, part2


def _panel(seed=0, n_states=14, periods=(1995, 2000, 2005, 2010, 2015)):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_states):
        base = rng.normal()
        for p in periods:
            trad = base - 0.1 * (p - 2005) / 5 + rng.normal(0, 0.3)
            row = {"unit_name": f"s{i}_{p}", "state": f"s{i}", "period": p,
                   "ours_occupation": -0.01 * trad, "ours_family_sphere": 0.01 * trad,
                   "ours_household": 0.005 * trad + rng.normal(0, 0.01),
                   "se_ours_occupation": 0.003, "se_ours_family_sphere": 0.006,
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


def test_part1_main_and_robustness_outputs(tmp_path):
    panel = _panel()
    picked = [sp for sp in ROBUSTNESS if sp.name in ("robustness-duncan", "robustness-iat",
                                                     "robustness-atus-household")]
    text = part1.run_part1(panel, _cells(), [MAIN] + picked, tmp_path)
    for f in ("1_1_occupation_validity.pdf", "1_2_temporal.pdf", "1_3_geography.pdf",
              "1_4_state_window.pdf", "1_5_between_within.pdf", "1_6_volume_error.pdf",
              "1_7_state_slopes.pdf"):
        assert (tmp_path / "main" / "figures" / f).exists(), f
    assert (tmp_path / "robustness-iat" / "tables" / "1_4_models.csv").exists()
    assert (tmp_path / "robustness-atus-household" / "figures" / "1_4_state_window.pdf").exists()
    models = pd.read_csv(tmp_path / "main" / "tables" / "1_4_models.csv")
    pooled = models[(models.model == "pooled") & (models.domain == "occupation")].iloc[0]
    assert pooled["beta_std"] > 0.5            # planted positive alignment (oriented)
    assert "1.1 Occupation-level validity" in text


def test_part2_heatmaps_and_change(tmp_path):
    panel = _panel()
    out = tmp_path / "main"
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True)
    part2.heatmaps(panel, out)
    md = part2.change_ranking(panel, out)
    assert (out / "figures" / "2_2_heatmap_occupation_region.pdf").exists()
    assert (out / "figures" / "2_3_change_family_sphere.pdf").exists()
    ch = pd.read_csv(out / "tables" / "2_3_change_occupation_2000_2015.csv")
    assert (ch["change"] < 0).mean() > 0.5     # planted trend toward less traditional
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
    part1.run_part1(panel, _cells(), [MAIN], tmp_path)
    t2b = part2b.run_part2b(panel, "", tmp_path)
    assert "Block fit" in t2b
    assert (tmp_path / "main" / "figures" / "2b_predictors_occupation.pdf").exists()
    t2c = part2c.run_part2c(panel, "", tmp_path)
    assert (tmp_path / "main" / "tables" / "2c_gaps.csv").exists() and "Model fit" in t2c
    terms = pd.DataFrame([{"state": f"s{s}", "period": p, "category": "family_sphere",
                           "term": t, "rnd": np.random.default_rng(s).normal(0.01, 0.01)}
                          for s in range(5) for p in (2005, 2015) for t in ("home", "family", "kitchen")])
    cells = pd.concat([_cells(0).assign(period=2005), _cells(0).assign(period=2015)])
    t3 = part3.run_part3(panel, cells, terms, tmp_path)
    cases = pd.read_csv(tmp_path / "main" / "tables" / "3_case_selection.csv")
    assert {"most stable", "largest move toward less traditional",
            "largest text-survey discrepancy"} <= set(cases["criterion"])
    assert (tmp_path / "main" / "figures" / "3_3_state_cases_family.pdf").exists()
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
    part1.run_part1(panel, _cells(), [MAIN], tmp_path)
    main = tmp_path / "main"
    part2.heatmaps(panel, main)
    part2.change_ranking(panel, main)
    part2b.run_part2b(panel, "", tmp_path)
    figures.run_figures(panel, tmp_path)
    for f in ("figure1_validation.pdf", "figure2_survey.pdf", "figure3_reliability.pdf",
              "figure5_dynamics_occupation.pdf", "figure6_explanatory.pdf"):
        assert (tmp_path / "figures_combined" / f).exists(), f
