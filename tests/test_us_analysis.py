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
    text = part1.run_part1(panel, _cells(), [MAIN] + ROBUSTNESS[:1] + ROBUSTNESS[-3:-1], tmp_path)
    for f in ("1_1_occupation_validity.pdf", "1_2_temporal.pdf", "1_3_geography.pdf",
              "1_4_state_window.pdf", "1_5_between_within.pdf", "1_6_volume_error.pdf",
              "1_7_state_slopes.pdf"):
        assert (tmp_path / "main" / "figures" / f).exists(), f
    assert (tmp_path / "robustness-iat" / "tables" / "1_4_models.csv").exists()
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
