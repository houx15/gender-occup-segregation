import gzip

import pandas as pd
import pytest

from scripts.data_prep.build_context_measures import (
    context_measures, gop_share_by_window, pfl_share_by_window, year_state_stats,
)

COLS = "YEAR,STATEFIP,PERWT,SEX,AGE,EDUC,METRO,IND1990,INCTOT,INCWAGE,CPI99,UHRSWORK,EMPSTAT,LABFORCE"


def _write(path, rows):
    with gzip.open(path, "wt") as f:
        f.write(COLS + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def test_context_measures_from_microdata(tmp_path):
    p = tmp_path / "usa_00005.csv.gz"
    # YEAR ST W SEX AGE EDUC METRO IND1990 INCTOT INCWAGE CPI99 UHRSWORK EMPSTAT LABFORCE
    _write(p, [
        (2005, 6, 10, 1, 40, 11, 2, 150, 60000, 60000, 1.0, 40, 1, 2),   # man, BA, metro, manuf, FT
        (2005, 6, 10, 2, 40, 6, 1, 700, 30000, 30000, 1.0, 40, 1, 2),    # woman, no BA, non-metro, services, FT
        (2005, 6, 10, 2, 35, 10, 3, 0, 0, 0, 1.0, 0, 3, 1),              # woman, BA, metro, not in LF
        (2006, 6, 10, 1, 50, 6, 0, 0, 1000, 0, 1.0, 0, 2, 2),            # man, unemployed, metro unknown
    ])
    stats = year_state_stats(p)
    m = context_measures(stats, period_start=2005, width=5).set_index(["STATEFIP", "period"])
    r = m.loc[(6, 2005)]
    assert r["ba_share"] == pytest.approx(0.5)
    assert r["metro_share"] == pytest.approx(2 / 3)                  # METRO 0 excluded
    assert r["unemployment_rate"] == pytest.approx(1 / 3)
    assert r["manufacturing_share"] == pytest.approx(0.5)            # of the employed
    assert r["women_lfp"] == pytest.approx(0.5)
    assert r["gender_wage_gap"] == pytest.approx(0.5)                # 1 - 30k/60k
    assert r["real_income_pc"] == pytest.approx((60000 + 30000 + 0 + 1000) / 4)


def test_pfl_share_by_window():
    pfl = pd.DataFrame({"state": ["california", "new_york"], "benefits_start_year": [2004, 2018]})
    out = pfl_share_by_window(pfl, ["california", "new_york", "texas"], periods=[2000, 2015], width=10)
    o = out.set_index(["state", "period"])["pfl_share"]
    assert o[("california", 2000)] == pytest.approx(0.6)    # 2004-2009 of 2000-2009
    assert o[("new_york", 2015)] == pytest.approx(0.7)      # 2018-2024
    assert o[("texas", 2015)] == 0


def test_gop_share_by_window():
    votes = pd.DataFrame([
        {"year": 2016, "state": "OHIO", "party_simplified": "REPUBLICAN", "candidatevotes": 60},
        {"year": 2016, "state": "OHIO", "party_simplified": "DEMOCRAT", "candidatevotes": 40},
        {"year": 2016, "state": "OHIO", "party_simplified": "LIBERTARIAN", "candidatevotes": 10},
        {"year": 2020, "state": "OHIO", "party_simplified": "REPUBLICAN", "candidatevotes": 50},
        {"year": 2020, "state": "OHIO", "party_simplified": "DEMOCRAT", "candidatevotes": 50},
    ])
    out = gop_share_by_window(votes, periods=[2015], width=10).set_index("state")
    assert out.loc["ohio", "gop_two_party_share"] == pytest.approx(0.55)


def test_window_mean_requires_half_the_years():
    from scripts.data_prep.build_context_measures import window_means
    yearly = pd.DataFrame({"state": ["ohio"] * 6, "year": [2005, 2006, 2007, 2008, 2009, 2016],
                           "x": [1, 1, 1, 1, 0, 9]})
    out = window_means(yearly, ["x"], periods=[2005, 2015], width=10).set_index(["state", "period"])
    assert out.loc[("ohio", 2005), "x"] == pytest.approx(0.8)    # 5 observed years of 10
    assert pd.isna(out.loc[("ohio", 2015), "x"])                  # 1 observed year: too few


def test_cspp_policy_coding_rules():
    from scripts.data_prep.build_context_measures import cspp_yearly
    rows = []
    for y in (2000, 2001):
        for i in range(45):
            st = f"S{i}"
            rows.append({"st": st, "year": y,
                         "universalprek": float(i % 2) if y == 2000 else (1.0 if i < 5 else None),
                         "fundslife": 1.0 if i < 10 else None, "equalpay": 0.0, "solaw": 1.0,
                         "infconsent": 1.0, "gagrule": 0.0, "medicalrest": 0.0, "insprivate": 0.0,
                         "inspublic": 0.0, "inswaiver": 0.0, "evangldsper": 20.0})
    y = cspp_yearly(pd.DataFrame(rows), state_key=lambda s: s.lower())
    a = y.set_index(["state", "year"])
    assert pd.isna(a.loc[("s20", 2001), "universal_prek"])       # 2001: < 40 states coded
    assert a.loc[("s20", 2000), "universal_prek"] == 0.0
    assert a.loc[("s20", 2000), "abortion_restrictions"] == 1     # infconsent only; fundslife blank -> 0
    assert a.loc[("s3", 2000), "abortion_restrictions"] == 2      # + fundslife


def test_abortion_index_scales_over_observed_items():
    from scripts.data_prep.build_context_measures import cspp_yearly
    rows = [{"st": f"S{i}", "year": 2000, "universalprek": 0.0, "equalpay": 0.0, "solaw": 0.0,
             "evangldsper": 1.0, "fundslife": 1.0, "infconsent": 1.0, "gagrule": None,
             "medicalrest": 0.0, "insprivate": 0.0, "inspublic": 0.0, "inswaiver": 0.0}
            for i in range(45)]
    for r in rows:                       # gagrule never coded -> 6 observed items
        r["gagrule"] = None
    y = cspp_yearly(pd.DataFrame(rows), state_key=lambda s: s.lower())
    assert y["abortion_restrictions"].iloc[0] == pytest.approx(2 / 6 * 7)
