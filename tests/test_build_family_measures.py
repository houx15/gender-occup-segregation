import gzip

import pandas as pd
import pytest

from scripts.data_prep.build_family_measures import family_measures, year_state_stats

COLS = "YEAR,STATEFIP,PERWT,NCHILD,NCHLT5,SEX,AGE,MARST,EMPSTAT,LABFORCE,UHRSWORK,INCWAGE,INCWAGE_SP"


def _write(path, rows):
    with gzip.open(path, "wt") as f:
        f.write(COLS + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")


def _rows():
    # YEAR, ST, W, NCHILD, NCHLT5, SEX, AGE, MARST, EMPSTAT, LABFORCE, UHRSWORK, INCWAGE, INCWAGE_SP
    return [
        (2005, 6, 10, 0, 0, 2, 30, 6, 1, 2, 40, 50000, None),       # childless woman, employed 40h
        (2005, 6, 10, 0, 0, 2, 35, 6, 3, 1, 0, 0, None),            # childless woman, not in LF
        (2005, 6, 10, 1, 1, 2, 31, 1, 1, 2, 20, 20000, 60000),      # mother <5, married, employed 20h
        (2005, 6, 10, 1, 1, 2, 33, 1, 3, 1, 0, 0, 80000),           # mother <5, married, not in LF
        (2006, 6, 20, 0, 0, 1, 40, 1, 1, 2, 45, 60000, 20000),      # married man, employed
        (2006, 6, 20, 0, 0, 1, 41, 6, 2, 2, 0, 0, None),            # man, unemployed (in LF)
    ]


def test_year_state_stats_and_window_measures(tmp_path):
    p = tmp_path / "usa_00002.csv.gz"
    _write(p, _rows())
    stats = year_state_stats(p, chunksize=2)
    m = family_measures(stats, period_start=2005, width=5).set_index(["STATEFIP", "period"])
    r = m.loc[(6, 2005)]
    # employment: childless women 1/2 = .5, mothers <5 1/2 = .5 -> gap 0
    assert r["motherhood_emp_gap"] == pytest.approx(0.0)
    # hours among employed: childless 40, mothers 20 -> gap 20
    assert r["motherhood_hours_gap"] == pytest.approx(20.0)
    # married women (MARST 1): one in LF, one not -> 0.5
    assert r["married_women_nilf"] == pytest.approx(0.5)
    # wives with positive couple income: 20k/(20k+60k)=.25 and 0/(0+80k)=0 -> mean .125
    assert r["wife_earnings_share"] == pytest.approx(0.125)
    assert r["wife_earns_more"] == pytest.approx(0.0)
    # men employed 1/2 = .5 (weights 20,20); women 2/4 = .5 -> gap 0
    assert r["gender_emp_gap"] == pytest.approx(0.0)
