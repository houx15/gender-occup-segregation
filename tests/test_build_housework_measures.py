import gzip

import pandas as pd
import pytest

from scripts.data_prep.build_housework_measures import housework_measures, year_state_stats

COLS = "YEAR,STATEFIP,SEX,AGE,SPOUSEPRES,HH_CHILD,WT06,WT20,BLS_HHACT,BLS_HHACT_HWORK,BLS_HHACT_FOOD,BLS_CAREHH,BLS_CAREHH_KID"


def _write(path, rows):
    with gzip.open(path, "wt") as f:
        f.write(COLS + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def test_housework_shares_and_weights(tmp_path):
    p = tmp_path / "atus_00001.csv.gz"
    # YEAR ST SEX AGE SPOUSE HHCHILD WT06 WT20 HHACT HWORK FOOD CAREHH KID
    _write(p, [
        (2005, 6, 2, 30, 1, 1, 10, 0, 150, 60, 50, 90, 80),   # woman, parent
        (2005, 6, 1, 32, 1, 1, 10, 0, 50, 20, 10, 30, 20),    # man, parent
        (2006, 6, 2, 60, 1, 0, 99, 0, 999, 999, 999, 0, 0),   # outside 25-54: excluded
        (2020, 6, 2, 40, 1, 0, 1, 10, 100, 40, 30, 0, 0),     # 2020: WT20 used
        (2020, 6, 1, 40, 1, 0, 50, 10, 100, 40, 30, 0, 0),
    ])
    stats = year_state_stats(p)
    m = housework_measures(stats, period_start=2005, width=5).set_index(["STATEFIP", "period"])
    r = m.loc[(6, 2005)]
    assert r["women_share_household"] == pytest.approx(150 / (150 + 50))
    assert r["women_share_housework"] == pytest.approx(60 / 80)
    assert r["women_share_childcare_parents"] == pytest.approx(80 / 100)
    assert r["n_respondents"] == 2
    r20 = m.loc[(6, 2020)]
    assert r20["women_share_household"] == pytest.approx(0.5)     # equal WT20 weights
