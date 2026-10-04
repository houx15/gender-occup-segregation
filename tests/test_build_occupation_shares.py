import gzip

import pandas as pd
import pytest

from scripts.data_prep.build_occupation_shares import (
    aggregate_extract, parse_occ_codes, word_female_shares,
)


def _write_extract(path):
    rows = [  # YEAR, STATEFIP, SEX (1=male, 2=female), OCC2010, PERWT
        (2005, 6, 2, 3255, 30), (2005, 6, 1, 3255, 10),     # CA nurses: 75% female
        (2006, 6, 2, 3255, 10), (2006, 6, 1, 3255, 10),
        (2005, 48, 1, 3740, 90), (2005, 48, 2, 3740, 10),   # TX firefighters
        (2005, 48, 2, 3255, 50), (2005, 48, 1, 3255, 50),
        (2010, 6, 2, 2310, 80), (2010, 6, 1, 2310, 20),
    ]
    with gzip.open(path, "wt") as f:
        f.write("YEAR,SAMPLE,SERIAL,PERNUM,PERWT,STATEFIP,SEX,AGE,EMPSTAT,OCC2010\n")
        for y, st, sex, occ, w in rows:
            f.write(f"{y},{y}01,1,1,{w},{st},{sex},40,1,{occ}\n")


def test_aggregate_extract_sums_weights_by_cell(tmp_path):
    p = tmp_path / "usa_00001.csv.gz"
    _write_extract(p)
    agg = aggregate_extract(p, chunksize=3).set_index(["YEAR", "STATEFIP", "OCC2010"])
    assert agg.loc[(2005, 6, 3255), "female"] == 30
    assert agg.loc[(2005, 6, 3255), "total"] == 40


def test_parse_occ_codes():
    assert parse_occ_codes("3255; 3256") == [3255, 3256]
    assert parse_occ_codes("") == []


def test_word_female_shares_by_period_national_and_state(tmp_path):
    p = tmp_path / "x.csv.gz"
    _write_extract(p)
    agg = aggregate_extract(p)
    mapping = pd.DataFrame({"word": ["nurse", "firefighter", "teacher"],
                            "occ2010": ["3255", "3740", "2310"]})
    nat, state = word_female_shares(agg, mapping, period_start=2005, width=5)
    n = nat.set_index(["word", "period"])
    # nurse 2005-09 national: female 30+10+50=90 of 40+20+100=160
    assert n.loc[("nurse", 2005), "female_share"] == pytest.approx(90 / 160)
    assert n.loc[("teacher", 2010), "female_share"] == pytest.approx(0.8)
    s = state.set_index(["word", "STATEFIP", "period"])
    assert s.loc[("nurse", 6, 2005), "female_share"] == pytest.approx(40 / 60)
    assert s.loc[("firefighter", 48, 2005), "female_share"] == pytest.approx(0.1)
    assert s.loc[("nurse", 6, 2005), "weighted_n"] == 60
