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


def test_state_labor_indicators_duncan_and_female_share():
    from scripts.data_prep.build_occupation_shares import state_labor_indicators
    agg = pd.DataFrame([
        # state 1: fully segregated (women only in A, men only in B)
        {"YEAR": 2005, "STATEFIP": 1, "OCC2010": 100, "female": 50, "total": 50},
        {"YEAR": 2005, "STATEFIP": 1, "OCC2010": 200, "female": 0, "total": 50},
        # state 2: identical distributions -> D = 0
        {"YEAR": 2006, "STATEFIP": 2, "OCC2010": 100, "female": 20, "total": 40},
        {"YEAR": 2006, "STATEFIP": 2, "OCC2010": 200, "female": 30, "total": 60},
        {"YEAR": 2005, "STATEFIP": 2, "OCC2010": 9920, "female": 99, "total": 99},  # not an occupation
    ])
    out = state_labor_indicators(agg, period_start=2005, width=5).set_index("STATEFIP")
    assert out.loc[1, "duncan"] == pytest.approx(1.0)
    assert out.loc[2, "duncan"] == pytest.approx(0.0)
    assert out.loc[2, "female_emp_share"] == pytest.approx(0.5)
    assert (out["period"] == 2005).all()


def test_ddi_labels_reads_value_labels(tmp_path):
    from scripts.data_prep.build_occupation_shares import ddi_labels
    xml = tmp_path / "x.xml"
    xml.write_text(
        '<codeBook xmlns="ddi:codebook:2_5"><dataDscr>'
        '<var name="STATEFIP"><catgry><catValu>06</catValu><labl>California</labl></catgry>'
        '<catgry><catValu>11</catValu><labl>District of Columbia</labl></catgry></var>'
        '</dataDscr></codeBook>')
    assert ddi_labels(xml, "STATEFIP") == {6: "California", 11: "District of Columbia"}


def test_rolling_windows_count_a_year_in_every_window(tmp_path):
    from scripts.data_prep.build_occupation_shares import state_labor_indicators
    p = tmp_path / "x.csv.gz"
    _write_extract(p)
    agg = aggregate_extract(p)
    mapping = pd.DataFrame({"word": ["teacher"], "occ2010": ["2310"]})
    nat, _ = word_female_shares(agg, mapping, period_start=2005, width=10, step=5,
                                last_year=2014)
    # 2010 teachers fall in the 2005-2014 window; 2010-2019 is not full -> dropped
    assert set(nat["period"]) == {2005}
    lab = state_labor_indicators(agg, period_start=2005, width=2, step=1, last_year=2006)
    assert set(lab["period"]) == {2005}  # windows 2005-06 only (2006-07 not full)


def test_find_extracts_reads_subfolders_and_tracks_sources(tmp_path):
    from scripts.data_prep.build_occupation_shares import find_extracts, sources_changed
    (tmp_path / "early_2000_2004").mkdir()
    for p in (tmp_path / "usa_00001.csv.gz", tmp_path / "early_2000_2004" / "usa_00005.csv.gz"):
        _write_extract(p)
    found = find_extracts(tmp_path)
    assert [f.name for f in found] == ["usa_00001.csv.gz", "usa_00005.csv.gz"]
    assert sources_changed(tmp_path, found) is True       # no record yet
    (tmp_path / "aggregate_sources.txt").write_text("\n".join(str(f) for f in found))
    assert sources_changed(tmp_path, found) is False
