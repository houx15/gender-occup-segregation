import json
from scripts.data_prep import us_state_mapper as m


def test_normalize_state_accepts_name_usps_and_messy_case():
    assert m.normalize_state("California") == "California"
    assert m.normalize_state("california") == "California"
    assert m.normalize_state("  CA ") == "California"
    assert m.normalize_state("New york") == "New York"
    assert m.normalize_state("District of Columbia") == "District of Columbia"
    assert m.normalize_state("DC") == "District of Columbia"
    assert m.normalize_state("Freedonia") is None
    assert m.normalize_state("") is None


def test_unit_state_tokenizes():
    assert m.unit_state("California") == "california"
    assert m.unit_state("New York") == "new_york"
    assert m.unit_state("District of Columbia") == "district_of_columbia"


def test_lccn_from_article_id():
    # American Stories ids embed the LCCN of the source title.
    assert m.lccn_from_article_id("sn83030214_1940-01-02_p1_a3") == "sn83030214"
    assert m.lccn_from_article_id("2012271201-1950-06-05-seq1-1") == "2012271201"
    assert m.lccn_from_article_id("no-lccn-here") is None


def test_build_and_resolve_state_table():
    records = [
        {"lccn": "sn83030214", "state": "New York"},
        {"lccn": "sn84020000", "state": "CA"},
        {"lccn": "sn00000001", "state": "Freedonia"},  # unknown -> dropped
    ]
    table = m.build_lccn_state_table(records)
    assert table == {"sn83030214": "New York", "sn84020000": "California"}
    assert m.resolve_state("sn83030214", table) == "New York"
    assert m.resolve_state("sn99999999", table) is None


def test_table_roundtrip(tmp_path):
    table = {"sn83030214": "New York"}
    p = tmp_path / "lccn_state.json"
    m.save_lccn_state_table(table, str(p))
    assert m.load_lccn_state_table(str(p)) == table
