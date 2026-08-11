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


def test_parse_newspapers_txt():
    # Real Chronicling America bulk-list format, verified against the live file
    # 2026-08-11: Newspapers|LCCN|OCLC|ISSN|State|County|City|...
    text = (
        "Newspapers|LCCN|OCLC|ISSN|State|County|City|Geo|Browse|N|First|Last|Essay|Lang|Eth\n"
        "The Abbeville Banner (Abbeville, S.C.) 1847-1869|sn85026945|12795764|2373-1370|"
        "South Carolina|Abbeville|Abbeville|34.18,-82.38|url|254|1847|1869|True|English|\n"
        "Some Paper|sn84020000|123|456|CA|LA|LA|0,0|url|1|x|y|False|English|\n"
        "malformed row with too few columns\n"
    )
    recs = m._parse_newspapers_txt(text)
    assert {"lccn": "sn85026945", "state": "South Carolina"} in recs
    assert {"lccn": "sn84020000", "state": "CA"} in recs
    assert len(recs) == 2  # header + malformed line skipped

    table = m.build_lccn_state_table(recs)
    assert table["sn85026945"] == "South Carolina"
    assert table["sn84020000"] == "California"  # USPS "CA" normalized to canonical


def test_table_roundtrip(tmp_path):
    table = {"sn83030214": "New York"}
    p = tmp_path / "lccn_state.json"
    m.save_lccn_state_table(table, str(p))
    assert m.load_lccn_state_table(str(p)) == table
