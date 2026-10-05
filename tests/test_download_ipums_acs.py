import pytest

from scripts.data_prep.download_ipums_acs import extract_body, read_api_key


def test_extract_body_samples_variables_and_case_selection():
    body = extract_body(years=[2005, 2024], variables=["YEAR", "SEX", "EMPSTAT", "OCC2010"],
                        case_selections={"EMPSTAT": ["1"]}, description="test")
    assert body["samples"] == {"us2005a": {}, "us2024a": {}}
    assert body["variables"]["EMPSTAT"] == {"caseSelections": {"general": ["1"]}}
    assert body["variables"]["OCC2010"] == {}
    assert body["dataFormat"] == "csv"
    assert body["dataStructure"] == {"rectangular": {"on": "P"}}


def test_read_api_key_prefers_env(tmp_path):
    f = tmp_path / "api_key"
    f.write_text("from-file\n")
    assert read_api_key({"IPUMS_API_KEY": "from-env"}, f) == "from-env"
    assert read_api_key({}, f) == "from-file"


def test_read_api_key_missing_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="IPUMS_API_KEY"):
        read_api_key({}, tmp_path / "nope")


def test_extract_body_attached_spouse_characteristics():
    body = extract_body(years=[2005], variables=["SEX", "INCWAGE", "AGE"],
                        case_selections={"AGE": ["025", "026"]}, description="f",
                        attached={"INCWAGE": ["spouse"]})
    assert body["variables"]["INCWAGE"] == {"attachedCharacteristics": ["spouse"]}
    assert body["variables"]["AGE"] == {"caseSelections": {"general": ["025", "026"]}}


def test_extract_body_atus_samples_and_time_use_variables():
    body = extract_body(years=[2005, 2024], variables=["SEX", "STATEFIP"], case_selections={},
                        description="atus", sample_template="at{year}",
                        time_use_variables=["BLS_HHACT", "BLS_CAREHH"],
                        sample_members={"includeNonRespondents": False})
    assert body["samples"] == {"at2005": {}, "at2024": {}}
    assert body["timeUseVariables"] == {"BLS_HHACT": {}, "BLS_CAREHH": {}}
    assert body["sampleMembers"] == {"includeNonRespondents": False}


def test_extract_body_usa_default_has_no_time_use_block():
    body = extract_body(years=[2005], variables=["SEX"], case_selections={}, description="x")
    assert "timeUseVariables" not in body and "sampleMembers" not in body


def test_extract_body_explicit_samples_override_years():
    body = extract_body(years=[], variables=["SEX"], case_selections={}, description="early",
                        samples=["us2000a", "us2001a"])
    assert body["samples"] == {"us2000a": {}, "us2001a": {}}
