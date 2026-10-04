import csv
import json
from pathlib import Path

WL = Path("wordlists/en/occupation_family")


def _grounding():
    with open(WL / "occupation_grounding.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_candidates_match_included_grounding_rows():
    rows = _grounding()
    included = [r["word"] + (f"|{r['plural']}" if r["plural"] else "")
                for r in rows if r["include"] == "1"]
    listed = [w.strip() for w in open(WL / "candidates_occupation.txt") if w.strip()]
    assert listed == included


def test_grounding_rows_are_documented():
    for r in _grounding():
        assert r["include"] in ("0", "1"), r
        if r["include"] == "1":
            assert r["census_2018_title"], f"{r['word']}: no census title"
        else:
            assert r["note"], f"{r['word']}: excluded without a reason"


def test_no_list_word_is_a_gender_anchor():
    g = json.load(open(WL / "gender_words.json"))
    anchors = set(g["male"]) | set(g["female"])
    for name in ("candidates_occupation.txt", "candidates_household.txt", "family_sphere.txt"):
        forms = {f for line in open(WL / name) for f in line.strip().split("|") if f}
        assert not forms & anchors, name


def test_occ2010_mapping_covers_candidates_with_valid_codes():
    with open(WL / "occupation_occ2010.csv", encoding="utf-8") as f:
        mapping = {r["word"]: r for r in csv.DictReader(f)}
    with open(WL / "occ2010_codes.csv", encoding="utf-8") as f:
        valid = {int(r["occ2010"]) for r in csv.DictReader(f)}
    included = [r["word"] for r in _grounding() if r["include"] == "1"]
    assert set(mapping) == set(included)
    for word, r in mapping.items():
        assert r["match"] in ("exact", "broad", "multi", "none"), word
        codes = [int(c) for c in r["occ2010"].split(";") if c]
        assert (r["match"] == "none") == (not codes), word
        assert set(codes) <= valid, f"{word}: unknown codes {set(codes) - valid}"
        assert r["match"] != "multi" or len(codes) > 1, word
