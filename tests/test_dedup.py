# tests/test_dedup.py
from scripts.data_prep.dedup import Deduper, normalize_for_hash


def test_normalize_for_hash():
    assert normalize_for_hash("The  QUICK, brown fox!") == "the quick brown fox"


def test_exact_dedup_catches_identical_only():
    d = Deduper(method="exact")
    a = "Senate passes the new farm bill today in Washington."
    assert d.is_duplicate(a) is False          # first sighting
    assert d.is_duplicate(a) is True           # exact repeat
    assert d.is_duplicate(a + " Extra clause.") is False  # exact = not a dup


def test_shingle_dedup_catches_near_duplicate_wire_copy():
    d = Deduper(method="shingle", shingle_k=4, n_perm=64, bands=16)
    base = ("washington the senate approved a sweeping new farm bill on tuesday "
            "sending the measure to the house for final consideration next week")
    near = ("washington the senate approved a sweeping new farm bill on tuesday "
            "sending the measure to the house for a final vote next week")  # minor edit
    far = ("local high school students won the regional science fair with a "
           "project on solar powered water purification for rural communities")
    assert d.is_duplicate(base) is False
    assert d.is_duplicate(near) is True   # near-dup wire copy -> caught
    assert d.is_duplicate(far) is False   # unrelated -> kept


def test_shingle_dedup_records_bands_even_when_flagged_duplicate():
    # Regression test: a drift chain doc1 -> doc2 (near-dup of doc1) ->
    # doc3 (near-dup of doc2, drifted from doc1) must have doc3 caught
    # because doc2's LSH bands are recorded even though doc2 itself was
    # reported as a duplicate ("records the text either way").
    d = Deduper(method="shingle", shingle_k=4, n_perm=64, bands=16)
    doc1 = ("washington the senate approved a sweeping new farm bill on tuesday "
            "sending the measure to the house for final consideration before "
            "the end of next month")
    doc2 = ("washington the senate approved a sweeping new farm bill on wednesday "
            "sending the measure to the house for final consideration before "
            "the end of next month")  # single-token edit of doc1 (tuesday -> wednesday)
    doc3 = ("washington the senate approved a sweeping new farm bill on wednesday "
            "sending the measure to the house for final review before "
            "the end of next month")  # single-token edit of doc2 (consideration -> review)
    assert d.is_duplicate(doc1) is False
    assert d.is_duplicate(doc2) is True
    assert d.is_duplicate(doc3) is True


def test_reset_clears_state():
    d = Deduper(method="exact")
    a = "same story"
    assert d.is_duplicate(a) is False
    d.reset()
    assert d.is_duplicate(a) is False  # after reset, first sighting again
