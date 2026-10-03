import pytest

from scripts.data_prep.download_dlnews import build_transfer_batch, write_batch_file

COLLECTIONS = {
    "google_newspaper": "/1-Google/1-Newspaper/preprocessed_state",
    "google_tv": "/1-Google/3-TV/preprocessed_state",
}
DEST = "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw"


def test_whole_collection_dirs_map_to_separate_subdirs():
    # Default (no state allow-list): one recursive pair per collection, so every
    # year and state is pulled without guessing per-media filenames, and each
    # media type lands in its own subdir (newspaper and TV are never merged).
    pairs = build_transfer_batch(COLLECTIONS, DEST)
    assert pairs == [
        ("/1-Google/1-Newspaper/preprocessed_state", f"{DEST}/google_newspaper"),
        ("/1-Google/3-TV/preprocessed_state", f"{DEST}/google_tv"),
    ]


def test_state_allow_list_narrows_to_state_dirs():
    pairs = build_transfer_batch(COLLECTIONS, DEST, states=["NY", "AK"])
    assert ("/1-Google/3-TV/preprocessed_state/NY", f"{DEST}/google_tv/NY") in pairs
    assert ("/1-Google/1-Newspaper/preprocessed_state/AK",
            f"{DEST}/google_newspaper/AK") in pairs
    assert len(pairs) == 4


def test_empty_collections_is_an_error():
    with pytest.raises(ValueError):
        build_transfer_batch({}, DEST)


def test_batch_file_lines_are_recursive(tmp_path):
    path = write_batch_file(build_transfer_batch(COLLECTIONS, DEST), str(tmp_path))
    lines = open(path, encoding="utf-8").read().splitlines()
    assert lines[0] == (
        f'--recursive "/1-Google/1-Newspaper/preprocessed_state" "{DEST}/google_newspaper"')
    assert all(line.startswith("--recursive ") for line in lines)
