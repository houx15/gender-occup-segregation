from scripts.data_prep.download_dlnews import build_transfer_batch


def test_build_transfer_batch_pairs_paths():
    # 3DLNews2 lays files out by 2-letter USPS code:
    #   preprocessed_state/NY/preprocessed_google_newspaper_NY_2000.jsonl.gz
    pairs = build_transfer_batch(
        "/Google/1-Newspapers/preprocessed_state",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw",
        years=[2000, 2020], states=["NY"])
    assert (
        "/Google/1-Newspapers/preprocessed_state/NY/preprocessed_google_newspaper_NY_2000.jsonl.gz",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw/preprocessed_google_newspaper_NY_2000.jsonl.gz",
    ) in pairs
    assert len(pairs) == 2


def test_build_transfer_batch_default_states_are_usps_codes():
    # main() defaults to USPS codes (values of _STATE_NAME_TO_USPS), so a smoke
    # check that the pure function threads 2-letter codes through unchanged.
    pairs = build_transfer_batch(
        "/root", "/dst", years=[1996], states=["AK", "WY"])
    srcs = {src for src, _ in pairs}
    assert "/root/AK/preprocessed_google_newspaper_AK_1996.jsonl.gz" in srcs
    assert "/root/WY/preprocessed_google_newspaper_WY_1996.jsonl.gz" in srcs
    assert len(pairs) == 2
