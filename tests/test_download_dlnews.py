from scripts.data_prep.download_dlnews import build_transfer_batch


def test_build_transfer_batch_pairs_paths():
    # 3DLNews2 real layout (verified against the live collection): state dirs are
    # 2-letter USPS codes; files are preprocessed_newspaper_articles_{USPS}_{YEAR}.
    pairs = build_transfer_batch(
        "/1-Google/1-Newspaper/preprocessed_state",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw",
        years=[2000, 2020], states=["NY"])
    assert (
        "/1-Google/1-Newspaper/preprocessed_state/NY/preprocessed_newspaper_articles_NY_2000.jsonl.gz",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw/preprocessed_newspaper_articles_NY_2000.jsonl.gz",
    ) in pairs
    assert len(pairs) == 2


def test_build_transfer_batch_default_states_are_usps_codes():
    # main() defaults to USPS codes (values of _STATE_NAME_TO_USPS), so a smoke
    # check that the pure function threads 2-letter codes through unchanged.
    pairs = build_transfer_batch(
        "/root", "/dst", years=[1996], states=["AK", "WY"])
    srcs = {src for src, _ in pairs}
    assert "/root/AK/preprocessed_newspaper_articles_AK_1996.jsonl.gz" in srcs
    assert "/root/WY/preprocessed_newspaper_articles_WY_1996.jsonl.gz" in srcs
    assert len(pairs) == 2
