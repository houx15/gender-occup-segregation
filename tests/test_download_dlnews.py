from scripts.data_prep.download_dlnews import build_transfer_batch


def test_build_transfer_batch_pairs_paths():
    pairs = build_transfer_batch(
        "/3dlnews2/Google/1-Newspapers/preprocessed_state",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw",
        years=[2000, 2020], states=["New York"])
    assert (
        "/3dlnews2/Google/1-Newspapers/preprocessed_state/New York/preprocessed_google_newspaper_New York_2000.jsonl.gz",
        "/scratch/network/yh6580/gender-occup/data/american_state/dlnews/raw/preprocessed_google_newspaper_New York_2000.jsonl.gz",
    ) in pairs
    assert len(pairs) == 2
