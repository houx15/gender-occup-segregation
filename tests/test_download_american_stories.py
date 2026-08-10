import logging
import sys
import types

import pytest

from scripts.data_prep import download_american_stories as d


def test_partial_file_removed_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "american_stories_1940.jsonl"

    class _Raising:
        def __iter__(self):
            yield {"article_id": "sn1", "article": "hello world"}
            raise RuntimeError("network died mid-stream")

    fake = types.ModuleType("datasets")
    def load_dataset(*args, **kwargs):
        return {"train": _Raising()}
    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)

    with pytest.raises(RuntimeError):
        d._download_year(1940, str(out), logging.getLogger("t"))
    assert not out.exists()  # partial file cleaned up, not left for silent skip
