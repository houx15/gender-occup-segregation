import numpy as np

from scripts.diagnose_word_senses import nearest_neighbors


class _KV:
    """Minimal KeyedVectors stand-in (other tests replace gensim with a fake)."""

    def __init__(self, vecs):
        self.words = list(vecs)
        self.mat = np.array([vecs[w] for w in self.words], dtype=float)
        self.key_to_index = {w: i for i, w in enumerate(self.words)}

    def __getitem__(self, w):
        return self.mat[self.key_to_index[w]]

    def similar_by_vector(self, vec, topn):
        unit = self.mat / np.linalg.norm(self.mat, axis=1, keepdims=True)
        sims = unit @ (vec / np.linalg.norm(vec))
        order = np.argsort(-sims)[:topn]
        return [(self.words[i], float(sims[i])) for i in order]


def _kv():
    return _KV({"weaver": [1.0, 0.1], "weavers": [0.9, 0.2], "john": [1.0, 0.0],
                "smith": [0.95, 0.05], "loom": [0.0, 1.0], "textile": [0.1, 0.9]})


def test_neighbors_exclude_the_entry_forms():
    out = nearest_neighbors(_kv(), "weaver|weavers", topn=2)
    words = [w for w, _ in out]
    assert "weaver" not in words and "weavers" not in words
    assert set(words) == {"john", "smith"}  # names, not trade words


def test_neighbors_empty_when_out_of_vocab():
    assert nearest_neighbors(_kv(), "plumber|plumbers", topn=3) == []
