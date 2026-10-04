import numpy as np

from scripts.common.metrics import entry_label, entry_vector, l2_normalize


class _KV:
    def __init__(self, vecs):
        self._v = {k: np.asarray(v, dtype=float) for k, v in vecs.items()}
        self.key_to_index = {k: i for i, k in enumerate(self._v)}

    def __getitem__(self, k):
        return self._v[k]


def test_entry_label_is_first_form():
    assert entry_label("nurse|nurses") == "nurse"
    assert entry_label("police") == "police"


def test_single_form_entry_is_normalized_vector():
    kv = _KV({"nurse": [3.0, 4.0]})
    assert np.allclose(entry_vector(kv, "nurse"), [0.6, 0.8])


def test_variant_entry_averages_in_vocab_forms():
    kv = _KV({"nurse": [1.0, 0.0], "nurses": [0.0, 2.0]})
    expected = l2_normalize(np.array([0.5, 0.5]))
    assert np.allclose(entry_vector(kv, "nurse|nurses"), expected)


def test_variant_entry_uses_whichever_form_exists():
    kv = _KV({"plumbers": [0.0, 5.0]})
    assert np.allclose(entry_vector(kv, "plumber|plumbers"), [0.0, 1.0])


def test_entry_absent_when_no_form_in_vocab():
    assert entry_vector(_KV({"x": [1.0]}), "plumber|plumbers") is None
