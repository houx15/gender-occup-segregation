import logging

import numpy as np

from scripts.analyze_state_tagged import analyze_period_model, states_in_vocab, tagged_gender_words


class _KV:
    def __init__(self, vecs):
        self._v = {k: np.asarray(v, dtype=float) for k, v in vecs.items()}
        self.key_to_index = {k: i for i, k in enumerate(self._v)}

    def __getitem__(self, k):
        return self._v[k]


GW = {"male": ["he", "man"], "female": ["she", "woman"]}


def test_states_in_vocab_from_tagged_anchors():
    keys = ["she__ohio", "he__new_york", "nurse", "x__ohio", "man__ohio"]
    assert states_in_vocab(keys, {"he", "she", "man", "woman"}) == {"ohio", "new_york"}


def test_tagged_gender_words():
    assert tagged_gender_words(GW, "ohio") == {"male": ["he__ohio", "man__ohio"],
                                               "female": ["she__ohio", "woman__ohio"]}


def test_analyze_period_model_scores_each_state_against_its_own_anchors():
    kv = _KV({
        "he__ohio": [1, 0], "man__ohio": [1, 0], "she__ohio": [-1, 0], "woman__ohio": [-1, 0],
        # Utah's anchors are rotated: 'nurse' sits near Utah's male pole
        "he__utah": [-1, 0], "man__utah": [-1, 0], "she__utah": [1, 0], "woman__utah": [1, 0],
        "he__iowa": [1, 0],                               # too few anchors -> skipped
        "nurse": [-0.9, 0.1],
    })
    frames, units = analyze_period_model(
        kv, period=2005, gender_words=GW, categories={"occupation": ["nurse|nurses"]},
        metrics=["rnd"], min_anchors=2, allowed_states=None, logger=logging.getLogger("t"))
    assert sorted(units["rnd"]) == ["ohio_2005", "utah_2005"]
    by_unit = {f["unit_name"].iloc[0]: f for f in frames["rnd"]}
    assert by_unit["ohio_2005"]["value"].iloc[0] > 0     # female-leaning in Ohio
    assert by_unit["utah_2005"]["value"].iloc[0] < 0     # male-leaning in Utah
    assert by_unit["ohio_2005"]["occupation"].iloc[0] == "nurse"


def test_allowed_states_filter():
    kv = _KV({"he__ohio": [1, 0], "she__ohio": [-1, 0], "he__utah": [1, 0],
              "she__utah": [-1, 0], "nurse": [0, 1]})
    _, units = analyze_period_model(
        kv, period=2005, gender_words={"male": ["he"], "female": ["she"]},
        categories={"occupation": ["nurse"]}, metrics=["rnd"], min_anchors=1,
        allowed_states={"utah"}, logger=logging.getLogger("t"))
    assert units["rnd"] == ["utah_2005"]
