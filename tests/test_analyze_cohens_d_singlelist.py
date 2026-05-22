"""Tests for projection_values (single-list Cohen's d per-word producer)."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

logger = logging.getLogger("test")


# ---------------------------------------------------------------------------
# Install a fake gensim before importing the analyzer.
#
# The host env's gensim is broken (scipy.linalg.triu was removed in newer
# scipy). Real gensim is not needed for these unit tests — the model is a
# StubKV. Install a minimal fake so ``from gensim.models import KeyedVectors``
# inside embedding_utils.py succeeds.  Pattern mirrors all other test files in
# this repo (e.g. test_analyze_garg.py).
# ---------------------------------------------------------------------------


def _install_fake_gensim() -> None:
    if "gensim" in sys.modules and not getattr(sys.modules["gensim"], "_fake", False):
        try:
            from gensim.models import KeyedVectors  # noqa: F401
            return
        except Exception:
            pass
    fake_gensim = types.ModuleType("gensim")
    fake_gensim._fake = True  # type: ignore[attr-defined]
    fake_models = types.ModuleType("gensim.models")
    fake_models.KeyedVectors = object  # type: ignore[attr-defined]
    fake_gensim.models = fake_models  # type: ignore[attr-defined]
    sys.modules["gensim"] = fake_gensim
    sys.modules["gensim.models"] = fake_models


_install_fake_gensim()


class StubKV:
    def __init__(self, vectors):
        self._v = {w: np.asarray(v, dtype=float) for w, v in vectors.items()}
        self.key_to_index = {w: i for i, w in enumerate(self._v)}

    def __getitem__(self, k):
        return self._v[k]

    def __contains__(self, k):
        return k in self._v


# After L2-normalize, c_male=(1,0), c_female=(-1,0); axis female-male = (-1,0).
_MALE = {"he": [1.0, 0.0], "man": [1.0, 0.0]}
_FEMALE = {"she": [-1.0, 0.0], "woman": [-1.0, 0.0]}


def _model():
    return StubKV({
        **_MALE, **_FEMALE,
        "president": [0.6, 0.0], "manager": [0.5, 0.0],   # +x → male-leaning
        "cooking": [-0.5, 0.0], "cleaning": [-0.6, 0.0],  # -x → female-leaning
    })


def _categories():
    return {"leadership": ["president", "manager"], "family": ["cooking", "cleaning"]}


def test_long_schema_and_sign_convention():
    from scripts.analyze_cohens_d_singlelist import projection_values
    gw = {"male": list(_MALE), "female": list(_FEMALE)}
    df = projection_values(_model(), "1990s", _categories(), gw, logger)
    assert set(df.columns) >= {"unit_name", "category", "occupation", "value", "in_vocab"}
    lead = df[df.category == "leadership"]["value"]
    fam = df[df.category == "family"]["value"]
    # axis points male→female (female - male), so male-leaning occupations
    # (+x, near c_male) project NEGATIVE; family (−x) projects POSITIVE.
    assert (lead < 0).all()
    assert (fam > 0).all()


def test_returns_none_when_gender_unobtainable():
    from scripts.analyze_cohens_d_singlelist import projection_values
    model = StubKV({"president": [0.6, 0.0]})  # no gender words
    gw = {"male": list(_MALE), "female": list(_FEMALE)}
    assert projection_values(model, "x", {"leadership": ["president"]}, gw, logger) is None


def test_oov_occupation_marked_not_in_vocab():
    from scripts.analyze_cohens_d_singlelist import projection_values
    gw = {"male": list(_MALE), "female": list(_FEMALE)}
    cats = {"leadership": ["president", "ceo_missing"]}
    df = projection_values(_model(), "1990s", cats, gw, logger)
    missing = df[df.occupation == "ceo_missing"].iloc[0]
    assert missing["in_vocab"] == False  # noqa: E712
    assert np.isnan(missing["value"])
