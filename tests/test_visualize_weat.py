"""
Regression tests for scripts.visualize WEAT plotting.

Locks the parse_year extension that recognises decade-style unit labels
('1990s') in addition to the rolling-window slice format ('1940_1949').
Without this, plot_weat_longitudinal_trend silently early-returns on every
COHA / HistWords run and no longitudinal trend PDF is written.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pytest

# matplotlib needs a non-interactive backend in CI / headless test envs.
os.environ.setdefault("MPLBACKEND", "Agg")


def _make_weat_df(units: Iterable[str], dimensions: Iterable[str]) -> pd.DataFrame:
    rows = []
    for u in units:
        for i, d in enumerate(dimensions):
            rows.append({"unit": u, "dimension": d, "cohens_d": 1.0 - 0.1 * i})
    return pd.DataFrame(rows)


def _pdf_names(d: Path) -> List[str]:
    return sorted(p.name for p in d.glob("*.pdf"))


def test_longitudinal_trend_recognises_decade_units(tmp_path):
    """COHA + HistWords units like '1910s', '1950s', '1990s' must parse."""
    from scripts.visualize import plot_weat_longitudinal_trend

    df = _make_weat_df(
        units=["1910s", "1950s", "1990s"],
        dimensions=["work_family", "leadership", "stem"],
    )
    plot_weat_longitudinal_trend(df, tmp_path, logging.getLogger("test"))

    names = _pdf_names(tmp_path)
    assert any("weat_longitudinal_trend" in n for n in names), (
        f"main trend PDF not written; got {names}"
    )
    # Per-dimension standalone plots should also land for each parsed dim.
    for dim in ("work_family", "leadership", "stem"):
        assert any(f"weat_timeline_{dim}" in n for n in names), (
            f"per-dim timeline PDF missing for {dim}; got {names}"
        )


def test_longitudinal_trend_still_recognises_window_units(tmp_path):
    """Regression: existing '1940_1949'-style ngram window slices keep working."""
    from scripts.visualize import plot_weat_longitudinal_trend

    df = _make_weat_df(
        units=["1940_1949", "1960_1969", "1980_1989"],
        dimensions=["work_family"],
    )
    plot_weat_longitudinal_trend(df, tmp_path, logging.getLogger("test"))

    names = _pdf_names(tmp_path)
    assert any("weat_longitudinal_trend" in n for n in names), (
        f"main trend PDF not written for window-format units; got {names}"
    )


def test_longitudinal_trend_skips_unparseable_units_with_warning(tmp_path, caplog):
    """When no units parse, the function must early-return AND log a warning
    that names the unparseable units (the previous bug was a silent return)."""
    from scripts.visualize import plot_weat_longitudinal_trend

    df = _make_weat_df(
        units=["alpha", "beta", "gamma"],
        dimensions=["work_family"],
    )
    with caplog.at_level(logging.WARNING):
        plot_weat_longitudinal_trend(df, tmp_path, logging.getLogger("test"))

    assert _pdf_names(tmp_path) == [], "no PDFs should be written for unparseable units"
    assert any(
        "plot_weat_longitudinal_trend" in r.message and "skipping" in r.message
        for r in caplog.records
    ), f"expected a skip warning in log; got {[r.message for r in caplog.records]}"
