"""
Tests for the cross-corpus (institutional vs public discourse) RND figures:
scripts.visualize._decade_start_year, plot_cross_corpus_category_trend,
and the cross_corpus entry point.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")


def test_decade_start_year_parses_windows_and_decades():
    from scripts.visualize import _decade_start_year

    assert _decade_start_year("1940_1949") == 1940
    assert _decade_start_year("1990s") == 1990
    assert _decade_start_year("北京") is None
    assert _decade_start_year("北京_2020") == 0 or _decade_start_year("北京_2020") is None
