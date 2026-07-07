"""Smoke tests for scripts.visualize_word_drivers figure writers."""

from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.analyze_word_drivers import build_long_table, build_summary_table


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _tables():
    data = {
        ("science", "a"): {1990: -0.4, 2000: -0.2, 2010: 0.0},
        ("science", "b"): {1990: 0.1, 2000: 0.2, 2010: 0.5},
        ("family", "d"): {1990: 0.2, 2000: 0.1, 2010: -0.1},
    }
    rows = []
    for (cat, occ), yv in data.items():
        for yr, v in yv.items():
            rows.append({"unit_name": f"{yr}s", "category": cat,
                         "occupation": occ, "rnd": v, "in_vocab": True})
    rnd_long = pd.DataFrame(rows)
    long_df = build_long_table(rnd_long, {"science": 1, "family": -1}, _Log())
    summary_df = build_summary_table(long_df, _Log())
    return long_df, summary_df


def test_all_four_forms_write_a_file(tmp_path):
    import scripts.visualize_word_drivers as vwd

    long_df, summary_df = _tables()
    figs = tmp_path / "figs"
    figs.mkdir()

    vwd.plot_contribution(summary_df, "science", figs, 20, _Log())
    vwd.plot_slope(summary_df, "science", figs, 20, _Log())
    vwd.plot_heatmap(long_df, summary_df, "science", figs, _Log())
    vwd.plot_trajectory(long_df, summary_df, "science", figs, 20, _Log())

    for form in ("contribution", "slope", "heatmap", "trajectory"):
        assert (figs / f"word_drivers_{form}_science.pdf").exists()
