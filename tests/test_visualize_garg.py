"""Tests for the Garg (2018) replication visualization branch.

Covers:
  - ``plot_garg_trend`` writes a non-empty PNG for a normal summary DataFrame.
  - ``plot_garg_trend`` handles empty DataFrame by logging a warning and
    returning early without writing the file.
  - ``main`` dispatcher routes ``analysis_mode=="garg"`` to ``plot_garg_trend``.
  - ``main`` dispatcher does NOT call prestige/weat plotters when mode is garg.
  - ``main`` dispatcher logs an ERROR when the summary parquet is missing.

``scripts.visualize`` does not import gensim, so no fake-gensim shim is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary_df() -> pd.DataFrame:
    """Build a 9-row decade summary spanning 1910s..1990s."""
    decades = [f"{d}s" for d in range(1910, 2000, 10)]
    rows = []
    # Synthetic monotonic-ish trajectory: starts strongly male-leaning, drifts
    # toward zero. ci_low/ci_high bracket mean_rnd by +/- 0.05.
    for idx, dec in enumerate(decades):
        mean = 0.20 - 0.02 * idx
        rows.append({
            "unit_name": dec,
            "mean_rnd": mean,
            "ci_low": mean - 0.05,
            "ci_high": mean + 0.05,
            "n_occupations": 50 + idx,
        })
    return pd.DataFrame(rows)


def _write_garg_config(tmp_path: Path, *, with_summary: bool) -> Path:
    """Write a minimal valid coha+garg config and (optionally) a fake summary
    parquet. Returns the config file path."""
    base_dir = tmp_path / "proj"
    (base_dir / "data" / "corpora").mkdir(parents=True)
    (base_dir / "data" / "models").mkdir(parents=True)
    results_dir = base_dir / "data" / "results"
    results_dir.mkdir(parents=True)
    (base_dir / "logs").mkdir(parents=True)
    (base_dir / "data" / "raw_coha").mkdir(parents=True)
    (base_dir / "data" / "coha_dec").mkdir(parents=True)

    config = {
        "language": "en",
        "data_source": "coha",
        "analysis_mode": "garg",
        "paths": {
            "base_dir": str(base_dir),
            "raw_coha_dir": str(base_dir / "data" / "raw_coha"),
            "coha_decompressed_dir": str(base_dir / "data" / "coha_dec"),
            "corpora_dir": str(base_dir / "data" / "corpora"),
            "models_dir": str(base_dir / "data" / "models"),
            "results_dir": str(results_dir),
            "log_dir": str(base_dir / "logs"),
        },
        "coha": {
            "ngram_order": 4,
            "source_archive_urls": ["http://example.com/coha.zip"],
            "decade_min": 1810,
            "decade_max": 2000,
        },
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    if with_summary:
        df = _make_summary_df()
        df.to_parquet(results_dir / "garg_average_bias_by_decade.parquet")

    return config_path


# ---------------------------------------------------------------------------
# plot_garg_trend behavior
# ---------------------------------------------------------------------------

def test_plot_garg_trend_writes_nonempty_png(tmp_path):
    from scripts.visualize import plot_garg_trend

    df = _make_summary_df()
    logger = logging.getLogger("test_plot_garg_trend_writes_nonempty_png")
    plot_garg_trend(df, tmp_path, logger)

    out = tmp_path / "fig2_garg_replication.png"
    assert out.exists(), f"Expected {out} to be created"
    assert out.stat().st_size > 0, "PNG file is empty"


def test_plot_garg_trend_empty_df_logs_warning_and_skips_write(tmp_path, caplog):
    """Empty-DataFrame contract: warn and return; do NOT write a file."""
    from scripts.visualize import plot_garg_trend

    df = pd.DataFrame(
        columns=["unit_name", "mean_rnd", "ci_low", "ci_high", "n_occupations"]
    )
    logger = logging.getLogger("test_plot_garg_trend_empty")
    logger.setLevel(logging.WARNING)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        plot_garg_trend(df, tmp_path, logger)

    out = tmp_path / "fig2_garg_replication.png"
    assert not out.exists(), (
        "Empty DataFrame should not produce a figure; got "
        f"{out} with size {out.stat().st_size if out.exists() else 'N/A'}"
    )
    # At least one WARNING-level record should mention emptiness.
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warned, "Expected at least one WARNING log record"
    assert any("empty" in r.getMessage().lower() for r in warned), (
        "Expected warning to mention 'empty'; got: "
        + "; ".join(r.getMessage() for r in warned)
    )


# ---------------------------------------------------------------------------
# main() dispatcher behavior
# ---------------------------------------------------------------------------

def test_main_dispatcher_routes_garg_to_plot_garg_trend(tmp_path):
    """analysis_mode='garg' -> main() should call plot_garg_trend exactly once
    with the loaded DataFrame."""
    import scripts.visualize as viz

    config_path = _write_garg_config(tmp_path, with_summary=True)

    with patch.object(viz, "plot_garg_trend", new=MagicMock()) as mock_garg:
        viz.main(config=str(config_path))

    assert mock_garg.call_count == 1, (
        f"Expected plot_garg_trend to be called once; got {mock_garg.call_count}"
    )
    # First positional arg should be the loaded DataFrame.
    args, _kwargs = mock_garg.call_args
    assert len(args) >= 1, "plot_garg_trend should receive a DataFrame as first arg"
    df_arg = args[0]
    assert isinstance(df_arg, pd.DataFrame), (
        f"Expected a DataFrame, got {type(df_arg)!r}"
    )
    assert len(df_arg) == 9, f"Expected 9 rows in summary, got {len(df_arg)}"
    assert "unit_name" in df_arg.columns
    assert "mean_rnd" in df_arg.columns


def test_main_dispatcher_does_not_call_prestige_or_weat_plotters_in_garg_mode(tmp_path):
    """In garg mode, the prestige/weat branches must not run."""
    import scripts.visualize as viz

    config_path = _write_garg_config(tmp_path, with_summary=True)

    with patch.object(viz, "plot_garg_trend", new=MagicMock()), \
         patch.object(viz, "plot_prestige_by_gender_over_time", new=MagicMock()) as mock_prest, \
         patch.object(viz, "plot_weat_heatmap", new=MagicMock()) as mock_weat:
        viz.main(config=str(config_path))

    assert mock_prest.call_count == 0, (
        f"plot_prestige_by_gender_over_time should NOT be called; was called "
        f"{mock_prest.call_count} times"
    )
    assert mock_weat.call_count == 0, (
        f"plot_weat_heatmap should NOT be called; was called "
        f"{mock_weat.call_count} times"
    )


def test_main_dispatcher_logs_error_when_garg_summary_missing(tmp_path, caplog):
    """If the parquet is missing, main() should log an ERROR mentioning the
    Garg summary, not crash."""
    import scripts.visualize as viz

    config_path = _write_garg_config(tmp_path, with_summary=False)

    with patch.object(viz, "plot_garg_trend", new=MagicMock()) as mock_garg, \
         caplog.at_level(logging.ERROR):
        viz.main(config=str(config_path))

    assert mock_garg.call_count == 0, (
        "plot_garg_trend should not be called when summary is missing"
    )
    err_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert err_records, "Expected at least one ERROR log record"
    assert any("Garg summary not found" in r.getMessage() for r in err_records), (
        "Expected error message to mention 'Garg summary not found'; got: "
        + "; ".join(r.getMessage() for r in err_records)
    )
