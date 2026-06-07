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
    assert _decade_start_year("北京_2020") is None


def _make_cross_summary(
    sources: Iterable[str],
    units: Iterable[str],
    categories: Iterable[str],
) -> pd.DataFrame:
    """Two-corpus long summary with the columns cross_corpus consumes."""
    rows = []
    for src_i, src in enumerate(sources):
        for u in units:
            for c_i, c in enumerate(categories):
                base = 0.5 - 0.1 * c_i + 0.05 * src_i
                rows.append({
                    "source": src,
                    "unit_name": u,
                    "category": c,
                    "mean_rnd": base,
                    "ci_low": base - 0.2,
                    "ci_high": base + 0.2,
                    "sub_low": base - 0.3,
                    "sub_high": base + 0.3,
                })
    return pd.DataFrame(rows)


def _pdfs(d: Path) -> List[str]:
    return sorted(p.name for p in d.glob("*.pdf"))


def test_cross_corpus_absolute_writes_one_pdf(tmp_path):
    from scripts.visualize import plot_cross_corpus_category_trend

    df = _make_cross_summary(
        sources=["renminribao", "ngram"],
        units=["1990_1999", "1995_2004", "2000_2009"],
        categories=["family"],
    )
    plot_cross_corpus_category_trend(
        df, tmp_path, logging.getLogger("test"),
        categories=["family"],
        source_labels={"renminribao": "People's Daily", "ngram": "Google Ngram"},
        band_cols=("ci_low", "ci_high"), band_tag="bootstrap",
        category_sign={"family": -1}, normalize_to=None,
        fig_stem="fig_crosscorpus_family",
    )
    assert _pdfs(tmp_path) == ["fig_crosscorpus_family__bootstrap.pdf"]


def test_cross_corpus_normalized_filename_has_rel2000(tmp_path):
    from scripts.visualize import plot_cross_corpus_category_trend

    df = _make_cross_summary(
        sources=["renminribao", "ngram"],
        units=["1990_1999", "1995_2004", "2000_2009"],
        categories=["leadership", "science"],
    )
    plot_cross_corpus_category_trend(
        df, tmp_path, logging.getLogger("test"),
        categories=["leadership", "science"],
        source_labels={"renminribao": "People's Daily", "ngram": "Google Ngram"},
        band_cols=("sub_low", "sub_high"), band_tag="subsample",
        category_sign={"leadership": 1, "science": 1},
        normalize_to="1995_2004", fig_stem="fig_crosscorpus_work_science",
    )
    assert _pdfs(tmp_path) == ["fig_crosscorpus_work_science_rel2000__subsample.pdf"]


def test_cross_corpus_missing_baseline_skips_line_not_crash(tmp_path, caplog):
    from scripts.visualize import plot_cross_corpus_category_trend

    # ngram lacks the 1995_2004 baseline window entirely.
    rmrb = _make_cross_summary(["renminribao"], ["1995_2004", "2000_2009"], ["family"])
    ngram = _make_cross_summary(["ngram"], ["2000_2009"], ["family"])
    df = pd.concat([rmrb, ngram], ignore_index=True)

    with caplog.at_level(logging.ERROR):
        plot_cross_corpus_category_trend(
            df, tmp_path, logging.getLogger("test"),
            categories=["family"],
            source_labels={"renminribao": "People's Daily", "ngram": "Google Ngram"},
            normalize_to="1995_2004", fig_stem="fig_crosscorpus_family",
        )
    # RMRB line still drawn -> a PDF exists; ngram line skipped with an error log.
    assert _pdfs(tmp_path) == ["fig_crosscorpus_family_rel2000__bootstrap.pdf"]
    assert any("baseline" in r.message and "ngram" in r.message for r in caplog.records)


def test_cross_corpus_empty_df_writes_nothing(tmp_path):
    from scripts.visualize import plot_cross_corpus_category_trend

    plot_cross_corpus_category_trend(
        pd.DataFrame(), tmp_path, logging.getLogger("test"),
        categories=["family"], source_labels={},
    )
    assert _pdfs(tmp_path) == []


def test_cross_corpus_all_nan_writes_nothing(tmp_path, caplog):
    import numpy as np
    from scripts.visualize import plot_cross_corpus_category_trend

    df = _make_cross_summary(
        sources=["renminribao", "ngram"],
        units=["1990_1999", "1995_2004", "2000_2009"],
        categories=["family"],
    )
    df["mean_rnd"] = np.nan  # every value NaN -> must refuse a blank figure

    with caplog.at_level(logging.ERROR):
        plot_cross_corpus_category_trend(
            df, tmp_path, logging.getLogger("test"),
            categories=["family"],
            source_labels={"renminribao": "People's Daily", "ngram": "Google Ngram"},
            band_cols=("ci_low", "ci_high"), band_tag="bootstrap",
            category_sign={"family": -1}, normalize_to=None,
            fig_stem="fig_crosscorpus_family",
        )
    assert _pdfs(tmp_path) == []
    assert any("NaN" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Task 3: cross_corpus entry point
# ---------------------------------------------------------------------------

def _write_profile(path: Path, *, data_source: str, results_dir: Path, log_dir: Path) -> None:
    import yaml
    # Use embedding.format=histwords (a pretrained format) so load_config does not
    # require corpora_dir, raw_ngram_dir, or time_slices — only models_dir is
    # additionally required alongside results_dir/log_dir/base_dir.
    cfg = {
        "language": "zh",
        "data_source": data_source,
        "analysis_mode": "garg_weat",
        "embedding": {"format": "histwords"},
        "paths": {
            "base_dir": str(results_dir.parent),
            "models_dir": str(results_dir.parent / "models"),
            "results_dir": str(results_dir),
            "log_dir": str(log_dir),
        },
        "analysis": {"ideation_sign": {"leadership": 1, "science": 1, "family": -1}},
    }
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))


def test_cross_corpus_emits_eight_pdfs(tmp_path, monkeypatch):
    import scripts.visualize as viz

    # Two corpus result dirs, each with a summary parquet covering the baseline.
    units = ["1990_1999", "1995_2004", "2000_2009", "2005_2014"]
    cats = ["leadership", "family", "science"]
    inst_dir = tmp_path / "results_rmrb"
    pub_dir = tmp_path / "results_ngram"
    for d, src in ((inst_dir, "renminribao"), (pub_dir, "ngram")):
        d.mkdir()
        _make_cross_summary([src], units, cats).drop(columns=["source"]).to_parquet(
            d / "garg_weat_summary_by_category.parquet"
        )

    inst_cfg = tmp_path / "inst.yml"
    pub_cfg = tmp_path / "pub.yml"
    _write_profile(inst_cfg, data_source="renminribao", results_dir=inst_dir, log_dir=tmp_path / "logs")
    _write_profile(pub_cfg, data_source="ngram", results_dir=pub_dir, log_dir=tmp_path / "logs")

    figs = tmp_path / "figs"
    viz.cross_corpus(
        institutional_config=str(inst_cfg),
        public_config=str(pub_cfg),
        figures_dir=str(figs),
        baseline_unit="1995_2004",
    )

    got = _pdfs(figs)
    assert got == sorted([
        "fig_crosscorpus_family__bootstrap.pdf",
        "fig_crosscorpus_family__subsample.pdf",
        "fig_crosscorpus_family_rel2000__bootstrap.pdf",
        "fig_crosscorpus_family_rel2000__subsample.pdf",
        "fig_crosscorpus_work_science__bootstrap.pdf",
        "fig_crosscorpus_work_science__subsample.pdf",
        "fig_crosscorpus_work_science_rel2000__bootstrap.pdf",
        "fig_crosscorpus_work_science_rel2000__subsample.pdf",
    ]), got


def test_cross_corpus_missing_parquet_raises(tmp_path):
    import scripts.visualize as viz

    inst_dir = tmp_path / "results_rmrb"   # never created
    pub_dir = tmp_path / "results_ngram"
    inst_cfg = tmp_path / "inst.yml"
    pub_cfg = tmp_path / "pub.yml"
    _write_profile(inst_cfg, data_source="renminribao", results_dir=inst_dir, log_dir=tmp_path / "logs")
    _write_profile(pub_cfg, data_source="ngram", results_dir=pub_dir, log_dir=tmp_path / "logs")

    with pytest.raises(FileNotFoundError):
        viz.cross_corpus(
            institutional_config=str(inst_cfg), public_config=str(pub_cfg),
            figures_dir=str(tmp_path / "figs"),
        )
