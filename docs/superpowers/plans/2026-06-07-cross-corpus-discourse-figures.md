# Cross-corpus discourse figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-corpus visualization that overlays institutional (People's Daily) and public (china_ngram) RND category trends in two figures (Family; Work & Science), each in absolute and Δ-from-2000 variants, with bootstrap and subsample bands — 8 PDFs total.

**Architecture:** Pure visualization. A new module-level helper lifts the existing nested year parser; a new plotting function (`plot_cross_corpus_category_trend`) overlays N categories across 2 corpora (color = category, linestyle = corpus); a new Fire entry point (`cross_corpus`) loads the two existing profiles' summary parquets, tags rows with `source`, and emits the 8 figures. A slurm script renders on the cluster.

**Tech Stack:** Python, pandas, matplotlib (Agg backend), Fire CLI, pytest, Slurm.

**Spec:** `docs/superpowers/specs/2026-06-07-cross-corpus-discourse-figures-design.md`

**Filename note:** The sibling `plot_garg_weat_categories_trend` writes **undated** filenames (no `YYYYMMDD_` prefix). For consistency the new function also writes undated names. (Spec mentioned dating; this plan supersedes that detail to match the closest existing convention.)

---

## File Structure

- `scripts/visualize.py` — add `_decade_start_year` (lifted helper), `plot_cross_corpus_category_trend`, `cross_corpus`; register `cross_corpus` in the Fire dict; add `from matplotlib.lines import Line2D` import.
- `tests/test_visualize_cross_corpus.py` — new test file (mirrors `tests/test_visualize_garg_weat.py` conventions).
- `slurm/garg_weat_cross_corpus_zh.slurm` — new slurm script (mirrors `slurm/garg_weat_zh.slurm`).

---

## Task 1: Lift the year-parser to module level (refactor)

The year parser is currently nested inside `plot_garg_weat_categories_trend` (`scripts/visualize.py:468`). Lift it to a module-level `_decade_start_year` so the new function can reuse it. Pure refactor — behavior unchanged.

**Files:**
- Modify: `scripts/visualize.py` (add helper near `apply_ideation_sign` at line ~419; replace nested def + call inside `plot_garg_weat_categories_trend`)
- Test: `tests/test_visualize_cross_corpus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualize_cross_corpus.py` with:

```python
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
```

Note on the last assertion: `"北京_2020".split("_")[0]` is `"北京"`, which `int()` rejects → `None`. The `or` keeps the test robust if someone later makes it numeric-aware; the real contract is "province units don't become a real year".

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_cross_corpus.py::test_decade_start_year_parses_windows_and_decades -v`
Expected: FAIL with `ImportError: cannot import name '_decade_start_year'`.

- [ ] **Step 3: Add the module-level helper**

In `scripts/visualize.py`, immediately **after** `apply_ideation_sign` (ends at line ~418) and **before** `def plot_garg_weat_categories_trend`, insert:

```python
def _decade_start_year(unit_name):
    """Start year from a longitudinal unit name. Handles decade labels
    ('1990s' -> 1990, COHA/HistWords) and rolling-window slices
    ('1940_1949' -> 1940, the ngram / renminribao pipelines). Province and
    province-year units ('北京', '北京_2020') don't parse to a real year and
    return None — those route to the provincial RND plots instead."""
    s = str(unit_name)
    if len(s) == 5 and s.endswith("s") and s[:4].isdigit():
        return int(s[:4])
    try:
        return int(s.split("_")[0])
    except (ValueError, IndexError):
        return None
```

- [ ] **Step 4: Replace the nested def with a call to the helper**

In `plot_garg_weat_categories_trend`, delete the nested `def _parse_decade(unit_name): ...` block (lines ~468–484) and change the line that uses it (line ~486) from:

```python
    df["start_year"] = df["unit_name"].apply(_parse_decade)
```

to:

```python
    df["start_year"] = df["unit_name"].apply(_decade_start_year)
```

- [ ] **Step 5: Run the new test and the existing garg_weat regression to verify no breakage**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_cross_corpus.py::test_decade_start_year_parses_windows_and_decades tests/test_visualize_garg_weat.py -v`
Expected: all PASS (the refactor preserves `plot_garg_weat_categories_trend` behavior).

- [ ] **Step 6: Commit**

```bash
git add scripts/visualize.py tests/test_visualize_cross_corpus.py
git commit -m "refactor(visualize): lift _decade_start_year to module level"
```

---

## Task 2: `plot_cross_corpus_category_trend` — the overlay plot

**Files:**
- Modify: `scripts/visualize.py` (add `from matplotlib.lines import Line2D` to the imports at top, ~line 16; add the function after `plot_garg_weat_categories_trend`, ~line 573)
- Test: `tests/test_visualize_cross_corpus.py`

- [ ] **Step 1: Add the import**

In `scripts/visualize.py` top imports (after `import matplotlib.pyplot as plt`, ~line 14), add:

```python
from matplotlib.lines import Line2D
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_visualize_cross_corpus.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_cross_corpus.py -k cross_corpus -v`
Expected: FAIL with `ImportError: cannot import name 'plot_cross_corpus_category_trend'`.

- [ ] **Step 4: Implement the function**

In `scripts/visualize.py`, after `plot_garg_weat_categories_trend` (ends ~line 573, before the `# ===== WEAT mode plots` banner), add:

```python
# Category colours/markers shared with plot_garg_weat_categories_trend.
_CROSS_CATEGORY_PALETTE = {
    "leadership": "#1f4e79",
    "family":     "#c0392b",
    "science":    "#2e7d32",
}
_CROSS_CATEGORY_MARKERS = {"leadership": "o", "family": "s", "science": "^"}
# Per-source line styling: institutional solid/filled, public dashed/open.
_CROSS_STYLE_CYCLE = [
    {"linestyle": "-",  "fillstyle": "full"},
    {"linestyle": "--", "fillstyle": "none"},
    {"linestyle": ":",  "fillstyle": "full"},
]


def plot_cross_corpus_category_trend(
    df, figures_dir, logger, *,
    categories,
    source_labels,
    source_styles=None,
    band_cols=("ci_low", "ci_high"),
    band_tag="bootstrap",
    band_label=None,
    line_col="mean_rnd",
    category_sign=None,
    normalize_to=None,
    fig_stem="fig_crosscorpus",
    ylabel=None,
):
    """Overlay one or more categories across two corpora on one axis.

    Encoding: colour = category, linestyle/marker-fill = source. With
    ``normalize_to`` set, each (source, category) series is re-expressed as
    ``line_col`` minus its value at that baseline unit (a *difference*, not a
    ratio — RND crosses zero), and the band is shifted by the same constant.
    A series missing the baseline is logged at ERROR and skipped (no silent
    neighbour substitution). Empty / no-category / nothing-plotted cases log
    and refuse to write a blank PDF, mirroring plot_garg_weat_categories_trend.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rel_tag = "_rel2000" if normalize_to else ""
    tag = f"__{band_tag}" if band_tag else ""
    out_path = figures_dir / f"{fig_stem}{rel_tag}{tag}.pdf"

    if df is None or df.empty:
        logger.warning(
            f"plot_cross_corpus_category_trend: empty DataFrame; skip {out_path}"
        )
        return

    df = df[df["category"].isin(categories)].copy()
    if df.empty:
        logger.warning(
            "plot_cross_corpus_category_trend: no rows for "
            f"categories={categories}; skip {out_path}"
        )
        return

    low_col, high_col = band_cols
    df = apply_ideation_sign(df, category_sign, [line_col, low_col, high_col])
    df["start_year"] = df["unit_name"].apply(_decade_start_year)
    df = df.dropna(subset=["start_year"]).sort_values("start_year")
    if df.empty:
        logger.warning(
            "plot_cross_corpus_category_trend: no unit_name parsed to a year; skip "
            f"{out_path}"
        )
        return

    reversed_cats = (
        [c for c, s in category_sign.items() if s < 0] if category_sign else []
    )
    source_styles = source_styles or {}
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    sources = sorted(df["source"].unique())

    def _style_for(source, idx):
        return source_styles.get(source, _CROSS_STYLE_CYCLE[idx % len(_CROSS_STYLE_CYCLE)])

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0
    for s_i, source in enumerate(sources):
        sstyle = _style_for(source, s_i)
        sdf = df[df["source"] == source]
        for c_i, cat in enumerate(categories):
            g = sdf[sdf["category"] == cat].sort_values("start_year")
            if g.empty:
                continue
            color = _CROSS_CATEGORY_PALETTE.get(cat) or default_cycle[c_i % len(default_cycle)]
            marker = _CROSS_CATEGORY_MARKERS.get(cat, "o")
            y = g[line_col]
            lo = g[low_col] if low_col in g.columns else None
            hi = g[high_col] if high_col in g.columns else None
            if normalize_to is not None:
                base = g.loc[g["unit_name"] == normalize_to, line_col]
                if base.empty or pd.isna(base.iloc[0]):
                    logger.error(
                        "plot_cross_corpus_category_trend: baseline "
                        f"'{normalize_to}' missing for source={source} "
                        f"category={cat}; skipping that line."
                    )
                    continue
                b = base.iloc[0]
                y = y - b
                lo = None if lo is None else lo - b
                hi = None if hi is None else hi - b
            ax.plot(
                g["start_year"], y, marker=marker, color=color,
                linestyle=sstyle["linestyle"], fillstyle=sstyle.get("fillstyle", "full"),
                linewidth=1.8,
            )
            if lo is not None and hi is not None:
                ax.fill_between(g["start_year"], lo, hi, color=color, alpha=0.12)
            plotted += 1

    if plotted == 0:
        logger.error(
            "plot_cross_corpus_category_trend: nothing plotted (all baselines "
            f"missing or empty); refusing blank figure {out_path}."
        )
        plt.close()
        return

    ax.axhline(0, color="lightgrey", linestyle="--", linewidth=1)
    ax.set_xlabel("Decade")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif normalize_to is not None:
        ax.set_ylabel(f"Δ oriented RND vs {normalize_to}\n(0 = {normalize_to} baseline)")
    elif reversed_cats:
        ax.set_ylabel("Gender ideation (oriented RND)\nhigher = less traditional")
    else:
        ax.set_ylabel("Mean RND\nlarger = more female-leaning")

    cat_handles = [
        Line2D(
            [0], [0],
            color=_CROSS_CATEGORY_PALETTE.get(c, default_cycle[i % len(default_cycle)]),
            marker=_CROSS_CATEGORY_MARKERS.get(c, "o"), linestyle="-",
            label=c.title() + (" (rev.)" if c in reversed_cats else ""),
        )
        for i, c in enumerate(categories)
    ]
    src_handles = [
        Line2D(
            [0], [0], color="grey", linestyle=_style_for(s, i)["linestyle"],
            label=source_labels.get(s, s),
        )
        for i, s in enumerate(sources)
    ]
    leg1 = ax.legend(handles=cat_handles, title="Category", loc="upper left", framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=src_handles, title="Discourse source", loc="upper right", framealpha=0.85)

    band_note = f"  [{band_label}]" if band_label else ""
    rel_note = f"  (Δ vs {normalize_to})" if normalize_to else ""
    topic = " + ".join(c.title() for c in categories)
    ax.set_title(f"{topic} — institutional vs public discourse{rel_note}{band_note}")
    if normalize_to is None:
        fig.text(
            0.5, -0.02,
            "Absolute RND levels are not directly comparable across corpora "
            "(embedding-geometry artifact); read direction and shape, not the gap.",
            ha="center", fontsize=8, color="grey",
        )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cross-corpus figure: {out_path}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_cross_corpus.py -k cross_corpus -v`
Expected: 4 PASS (`absolute_writes_one_pdf`, `normalized_filename_has_rel2000`, `missing_baseline_skips_line_not_crash`, `empty_df_writes_nothing`).

- [ ] **Step 6: Commit**

```bash
git add scripts/visualize.py tests/test_visualize_cross_corpus.py
git commit -m "feat(visualize): add plot_cross_corpus_category_trend overlay"
```

---

## Task 3: `cross_corpus` Fire entry point

**Files:**
- Modify: `scripts/visualize.py` (add `cross_corpus` after `main`, ~line 2799; register in the Fire dict at line ~2802)
- Test: `tests/test_visualize_cross_corpus.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_visualize_cross_corpus.py`:

```python
def _write_profile(path: Path, *, data_source: str, results_dir: Path, log_dir: Path) -> None:
    import yaml
    cfg = {
        "language": "zh",
        "data_source": data_source,
        "analysis_mode": "garg_weat",
        "paths": {
            "base_dir": str(results_dir.parent),
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
```

(`load_config` must accept these minimal profiles. If it rejects them for missing required keys, the test's `_write_profile` is the place to add those keys — check `scripts/common/config_loader.py:load_config` before implementing and extend the dict to whatever it validates. The four keys above — language, data_source, analysis_mode, paths — match what `main` reads.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_cross_corpus.py -k "eight_pdfs or missing_parquet" -v`
Expected: FAIL with `AttributeError: module 'scripts.visualize' has no attribute 'cross_corpus'`.

- [ ] **Step 3: Implement `cross_corpus`**

In `scripts/visualize.py`, immediately **before** `if __name__ == "__main__":` (line ~2801), add:

```python
def cross_corpus(
    institutional_config="config/profiles/garg_weat_renminribao.yml",
    public_config="config/profiles/garg_weat_china_ngram.yml",
    figures_dir=None,
    baseline_unit="1995_2004",
):
    """Overlay institutional (People's Daily) vs public (china_ngram) RND
    category trends.

    Loads each profile's ``garg_weat_summary_by_category.parquet``, tags rows
    with ``source`` (= the profile's data_source), and writes 8 figures:
    {family, work&science} × {absolute, Δ-vs-baseline} × {bootstrap, subsample}.

    Args:
        institutional_config: profile for the 制度话语 corpus (People's Daily).
        public_config: profile for the 公众话语 corpus (china_ngram).
        figures_dir: output dir; defaults to a sibling of the institutional
            results_dir, ``figures_garg_weat_cross_corpus_zh``.
        baseline_unit: window the Δ figures index to (default ``1995_2004``).
    """
    inst = load_config(institutional_config)
    pub = load_config(public_config)
    logger = setup_logging(Path(inst["paths"]["log_dir"]), "visualize_cross_corpus.log")
    logger.info("=" * 80)
    logger.info("Cross-corpus discourse figures: %s vs %s",
                inst["data_source"], pub["data_source"])
    sns.set_style("whitegrid")
    _configure_fonts(inst)  # both profiles are zh; register the CJK font once

    frames = []
    for cfg in (inst, pub):
        parquet = Path(cfg["paths"]["results_dir"]) / "garg_weat_summary_by_category.parquet"
        if not parquet.exists():
            raise FileNotFoundError(
                f"cross_corpus: summary parquet not found: {parquet}. Run "
                "analyze_category_bias for this corpus first."
            )
        d = pd.read_parquet(parquet)
        d["source"] = cfg["data_source"]
        logger.info("Loaded %s: %d rows (source=%s)", parquet, len(d), cfg["data_source"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    source_labels = {
        cfg["data_source"]: DATA_SOURCE_LABELS.get(cfg["data_source"], cfg["data_source"])
        for cfg in (inst, pub)
    }
    source_styles = {
        inst["data_source"]: {"linestyle": "-",  "fillstyle": "full"},
        pub["data_source"]:  {"linestyle": "--", "fillstyle": "none"},
    }
    category_sign = inst.get("analysis", {}).get("ideation_sign")

    if figures_dir is None:
        figures_dir = Path(inst["paths"]["results_dir"]).parent / "figures_garg_weat_cross_corpus_zh"
    figures_dir = Path(figures_dir)

    topics = [
        (["family"], "fig_crosscorpus_family"),
        (["leadership", "science"], "fig_crosscorpus_work_science"),
    ]
    bands = [
        (("ci_low", "ci_high"), "bootstrap", "bootstrap CI (Garg, 68%)"),
        (("sub_low", "sub_high"), "subsample", "80% word-subsample band"),
    ]
    for cats, stem in topics:
        for bcols, btag, blabel in bands:
            for norm in (None, baseline_unit):
                plot_cross_corpus_category_trend(
                    df, figures_dir, logger,
                    categories=cats, source_labels=source_labels,
                    source_styles=source_styles, band_cols=bcols,
                    band_tag=btag, band_label=blabel, line_col="mean_rnd",
                    category_sign=category_sign, normalize_to=norm, fig_stem=stem,
                )
    logger.info("cross_corpus: wrote figures under %s", figures_dir)
```

- [ ] **Step 4: Register the entry point in the Fire dict**

Change the bottom of `scripts/visualize.py` (line ~2802) from:

```python
    fire.Fire({"main": main, "composite": main_composite})
```

to:

```python
    fire.Fire({"main": main, "composite": main_composite, "cross_corpus": cross_corpus})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_cross_corpus.py -v`
Expected: all tests PASS (8-PDF emission + missing-parquet raise + Task 1/2 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/visualize.py tests/test_visualize_cross_corpus.py
git commit -m "feat(visualize): add cross_corpus entry point (8 discourse figures)"
```

---

## Task 4: Slurm script

**Files:**
- Create: `slurm/garg_weat_cross_corpus_zh.slurm`

- [ ] **Step 1: Read the reference slurm for the exact header + helpers**

Run: `sed -n '1,60p' slurm/garg_weat_zh.slurm`
Note the `#SBATCH` block, the `module load` line, the conda env name (`llm`), and the `read_config_value` helper (a small python one-liner that prints a YAML path). Reuse them verbatim so this script matches the cluster's environment.

- [ ] **Step 2: Create the slurm script**

Create `slurm/garg_weat_cross_corpus_zh.slurm`. Use the **same** `module load` line and conda env discovered in Step 1 (shown here as placeholders `<<MODULE_LOAD_LINE>>` / `<<CONDA_ACTIVATE_LINE>>` — replace with the literal lines copied from `garg_weat_zh.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=garg_weat_cross_corpus_zh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/garg_weat_cross_corpus_zh_%j.out
#SBATCH --error=logs/garg_weat_cross_corpus_zh_%j.err

set -euo pipefail

<<MODULE_LOAD_LINE>>
<<CONDA_ACTIVATE_LINE>>

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

INST_CONFIG="config/profiles/garg_weat_renminribao.yml"
PUB_CONFIG="config/profiles/garg_weat_china_ngram.yml"

read_config_value() {
    # $1 = config path, $2 = python expression over loaded dict `c`
    python - "$1" "$2" <<'PY'
import sys, yaml
with open(sys.argv[1]) as fh:
    c = yaml.safe_load(fh)
print(eval(sys.argv[2]))
PY
}

# Output dir: sibling of the institutional results_dir (where cross_corpus
# defaults), so the validation below looks in the right place.
INST_RESULTS=$(read_config_value "$INST_CONFIG" "c['paths']['results_dir']")
FIG_DIR="$(dirname "$INST_RESULTS")/figures_garg_weat_cross_corpus_zh"

echo "Rendering cross-corpus discourse figures into: $FIG_DIR"

python -m scripts.visualize cross_corpus \
    --institutional_config="$INST_CONFIG" \
    --public_config="$PUB_CONFIG"

# Validation: exactly 8 cross-corpus PDFs.
NPDF=$(find "$FIG_DIR" -maxdepth 1 -name 'fig_crosscorpus_*.pdf' | wc -l | tr -d ' ')
echo "Found $NPDF cross-corpus PDF(s) in $FIG_DIR"
if [ "$NPDF" -ne 8 ]; then
    echo "ERROR: expected 8 cross-corpus PDFs, found $NPDF" >&2
    exit 1
fi
echo "OK: 8 cross-corpus figures rendered."
```

- [ ] **Step 3: Syntax-check the script**

Run: `bash -n slurm/garg_weat_cross_corpus_zh.slurm && echo "syntax ok"`
Expected: `syntax ok`.

- [ ] **Step 4: Commit**

```bash
git add slurm/garg_weat_cross_corpus_zh.slurm
git commit -m "slurm(visualize): cross-corpus discourse figures render job"
```

---

## Task 5: Full-suite regression + wrap-up

- [ ] **Step 1: Run the new test file plus the touched regressions**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_cross_corpus.py tests/test_visualize_garg_weat.py tests/test_visualize_garg.py -v`
Expected: all PASS. If `test_visualize_garg.py` imports nothing changed here, it should be unaffected — its inclusion just guards the shared `apply_ideation_sign` / parser refactor.

- [ ] **Step 2: Confirm the Fire CLI exposes the new command**

Run: `python -m scripts.visualize cross_corpus -- --help`
Expected: Fire prints the `cross_corpus` signature (institutional_config, public_config, figures_dir, baseline_unit) without error.

- [ ] **Step 3 (cluster, manual): submit the job**

Run on the cluster: `sbatch slurm/garg_weat_cross_corpus_zh.slurm`
Then confirm 8 `fig_crosscorpus_*.pdf` files appear under
`<gender-occup>/figures_garg_weat_cross_corpus_zh/` and CJK glyphs render. If the
`china_ngram` summary parquet is absent there (only the *subsampled* build was
run), re-submit with
`--public_config=config/profiles/garg_weat_china_ngram_subsampled.yml` added to
the `python -m scripts.visualize cross_corpus` line.

---

## Self-Review notes

- **Spec coverage:** family fig (Task 3 topics), work&science fig (Task 3 topics), absolute + Δ-from-2000 (Task 2 `normalize_to`, Task 3 loop), bootstrap + subsample → 8 PDFs (Task 3 bands × norms, asserted in `test_cross_corpus_emits_eight_pdfs`), color=category/linestyle=source (Task 2), comparability footnote (Task 2 absolute branch), Δ = difference not ratio (Task 2 subtraction), no-silent-baseline (Task 2 skip + ERROR log, `test_cross_corpus_missing_baseline_skips_line_not_crash`), ideation_sign applied (Task 2 `apply_ideation_sign`), slurm render+validate (Task 4). All covered.
- **Placeholder scan:** the only intentional placeholders are `<<MODULE_LOAD_LINE>>` / `<<CONDA_ACTIVATE_LINE>>` in Task 4, which Step 1 instructs the engineer to copy verbatim from `garg_weat_zh.slurm` (environment-specific lines that must not be guessed).
- **Type/name consistency:** `_decade_start_year`, `plot_cross_corpus_category_trend`, `cross_corpus`, the `source` column, and the 8 filename stems are used identically across tasks and tests.
