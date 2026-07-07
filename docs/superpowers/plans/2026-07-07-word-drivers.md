# Word-level Ideation Drivers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose each Garg-WEAT dimension's plotted gender-ideation line into per-word RND — a year×word "show-your-work" table plus four driver figures per dimension — and wire it non-fatally into the existing garg_weat slurm loops.

**Architecture:** Two separate, config-driven scripts mirroring the repo's `analyze_* → visualize` split. `analyze_word_drivers.py` is pure pandas over the already-written `garg_weat_rnd_long.parquet` (no model loading); it emits two tables. `visualize_word_drivers.py` renders 12 figures (4 forms × 3 dimensions) from those tables, reusing font/path helpers from `scripts/visualize.py`.

**Tech Stack:** Python, pandas, numpy, matplotlib/seaborn, fire CLI, pytest. Same toolchain as the rest of `scripts/`.

## Global Constraints

- **Input file:** `results_dir/garg_weat_rnd_long.parquet`, columns exactly `unit_name, category, occupation, rnd, in_vocab`. Written by `analyze_category_bias` whenever `analysis.metrics` includes `rnd` (true for all four target profiles).
- **Sign convention:** positive RND = female-leaning (Garg). `signed_rnd = rnd × analysis.ideation_sign[category]`; default sign 1 for any category absent from the map. Config `ideation_sign` is `leadership: 1, science: 1, family: -1`.
- **`cat_mean_signed` must reproduce the plotted line:** mean of `signed_rnd` over the same word set the summary/plot uses (per-slice in-vocab, or the global consistent set when `analysis.consistent_occupations: true`).
- **Contribution decomposition:** `contribution = delta / N` over the endpoint-consistent set only, so `Σ contribution` = change in that set's mean.
- **Time-slice parsing:** `1990s → 1990`, `1940_1949 → 1940`; province / province-year units don't parse → dropped (out of scope).
- **Outputs:** parquet **and** CSV for both tables; figures as PDF, one file per `(form × dimension)`.
- **Failure isolation:** the word-driver step is secondary. In slurm it runs only after the primary figures are validated, and its failure logs a WARN without touching status arrays or skipping anything.
- Commit after each task once tests are green; stage only related files; stay on `main`; don't push (per repo convention).
- Run tests with `MPLBACKEND=Agg` (set at test-module top) so matplotlib never needs a display.

---

### Task 1: Long driver table (`build_long_table`)

**Files:**
- Create: `scripts/analyze_word_drivers.py`
- Test: `tests/test_analyze_word_drivers.py`

**Interfaces:**
- Consumes: a `garg_weat_rnd_long`-shaped DataFrame (`unit_name, category, occupation, rnd, in_vocab`).
- Produces:
  - `_slice_start_year(unit_name) -> Optional[int]`
  - `_consistent_words_per_category(df: pd.DataFrame) -> Dict[str, set]`
  - `build_long_table(rnd_long: pd.DataFrame, ideation_sign: Dict[str, int], consistent_only: bool, logger) -> pd.DataFrame` with columns `category, year, unit_name, occupation, rnd, signed_rnd, cat_mean_signed, deviation`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_word_drivers.py`:

```python
"""Tests for scripts.analyze_word_drivers — per-word ideation decomposition."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.analyze_word_drivers import (
    _slice_start_year,
    build_long_table,
    build_summary_table,
)


def _rnd_long_fixture() -> pd.DataFrame:
    """science sign +1, family sign -1. 'c' churns (only middle slice)."""
    data = {
        ("science", "a"): {1990: -0.4, 2000: -0.2, 2010: 0.0},
        ("science", "b"): {1990: 0.1, 2000: 0.2, 2010: 0.5},
        ("science", "c"): {2000: 0.3},   # churn — absent at both endpoints
        ("family", "d"): {1990: 0.2, 2000: 0.1, 2010: -0.1},
    }
    rows = []
    for (cat, occ), yv in data.items():
        for yr, v in yv.items():
            rows.append({
                "unit_name": f"{yr}s", "category": cat,
                "occupation": occ, "rnd": v, "in_vocab": True,
            })
    return pd.DataFrame(rows)


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def test_slice_start_year_formats():
    assert _slice_start_year("1990s") == 1990
    assert _slice_start_year("1940_1949") == 1940
    assert _slice_start_year("北京") is None
    assert _slice_start_year("北京_2020") == 2020  # parses leading int; fine


def test_long_table_cat_mean_and_deviation_and_sign():
    df = build_long_table(
        _rnd_long_fixture(), {"science": 1, "family": -1}, False, _Log()
    )
    # science 1990: in-vocab words a,b -> signed mean = (-0.4 + 0.1)/2 = -0.15
    sci90 = df[(df.category == "science") & (df.year == 1990)]
    assert sci90["cat_mean_signed"].round(6).eq(-0.15).all()
    a90 = sci90[sci90.occupation == "a"].iloc[0]
    assert a90["signed_rnd"] == pytest.approx(-0.4)   # sign +1
    assert a90["deviation"] == pytest.approx(-0.4 - (-0.15))
    # family sign flip: d rnd 0.2 -> signed -0.2
    d90 = df[(df.category == "family") & (df.year == 1990)].iloc[0]
    assert d90["signed_rnd"] == pytest.approx(-0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg pytest tests/test_analyze_word_drivers.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (`scripts.analyze_word_drivers` doesn't exist).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/analyze_word_drivers.py`:

```python
#!/usr/bin/env python3
"""Decompose each Garg-WEAT dimension's plotted ideation line into per-word RND.

Reads the per-word RND long table already written by analyze_category_bias
(garg_weat_rnd_long.parquet) and produces two per-corpus driver tables:

  word_drivers_long.{parquet,csv}     one row per (category, year, word):
      rnd, signed_rnd, cat_mean_signed, deviation
  word_drivers_summary.{parquet,csv}  one row per (category, word):
      first/last year, signed_first/last, delta, contribution, slope, present_both

No model loading — pure pandas over an existing parquet.

Usage:
  python -m scripts.analyze_word_drivers --config=config/profiles/garg_weat_renminribao.yml
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging


def _slice_start_year(unit_name) -> Optional[int]:
    """Start year from a longitudinal unit label. Mirrors
    scripts.visualize._decade_start_year (kept local so this data script needs
    no matplotlib import): '1990s' -> 1990, '1940_1949' -> 1940. Province units
    that don't start with a year return None and are dropped downstream."""
    s = str(unit_name)
    if len(s) == 5 and s.endswith("s") and s[:4].isdigit():
        return int(s[:4])
    try:
        return int(s.split("_")[0])
    except (ValueError, IndexError):
        return None


def _consistent_words_per_category(df: pd.DataFrame) -> Dict[str, set]:
    """Words in vocab in EVERY slice of their category (df already in_vocab-only)."""
    out: Dict[str, set] = {}
    for cat, g in df.groupby("category"):
        n_slices = g["year"].nunique()
        counts = g.groupby("occupation")["year"].nunique()
        out[cat] = set(counts[counts == n_slices].index)
    return out


def build_long_table(
    rnd_long: pd.DataFrame,
    ideation_sign: Dict[str, int],
    consistent_only: bool,
    logger,
) -> pd.DataFrame:
    """One row per (category, year, word): rnd, signed_rnd, cat_mean_signed, deviation.

    cat_mean_signed is the plotted line's value for that (category, slice): the
    mean signed_rnd over the retained word set. When consistent_only is True the
    set is the global consistent set (words in vocab in ALL of the category's
    slices), matching analysis.consistent_occupations; else per-slice in-vocab.
    """
    df = rnd_long[rnd_long["in_vocab"]].copy()
    df["year"] = df["unit_name"].map(_slice_start_year)
    dropped = int(df["year"].isna().sum())
    if dropped:
        logger.info(
            f"  word_drivers: dropping {dropped} rows with non-longitudinal "
            f"unit_name (provincial units are out of scope)"
        )
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df["signed_rnd"] = df["rnd"] * df["category"].map(
        lambda c: ideation_sign.get(c, 1)
    )

    if consistent_only:
        keep = _consistent_words_per_category(df)
        mask = df.apply(
            lambda r: r["occupation"] in keep.get(r["category"], set()), axis=1
        )
        df = df[mask].copy()

    df["cat_mean_signed"] = df.groupby(["category", "year"])["signed_rnd"].transform(
        "mean"
    )
    df["deviation"] = df["signed_rnd"] - df["cat_mean_signed"]
    return (
        df[[
            "category", "year", "unit_name", "occupation",
            "rnd", "signed_rnd", "cat_mean_signed", "deviation",
        ]]
        .sort_values(["category", "year", "occupation"])
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg pytest tests/test_analyze_word_drivers.py::test_slice_start_year_formats tests/test_analyze_word_drivers.py::test_long_table_cat_mean_and_deviation_and_sign -q`
Expected: 2 passed. (`build_summary_table` is imported but not yet exercised — it must exist as a stub or Task 2 must land before the full-file run. To keep this task's file runnable, also add the Task-2 stub now: append `def build_summary_table(long_df, logger): raise NotImplementedError` — Task 2 replaces it. Alternatively run only the two named tests as above, which don't import-time-fail because the symbol exists after Task 2. Use the named-test command here.)

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_word_drivers.py tests/test_analyze_word_drivers.py
git commit -m "feat(word-drivers): long table — signed RND, cat mean, deviation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Summary driver table (`build_summary_table`)

**Files:**
- Modify: `scripts/analyze_word_drivers.py` (add `_ols_slope`, `build_summary_table`)
- Test: `tests/test_analyze_word_drivers.py` (add cases)

**Interfaces:**
- Consumes: `build_long_table` output.
- Produces:
  - `_ols_slope(x: np.ndarray, y: np.ndarray) -> float`
  - `build_summary_table(long_df: pd.DataFrame, logger) -> pd.DataFrame` with columns `category, occupation, first_year, last_year, signed_first, signed_last, delta, contribution, slope, present_both, n_slices`, ranked within category by `|contribution|` (NaN last).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analyze_word_drivers.py`:

```python
def test_summary_contribution_sums_to_delta_of_endpoint_mean():
    long_df = build_long_table(
        _rnd_long_fixture(), {"science": 1, "family": -1}, False, _Log()
    )
    summ = build_summary_table(long_df, _Log())
    sci = summ[summ.category == "science"]
    both = sci[sci.present_both]
    # endpoint-consistent set is {a, b}; 'c' churns
    assert set(both.occupation) == {"a", "b"}
    assert sci[sci.occupation == "c"]["present_both"].iloc[0] == False
    assert np.isnan(sci[sci.occupation == "c"]["contribution"].iloc[0])
    # Σ contribution == mean_both(2010) - mean_both(1990)
    mean_last = (0.0 + 0.5) / 2
    mean_first = (-0.4 + 0.1) / 2
    assert both["contribution"].sum() == pytest.approx(mean_last - mean_first)  # 0.4


def test_summary_slope_is_ols():
    long_df = build_long_table(
        _rnd_long_fixture(), {"science": 1, "family": -1}, False, _Log()
    )
    summ = build_summary_table(long_df, _Log())
    # a: years [1990,2000,2010], signed [-0.4,-0.2,0.0] -> slope 0.02 / yr
    a_slope = summ[(summ.category == "science") & (summ.occupation == "a")]["slope"].iloc[0]
    assert a_slope == pytest.approx(0.02)


def test_consistent_only_drops_churning_word_from_mean():
    long_true = build_long_table(
        _rnd_long_fixture(), {"science": 1, "family": -1}, True, _Log()
    )
    # 'c' removed entirely under consistent_only
    assert "c" not in set(long_true[long_true.category == "science"]["occupation"])
    # science 2000 mean over {a,b} only = (-0.2 + 0.2)/2 = 0.0 (not 0.1 with c)
    sci00 = long_true[(long_true.category == "science") & (long_true.year == 2000)]
    assert sci00["cat_mean_signed"].round(6).eq(0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg pytest tests/test_analyze_word_drivers.py -q`
Expected: FAIL — `build_summary_table` missing / `NotImplementedError`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/analyze_word_drivers.py`, add after `build_long_table`:

```python
def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x; NaN if fewer than 2 distinct x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.unique(x).size < 2:
        return float("nan")
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum() / denom)


def build_summary_table(long_df: pd.DataFrame, logger) -> pd.DataFrame:
    """Per (category, word): endpoint delta, contribution to Δmean, OLS slope.

    Endpoints are per category (min/max year). delta/contribution are defined
    only for words in vocab at BOTH endpoints (present_both); contribution =
    delta / N with N = size of that endpoint-consistent set, so Σ contribution
    equals the change in that set's mean.
    """
    rows = []
    for cat, g in long_df.groupby("category"):
        years = sorted(g["year"].unique())
        first_year, last_year = years[0], years[-1]
        at_first = g[g["year"] == first_year].set_index("occupation")["signed_rnd"]
        at_last = g[g["year"] == last_year].set_index("occupation")["signed_rnd"]
        both = set(at_first.index) & set(at_last.index)
        n = len(both)
        for occ, gg in g.groupby("occupation"):
            present_both = occ in both
            s_first = float(at_first[occ]) if occ in at_first.index else np.nan
            s_last = float(at_last[occ]) if occ in at_last.index else np.nan
            delta = (s_last - s_first) if present_both else np.nan
            contribution = (delta / n) if (present_both and n) else np.nan
            slope = _ols_slope(
                gg["year"].to_numpy(dtype=float),
                gg["signed_rnd"].to_numpy(dtype=float),
            )
            rows.append({
                "category": cat, "occupation": occ,
                "first_year": int(first_year), "last_year": int(last_year),
                "signed_first": s_first, "signed_last": s_last,
                "delta": delta, "contribution": contribution, "slope": slope,
                "present_both": present_both, "n_slices": int(gg["year"].nunique()),
            })
    summary = pd.DataFrame(rows)
    summary["_absc"] = summary["contribution"].abs()
    summary = (
        summary.sort_values(
            ["category", "_absc"], ascending=[True, False], na_position="last"
        )
        .drop(columns="_absc")
        .reset_index(drop=True)
    )
    return summary
```

If a `build_summary_table` stub was added in Task 1, replace it with the above.

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg pytest tests/test_analyze_word_drivers.py -q`
Expected: all tests pass (5+ tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_word_drivers.py tests/test_analyze_word_drivers.py
git commit -m "feat(word-drivers): summary table — delta, contribution, OLS slope

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Analyze CLI (`main`) — read parquet, write both tables ×2 formats

**Files:**
- Modify: `scripts/analyze_word_drivers.py` (add `main`, `fire` entry)
- Test: `tests/test_analyze_word_drivers.py` (add an end-to-end write test)

**Interfaces:**
- Consumes: config path; reads `results_dir/garg_weat_rnd_long.parquet`.
- Produces: `main(config: str = "config/config.yml") -> None`; writes `word_drivers_long.{parquet,csv}` and `word_drivers_summary.{parquet,csv}` to `results_dir`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_analyze_word_drivers.py`:

```python
def test_main_writes_four_files(tmp_path, monkeypatch):
    import scripts.analyze_word_drivers as awd

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _rnd_long_fixture().to_parquet(results_dir / "garg_weat_rnd_long.parquet", index=False)

    cfg = {
        "paths": {"results_dir": str(results_dir), "log_dir": str(tmp_path / "logs")},
        "analysis": {"ideation_sign": {"science": 1, "family": -1}},
    }
    monkeypatch.setattr(awd, "load_config", lambda _p: cfg)
    monkeypatch.setattr(awd, "setup_logging", lambda *_a, **_k: _Log())

    awd.main(config="ignored.yml")

    for name in ("word_drivers_long", "word_drivers_summary"):
        assert (results_dir / f"{name}.parquet").exists()
        assert (results_dir / f"{name}.csv").exists()
    # long CSV round-trips and cat mean reproduces
    long_df = pd.read_parquet(results_dir / "word_drivers_long.parquet")
    assert {"signed_rnd", "cat_mean_signed", "deviation"} <= set(long_df.columns)


def test_main_errors_without_input(tmp_path, monkeypatch):
    import scripts.analyze_word_drivers as awd

    cfg = {"paths": {"results_dir": str(tmp_path), "log_dir": str(tmp_path)},
           "analysis": {}}
    monkeypatch.setattr(awd, "load_config", lambda _p: cfg)
    monkeypatch.setattr(awd, "setup_logging", lambda *_a, **_k: _Log())
    with pytest.raises(FileNotFoundError):
        awd.main(config="ignored.yml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg pytest tests/test_analyze_word_drivers.py::test_main_writes_four_files tests/test_analyze_word_drivers.py::test_main_errors_without_input -q`
Expected: FAIL — `main` not defined.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/analyze_word_drivers.py`:

```python
def main(config: str = "config/config.yml") -> None:
    """Read garg_weat_rnd_long.parquet, write the two driver tables (parquet+csv)."""
    cfg = load_config(config)
    logger = setup_logging(
        Path(cfg["paths"]["log_dir"]), "analyze_word_drivers.log"
    )
    results_dir = Path(cfg["paths"]["results_dir"])
    long_path = results_dir / "garg_weat_rnd_long.parquet"
    if not long_path.exists():
        raise FileNotFoundError(
            f"analyze_word_drivers: {long_path} not found. Run "
            f"analyze_category_bias (analysis.metrics must include 'rnd') first."
        )
    rnd_long = pd.read_parquet(long_path)
    required = {"unit_name", "category", "occupation", "rnd", "in_vocab"}
    missing = required - set(rnd_long.columns)
    if missing:
        raise ValueError(f"{long_path} missing columns: {sorted(missing)}")

    ideation_sign = cfg.get("analysis", {}).get("ideation_sign", {})
    consistent_only = bool(
        cfg.get("analysis", {}).get("consistent_occupations", False)
    )
    logger.info(
        f"word_drivers: {len(rnd_long)} rnd rows; ideation_sign={ideation_sign}; "
        f"consistent_only={consistent_only}"
    )

    long_df = build_long_table(rnd_long, ideation_sign, consistent_only, logger)
    summary_df = build_summary_table(long_df, logger)

    results_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("word_drivers_long", long_df),
        ("word_drivers_summary", summary_df),
    ]:
        frame.to_parquet(results_dir / f"{name}.parquet", index=False)
        frame.to_csv(results_dir / f"{name}.csv", index=False)
        logger.info(
            f"Saved: {results_dir / name}.parquet / .csv ({len(frame)} rows)"
        )


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg pytest tests/test_analyze_word_drivers.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_word_drivers.py tests/test_analyze_word_drivers.py
git commit -m "feat(word-drivers): analyze CLI writes long+summary (parquet+csv)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Figures (`scripts/visualize_word_drivers.py`)

**Files:**
- Create: `scripts/visualize_word_drivers.py`
- Test: `tests/test_visualize_word_drivers.py`

**Interfaces:**
- Consumes: `word_drivers_long` / `word_drivers_summary` DataFrames; `scripts.visualize._configure_fonts`, `scripts.visualize.get_figure_path`.
- Produces: `plot_contribution`, `plot_slope`, `plot_heatmap`, `plot_trajectory` (each `(…, category, figures_dir, [top_n,] logger) -> None`) and `main(config: str = "config/config.yml") -> None`. Writes `word_drivers_{contribution,slope,heatmap,trajectory}_<category>.pdf`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_visualize_word_drivers.py`:

```python
"""Smoke tests for scripts.visualize_word_drivers figure writers."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.analyze_word_drivers import build_long_table, build_summary_table


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _tables():
    # Inline fixture (kept independent of the analyze test module).
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
    long_df = build_long_table(rnd_long, {"science": 1, "family": -1}, False, _Log())
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_word_drivers.py -q`
Expected: FAIL — `scripts.visualize_word_drivers` doesn't exist.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/visualize_word_drivers.py`:

```python
#!/usr/bin/env python3
"""Render word-level ideation-driver figures from word_drivers_* tables.

One PDF per (dimension × form): contribution bars, slope/dumbbell, word×year
heatmap, trajectory small-multiples. Reads the tables produced by
scripts.analyze_word_drivers.

Usage:
  python -m scripts.visualize_word_drivers --config=config/profiles/garg_weat_renminribao.yml
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import fire

from scripts.common.config_loader import load_config
from scripts.common.logging_utils import setup_logging
from scripts.visualize import _configure_fonts, get_figure_path

_NEG = "#c0392b"   # male-leaning / negative change
_POS = "#2c7fb8"   # female-leaning / positive change


def _top_n(cfg: dict) -> int:
    return int(cfg.get("analysis", {}).get("word_drivers", {}).get("top_n", 20))


def plot_contribution(summary_df, category, figures_dir, top_n, logger):
    sub = summary_df[
        (summary_df["category"] == category) & summary_df["present_both"]
    ].dropna(subset=["contribution"]).copy()
    if sub.empty:
        logger.warning(f"  contribution[{category}]: no present_both words — skipped")
        return
    sub["_absc"] = sub["contribution"].abs()
    sub = sub.sort_values("_absc", ascending=False).head(top_n).sort_values("contribution")
    colors = [_NEG if v < 0 else _POS for v in sub["contribution"]]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(sub))))
    ax.barh(sub["occupation"].astype(str), sub["contribution"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    total = float(sub["contribution"].sum())
    ax.set_xlabel("contribution to Δ ideation (signed RND / N)")
    ax.set_title(
        f"{category}: word contributions to change "
        f"({int(sub['first_year'].iloc[0])}→{int(sub['last_year'].iloc[0])}); "
        f"shown Σ={total:+.3f}"
    )
    fig.tight_layout()
    out = get_figure_path(f"word_drivers_contribution_{category}.pdf", figures_dir)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def plot_slope(summary_df, category, figures_dir, top_n, logger):
    sub = summary_df[
        (summary_df["category"] == category) & summary_df["present_both"]
    ].dropna(subset=["delta"]).copy()
    if sub.empty:
        logger.warning(f"  slope[{category}]: no present_both words — skipped")
        return
    sub["_absd"] = sub["delta"].abs()
    sub = sub.sort_values("_absd", ascending=False).head(top_n).sort_values("delta")
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(sub))))
    y = np.arange(len(sub))
    for yi, (_, r) in zip(y, sub.iterrows()):
        color = _POS if r["delta"] >= 0 else _NEG
        ax.plot([r["signed_first"], r["signed_last"]], [yi, yi], color=color, lw=1.5, zorder=1)
        ax.scatter([r["signed_first"]], [yi], color="#bbbbbb", s=30, zorder=2)
        ax.scatter([r["signed_last"]], [yi], color=color, s=30, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["occupation"].astype(str))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("signed RND (start ○ → end ●)")
    ax.set_title(
        f"{category}: per-word change "
        f"{int(sub['first_year'].iloc[0])}→{int(sub['last_year'].iloc[0])}"
    )
    fig.tight_layout()
    out = get_figure_path(f"word_drivers_slope_{category}.pdf", figures_dir)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def plot_heatmap(long_df, summary_df, category, figures_dir, logger):
    sub = long_df[long_df["category"] == category]
    if sub.empty:
        logger.warning(f"  heatmap[{category}]: no rows — skipped")
        return
    pivot = sub.pivot_table(index="occupation", columns="year", values="signed_rnd")
    order = summary_df[summary_df["category"] == category].set_index("occupation")["delta"]
    order = order.sort_values(ascending=False, na_position="last").index
    pivot = pivot.reindex([o for o in order if o in pivot.index]).dropna(how="all")
    if pivot.empty:
        logger.warning(f"  heatmap[{category}]: empty pivot — skipped")
        return
    arr = pivot.to_numpy()
    finite = arr[np.isfinite(arr)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    if vmax == 0.0:
        vmax = 1.0
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.5 * pivot.shape[1]), max(4.0, 0.3 * pivot.shape[0]))
    )
    sns.heatmap(
        pivot, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={"label": "signed RND"}, ax=ax,
    )
    ax.set_title(f"{category}: signed RND by word × year")
    ax.set_xlabel("year")
    ax.set_ylabel("")
    fig.tight_layout()
    out = get_figure_path(f"word_drivers_heatmap_{category}.pdf", figures_dir)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def plot_trajectory(long_df, summary_df, category, figures_dir, top_n, logger):
    sub = long_df[long_df["category"] == category]
    if sub.empty:
        logger.warning(f"  trajectory[{category}]: no rows — skipped")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for _, g in sub.groupby("occupation"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["signed_rnd"], color="#cccccc", lw=0.7, zorder=1)
    movers = summary_df[
        (summary_df["category"] == category) & summary_df["present_both"]
    ].dropna(subset=["delta"]).copy()
    movers["_absd"] = movers["delta"].abs()
    movers = movers.sort_values("_absd", ascending=False).head(top_n)
    cmap = plt.get_cmap("tab10")
    for i, occ in enumerate(movers["occupation"]):
        g = sub[sub["occupation"] == occ].sort_values("year")
        ax.plot(g["year"], g["signed_rnd"], color=cmap(i % 10), lw=1.8, zorder=3, label=str(occ))
        ax.annotate(
            str(occ), (g["year"].iloc[-1], g["signed_rnd"].iloc[-1]),
            fontsize=7, xytext=(3, 0), textcoords="offset points",
        )
    mean_line = sub.groupby("year")["cat_mean_signed"].first().sort_index()
    ax.plot(
        mean_line.index, mean_line.values, color="k", lw=2.2, ls="--",
        zorder=4, label="dimension mean",
    )
    ax.set_xlabel("year")
    ax.set_ylabel("signed RND")
    ax.set_title(f"{category}: per-word trajectories (top {top_n} movers bold)")
    ax.legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    out = get_figure_path(f"word_drivers_trajectory_{category}.pdf", figures_dir)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved {out}")


def main(config: str = "config/config.yml") -> None:
    """Render all four driver figures per dimension from the driver tables."""
    cfg = load_config(config)
    logger = setup_logging(
        Path(cfg["paths"]["log_dir"]), "visualize_word_drivers.log"
    )
    sns.set_style("whitegrid")
    _configure_fonts(cfg)  # must run after sns.set_style resets rcParams
    results_dir = Path(cfg["paths"]["results_dir"])
    figures_dir = Path(
        cfg["paths"].get("figures_dir", cfg["paths"]["results_dir"] + "/figures")
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    long_path = results_dir / "word_drivers_long.parquet"
    summ_path = results_dir / "word_drivers_summary.parquet"
    if not long_path.exists() or not summ_path.exists():
        raise FileNotFoundError(
            f"visualize_word_drivers: expected {long_path} and {summ_path}. "
            f"Run scripts.analyze_word_drivers first."
        )
    long_df = pd.read_parquet(long_path)
    summary_df = pd.read_parquet(summ_path)
    top_n = _top_n(cfg)

    categories = sorted(long_df["category"].unique())
    logger.info(f"word_drivers figures: categories={categories}, top_n={top_n}")
    for cat in categories:
        plot_contribution(summary_df, cat, figures_dir, top_n, logger)
        plot_slope(summary_df, cat, figures_dir, top_n, logger)
        plot_heatmap(long_df, summary_df, cat, figures_dir, logger)
        plot_trajectory(long_df, summary_df, cat, figures_dir, top_n, logger)


if __name__ == "__main__":
    fire.Fire(main)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MPLBACKEND=Agg pytest tests/test_visualize_word_drivers.py -q`
Expected: 1 passed (4 PDFs written).

- [ ] **Step 5: Commit**

```bash
git add scripts/visualize_word_drivers.py tests/test_visualize_word_drivers.py
git commit -m "feat(word-drivers): 4 driver figures per dimension (bars/slope/heatmap/trajectory)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Wire into both garg_weat slurm loops (non-fatal)

**Files:**
- Modify: `slurm/garg_weat_all_sources.slurm` (insert before the loop's closing `done`, line ~144)
- Modify: `slurm/garg_weat_zh.slurm` (insert before the loop's closing `done`, line ~111)

**Interfaces:**
- Consumes: `$CONFIG` (loop variable), the two new CLIs.
- Produces: driver tables + figures beside the existing per-source outputs. No new status entries (keeps the STATUSES/SOURCES arrays aligned by config index).

Rationale for placement: the driver step runs at the **ok-tail** of each iteration — only after the primary figures are validated and `STATUSES+=("ok")` is recorded. On failure it logs a WARN and lets the loop advance to `done`; it must NOT append to any status array or `continue` (that would desync the per-config arrays). This supersedes the illustrative snippet in the spec, which appended a status.

- [ ] **Step 1: Add the block to `garg_weat_all_sources.slurm`**

Find the end of the ok branch (after the `echo "  fig (subsample): $FIG_SUB"` line, line ~143, and before `done` on line ~144). Insert:

```bash

    # Word-level ideation drivers (secondary, NON-FATAL). Runs only here — after
    # the primary figures are validated and status recorded — so a driver bug
    # never regresses the main deliverable. Do not touch STATUSES here.
    if python -m scripts.analyze_word_drivers --config="$CONFIG" \
        && python -m scripts.visualize_word_drivers --config="$CONFIG"; then
        echo "  word_drivers: ok ($RESULTS_DIR/word_drivers_*.{parquet,csv})"
    else
        echo "  WARN: word_drivers step failed (primary figures unaffected)"
    fi
```

- [ ] **Step 2: Add the block to `garg_weat_zh.slurm`**

Find the end of the ok branch (after `echo "  ok: $SUMMARY_PATH  ($NPDF figure(s) under $FIG_DIR)"` and `STATUSES+=("ok")`, line ~110, before `done` on line ~111). Insert:

```bash

    # Word-level ideation drivers (secondary, NON-FATAL). See garg_weat_all_sources.slurm.
    if python -m scripts.analyze_word_drivers --config="$CONFIG" \
        && python -m scripts.visualize_word_drivers --config="$CONFIG"; then
        echo "  word_drivers: ok ($RESULTS_DIR/word_drivers_*.{parquet,csv})"
    else
        echo "  WARN: word_drivers step failed (primary figures unaffected)"
    fi
```

- [ ] **Step 3: Syntax-check both scripts**

Run: `bash -n slurm/garg_weat_all_sources.slurm && bash -n slurm/garg_weat_zh.slurm && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 4: Verify the CLIs are importable/callable locally**

Run: `MPLBACKEND=Agg python -c "import scripts.analyze_word_drivers, scripts.visualize_word_drivers; print('import ok')"`
Expected: `import ok`.

- [ ] **Step 5: Commit**

```bash
git add slurm/garg_weat_all_sources.slurm slurm/garg_weat_zh.slurm
git commit -m "slurm(garg_weat): call word_drivers analyze+visualize (non-fatal) per source

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Full-suite regression check

**Files:** none (verification only).

- [ ] **Step 1: Run the two new test modules plus the existing garg_weat/visualize tests**

Run:
```bash
MPLBACKEND=Agg pytest tests/test_analyze_word_drivers.py \
  tests/test_visualize_word_drivers.py \
  tests/test_analyze_garg_weat.py tests/test_visualize_garg_weat.py -q
```
Expected: all pass. If a pre-existing test fails, confirm it also fails on `HEAD~5` before touching it (it is unrelated to this change).

- [ ] **Step 2: Commit (only if any incidental fixups were needed)**

```bash
git add -A
git commit -m "test(word-drivers): green full driver + garg_weat suite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Long table (`cat_mean_signed`, `deviation`, sign, year parse, consistent_occupations) → Task 1 + Task 2 (consistent test).
- Summary table (delta=lens2, contribution=lens1, slope, endpoint-consistent, present_both) → Task 2.
- Parquet+CSV output, input validation, missing-file error → Task 3.
- 4 figures × per-dimension files, top_n config, non-fatal skips → Task 4.
- Non-fatal slurm wiring in both loops → Task 5.
- Tests for all six spec test cases → Tasks 1–3 (analyze) + Task 4 (viz smoke). Mapping: (1) cat_mean reproduces → `test_long_table_cat_mean_and_deviation_and_sign`; (2) Σcontribution=Δmean → `test_summary_contribution_sums_to_delta_of_endpoint_mean`; (3) sign flip → same long test; (4) churn present_both=False/NaN → contribution test; (5) deviation → long test; (6) consistent_occupations → `test_consistent_only_drops_churning_word_from_mean`.

**Placeholder scan:** none — every code and test step is complete and self-contained.

**Type consistency:** `build_long_table(rnd_long, ideation_sign, consistent_only, logger)` and `build_summary_table(long_df, logger)` are used identically across Tasks 1–4 and both test modules. Figure functions take `(summary_df|long_df, category, figures_dir, [top_n,] logger)` consistently. Output filenames `word_drivers_{long,summary}.{parquet,csv}` and `word_drivers_{form}_<category>.pdf` match between analyze, visualize, tests, and slurm echoes.

**Out-of-scope confirmed dropped:** provincial/weibo units filtered by `_slice_start_year` returning None; no run_pipeline.sh edits (garg_weat doesn't run through it).
