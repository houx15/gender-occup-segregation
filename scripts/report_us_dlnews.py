#!/usr/bin/env python3
"""Measurement and data statistics for the 3DLNews2 US state-window report.

Documentation only (no validation results): dataset volume, training set-up,
word-list coverage, survey measures and text scores per window. Reads the
finished outputs of one profile (corpus build, analysis, check_dlnews) and
writes to --out_dir:
  figures/*.pdf   dataset volume, word coverage
  tables/*.csv    every statistic in the report
  tables.md       the same tables as markdown

Usage:
  python -m scripts.report_us_dlnews --config=config/profiles/garg_weat_dlnews_w10.yml \
      --out_dir=/scratch/network/yh6580/gender-occup/results/report_dlnews_w10
"""

from __future__ import annotations

from pathlib import Path

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.common.config_loader import load_config  # noqa: E402

OCCUPATION = ["matched_female_share", "duncan", "female_emp_share"]
FAMILY = ["family_index_acs", "motherhood_emp_gap", "motherhood_hours_gap",
          "married_women_nilf", "wife_earnings_share", "wife_earns_more", "gender_emp_gap"]
SUBJECTIVE = ["iat_sex_balanced", "explicit_sex_balanced"]
OURS = ["ours_occupation", "ours_family_sphere", "ours_household"]


def window_label(p: int, width: int) -> str:
    return f"{p}–{(p + width - 1) % 100:02d}"


def count_tokens(unit_dir: Path) -> int:
    """Exact token count of a unit's corpus (space-separated, one doc per line)."""
    n = 0
    for f in sorted(unit_dir.glob("corpus_*")):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                n += chunk.count(b" ") + chunk.count(b"\n")
    return n


def _md(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(4)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _stats_by_window(t: pd.DataFrame, cols, lab) -> pd.DataFrame:
    rows = []
    for c in [c for c in cols if c in t.columns]:
        for p, g in t.groupby("period"):
            v = g[c].dropna()
            if v.empty:
                continue
            rows.append({"measure": c, "window": lab(p), "states": len(v), "mean": v.mean(),
                         "sd": v.std(), "min": v.min(), "max": v.max()})
    return pd.DataFrame(rows)


def main(config: str, out_dir: str) -> None:
    cfg = load_config(config)
    res = Path(cfg["paths"]["results_dir"])
    width = int(cfg["us_states"].get("year_bins") or 1)
    start, end = cfg["analysis"]["decade_range"]
    lab = lambda p: window_label(p, width)  # noqa: E731
    out = Path(out_dir)
    for sub in ("figures", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    md = []

    def table(name, df, title):
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
        md.append(f"### {title}\n\n{_md(df)}\n")

    # 1. dataset volume ---------------------------------------------------------
    cov = pd.read_csv(res / "coverage_dlnews.csv")
    cov = cov[(cov["year"] >= start) & (cov["year"] <= end)].copy()
    corpora = Path(cfg["paths"]["corpora_dir"])
    cov["tokens"] = [count_tokens(corpora / u) if (corpora / u).is_dir() else np.nan
                     for u in cov["unit_name"]]
    cov["window"] = cov["year"].map(lab)
    cov.to_csv(out / "tables" / "volume_by_state_window.csv", index=False)
    vol = (cov.groupby("window")
           .agg(states_with_articles=("state", "nunique"), states_modelled=("kept", "sum"),
                articles_total=("n_docs", "sum"), articles_median=("n_docs", "median"),
                articles_min=("n_docs", "min"), articles_max=("n_docs", "max"),
                tokens_total_millions=("tokens", lambda x: x.sum() / 1e6),
                tokens_median_millions=("tokens", lambda x: x.median() / 1e6),
                tokens_min_millions=("tokens", lambda x: x.min() / 1e6),
                tokens_max_millions=("tokens", lambda x: x.max() / 1e6))
           .reset_index())
    table("volume", vol, "Dataset volume per window (state models: states with >= "
          f"{cfg['us_states'].get('min_documents', 500)} articles)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    m = cov.dropna(subset=["tokens"])
    wins = sorted(m["window"].unique())
    axes[0].boxplot([m[m["window"] == w]["tokens"] / 1e6 for w in wins], labels=wins)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tokens per state model (millions, log)")
    axes[0].set_title("Training text per state-window")
    axes[1].bar(vol["window"], vol["states_modelled"], color="#4c72b0")
    axes[1].set_ylabel("States with a model")
    axes[1].set_title("States modelled per window")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(out / "figures" / "01_dataset_volume.pdf")
    plt.close()

    # 2. training set-up ---------------------------------------------------------
    e, c, u = cfg["embedding"], cfg["corpus"], cfg["us_states"]
    setup = pd.DataFrame([
        ("Text", "3DLNews2, Google-platform newspaper articles"),
        ("Unit", f"state x {width}-year window, step {u.get('year_step') or width} "
                 f"({', '.join(lab(p) for p in sorted(cov['year'].unique()))})"),
        ("Minimum articles for a state model", u.get("min_documents", 500)),
        ("Preprocessing", f"tokenizer {c['tokenizer']}, stopwords {c['stopwords']}, "
                          f"lowercase {c['lowercase']}, min {c['min_words']} tokens per article"),
        ("De-duplication", f"{c['dedup']['method']} (k={c['dedup'].get('shingle_k')}), "
                           "within window, across states"),
        ("Model", f"word2vec, {'skip-gram' if e['sg'] else 'CBOW'}"),
        ("vector_size / window / min_count", f"{e['vector_size']} / {e['window']} / {e['min_count']}"),
        ("negative / epochs / seed", f"{e['negative']} / {e['epochs']} / {e['seed']}"),
    ], columns=["setting", "value"])
    table("training_setup", setup, "Training set-up")

    # 3. word-list coverage ------------------------------------------------------
    wc = pd.read_csv(res / "word_coverage.csv")
    long_df = pd.read_parquet(res / "garg_weat_rnd_long.parquet")
    long_df["period"] = long_df["unit_name"].str.rsplit("_", n=1).str[1].astype(int)
    by_win = (long_df.groupby(["category", "occupation", "period"])["in_vocab"].mean()
              .unstack("period"))
    by_win.columns = [f"coverage_{lab(p)}" for p in by_win.columns]
    words = wc.merge(by_win.reset_index(), on=["category", "occupation"], how="left")
    words = words.rename(columns={"occupation": "word", "coverage": "coverage_all"})
    words.to_csv(out / "tables" / "word_coverage.csv", index=False)
    summ = (words.groupby("category")
            .agg(candidates=("word", "count"), used=("used", "sum"),
                 median_coverage=("coverage_all", "median")).reset_index())
    table("wordlist_summary", summ, "Word lists: candidates, words used (in vocabulary of "
          f">= {cfg['analysis'].get('min_word_coverage')} of state models), median coverage")
    cats = list(summ["category"])
    fig, axes = plt.subplots(1, len(cats), figsize=(4.5 * len(cats), 0.13 * len(words) / len(cats) * 3 + 2))
    for ax, cat in zip(np.atleast_1d(axes), cats):
        g = words[words["category"] == cat].sort_values("coverage_all")
        colors = np.where(g["used"], "#4c72b0", "#c0c0c0")
        ax.barh(g["word"], g["coverage_all"], color=colors)
        ax.axvline(cfg["analysis"].get("min_word_coverage", 0.5), color="black", lw=0.8, ls="--")
        ax.set_xlim(0, 1)
        ax.set_title(f"{cat}: {int(g['used'].sum())}/{len(g)} used")
        ax.set_xlabel("Share of state models with the word in vocabulary")
        ax.tick_params(axis="y", labelsize=5)
    plt.tight_layout()
    plt.savefig(out / "figures" / "02_word_coverage.pdf")
    plt.close()

    # 4. survey measures -----------------------------------------------------------
    t = pd.read_csv(res / "state_benchmark_table.csv")
    sv = _stats_by_window(t, OCCUPATION + FAMILY + SUBJECTIVE, lab)
    table("survey_measures", sv, "Survey measures per window (across states)")
    shares = Path(cfg["census_check"]["shares_dir"])
    n_rows = []
    for name, path, col in (
            ("ACS occupation (employed persons)", shares / "state_labor_indicators.csv", "n_persons"),
            ("ACS family (adults 25-54)", Path(cfg["census_check"]["family_file"]), "n_persons"),
            ("Project Implicit IAT (US respondents)", Path(cfg["census_check"]["attitude_file"]), "n")):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if col not in d:
            continue
        for p, g in d.groupby("period"):
            n_rows.append({"source": name, "window": lab(p), "states": len(g),
                           "respondents_total": int(g[col].sum()),
                           "respondents_median_per_state": int(g[col].median()),
                           "respondents_min_per_state": int(g[col].min())})
    table("survey_sample_sizes", pd.DataFrame(n_rows), "Survey sample sizes (unweighted respondents)")

    # 5. text scores ------------------------------------------------------------------
    s = pd.read_parquet(res / "garg_weat_summary_by_category.parquet")
    s["period"] = s["unit_name"].str.rsplit("_", n=1).str[1].astype(int)
    ts = (s.groupby(["category", "period"])
          .agg(states=("unit_name", "count"), words_in_set=("n_consistent", "max"),
               words_per_state_median=("n_occupations", "median"),
               mean_rnd=("mean_value", "mean"), sd_rnd_across_states=("mean_value", "std"),
               median_ci_halfwidth=("mean_ci_high", lambda x: np.nan))
          .reset_index())
    ci = (s.assign(hw=(s["mean_ci_high"] - s["mean_ci_low"]) / 2)
          .groupby(["category", "period"])["hw"].median().rename("median_ci_halfwidth"))
    ts = ts.drop(columns="median_ci_halfwidth").merge(ci.reset_index(), on=["category", "period"])
    ts["window"] = ts["period"].map(lab)
    table("text_scores", ts[["category", "window", "states", "words_in_set",
                             "words_per_state_median", "mean_rnd", "sd_rnd_across_states",
                             "median_ci_halfwidth"]],
          "Text scores (RND, word fixed effects) per window")

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    fire.Fire(main)
