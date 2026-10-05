#!/usr/bin/env python3
"""Figures and statistics tables for the 3DLNews2 US state-window report.

Reads the finished outputs of one profile (analysis, check_occupation_census,
check_state_benchmarks) and writes, to --out_dir:
  figures/*.png   result visualizations
  tables/*.csv    the statistics behind them
  tables.md       the same tables as markdown, for the report note

Measures are RND (Garg et al. 2018; > 0 = closer to the female centroid),
estimated per state-window with word fixed effects. "National" numbers are
aggregates of the state models (no separately trained national model).
Only objective ACS benchmarks are reported (Project Implicit is left out).

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

from scripts.check_state_benchmarks import TRADITIONAL_SIGN, add_agreement  # noqa: E402
from scripts.common.config_loader import load_config  # noqa: E402

OCC_ACS = ["matched_female_share", "duncan", "female_emp_share"]
FAM_ACS = ["motherhood_emp_gap", "motherhood_hours_gap", "married_women_nilf",
           "wife_earnings_share", "wife_earns_more", "gender_emp_gap"]
OURS = ["ours_occupation", "ours_family_sphere", "ours_household"]
LABEL = {
    "ours_occupation": "Ours: occupation", "ours_family_sphere": "Ours: family sphere",
    "ours_household": "Ours: household",
    "matched_female_share": "Female share, our occupations", "duncan": "Segregation (Duncan)",
    "female_emp_share": "Women's share of employment",
    "motherhood_emp_gap": "Motherhood employment gap", "motherhood_hours_gap": "Motherhood hours gap",
    "married_women_nilf": "Married women not in LF", "wife_earnings_share": "Wife's earnings share",
    "wife_earns_more": "Wife earns more", "gender_emp_gap": "Gender employment gap",
}


def window_label(p: int, width: int) -> str:
    return f"{p}–{(p + width - 1) % 100:02d}"


def count_tokens(unit_dir: Path) -> int:
    n = 0
    for f in sorted(unit_dir.glob("corpus_*")):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                n += chunk.count(b" ") + chunk.count(b"\n")
    return n


def _md(df: pd.DataFrame, floatfmt: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(floatfmt)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def main(config: str, out_dir: str) -> None:
    cfg = load_config(config)
    res = Path(cfg["paths"]["results_dir"])
    width = int(cfg["us_states"].get("year_bins") or 1)
    start, end = cfg["analysis"].get("decade_range", [None, None])
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    md = []

    def save_table(name, df, title):
        df.to_csv(out / "tables" / f"{name}.csv", index=False)
        md.append(f"### {title}\n\n{_md(df)}\n")

    def save_fig(name):
        plt.tight_layout()
        plt.savefig(out / "figures" / f"{name}.png", dpi=150)
        plt.close()

    # ---- 1. corpus ---------------------------------------------------------
    cov = pd.read_csv(res / "coverage_dlnews.csv")
    cov = cov[(cov["year"] >= start) & (cov["year"] <= end)]
    corpora = Path(cfg["paths"]["corpora_dir"])
    cov["tokens"] = [count_tokens(corpora / u) if (corpora / u).is_dir() else np.nan
                     for u in cov["unit_name"]]
    cov["window"] = cov["year"].map(lambda p: window_label(p, width))
    corpus = (cov.groupby("window")
              .agg(states=("state", "nunique"), states_trained=("kept", "sum"),
                   articles_median=("n_docs", "median"), articles_total=("n_docs", "sum"),
                   tokens_median_millions=("tokens", lambda x: x.median() / 1e6),
                   tokens_min_millions=("tokens", lambda x: x.min() / 1e6),
                   tokens_max_millions=("tokens", lambda x: x.max() / 1e6))
              .reset_index())
    save_table("corpus", corpus, "Corpus per state-window (3DLNews2, Google newspaper articles)")
    fig, ax = plt.subplots(figsize=(7, 4))
    for w, g in cov.dropna(subset=["tokens"]).groupby("window"):
        ax.hist(g["tokens"] / 1e6, bins=30, alpha=0.5, label=w)
    ax.set_xlabel("Tokens per state-window model (millions)")
    ax.set_ylabel("Number of states")
    ax.legend(title="Window")
    ax.set_title("Training text per state-window")
    save_fig("01_corpus_tokens")

    # ---- 2. word lists actually used --------------------------------------
    wc = pd.read_csv(res / "word_coverage.csv")
    used = (wc[wc["used"]].groupby("category")
            .agg(n_used=("occupation", "count"),
                 words=("occupation", lambda x: ", ".join(x))).reset_index())
    cand = wc.groupby("category")["occupation"].count().rename("n_candidates").reset_index()
    save_table("wordlists_used", cand.merge(used, on="category"),
               "Word lists: candidates and words used (in vocab in >= 50% of state-window models)")

    # ---- 3. national, by occupation ----------------------------------------
    occ = pd.read_csv(res / "occupation_census_scores.csv").dropna(subset=["female_share"])
    occ_r = pd.read_csv(res / "occupation_census_correlations.csv")
    occ_r["window"] = occ_r["period"].map(lambda p: window_label(p, width))
    save_table("national_by_occupation_r",
               occ_r[["window", "n_occupations", "pearson_r", "spearman_r", "pearson_r_logit"]],
               "National, by occupation: occupation RND vs ACS female share")
    occ.to_csv(out / "tables" / "national_by_occupation_scores.csv", index=False)
    periods = sorted(occ["period"].unique())
    fig, axes = plt.subplots(1, len(periods), figsize=(5.2 * len(periods), 4.6), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), periods):
        g = occ[occ["period"] == p]
        r = occ_r.set_index("period").loc[p, "pearson_r"]
        ax.scatter(g["female_share"], g["rnd"], s=14)
        for _, row in g.iterrows():
            ax.annotate(row["occupation"], (row["female_share"], row["rnd"]), fontsize=5.5)
        b = np.polyfit(g["female_share"], g["rnd"], 1)
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, np.polyval(b, xs), color="grey", lw=1)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_title(f"{window_label(p, width)}: r = {r:.2f} (n = {len(g)})")
        ax.set_xlabel("ACS female share of the occupation")
    np.atleast_1d(axes)[0].set_ylabel("Occupation RND (> 0 = female-leaning)")
    save_fig("02_national_by_occupation")

    # ---- 4. national trend ---------------------------------------------------
    trend = pd.read_csv(res / "state_benchmark_trend.csv")
    cols = [c for c in OURS + OCC_ACS + FAM_ACS if c in trend.columns]
    trend["window"] = trend["period"].map(lambda p: window_label(p, width))
    save_table("national_trend", trend[["window", "n_states"] + cols],
               "National trend: mean over a balanced panel of states (RND for ours; ACS shares / gaps)")
    groups = [("Our scores (RND)", OURS), ("ACS: occupation", OCC_ACS), ("ACS: family", FAM_ACS)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, cs) in zip(axes, groups):
        for c in [c for c in cs if c in trend.columns]:
            s = trend[c]
            z = (s - s.iloc[0]) / (s.abs().max() or 1)   # change from first window, scaled
            ax.plot(trend["window"], z, marker="o", label=LABEL[c])
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(title)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Change from first window (scaled)")
    save_fig("03_national_trend")

    # ---- 5. state level ------------------------------------------------------
    t = pd.read_csv(res / "state_benchmark_table.csv")
    t["window"] = t["period"].map(lambda p: window_label(p, width))
    acs = [c for c in OCC_ACS + FAM_ACS if c in t.columns]
    corr = add_agreement(pd.read_csv(res / "state_benchmark_correlations.csv")
                         .drop(columns="agreement_r", errors="ignore"))
    corr = corr[corr["survey"].isin(acs) & corr["ours"].isin(OURS)]
    corr.to_csv(out / "tables" / "state_correlations_all.csv", index=False)
    within = (corr[corr["scope"].str.startswith("period")]
              .pivot_table(index="survey", columns="ours", values="agreement_r", aggfunc="mean"))
    change = (corr[corr["scope"].str.startswith("change")]
              .pivot_table(index="survey", columns="ours", values="agreement_r"))
    st = within.add_suffix(" (within window)").join(change.add_suffix(" (change)"))
    st = st.reindex(acs).reset_index().rename(columns={"survey": "ACS measure"})
    save_table("state_agreement", st,
               "State level: agreement r between our scores and ACS measures "
               "(> 0 = both point to more, or both to less, traditional)")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    m = within.reindex(acs)[[c for c in OURS if c in within.columns]]
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(m.shape[1]), [LABEL[c] for c in m.columns], rotation=20, ha="right")
    ax.set_yticks(range(m.shape[0]), [LABEL[c] for c in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="agreement r (> 0 = agree)")
    ax.set_title("State level: our scores vs ACS, mean r across windows")
    save_fig("04_state_agreement_heatmap")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (x, y) in zip(axes, [("matched_female_share", "ours_occupation"),
                                 ("married_women_nilf", "ours_family_sphere")]):
        for w, g in t.groupby("window"):
            ax.scatter(g[x], g[y], s=12, label=w)
        ax.set_xlabel(f"ACS: {LABEL[x]}")
        ax.set_ylabel(f"{LABEL[y]} (RND)")
        ax.legend(title="Window", fontsize=7)
    fig.suptitle("State-windows: our score vs the matching ACS measure")
    save_fig("05_state_scatter")

    # ---- 6. reliability --------------------------------------------------------
    rel_cols = [c for c in OURS + acs if c in t.columns]
    a, b = t["period"].min(), t["period"].max()
    w = t.pivot_table(index="state", columns="period", values=rel_cols)
    rel = pd.DataFrame({"measure": rel_cols,
                        f"r_{a}_vs_{b}": [w[(c, a)].corr(w[(c, b)]) for c in rel_cols]})
    save_table("reliability", rel,
               f"Reliability: correlation of state values, window {window_label(a, width)} vs "
               f"{window_label(b, width)} (no shared years)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, c in zip(axes, ["ours_occupation", "matched_female_share"]):
        ax.scatter(w[(c, a)], w[(c, b)], s=14)
        ax.set_xlabel(window_label(a, width))
        ax.set_ylabel(window_label(b, width))
        ax.set_title(f"{LABEL[c]}: r = {w[(c, a)].corr(w[(c, b)]):.2f}")
    fig.suptitle("Does a state's value repeat across non-overlapping windows?")
    save_fig("06_reliability")

    # ---- 7. maps -----------------------------------------------------------------
    try:
        import geopandas as gpd
        from scripts.data_prep.us_state_mapper import normalize_state
        shp = Path(cfg["us_states"].get("shapefile", "data/shapefiles/us_states.shp"))
        states = gpd.read_file(shp)
        states = states[~states["NAME"].isin(["Alaska", "Hawaii", "Puerto Rico"])]
        t["NAME"] = t["state"].str.replace("_", " ").map(normalize_state)
        vmax = t["ours_occupation"].abs().max()
        wins = sorted(t["period"].unique())
        fig, axes = plt.subplots(1, len(wins), figsize=(6 * len(wins), 4.2))
        for ax, p in zip(np.atleast_1d(axes), wins):
            g = states.merge(t[t["period"] == p][["NAME", "ours_occupation"]], on="NAME", how="left")
            g.plot(column="ours_occupation", ax=ax, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   edgecolor="black", linewidth=0.2,
                   missing_kwds={"color": "lightgrey"}, legend=(p == wins[-1]))
            ax.set_title(f"Occupation RND, {window_label(p, width)}")
            ax.set_axis_off()
        save_fig("07_map_occupation")
    except Exception as e:  # noqa: BLE001
        print(f"  (map skipped: {e!r})")

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {len(list((out / 'figures').glob('*.png')))} figures and tables to {out}")


if __name__ == "__main__":
    fire.Fire(main)
