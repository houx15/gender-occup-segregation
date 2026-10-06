# US 3DLNews2 report — where to start

Gender norms in US local news, by state and 10-year window (1995–2004,
2000–09, 2005–14, 2010–19, 2015–24), measured with word embeddings (Garg RND)
and compared with surveys (ACS, ATUS, Project Implicit).
Two text domains: **occupation** and **domestic and care work** (unpaid
housework and care; `household` in file and column names).

**Orientation.** Everything in `analysis/` reads **higher = less traditional**
(maps: blue = less, red = more traditional). The methods tables (`tables.md`,
notes §3.3) show **raw RND**: > 0 = closer to female words.

## Reading order

1. **Methods and data** — [`../2026-10-05-us-3dlnews-notes.md`](../2026-10-05-us-3dlnews-notes.md):
   units and windows, corpus, word lists, how the score is computed, survey
   measures, weights, missing data, sample sizes, context predictors.
2. **Headline figures** — `analysis/figures_combined/` (Figures 1–6, below).
3. **Results text** — [`analysis/analysis_summary.md`](analysis/analysis_summary.md):
   all parts in one file, numbers and tables. The per-part files
   (`analysis/part*_summary.md`) are the same text split by part.
4. **Detail** — step-by-step figures and tables in `analysis/main/`
   (validation and correlates tables: `main/tables/1_4_validation_table.md`,
   `1_9_correlates_table.md`, LaTeX versions `.tex`), then the reliability
   checks per benchmark in `analysis/validation-*/` as needed.
5. **The plan** these results follow — [`2026-10-05-analysis-plan.md`](2026-10-05-analysis-plan.md).

## Headline figures (`analysis/figures_combined/`)

| File | Content | Plan step |
|---|---|---|
| `figure1_validation.pdf` | top: occupation RND vs ACS female share, state means; bottom: national trends (occupation, domestic and care work) | 1.1–1.3 |
| `figure2_validation_{occupation,household}.pdf` | validation against every direct benchmark, one row each: A pooled state-windows, B state dimension (state averages), C time dimension (net of state and window means), D national trend | 1.4 (1.5, 1.8) |
| `figure3_reliability.pdf` | every direct benchmark: mismatch vs text volume; hierarchical alignment slope and its spread across states | 1.6–1.7 |
| `figure4_maps_{occupation,household}.pdf` | state maps, one per window (`household` = domestic and care work) | 2.1 |
| `figure5_dynamics_{occupation,household}.pdf` | state × window heatmap; change 2000–09 → 2015–24 by state | 2.2–2.3 |
| `figure6_explanatory.pdf` | state predictors of the text score: between states (state averages) and within states (state + window FE) | II-B |

## Folder structure

```
us_3dlnews_report/
├── README.md                    this file
├── 2026-10-05-analysis-plan.md  analysis plan (PI)
├── tables.md, tables/, figures/ methods statistics: data volume, training set-up,
│                                word-list coverage, survey measures, sample sizes,
│                                text scores per window (scripts/report_us_dlnews.py)
├── analysis/                    results (scripts/us_analysis/run.py)
│   ├── analysis_summary.md      all results in one file
│   ├── part1_summary.md         I    measurement validation (1.1–1.4), correlates (1.9),
│   │                                 reliability per benchmark (1.6–1.7)
│   ├── part2_summary.md         II   geography and dynamics (2.1–2.3, 2.6 policy timing)
│   ├── part2b_summary.md        II-B state predictors (socioeconomic, labour market,
│   │                                 policy, political & cultural)
│   ├── part2c_summary.md        II-C text–survey gap as an outcome
│   ├── part3_summary.md         III  case selection + state profiles (3.1–3.3)
│   ├── part3s_summary.md        III  nearest-neighbour words of selected terms
│   ├── part_balanced_summary.md robustness: national trends on a balanced panel
│   ├── figures_combined/        Figures 1–6
│   ├── main/figures, main/tables  every step, main measures; file names start
│   │                              with the plan step (1_4_..., 2b_..., 3_3_...)
│   ├── validation-MEASURE/      reliability (1.6 mismatch vs volume, 1.7 state
│   │                            slopes) for each direct benchmark (list below)
│   ├── robustness-balanced-2000/ national trends (1.2, 3.1, 3.2) on states observed
│   │                            in every window from 2000–09
│   └── data/                    analysis inputs: state × window panel,
│                                occupation cells, domestic- and care-work terms
└── archive/                     deprecated tables and outputs (first report version;
                                 family-sphere-as-main analysis; per-measure
                                 robustness folders); do not cite
                                 (see archive/README.md)
```

## Survey measures: validation and correlates

**Validation (direct benchmarks: the same concept as the text measure).** Every
benchmark is reported, none is singled out; each in three dimensions (pooled
state-windows; between states; within states with state + window fixed effects)
plus the national trend.

| Domain | Text | Direct benchmarks |
|---|---|---|
| Occupation | occupation words (65 used) | ACS female share of our occupations; IAT; explicit stereotype |
| Domestic and care work (`household` in file names) | domestic- and care-work words (21 used) | ATUS women's share of housework, household activities, childcare; IAT; explicit stereotype |

**Correlates (related concepts: associations, a mechanism question, not
validation)**, both domains, same three dimensions, one coefficient table
(`1_9_correlates_table`; without DC: `_no_dc`): ACS family index, motherhood
employment and hours gaps, married women not in the labour force, wife's
earnings share, wife earns more, gender employment gap, women's share of
employment, occupational segregation (Duncan).

Where one survey value per state-window is needed (II-C text–survey gap,
Part III state cases), the **survey composite** is used: the mean of the
domain's z-scored direct benchmarks.

`robustness-balanced-2000` checks the national average trends (1.2, 3.1, 3.2)
on a balanced panel, 2000–09 to 2015–24, with only states (or, per word,
states with the word) observed in every window. The earlier family-sphere list
(home, kids, marriage, ...) is no longer used.

## Related documents (outside this folder)

- Word-list decisions and evidence: [`../research-log-wordlists.md`](../research-log-wordlists.md)
- Word lists and screening tables: `wordlists/en/occupation_family/`
  (`occupation_grounding.csv`, `household_screening.csv`)
