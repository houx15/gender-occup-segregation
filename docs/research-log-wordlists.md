# Research log: wordlists for the US state-period arm (3DLNews2)

What we decided about the words we measure, why, and what the data said.
Newest entries at the bottom. Each entry gives the date, the decision, the
evidence (with the job / commit that produced it), and open questions.

Setup common to all entries unless stated otherwise:

- Corpus: 3DLNews2, Google-platform local **newspaper** articles (TV kept as a
  separate, thin robustness arm). Twitter downloaded but not used.
- Units: state x 5-year period (`us_states.year_bins: 5`), labelled by the
  period start (`ohio_2010` = 2010-2014). One word2vec model per unit
  (300d, window 5, `min_count` 20); units with < 500 articles are not trained.
- Gender anchors: Garg et al. (2018) male / female word lists
  (`gender_words.json`, 20 + 20 words).
- Score: relative norm distance (RND, Garg et al. 2018) of each list word to
  the female vs male centroid; > 0 = female-leaning. `ideation_sign` flips
  family-type categories so that higher = less traditional.
- Research focus: **change** in gender ideation over time, by state. Low
  correlation of state rankings between adjacent periods is acceptable.

---

## 2026-10-04 — Baseline: COHA-derived lists (leadership / family / science)

**Lists.** `wordlists/en/garg_weat/cleaned_{leadership,family,science}.txt`
(70 / 21 / 54 words), originally pruned against COHA models.

**Rule.** Consistent set: a word is used only if it is in vocab in *every*
analyzed unit.

**Evidence** (237 newspaper models, 1995-2024):

| Category   | words used | adjacent-period state corr. |
|------------|-----------:|----------------------------:|
| family     | 2 / 21     | 0.36 |
| leadership | 12 / 71    | 0.52 |
| science    | 0 / 55     | — |

The thinnest units (`*_1995`, ~100 articles) veto most words.

## 2026-10-04 — Re-prune the large candidate pools against 3DLNews2

**Change.** `prepare_wordlists` on `candidates_*.txt` (445 / 213 / 442 words),
keeping words in vocab in all 237 newspaper models -> `dlnews_cleaned_*`.

**Evidence** (job 3391352): leadership 27, family 3, science 7 words.
Stability did not improve (leadership 0.42, family -0.01, science 0.05).
Words like *research, science, teacher, technology* were in 236/237 models
and dropped for missing in one thin 1995 unit.

**Takeaway.** Coverage is the binding constraint; the family candidates are
mostly rare chores/objects that local news seldom uses.

## 2026-10-04 — Add `family_sphere` (common family words)

**Change.** New category `family_sphere`: Caliskan et al. (2017) WEAT-6
family words (home, parents, children, family, cousins, marriage, wedding,
relatives) + household, kitchen. Gender-neutral, frequent in news. Sign -1.

**Evidence** (job 3391366): only 4/10 survived the every-unit rule
(children, family, home, parents); still better stability (0.35) than the
chore-based family list.

## 2026-10-04 — Refocus: occupation (primary) + family

**Decision (PI).** Drop leadership / science / chore-family as headline
measures. Focus on **occupation** (most important; start from Garg et al.
2018) and **family**. Words must be grounded and must have good coverage in
this corpus.

**Occupation list** — `wordlists/en/occupation_family/occupation_grounding.csv`
documents every candidate (word, plural, include, 2018 Census occupation title,
Garg source word, note):

- Start: Garg's 76-word list plus the 124 occupations in his census
  female-share file (`wordlists/en/garg/occupation_percentages_gender.csv`,
  1850-2015). Keeping Garg's word in `garg_word` lets us reuse his female
  shares to validate the embedding scores.
- Replaced Garg's artificial neutral forms with the words news actually uses:
  fireperson -> firefighter, newsperson -> journalist, bankteller -> teller.
- Excluded (with reason in the CSV): obsolete (bootblack, milliner,
  typesetter...), common surnames (smith, mason, porter, baker, cook),
  ambiguous (pilot, guard, conductor, collector, server, developer),
  gendered pairs (actor, waiter), not occupations (student, retired, sales,
  official, clerical, unemployed).
- Added common modern occupations (programmer, paramedic, receptionist,
  paralegal, nanny, caregiver, realtor, reporter, prosecutor, ...).
- Result: 134 candidates. Kept with a caveat: *secretary* (Secretary of
  State sense), *supervisor* (elected county supervisor) — Garg comparability.

**Household list** — `candidates_household.txt`: 20 unpaid-work / care words
from ATUS activity categories (housework, laundry, cooking, childcare,
daycare, caregiving, ...).

**Window.** Analyze 2005-09 .. 2020-24 only (`analysis.decade_range`); earlier
periods cover 8-32 states. `decade_range` was extended to parse
`{state}_{period}` units, and `prepare_wordlists` now uses the same clip.

**Evidence** (job 3391382, every-unit rule over 197 models):

| Category      | kept |
|---------------|-----:|
| occupation    | 10 / 134 (attorney, coach, driver, editor, executive, judge, manager, police, secretary, writer) |
| household     | 0 / 20 |
| family_sphere | 4 / 10 |

Singular coverage (share of models with the word): teacher 0.99,
reporter 0.99, nurse 0.91, engineer 0.92, firefighter 0.83, cashier 0.16,
plumber 0.05; household: cooking 0.89, cleaning 0.86, laundry 0.62,
housework / homemaking 0.00.

**Takeaway.** The every-unit rule is unworkable with small state-period
models; the surviving occupations are generic titles that cannot measure
segregation.

## 2026-10-04 — Plural pooling + word fixed effects

**Change 1 — plural pooling.** Wordlist entries may list surface forms,
`nurse|nurses`. The entry vector is the mean of the L2-normalized in-vocab
forms (re-normalized); the entry counts as in vocab if any form is. Local
news often uses the plural (*nurses, teachers, plumbers*). Plurals are in the
`plural` column of `occupation_grounding.csv`; family/household lists carry
their own variants. (`scripts/common/metrics.py: entry_vector`)

**Change 2 — word fixed effects** (`analysis.word_set: fixed_effects`,
`min_word_coverage: 0.5`). Use every word in vocab in >= 50% of analyzed
units. Each unit's level comes from `value[u, w] = a[u] + b[w] + e` with
mean(b) = 0, so a unit is not shifted by which words it happens to have; on a
balanced panel `a[u]` is exactly the plain mean used before. CIs resample
words and refit. Per-word coverage is written to
`<results_dir>/word_coverage.csv`. (`scripts/common/fixed_effects.py`)

**Evidence** (job 3391395, commit 3023aef; 197 newspaper models 2005-2024):

| Category      | words used (cov. >= 0.5) | before (every-unit rule) | adjacent-period corr. | states w/ significant change 2005-09 -> 2020-24 |
|---------------|---------------:|-----:|-----:|--------:|
| occupation    | 56 / 134 | 10 | 0.22 | 21 / 46 |
| family_sphere | 10 / 10  | 4  | 0.29 | 5 / 46  |
| household     | 6 / 20   | 0  | -0.02 | 9 / 37 |

- Occupation words used: attorney coach driver editor executive judge manager
  police secretary writer teacher reporter artist author doctor lawyer
  professor prosecutor administrator sheriff athlete soldier farmer
  firefighter pastor engineer nurse instructor clerk detective photographer
  operator supervisor journalist musician physician contractor chef scientist
  trooper consultant counselor carpenter architect designer dancer technician
  weaver inspector mechanic therapist entrepreneur surgeon veterinarian
  painter paramedic.
- Just below the bar (coverage): biologist 0.50, miner 0.50, librarian 0.49,
  dispatcher 0.49, sailor 0.43, dentist 0.42, broker 0.42, caregiver 0.38,
  realtor 0.38.
- Household words used: groceries, cooking, cleaning, dishes, gardening,
  laundry (care words — parenting 0.37, daycare 0.32, childcare 0.26 — miss).
- National trend (balanced panel of 46 states, oriented RND): occupation
  rises steadily, -0.0122 -> -0.0091 -> -0.0067 -> -0.0043 (occupations
  remain male-leaning but less so); family_sphere stays female-leaning
  (about -0.009 to -0.012), no clear trend; household is noisy.
- TV arm: occupation 21 words, family_sphere 4, household 0 — confirms TV is
  too thin for these lists.

**Follow-ups flagged.** *weaver* (coverage 0.69 vs ~0.05 for comparable
crafts) is almost certainly the surname Weaver; *carpenter* and *painter*
may carry surname / artist senses too. Check their per-word RND and likely
exclude weaver.

## 2026-10-04 — Census check and word-sense check

**Framing (PI).** Comparing scores with census / survey female shares is a
descriptive *check* of how our measure lines up with real occupational
composition, not a validation.

**Census check** (`scripts/check_occupation_census.py`, job 3391412). Per
period: occupation RND (mean over the state-period units that have it) vs the
census female share from Garg's file (period mean; 2020-24 uses 2015).
39 of the 56 used occupations have a Garg census counterpart.

| Period  | n  | Pearson r | Pearson r (logit share) | Spearman r |
|---------|---:|----:|----:|----:|
| 2005-09 | 39 | 0.57 | 0.51 | 0.54 |
| 2010-14 | 39 | 0.60 | 0.55 | 0.55 |
| 2015-19 | 39 | 0.66 | 0.61 | 0.60 |
| 2020-24 | 39 | 0.67 | 0.62 | 0.64 |

**Word-sense check** (`scripts/diagnose_word_senses.py`; nearest neighbours in
the 4 largest models: CA 2010/2015/2020, TX 2020):

| word | neighbours | reading |
|------|------------|---------|
| weaver | personal names in 4/4 (hubbell, burton, wiggins, ...) | surname |
| carpenter | unions / trades in CA 2010, 2020; names in CA 2015, TX 2020 | mixed |
| painter | artist, sculptor, watercolor | fine artist, not the census trade it is mapped to |
| secretary | treasurer, Mattis, Pompeo, cabinet | political office |
| supervisor | county board, Antonovich, "supes" (CA); job sense in TX | mostly elected office |
| operator | owner, company, operations | business operator |
| driver | vehicle, swerved, seatbelt, Camry | motorist in crash reports |
| coach / judge / executive | coaching, players / court, magistrate / CEO, president | job sense |

**Decision.** Drop *weaver* (`include = 0` in the grounding table, with this
evidence). Keep carpenter, painter, secretary, supervisor, operator and driver
(PI: words with adequate coverage stay; their mixed senses are documented
here and in the grounding notes).

## 2026-10-04 — Rerun without weaver (current results)

Jobs 3391418 (analysis) + 3391419 (checks), commit fd2cd9d.

- Occupation: 55 / 133 words used (coverage >= 0.5 of 197 units); 38 have a
  census share.
- Census check: Pearson r = 0.57 (2005-09), 0.61, 0.66, 0.68 (2020-24);
  Spearman 0.55 -> 0.65. Female end: nurse, therapist, teacher; male end:
  mechanic, engineer, soldier, firefighter. Clear outlier: *secretary*
  (~95% female in the census, RND <= 0) — consistent with its political-office
  sense in news. Doctor / physician sit above their census share.
- National occupation trend (46-state balanced panel): -0.0122 -> -0.0091 ->
  -0.0066 -> -0.0042 — occupations stay male-leaning but steadily less so.
- State change 2005-09 -> 2020-24: 34 / 46 states move toward
  female-leaning; 21 / 46 significant.

## 2026-10-04 — Census check from the source: ACS microdata (IPUMS)

**Why.** Garg's female-share file ends in 2015 (the 2020-24 check used 2015
shares) and is national only. We now use the IPUMS USA ACS 1-year samples
2005-2024 directly (extract 1: employed persons; OCC2010, SEX, STATEFIP,
PERWT; `config/ipums_acs.yml`).

**Word -> occupation codes.** `occupation_occ2010.csv` maps each candidate to
OCC2010 codes, checked against the extract's codebook: 70 exact, 35 broad
(code wider than the word, e.g. judge -> "lawyers, and judges"), 24 multi
(e.g. engineer -> 13 engineering codes), 4 none (entrepreneur, operator,
proprietor, supervisor).

**Design (PI).** Compare at the state-period level, like Garg compares by
decade: our occupation gender norm per state-period (word fixed-effects score)
vs a survey-based norm for the same state-period: (a) matched female share =
equal-weight mean female share of our used occupations; (b) Duncan
occupational segregation index over all occupations; (c) women's share of
employment.

**Evidence** (job 3391503, commit f21f419):

National, by occupation (52 of 55 used occupations have ACS shares):

| Period  | Pearson r | Pearson r (logit) | Spearman r |
|---------|----:|----:|----:|
| 2005-09 | 0.70 | 0.66 | 0.69 |
| 2010-14 | 0.72 | 0.68 | 0.77 |
| 2015-19 | 0.76 | 0.72 | 0.77 |
| 2020-24 | 0.78 | 0.74 | 0.77 |

Stronger than with Garg's file (0.57-0.68): more occupations matched and
period-specific shares.

Over time (national means across the 46-51 states):

| Period  | our score | matched female share | Duncan | female emp. share |
|---------|------:|------:|------:|------:|
| 2005-09 | -0.0122 | 0.385 | 0.528 | 0.471 |
| 2010-14 | -0.0085 | 0.392 | 0.521 | 0.476 |
| 2015-19 | -0.0068 | 0.403 | 0.506 | 0.475 |
| 2020-24 | -0.0049 | 0.416 | 0.489 | 0.475 |

Our score moves toward female-leaning as women's share in these occupations
rises and segregation falls — same direction every period.

Across states (197 state-periods): no meaningful relationship. Pooled r with
matched share 0.17, Duncan 0.02, female employment share -0.04; within-period
|r| <= 0.22 with signs flipping; change 2005->2020 |r| <= 0.15. The survey
measures barely differ across states (matched share SD 1.4-1.7 pp; female
employment share SD 1.4 pp), while our state scores are noisy (adjacent-period
stability ~0.2).

**Reading.** The embedding measure tracks occupational gender composition
across occupations and over time, but its cross-state differences do not
track cross-state differences in composition. Either the state signal is
mostly estimation noise, or local news norms and local labor-market
composition genuinely diverge — not separable yet.

## 2026-10-04 — More benchmarks: family behaviour (ACS) and attitudes (Project Implicit)

**Why.** ACS only records behaviour; no housework time, no opinions. Combined
benchmark set per state-window:

- objective, occupation: matched female share, Duncan index, female
  employment share (ACS extract 1);
- objective, family: motherhood employment gap, motherhood hours gap, married
  women not in the labor force, wife's share of couple earnings, wife earns
  more, gender employment gap (ACS extract 2, adults 25-54, PERWT);
- objective, housework: ATUS (pending — the account needs a separate IPUMS
  ATUS registration);
- subjective: Project Implicit Gender-Career IAT 2005-2024 (2.1M US
  respondents with state): implicit D score and explicit career-family
  stereotype, raw and sex-balanced. Volunteers, not a probability sample.

All ACS measures use person weights (PERWT) summed over the window's years, so
they are state-representative; replicate weights (SEs) are not used yet.

**Evidence** (5-year units, job 3391526, commit a4ab042; balanced panel of 46
states):

| Period | ours occupation | ours family_sphere (raw) | IAT (sex-bal.) | explicit (sex-bal.) | motherhood emp. gap | wife earnings share |
|--------|------:|------:|------:|------:|------:|------:|
| 2005-09 | -0.0122 | 0.0090 | 0.369 | 1.56 | 0.134 | 0.359 |
| 2010-14 | -0.0091 | 0.0113 | 0.374 | 1.64 | 0.113 | 0.366 |
| 2015-19 | -0.0066 | 0.0118 | 0.341 | 1.05 | 0.106 | 0.366 |
| 2020-24 | -0.0042 | 0.0093 | 0.316 | 0.82 | 0.109 | 0.380 |

Nationally, every benchmark moves the less-traditional way from 2005-09 to
2020-24, as does our occupation score; family_sphere stays female-leaning
with no clear trend.

Across states, occupation score vs attitudes, within each window:

| | 2005-09 | 2010-14 | 2015-19 | 2020-24 | change |
|---|---:|---:|---:|---:|---:|
| IAT (sex-balanced) | -0.26 | -0.38 | -0.43 | +0.33 | 0.10 |
| explicit (sex-balanced) | -0.11 | -0.22 | -0.42 | +0.08 | 0.04 |

Negative = states whose news treats occupations as less male have weaker
career-male stereotypes (the expected direction) in three of four windows,
reversing in 2020-24. All other ours x benchmark pairs (family behaviour,
occupational composition) are |r| <= 0.3 with unstable signs; the one
notable change correlation is family_sphere vs IAT (0.44, n = 46) — one of
~40 pairs tested, so treat as exploratory.

## 2026-10-04 — Approach A (shared model per period, state-tagged anchors) set up

**Why.** Per-state models train on ~1.5M tokens (median 5-year unit) vs
billions for Google Ngram; state scores are dominated by estimation noise
(adjacent-period stability ~0.2 vs ~0.97 for ACS state measures). Approach A
keeps the data but pools it: one model per 5-year period over all states
(~110M tokens), where only the 40 gender anchors are tagged by state
('she__ohio'). List words learn from all text; each state contributes its own
gender centroids. Separate arm: `garg_weat_dlnews_tagged.yml`,
`slurm/tagged_dlnews.slurm`, `scripts/analyze_state_tagged.py`. In parallel:
10-year windows every 5 years for the per-state arm (`_w10.yml`).

**Finding while building it: pronouns were never in the English corpora.**
NLTK's English stopword list (our `en_default`) contains he, him, his,
himself, she, her, hers, herself, so every English corpus built with
`en_default` (3DLNews2 per-state arms; likely COHA-trained and others) dropped
8 of the 40 gender anchors; gender centroids came from the remaining 32 nouns
(man, woman, father, mother, ...). Approach A exempts the anchors from
stopword removal (`preprocess(..., keep_words=...)`) because pronouns are what
keep small states' tagged anchors frequent. Existing arms left unchanged
pending a PI decision.

## 2026-10-05 — Results: 10-year rolling windows vs approach A vs 5-year bins

Jobs 3391515 / 3391525 (10-year, per-state models), 3391880 (approach A),
comparison 3393260 (`scripts/compare_arms.py`, commit 438d4c0+).

**Stability of state scores** (correlation of state scores between windows):

| | 5-year bins | 10-year rolling | approach A (5-year) |
|---|---:|---:|---:|
| occupation words used | 55 | 75 | 129 |
| occupation, adjacent windows | 0.23 | 0.54 | 0.01 |
| family_sphere, adjacent windows | 0.29 | 0.48 | 0.05 |
| occupation, 2005 vs 2015 (no shared years) | 0.54 | 0.16 | 0.03 |
| family_sphere, 2005 vs 2015 (no shared years) | 0.17 | 0.21 | 0.02 |

Adjacent 10-year windows share 5 years of text, so their 0.5 stability is
largely mechanical; on non-overlapping windows the 10-year arm is no more
stable than the 5-year arm. With ~46 states a correlation's SE is ~0.14, so
the per-state arms' values (0.16-0.54) are all consistent with a modest true
reliability; none reaches the ~0.97 of ACS state measures.

**Approach A did not work as built.** Stability ~0 and a flat national
occupation trend (0.0003 -> 0.0005, where both per-state arms rise
monotonically). Likely reason: each state's tagged anchors still learn only
from that state's text, so the estimation noise moved from the list words to
the gender centroids; rare tagged tokens may also carry frequency artifacts.
Not pursued further as is (possible variants: shrink state centroids toward
the national one; national model fine-tuned per state).

**Agreement with benchmarks** (agreement_r, > 0 = agree; mean over windows):
weak and inconsistent in every arm. 10-year occupation: IAT +0.24, explicit
+0.09, but Duncan -0.24, female employment share -0.17. No ours x benchmark
pair agrees consistently across arms and scopes.

**National trend** (balanced states): occupation rises monotonically in both
per-state arms (5-year -0.0122 -> -0.0042; 10-year -0.0092 -> -0.0035), in
line with every survey benchmark; approach A shows no trend.

**Reading.** With 3DLNews2's volume, the national over-time trend is robust
and matches the surveys; state-level differences are not reliably measured
by any of the three designs.

---

## Open questions

- Validate occupation scores against census female shares (Garg's file covers
  the words with a `garg_word`; ACS state x occupation shares would allow a
  state-level validation).
- Do *secretary* / *supervisor* behave as outliers? If so, drop them.
- TV arm: keep as robustness only (≤ 8 states present in every period).

## 2026-10-05 — Analysis plan executed (Parts I, II, II-B, II-C, III)

10-year windows every 5 years, 5 windows (1995–2004 … 2015–24); survey data
now Census 2000 + ACS 2001–2024 for every window; subjective = Project
Implicit IAT (robustness). Code `scripts/us_analysis/`, job 3393431; results
`docs/us_3dlnews_report/analysis/` (`analysis_summary.md`, PDFs in `main/`
and `robustness-*/`). Orientation everywhere: higher = more traditional.

- I.1 occupations (62, pooled): r = 0.75 between occupation RND and ACS
  female share.
- I.2 occupation score falls from 2000–09 (+0.0122) to 2015–24 (+0.0049);
  family flat (~+0.010).
- I.4 state-window: pooled β = 0.09 (occupation), 0.04 (family), n.s.; with
  state + window FE ≈ 0. I.5: within-state occupation r = 0.28 reflects the
  common time trend. I.6: more text → smaller discrepancy (occupation r =
  −0.13, p = 0.04). I.7: partially pooled slopes, occupation 0.25 (SE 0.08),
  family −0.07.
- II: occupation 2000–09 → 2015–24: 22/48 states significantly less
  traditional, 3 more; family: 3 less, 2 more.
- II-B (between states, one block at a time): occupation text score higher
  where metro share and women's share of professionals are higher, lower with
  occupational segregation and Republican vote share; within states nothing
  significant.
- II-C: text-survey gap largest where women's share of professionals is high
  (both domains); the survey side of the gap is itself explained by context.
- III: case-selection table `main/tables/3_case_selection.csv`.

## 2026-10-05 — ATUS housework benchmark added

IPUMS ATUS (respondents 25–54, WT06 / WT20 for 2020): women's share of
household activities, housework, and childcare (parents) per state-window
(`scripts/data_prep/build_housework_measures.py`; jobs 3393466–3393468).
Women's share of household activities 0.633 (2000–09) → 0.611 (2015–24);
smallest state cells 66–118 respondents.

Robustness specs (family domain): family text vs household share β = 0.02,
vs childcare share β = 0.11 (both n.s., ≈ 0 with FE); household-work text vs
housework share β = −0.09, within-state r = −0.20 (p = 0.007, opposite
direction: household words drift female-ward while women's housework share
falls). No ATUS measure aligns with the text measures across states.
