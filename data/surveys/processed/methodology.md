# Gender Ideation Index: Methodology

## Overview

The gender ideation index measures traditional vs. progressive gender attitudes on a **[0, 1]** scale, where **0 = most progressive** and **1 = most traditional**.

For each respondent, the score is the mean of all valid (non-missing) items after recoding. The province-year aggregates are simple means of individual scores.

---

## Missing Value Handling

All non-substantive response codes are converted to missing (NaN) before scoring. No imputation is applied. A respondent's score is the mean of their valid items only.

| Dataset   | Missing codes treated as NaN              |
|-----------|-------------------------------------------|
| ACWF 1990 | 0 (no response), Stata system-missing     |
| ACWF 2000 | 8 (don't know), 9 (refused), system-missing |
| ACWF 2010 | 8 (don't know), system-missing            |
| CFPS 2014 | -1 (don't know), -2 (refused), system-missing |
| CFPS 2020 | -8 (not applicable), -1 (don't know), -2 (refused), system-missing |

Implementation: `df.loc[~df[col].between(1, scale_max), col] = NaN` — this catches all codes outside the valid Likert range in a single step.

---

## ACWF 1990 (Chinese Women's Status Survey)

**Source file:** `中国妇女地位调查/1990/w1990_fixed.dta`
**Respondents:** 23,722 (after dropping all-missing)
**Provinces:** 11
**Scale:** 5-point (1 = strongly agree, 2 = agree, 3 = indifferent, 4 = not necessarily, 5 = disagree)

### Items and recoding

All items are traditional-leaning statements except w615. For traditional items, agreeing (raw 1) = traditional, so we reverse: **(6 - x - 1) / 4**.  For progressive items, agreeing (raw 1) = progressive = low score, so we keep direction: **(x - 1) / 4**.

| Item | Statement | Direction | Formula |
|------|-----------|-----------|---------|
| w611 | Men focus on society, women on family | Traditional | (6 - x - 1) / 4 |
| w612 | Men are innately more capable than women | Traditional | (6 - x - 1) / 4 |
| w613 | Women should avoid surpassing husband's social status | Traditional | (6 - x - 1) / 4 |
| w614 | Husband's success is wife's success, wife should fully support husband | Traditional | (6 - x - 1) / 4 |
| w615 | Let your child take the mother's surname | Progressive | (x - 1) / 4 |
| w616 | Women haven't really played a half-the-sky role in political/economic life | Traditional | (6 - x - 1) / 4 |
| w617 | Men should be responsible for external family affairs | Traditional | (6 - x - 1) / 4 |
| w618 | A widow who remarries should leave property to ex-husband's children/family | Traditional | (6 - x - 1) / 4 |

**Score** = mean of 8 normalized items per respondent.

---

## ACWF 2000

**Source file:** `中国妇女地位调查/2000/w2000.dta`
**Respondents:** 19,283
**Provinces:** 30
**Scale:** 4-point (1 = strongly agree, 2 = somewhat agree, 3 = somewhat disagree, 4 = strongly disagree)

### Items and recoding

For traditional items: agreeing (raw 1) = traditional, reverse: **(5 - x - 1) / 3**. For progressive items: agreeing (raw 1) = progressive = low score, keep: **(x - 1) / 3**.

Items i3_f and i3_j are excluded (not in the analysis plan).

| Item | Statement | Direction | Formula |
|------|-----------|-----------|---------|
| i3_a | Men focus on society, women on family | Traditional | (5 - x - 1) / 3 |
| i3_b | Men are innately more capable than women | Traditional | (5 - x - 1) / 3 |
| i3_c | Doing well is not as good as marrying well | Traditional | (5 - x - 1) / 3 |
| i3_d | A woman without children is not a complete woman | Traditional | (5 - x - 1) / 3 |
| i3_e | Women should avoid surpassing husband's social status | Traditional | (5 - x - 1) / 3 |
| i3_g | For women seeking jobs, looks matter more than ability | Traditional | (5 - x - 1) / 3 |
| i3_h | At least 30% of senior government leaders should be women | Progressive | (x - 1) / 3 |
| i3_i | Men should share half the housework | Progressive | (x - 1) / 3 |

**Score** = mean of 8 normalized items per respondent.

---

## ACWF 2010

**Source file:** `中国妇女地位调查/2010/w2010.DTA`
**Respondents:** 26,021
**Provinces:** 31
**Scale:** 4-point (1 = strongly agree, 2 = somewhat agree, 3 = somewhat disagree, 4 = strongly disagree; 8 = don't know)

### Items and recoding

Same logic as 2000. Traditional items: **(5 - x - 1) / 3**. Progressive items: **(x - 1) / 3**.

| Item | Statement | Direction | Formula |
|------|-----------|-----------|---------|
| J2A | Women are no less capable than men | Progressive | (x - 1) / 3 |
| J2B | Men should focus on society, women on family | Traditional | (5 - x - 1) / 3 |
| J2C | Breadwinning is mainly men's responsibility | Traditional | (5 - x - 1) / 3 |
| J2D | Husband's career development is more important than wife's | Traditional | (5 - x - 1) / 3 |
| J2E | Men should also actively share housework | Progressive | (x - 1) / 3 |
| J2F | Boys should act like boys, girls like girls | Traditional | (5 - x - 1) / 3 |
| J2G | Doing well is not as good as marrying well | Traditional | (5 - x - 1) / 3 |
| J2H | Leadership positions should be roughly gender-equal | Progressive | (x - 1) / 3 |
| J2I | Gender equality won't happen naturally, needs active promotion | Progressive | (x - 1) / 3 |

**Score** = mean of 9 normalized items per respondent.

---

## CFPS 2014 (China Family Panel Studies)

**Source file:** `CFPS/cfps2014_adult.dta`
**Respondents:** 31,554
**Provinces:** 28
**Scale:** 5-point (1 = strongly disagree, 5 = strongly agree)

Note: CFPS scale runs **opposite** to ACWF (1 = disagree here vs. 1 = agree in ACWF).

### Items and recoding

For traditional items: agreeing (raw 5) = traditional, so we keep direction: **(x - 1) / 4**. For progressive items: agreeing (raw 5) = progressive = low score, reverse: **(6 - x - 1) / 4**.

| Item | Statement | Direction | Formula |
|------|-----------|-----------|---------|
| qm1101 | Men for career, women for family | Traditional | (x - 1) / 4 |
| qm1102 | Doing well is not as good as marrying well | Traditional | (x - 1) / 4 |
| qm1103 | A woman needs children to be complete | Traditional | (x - 1) / 4 |
| qm1104 | Men should share half the housework | Progressive | (6 - x - 1) / 4 |

**Score** = mean of 4 normalized items per respondent.

---

## CFPS 2020

**Source file:** `CFPS/cfps2020_adult.dta`
**Respondents:** 22,692
**Provinces:** 31
**Scale:** 5-point (same as CFPS 2014)

### Items and recoding

Identical to CFPS 2014.

| Item | Statement | Direction | Formula |
|------|-----------|-----------|---------|
| qm1101 | Men for career, women for family | Traditional | (x - 1) / 4 |
| qm1102 | Doing well is not as good as marrying well | Traditional | (x - 1) / 4 |
| qm1103 | A woman needs children to be complete | Traditional | (x - 1) / 4 |
| qm1104 | Men should share half the housework | Progressive | (6 - x - 1) / 4 |

**Score** = mean of 4 normalized items per respondent.

---

## Cross-survey comparability

ACWF and CFPS are **not directly comparable** in absolute levels because:

1. Different number of items (8-9 vs. 4)
2. Different item content coverage (ACWF covers leadership, family name, property; CFPS is narrower)
3. Different scale anchors (ACWF 1990 has a 5-point scale with "indifferent" and "not necessarily" as midpoints; ACWF 2000/2010 use a clean 4-point agree-disagree; CFPS uses a 5-point disagree-agree)
4. Different sampling frames and provincial coverage

**Within-dataset temporal comparisons are valid** (e.g., ACWF 1990 vs. 2000 vs. 2010; CFPS 2014 vs. 2020). Cross-dataset comparisons should be limited to rank-ordering of provinces or directional trends rather than absolute score differences.

---

## Output files

| File | Description |
|------|-------------|
| `gender_ideation_by_province_year.csv` | Province-level mean, SD, and N per dataset-year |
| `gender_ideation_by_year.csv` | National-level mean, SD, N, and province count per dataset-year |
| `gender_ideation_individual.parquet` | Individual-level scores with province code, for custom aggregation |

Province codes follow the GB/T 2260 standard (11 = Beijing, 12 = Tianjin, ..., 65 = Xinjiang).
