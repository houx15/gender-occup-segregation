# Provincial Data Calculation Method

This document records how each output column is calculated for the cleaned
provincial dataset.

## Output Columns

- `province`: Short province key (e.g., 北京、河北、内蒙古) that matches the keys in
  `PROVINCE_NAME_MAPPING`.
- `gdp_2024`: 2024 GDP for each province, read from `gdp.csv`.
- `eduy_gt25_2020`: Average years of education for population age 25+ (total).
- `eduy_m_gt25_2020`: Average years of education for males age 25+.
- `eduy_f_gt25_2020`: Average years of education for females age 25+.
- `emp_2020`: Overall employment rate based on age- and sex-specific ranges.
- `emp_m_2020`: Male employment rate for ages 16–60.
- `emp_f_2020`: Female employment rate for ages 16–55.
- `avg_income_2024`: 2024 urban unit employed persons' average wage.

## 1) GDP (2024)

Source: `gender_norms/provincial/gdp.csv`

- The file contains 3 metadata rows; data starts after them.
- Use the column `2024年` and rename it to `gdp_2024`.
- Province names are normalized to the short key used in
  `PROVINCE_NAME_MAPPING`.

## 2) Education (25+ average years, 2020)

Source: `gender_norms/provincial/gender_education_25.xls`

Education years mapping:

- 未上过学 = 0
- 学前教育 = 0
- 小学 = 6
- 初中 = 9
- 高中 = 12
- 大学专科 = 15
- 大学本科 = 16
- 硕士研究生 = 19
- 博士研究生 = 23

For each province:

- Total average years:

  `eduy_gt25_2020 = Σ(level_pop_total × years) / total_pop_25plus`

- Male average years:

  `eduy_m_gt25_2020 = Σ(level_pop_male × years) / male_pop_25plus`

- Female average years:

  `eduy_f_gt25_2020 = Σ(level_pop_female × years) / female_pop_25plus`

Where counts come from the “25岁及以上人口” block of the table.

## 3) Employment Rate (2020)

Sources:

- `gender_norms/provincial/gender_employment.xls` (employment counts)
- `gender_norms/provincial/gender_population.xls` (population counts)

Employment numerator (from `gender_employment.xls`):

- Male employed, ages 16–60: sum of
  `16-19`, `20-24`, `25-29`, `30-34`, `35-39`, `40-44`, `45-49`, `50-54`,
  `55-59`, `60-64`
- Female employed, ages 16–55: sum of
  `16-19`, `20-24`, `25-29`, `30-34`, `35-39`, `40-44`, `45-49`, `50-54`,
  `55-59`

Population denominator (from `gender_population.xls`):

- Male population, ages 16–60: sum of
  `15-19`, `20-24`, `25-29`, `30-34`, `35-39`, `40-44`, `45-49`, `50-54`,
  `55-59`, `60-64`
- Female population, ages 16–55: sum of
  `15-19`, `20-24`, `25-29`, `30-34`, `35-39`, `40-44`, `45-49`, `50-54`,
  `55-59`

Notes on age-bin alignment:

- Employment uses `16-19` while population uses `15-19`. We approximate
  `16-19` with the full `15-19` population bin.
- For 55–59 and 60–64, full bins are used to approximate single-year endpoints.

Rates:

- `emp_m_2020 = male_employed_16_60 / male_population_16_60`
- `emp_f_2020 = female_employed_16_55 / female_population_16_55`
- `emp_2020 = (male_employed_16_60 + female_employed_16_55) / (male_population_16_60 + female_population_16_55)`

## 4) Average Income (2024)

Source: `gender_norms/provincial/income.csv`

- The file contains 3 metadata rows; data starts after them.
- Use the column `2024年` and rename it to `avg_income_2024`.
- Province names are normalized to the short key used in
  `PROVINCE_NAME_MAPPING`.
