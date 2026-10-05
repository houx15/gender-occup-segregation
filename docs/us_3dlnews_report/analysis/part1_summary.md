# Part I — measurement validation

Orientation: state-level scores are higher = more traditional.

**1.1 Occupation-level validity.** 62 occupations, pooled over states and windows: r = 0.75 (p = 1.66e-12). No family analogue: there is no per-term external benchmark for family words.

**1.2 Temporal variation** (higher = more traditional). Text: occupation: 1995–04 +0.0093, 2000–09 +0.0122, 2005–14 +0.0101, 2010–19 +0.0065, 2015–24 +0.0049; Text: family sphere: 1995–04 +0.0106, 2000–09 +0.0090, 2005–14 +0.0098, 2010–19 +0.0104, 2015–24 +0.0109; Text: household work: 1995–04 -0.0052, 2000–09 -0.0030, 2005–14 -0.0019, 2010–19 +0.0024, 2015–24 +0.0052.

**1.3 Geographic heterogeneity.** occupation: SD across states 0.0056, median 95% half-width 0.0032; family: SD across states 0.0059, median 95% half-width 0.0063.

## main

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | matched_female_share | pooled | 0.094 | 0.096 | 0.323 | 232 | 0.009 |
| occupation | ours_occupation | matched_female_share | state + window FE | -0.019 | 0.032 | 0.551 | 232 | 0.925 |
| family | ours_family_sphere | family_index_acs | pooled | 0.038 | 0.06 | 0.524 | 232 | 0.001 |
| family | ours_family_sphere | family_index_acs | state + window FE | -0.018 | 0.026 | 0.479 | 232 | 0.961 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | -0.15 | 0.295 | 51 | -0.062 |
| occupation | within states | 0.276 | 0.0 | 232 | 0.14 |
| family | between states | 0.149 | 0.297 | 51 | 0.001 |
| family | within states | -0.084 | 0.201 | 232 | -0.001 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | -0.134 | 0.041 | -0.175 | 232 | 0.98 | 0.812 |
| family | -0.065 | 0.327 | -0.09 | 232 | 0.8 | 0.597 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.25 | 0.079 | 0.287 | 3 |
| family | -0.067 | 0.061 | 0.237 | 0 |

## robustness-duncan

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | duncan | pooled | -0.135 | 0.125 | 0.282 | 232 | 0.018 |
| occupation | ours_occupation | duncan | state + window FE | 0.006 | 0.018 | 0.75 | 232 | 0.986 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | -0.403 | 0.003 | 51 | -0.056 |
| occupation | within states | 0.32 | 0.0 | 232 | 0.153 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | -0.074 | 0.264 | -0.115 | 232 | 0.694 | 0.669 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.133 | 0.034 | 0.072 | 35 |

## robustness-female_emp_share

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | female_emp_share | pooled | -0.101 | 0.094 | 0.286 | 232 | 0.01 |
| occupation | ours_occupation | female_emp_share | state + window FE | 0.009 | 0.035 | 0.803 | 232 | 0.956 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | -0.265 | 0.06 | 51 | -0.119 |
| occupation | within states | 0.23 | 0.0 | 232 | 0.364 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | 0.006 | 0.922 | 0.009 | 232 | 0.727 | 0.747 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.114 | 0.039 | 0.138 | 1 |

## robustness-motherhood_emp_gap

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | motherhood_emp_gap | pooled | 0.002 | 0.055 | 0.973 | 232 | 0.0 |
| family | ours_family_sphere | motherhood_emp_gap | state + window FE | -0.02 | 0.041 | 0.627 | 232 | 0.941 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.065 | 0.653 | 51 | 0.009 |
| family | within states | -0.083 | 0.207 | 232 | -0.03 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.066 | 0.316 | -0.088 | 232 | 0.841 | 0.58 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.052 | 0.048 | 0.164 | 1 |

## robustness-motherhood_hours_gap

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | motherhood_hours_gap | pooled | 0.013 | 0.061 | 0.837 | 232 | 0.0 |
| family | ours_family_sphere | motherhood_hours_gap | state + window FE | -0.06 | 0.029 | 0.041 | 232 | 0.934 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.159 | 0.265 | 51 | 0.001 |
| family | within states | -0.137 | 0.037 | 232 | -0.001 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.135 | 0.041 | -0.178 | 232 | 0.972 | 0.597 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.075 | 0.275 | 0.085 | 0 |

## robustness-married_women_nilf

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | married_women_nilf | pooled | 0.009 | 0.084 | 0.912 | 232 | 0.0 |
| family | ours_family_sphere | married_women_nilf | state + window FE | -0.005 | 0.023 | 0.827 | 232 | 0.973 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.044 | 0.759 | 51 | 0.006 |
| family | within states | -0.063 | 0.336 | 232 | -0.031 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.088 | 0.18 | -0.111 | 232 | 0.846 | 0.688 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.039 | 0.037 | 0.121 | 0 |

## robustness-wife_earnings_share

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | wife_earnings_share | pooled | 0.041 | 0.067 | 0.541 | 232 | 0.002 |
| family | ours_family_sphere | wife_earnings_share | state + window FE | -0.022 | 0.025 | 0.377 | 232 | 0.962 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.133 | 0.351 | 51 | 0.031 |
| family | within states | -0.102 | 0.122 | 232 | -0.087 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.101 | 0.127 | -0.141 | 232 | 0.792 | 0.575 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.05 | 0.04 | 0.165 | 1 |

## robustness-wife_earns_more

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | wife_earns_more | pooled | 0.056 | 0.071 | 0.432 | 232 | 0.003 |
| family | ours_family_sphere | wife_earns_more | state + window FE | -0.009 | 0.025 | 0.716 | 232 | 0.954 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.182 | 0.202 | 51 | 0.042 |
| family | within states | -0.071 | 0.283 | 232 | -0.034 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.124 | 0.06 | -0.173 | 232 | 0.813 | 0.563 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.068 | 0.065 | 0.286 | 1 |

## robustness-gender_emp_gap

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | gender_emp_gap | pooled | 0.073 | 0.075 | 0.331 | 232 | 0.005 |
| family | ours_family_sphere | gender_emp_gap | state + window FE | 0.023 | 0.025 | 0.339 | 232 | 0.958 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.155 | 0.277 | 51 | 0.034 |
| family | within states | -0.02 | 0.758 | 232 | -0.009 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.06 | 0.364 | -0.079 | 232 | 0.835 | 0.677 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.042 | 0.068 | 0.29 | 0 |

## robustness-iat

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | iat_sex_balanced | pooled | 0.405 | 0.083 | 0.0 | 199 | 0.164 |
| occupation | ours_occupation | iat_sex_balanced | state + window FE | -0.008 | 0.062 | 0.894 | 199 | 0.857 |
| family | ours_family_sphere | iat_sex_balanced | pooled | -0.035 | 0.077 | 0.65 | 199 | 0.001 |
| family | ours_family_sphere | iat_sex_balanced | state + window FE | 0.115 | 0.05 | 0.023 | 199 | 0.864 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | 0.493 | 0.0 | 51 | 0.237 |
| occupation | within states | 0.361 | 0.0 | 199 | 0.133 |
| family | between states | -0.128 | 0.369 | 51 | -0.069 |
| family | within states | 0.013 | 0.855 | 199 | 0.005 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | -0.054 | 0.448 | -0.068 | 199 | 0.868 | 0.813 |
| family | -0.032 | 0.654 | -0.041 | 199 | 0.96 | 0.912 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.439 | 0.082 | 0.28 | 17 |
| family | -0.044 | 0.084 | 0.105 | 0 |

## robustness-explicit

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | explicit_sex_balanced | pooled | 0.329 | 0.067 | 0.0 | 199 | 0.108 |
| occupation | ours_occupation | explicit_sex_balanced | state + window FE | -0.005 | 0.024 | 0.82 | 199 | 0.975 |
| family | ours_family_sphere | explicit_sex_balanced | pooled | -0.067 | 0.064 | 0.3 | 199 | 0.004 |
| family | ours_family_sphere | explicit_sex_balanced | state + window FE | -0.018 | 0.02 | 0.355 | 199 | 0.975 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | 0.112 | 0.433 | 51 | 0.006 |
| occupation | within states | 0.414 | 0.0 | 199 | 0.01 |
| family | between states | 0.05 | 0.729 | 51 | 0.003 |
| family | within states | -0.111 | 0.118 | 199 | -0.003 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | 0.027 | 0.708 | 0.032 | 199 | 0.815 | 0.839 |
| family | 0.085 | 0.232 | 0.105 | 199 | 0.842 | 0.928 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.331 | 0.134 | 0.041 | 51 |
| family | -0.079 | 0.063 | 0.099 | 0 |

## robustness-household-text

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | family_index_acs | pooled | -0.003 | 0.075 | 0.966 | 217 | 0.0 |
| family | ours_household | family_index_acs | state + window FE | 0.022 | 0.026 | 0.405 | 217 | 0.967 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.13 | 0.361 | 51 | 0.002 |
| family | within states | -0.239 | 0.0 | 217 | -0.007 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.139 | 0.04 | -0.208 | 217 | 0.858 | 0.599 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.123 | 0.046 | 0.045 | 39 |
