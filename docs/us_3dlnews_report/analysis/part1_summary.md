# Part I — measurement validation

Orientation: state-level scores are higher = less traditional.

**1.1 Occupation-level validity.** 62 occupations, pooled over states and windows: r = 0.75 (p = 1.66e-12). No family analogue: there is no per-term external benchmark for family words.

**1.2 Temporal variation** (higher = less traditional). Text: occupation: 1995–04 -0.0093, 2000–09 -0.0122, 2005–14 -0.0101, 2010–19 -0.0065, 2015–24 -0.0049; Text: family sphere: 1995–04 -0.0115, 2000–09 -0.0112, 2005–14 -0.0118, 2010–19 -0.0132, 2015–24 -0.0120; Text: household work: 1995–04 +0.0052, 2000–09 +0.0030, 2005–14 +0.0019, 2010–19 -0.0024, 2015–24 -0.0052.

**1.3 Geographic heterogeneity.** occupation: SD across states 0.0056, median 95% half-width 0.0032; family: SD across states 0.0051, median 95% half-width 0.0049.

## main

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | matched_female_share | pooled | 0.094 | 0.096 | 0.323 | 232 | 0.009 |
| occupation | ours_occupation | matched_female_share | state + window FE | -0.019 | 0.032 | 0.551 | 232 | 0.925 |
| family | ours_family_sphere | family_index_acs | pooled | 0.032 | 0.071 | 0.651 | 232 | 0.001 |
| family | ours_family_sphere | family_index_acs | state + window FE | -0.045 | 0.022 | 0.037 | 232 | 0.962 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | -0.15 | 0.295 | 51 | -0.062 |
| occupation | within states | 0.276 | 0.0 | 232 | 0.14 |
| family | between states | 0.181 | 0.204 | 51 | 0.001 |
| family | within states | -0.135 | 0.04 | 232 | -0.002 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | -0.134 | 0.041 | -0.175 | 232 | 0.98 | 0.812 |
| family | -0.067 | 0.308 | -0.093 | 232 | 0.802 | 0.595 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.25 | 0.079 | 0.287 | 3 |
| family | -0.084 | 0.06 | 0.254 | 0 |

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
| family | ours_family_sphere | motherhood_emp_gap | pooled | -0.035 | 0.068 | 0.608 | 232 | 0.001 |
| family | ours_family_sphere | motherhood_emp_gap | state + window FE | -0.05 | 0.034 | 0.14 | 232 | 0.942 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.057 | 0.694 | 51 | 0.007 |
| family | within states | -0.156 | 0.018 | 232 | -0.052 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.071 | 0.284 | -0.094 | 232 | 0.843 | 0.574 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.072 | 0.049 | 0.189 | 1 |

## robustness-motherhood_hours_gap

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | motherhood_hours_gap | pooled | -0.024 | 0.072 | 0.743 | 232 | 0.001 |
| family | ours_family_sphere | motherhood_hours_gap | state + window FE | -0.073 | 0.027 | 0.007 | 232 | 0.935 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.127 | 0.374 | 51 | 0.001 |
| family | within states | -0.167 | 0.011 | 232 | -0.002 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.132 | 0.044 | -0.175 | 232 | 0.969 | 0.594 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.118 | 0.062 | 0.212 | 0 |

## robustness-married_women_nilf

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | married_women_nilf | pooled | 0.052 | 0.081 | 0.521 | 232 | 0.003 |
| family | ours_family_sphere | married_women_nilf | state + window FE | -0.023 | 0.02 | 0.258 | 232 | 0.974 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.132 | 0.355 | 51 | 0.015 |
| family | within states | -0.106 | 0.107 | 232 | -0.048 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.078 | 0.234 | -0.099 | 232 | 0.837 | 0.692 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.046 | 0.037 | 0.135 | 0 |

## robustness-wife_earnings_share

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | wife_earnings_share | pooled | 0.056 | 0.073 | 0.446 | 232 | 0.003 |
| family | ours_family_sphere | wife_earnings_share | state + window FE | -0.049 | 0.019 | 0.012 | 232 | 0.963 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.188 | 0.186 | 51 | 0.038 |
| family | within states | -0.166 | 0.011 | 232 | -0.133 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.103 | 0.117 | -0.145 | 232 | 0.798 | 0.572 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.066 | 0.039 | 0.16 | 2 |

## robustness-wife_earns_more

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | wife_earns_more | pooled | 0.056 | 0.08 | 0.484 | 232 | 0.003 |
| family | ours_family_sphere | wife_earns_more | state + window FE | -0.031 | 0.022 | 0.158 | 232 | 0.955 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.214 | 0.131 | 51 | 0.042 |
| family | within states | -0.113 | 0.087 | 232 | -0.051 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.128 | 0.051 | -0.178 | 232 | 0.821 | 0.563 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.074 | 0.063 | 0.281 | 2 |

## robustness-gender_emp_gap

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | gender_emp_gap | pooled | 0.056 | 0.078 | 0.473 | 232 | 0.003 |
| family | ours_family_sphere | gender_emp_gap | state + window FE | -0.002 | 0.02 | 0.938 | 232 | 0.958 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.162 | 0.257 | 51 | 0.031 |
| family | within states | -0.068 | 0.306 | 232 | -0.027 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.065 | 0.324 | -0.086 | 232 | 0.842 | 0.675 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.058 | 0.065 | 0.277 | 0 |

## robustness-iat

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | iat_sex_balanced | pooled | 0.405 | 0.083 | 0.0 | 199 | 0.164 |
| occupation | ours_occupation | iat_sex_balanced | state + window FE | -0.008 | 0.062 | 0.894 | 199 | 0.857 |
| family | ours_family_sphere | iat_sex_balanced | pooled | -0.044 | 0.081 | 0.591 | 199 | 0.002 |
| family | ours_family_sphere | iat_sex_balanced | state + window FE | 0.022 | 0.059 | 0.71 | 199 | 0.857 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | 0.493 | 0.0 | 51 | 0.237 |
| occupation | within states | 0.361 | 0.0 | 199 | 0.133 |
| family | between states | -0.101 | 0.481 | 51 | -0.047 |
| family | within states | -0.016 | 0.819 | 199 | -0.005 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | -0.054 | 0.448 | -0.068 | 199 | 0.868 | 0.813 |
| family | -0.032 | 0.658 | -0.04 | 199 | 0.957 | 0.908 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.439 | 0.082 | 0.28 | 17 |
| family | -0.067 | 0.089 | 0.313 | 1 |

## robustness-explicit

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | explicit_sex_balanced | pooled | 0.329 | 0.067 | 0.0 | 199 | 0.108 |
| occupation | ours_occupation | explicit_sex_balanced | state + window FE | -0.005 | 0.024 | 0.82 | 199 | 0.975 |
| family | ours_family_sphere | explicit_sex_balanced | pooled | -0.051 | 0.067 | 0.443 | 199 | 0.003 |
| family | ours_family_sphere | explicit_sex_balanced | state + window FE | -0.011 | 0.02 | 0.597 | 199 | 0.975 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | 0.112 | 0.433 | 51 | 0.006 |
| occupation | within states | 0.414 | 0.0 | 199 | 0.01 |
| family | between states | -0.034 | 0.81 | 51 | -0.002 |
| family | within states | -0.057 | 0.426 | 199 | -0.001 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | 0.027 | 0.708 | 0.032 | 199 | 0.815 | 0.839 |
| family | 0.083 | 0.245 | 0.102 | 199 | 0.845 | 0.923 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.331 | 0.134 | 0.041 | 51 |
| family | -0.067 | 0.069 | 0.198 | 0 |

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

## robustness-atus-household

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | women_share_household | pooled | 0.063 | 0.065 | 0.337 | 199 | 0.004 |
| family | ours_family_sphere | women_share_household | state + window FE | 0.071 | 0.082 | 0.384 | 199 | 0.652 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.069 | 0.631 | 51 | 0.011 |
| family | within states | 0.062 | 0.386 | 199 | 0.014 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.337 | 0.0 | -0.499 | 199 | 1.133 | 0.562 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | 0.056 | 0.068 | 0.029 | 0 |

## robustness-atus-childcare

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | women_share_childcare_parents | pooled | 0.05 | 0.078 | 0.52 | 199 | 0.002 |
| family | ours_family_sphere | women_share_childcare_parents | state + window FE | 0.096 | 0.118 | 0.418 | 199 | 0.476 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.001 | 0.996 | 51 | 0.0 |
| family | within states | 0.077 | 0.279 | 199 | 0.012 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.329 | 0.0 | -0.51 | 199 | 1.027 | 0.513 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | 0.026 | 0.085 | 0.297 | 1 |

## robustness-household-text-atus

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | women_share_housework | pooled | -0.094 | 0.092 | 0.309 | 191 | 0.009 |
| family | ours_household | women_share_housework | state + window FE | -0.09 | 0.121 | 0.457 | 191 | 0.455 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.062 | 0.671 | 50 | 0.015 |
| family | within states | -0.195 | 0.007 | 191 | -0.051 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.387 | 0.0 | -0.618 | 191 | 1.197 | 0.513 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.141 | 0.081 | 0.189 | 3 |
