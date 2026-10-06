# Part I — measurement validation

Orientation: state-level scores are higher = less traditional.

**1.1 Occupation-level validity.** 62 occupations, pooled over states and windows: r = 0.75 (p = 1.66e-12). No family analogue: there is no per-term external benchmark for family words.

**1.2 Temporal variation** (higher = less traditional). Text: occupation: 1995–04 -0.0093, 2000–09 -0.0122, 2005–14 -0.0101, 2010–19 -0.0065, 2015–24 -0.0049; Text: household work: 1995–04 -0.0022, 2000–09 -0.0010, 2005–14 -0.0038, 2010–19 -0.0081, 2015–24 -0.0096; Text: family sphere: 1995–04 -0.0115, 2000–09 -0.0112, 2005–14 -0.0118, 2010–19 -0.0132, 2015–24 -0.0120.

**1.3 Geographic heterogeneity.** occupation: SD across states 0.0056, median 95% half-width 0.0032; family: SD across states 0.0082, median 95% half-width 0.0053.

## main

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | matched_female_share | pooled | 0.094 | 0.096 | 0.323 | 232 | 0.009 |
| occupation | ours_occupation | matched_female_share | state + window FE | -0.019 | 0.032 | 0.551 | 232 | 0.925 |
| family | ours_household | family_index_acs | pooled | 0.071 | 0.077 | 0.354 | 228 | 0.005 |
| family | ours_household | family_index_acs | state + window FE | 0.018 | 0.034 | 0.599 | 228 | 0.962 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | -0.15 | 0.295 | 51 | -0.062 |
| occupation | within states | 0.276 | 0.0 | 232 | 0.14 |
| family | between states | 0.38 | 0.006 | 51 | 0.004 |
| family | within states | -0.228 | 0.001 | 228 | -0.007 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | -0.134 | 0.041 | -0.175 | 232 | 0.98 | 0.812 |
| family | -0.078 | 0.242 | -0.11 | 228 | 0.803 | 0.609 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.25 | 0.079 | 0.287 | 3 |
| family | -0.094 | 0.053 | 0.091 | 0 |

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
| family | ours_household | motherhood_emp_gap | pooled | 0.027 | 0.087 | 0.76 | 228 | 0.001 |
| family | ours_household | motherhood_emp_gap | state + window FE | 0.038 | 0.031 | 0.221 | 228 | 0.942 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.18 | 0.207 | 51 | 0.033 |
| family | within states | -0.154 | 0.02 | 228 | -0.096 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.069 | 0.298 | -0.095 | 228 | 0.851 | 0.588 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.069 | 0.031 | 0.049 | 18 |

## robustness-motherhood_hours_gap

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | motherhood_hours_gap | pooled | -0.052 | 0.073 | 0.48 | 228 | 0.003 |
| family | ours_household | motherhood_hours_gap | state + window FE | -0.018 | 0.029 | 0.532 | 228 | 0.935 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.19 | 0.183 | 51 | 0.002 |
| family | within states | -0.275 | 0.0 | 228 | -0.005 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.115 | 0.082 | -0.158 | 228 | 0.946 | 0.589 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.166 | 0.05 | 0.007 | 51 |

## robustness-married_women_nilf

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | married_women_nilf | pooled | 0.065 | 0.082 | 0.428 | 228 | 0.004 |
| family | ours_household | married_women_nilf | state + window FE | -0.008 | 0.039 | 0.83 | 228 | 0.974 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.282 | 0.045 | 51 | 0.051 |
| family | within states | -0.256 | 0.0 | 228 | -0.217 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.093 | 0.159 | -0.12 | 228 | 0.839 | 0.705 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.066 | 0.034 | 0.061 | 3 |

## robustness-wife_earnings_share

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | wife_earnings_share | pooled | 0.149 | 0.085 | 0.078 | 228 | 0.022 |
| family | ours_household | wife_earnings_share | state + window FE | 0.012 | 0.036 | 0.738 | 228 | 0.962 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.417 | 0.002 | 51 | 0.137 |
| family | within states | -0.192 | 0.004 | 228 | -0.286 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.096 | 0.149 | -0.137 | 228 | 0.781 | 0.575 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.045 | 0.036 | 0.071 | 1 |

## robustness-wife_earns_more

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | wife_earns_more | pooled | 0.086 | 0.081 | 0.29 | 228 | 0.007 |
| family | ours_household | wife_earns_more | state + window FE | 0.023 | 0.032 | 0.466 | 228 | 0.955 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.398 | 0.004 | 51 | 0.127 |
| family | within states | -0.217 | 0.001 | 228 | -0.183 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.132 | 0.047 | -0.187 | 228 | 0.824 | 0.565 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.086 | 0.049 | 0.025 | 10 |

## robustness-gender_emp_gap

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | gender_emp_gap | pooled | 0.083 | 0.076 | 0.271 | 228 | 0.007 |
| family | ours_household | gender_emp_gap | state + window FE | 0.042 | 0.028 | 0.135 | 228 | 0.962 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.381 | 0.006 | 51 | 0.115 |
| family | within states | -0.193 | 0.003 | 228 | -0.143 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.071 | 0.285 | -0.096 | 228 | 0.83 | 0.689 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.091 | 0.053 | 0.003 | 0 |

## robustness-iat

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | iat_sex_balanced | pooled | 0.405 | 0.083 | 0.0 | 199 | 0.164 |
| occupation | ours_occupation | iat_sex_balanced | state + window FE | -0.008 | 0.062 | 0.894 | 199 | 0.857 |
| family | ours_household | iat_sex_balanced | pooled | -0.152 | 0.083 | 0.066 | 198 | 0.023 |
| family | ours_household | iat_sex_balanced | state + window FE | 0.112 | 0.077 | 0.149 | 198 | 0.865 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | 0.493 | 0.0 | 51 | 0.237 |
| occupation | within states | 0.361 | 0.0 | 199 | 0.133 |
| family | between states | -0.111 | 0.437 | 51 | -0.071 |
| family | within states | -0.18 | 0.011 | 198 | -0.102 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | -0.054 | 0.448 | -0.068 | 199 | 0.868 | 0.813 |
| family | 0.001 | 0.988 | 0.001 | 198 | 0.951 | 0.909 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.439 | 0.082 | 0.28 | 17 |
| family | -0.179 | 0.082 | 0.164 | 1 |

## robustness-explicit

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| occupation | ours_occupation | explicit_sex_balanced | pooled | 0.329 | 0.067 | 0.0 | 199 | 0.108 |
| occupation | ours_occupation | explicit_sex_balanced | state + window FE | -0.005 | 0.024 | 0.82 | 199 | 0.975 |
| family | ours_household | explicit_sex_balanced | pooled | -0.205 | 0.063 | 0.001 | 198 | 0.042 |
| family | ours_household | explicit_sex_balanced | state + window FE | 0.001 | 0.022 | 0.973 | 198 | 0.975 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| occupation | between states | 0.112 | 0.433 | 51 | 0.006 |
| occupation | within states | 0.414 | 0.0 | 199 | 0.01 |
| family | between states | 0.14 | 0.328 | 51 | 0.01 |
| family | within states | -0.309 | 0.0 | 198 | -0.012 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| occupation | 0.027 | 0.708 | 0.032 | 199 | 0.815 | 0.839 |
| family | 0.102 | 0.152 | 0.128 | 198 | 0.762 | 0.893 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| occupation | 0.331 | 0.134 | 0.041 | 51 |
| family | -0.205 | 0.094 | 0.008 | 51 |

## robustness-atus-housework

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | women_share_housework | pooled | -0.03 | 0.104 | 0.774 | 198 | 0.001 |
| family | ours_household | women_share_housework | state + window FE | 0.009 | 0.096 | 0.925 | 198 | 0.469 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.123 | 0.388 | 51 | 0.028 |
| family | within states | -0.121 | 0.09 | 198 | -0.03 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.382 | 0.0 | -0.58 | 198 | 1.204 | 0.487 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.105 | 0.084 | 0.221 | 5 |

## robustness-atus-household

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | women_share_household | pooled | -0.032 | 0.115 | 0.779 | 198 | 0.001 |
| family | ours_household | women_share_household | state + window FE | -0.059 | 0.104 | 0.572 | 198 | 0.659 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.11 | 0.443 | 51 | 0.024 |
| family | within states | -0.171 | 0.016 | 198 | -0.068 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.304 | 0.0 | -0.44 | 198 | 1.101 | 0.586 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.079 | 0.08 | 0.325 | 1 |

## robustness-atus-childcare

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_household | women_share_childcare_parents | pooled | 0.099 | 0.103 | 0.336 | 198 | 0.01 |
| family | ours_household | women_share_childcare_parents | state + window FE | 0.156 | 0.119 | 0.191 | 198 | 0.499 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.159 | 0.264 | 51 | 0.034 |
| family | within states | 0.059 | 0.409 | 198 | 0.016 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.304 | 0.0 | -0.482 | 198 | 1.02 | 0.503 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | 0.034 | 0.095 | 0.386 | 3 |

## robustness-family-sphere-text

### 1.4 State-window alignment (standardized; SE clustered by state)

| domain | text | survey | model | beta_std | se | p | n | r2 |
|---|---|---|---|---|---|---|---|---|
| family | ours_family_sphere | family_index_acs | pooled | 0.032 | 0.071 | 0.651 | 232 | 0.001 |
| family | ours_family_sphere | family_index_acs | state + window FE | -0.045 | 0.022 | 0.037 | 232 | 0.962 |

### 1.5 Between- and within-state alignment

| domain | component | r | p | n | slope |
|---|---|---|---|---|---|
| family | between states | 0.181 | 0.204 | 51 | 0.001 |
| family | within states | -0.135 | 0.04 | 232 | -0.002 |

### 1.6 Discrepancy vs text volume

| domain | r_abs_error_log_tokens | p | slope_per_log10_tokens | n | mean_abs_error_q1 | mean_abs_error_q5 |
|---|---|---|---|---|---|---|
| family | -0.067 | 0.308 | -0.093 | 232 | 0.802 | 0.595 |

### 1.7 Hierarchical state-specific slopes

| domain | global_slope | global_se | slope_sd | states_ci_excl_0 |
|---|---|---|---|---|
| family | -0.084 | 0.06 | 0.254 | 0 |
