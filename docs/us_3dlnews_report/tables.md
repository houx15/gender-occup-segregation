### Dataset volume per window (state models: states with >= 500 articles)

| window | states_with_articles | states_modelled | articles_total | articles_median | articles_min | articles_max | tokens_total_millions | tokens_median_millions | tokens_min_millions | tokens_max_millions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1995–04 | 51 | 33 | 60972 | 840.0 | 46 | 5560 | 25.1319 | 0.4293 | 0.2213 | 3.0918 |
| 2000–09 | 51 | 48 | 157185 | 2117.0 | 73 | 19166 | 64.6202 | 0.8805 | 0.2084 | 8.5406 |
| 2005–14 | 51 | 49 | 342228 | 4259.0 | 108 | 40486 | 131.8669 | 1.7897 | 0.4151 | 15.8804 |
| 2010–19 | 51 | 51 | 588326 | 7841.0 | 970 | 61478 | 226.972 | 3.1036 | 0.4077 | 24.4076 |
| 2015–24 | 51 | 51 | 780648 | 11559.0 | 983 | 66627 | 302.2031 | 4.1043 | 0.4117 | 27.4561 |

### Training set-up

| setting | value |
|---|---|
| Text | 3DLNews2, Google-platform newspaper articles |
| Unit | state x 10-year window, step 5 (1995–04, 2000–09, 2005–14, 2010–19, 2015–24) |
| Minimum articles for a state model | 500 |
| Preprocessing | tokenizer nltk_en, stopwords en_default, lowercase True, min 5 tokens per article |
| De-duplication | shingle (k=8), within window, across states |
| Model | word2vec, skip-gram |
| vector_size / window / min_count | 300 / 5 / 20 |
| negative / epochs / seed | 10 / 10 / 42 |

### Word lists: candidates, words used (in vocabulary of >= 0.5 of state models), median coverage

| category | candidates | used | median_coverage |
|---|---|---|---|
| family_sphere | 10 | 10 | 0.9655 |
| household | 20 | 6 | 0.2996 |
| occupation | 133 | 65 | 0.4871 |

### Survey measures per window (across states)

| measure | window | states | mean | sd | min | max |
|---|---|---|---|---|---|---|
| matched_female_share | 1995–04 | 33 | 0.3793 | 0.0172 | 0.3487 | 0.4275 |
| matched_female_share | 2000–09 | 48 | 0.3866 | 0.015 | 0.3511 | 0.4344 |
| matched_female_share | 2005–14 | 49 | 0.3964 | 0.0157 | 0.3563 | 0.446 |
| matched_female_share | 2010–19 | 51 | 0.406 | 0.0142 | 0.3738 | 0.4587 |
| matched_female_share | 2015–24 | 51 | 0.4183 | 0.0124 | 0.3878 | 0.4483 |
| duncan | 1995–04 | 33 | 0.5263 | 0.039 | 0.3565 | 0.5921 |
| duncan | 2000–09 | 48 | 0.5289 | 0.0394 | 0.3318 | 0.5955 |
| duncan | 2005–14 | 49 | 0.5217 | 0.0432 | 0.3042 | 0.5924 |
| duncan | 2010–19 | 51 | 0.5093 | 0.0433 | 0.2939 | 0.5861 |
| duncan | 2015–24 | 51 | 0.4933 | 0.044 | 0.2745 | 0.5642 |
| female_emp_share | 1995–04 | 33 | 0.4671 | 0.0125 | 0.4391 | 0.5056 |
| female_emp_share | 2000–09 | 48 | 0.4684 | 0.0127 | 0.4392 | 0.5073 |
| female_emp_share | 2005–14 | 49 | 0.4731 | 0.0132 | 0.4407 | 0.5115 |
| female_emp_share | 2010–19 | 51 | 0.4755 | 0.0134 | 0.4431 | 0.5151 |
| female_emp_share | 2015–24 | 51 | 0.475 | 0.0135 | 0.4445 | 0.5192 |
| family_index_acs | 1995–04 | 33 | 0.6887 | 0.7133 | -0.598 | 3.1717 |
| family_index_acs | 2000–09 | 48 | 0.3895 | 0.7314 | -1.0197 | 3.1609 |
| family_index_acs | 2005–14 | 49 | -0.0289 | 0.7251 | -1.5121 | 2.9498 |
| family_index_acs | 2010–19 | 51 | -0.288 | 0.7494 | -1.5715 | 2.8566 |
| family_index_acs | 2015–24 | 51 | -0.4965 | 0.7504 | -2.1426 | 2.5843 |
| motherhood_emp_gap | 1995–04 | 33 | 0.1586 | 0.0511 | 0.0448 | 0.2744 |
| motherhood_emp_gap | 2000–09 | 48 | 0.1435 | 0.0448 | 0.0404 | 0.2776 |
| motherhood_emp_gap | 2005–14 | 49 | 0.124 | 0.0424 | 0.033 | 0.2711 |
| motherhood_emp_gap | 2010–19 | 51 | 0.1107 | 0.0449 | 0.02 | 0.2736 |
| motherhood_emp_gap | 2015–24 | 51 | 0.1082 | 0.0432 | 0.0132 | 0.2689 |
| motherhood_hours_gap | 1995–04 | 33 | 4.5367 | 1.2526 | 2.0875 | 8.1019 |
| motherhood_hours_gap | 2000–09 | 48 | 4.2376 | 1.0882 | 1.9646 | 7.6953 |
| motherhood_hours_gap | 2005–14 | 49 | 3.5516 | 0.9558 | 1.5972 | 6.9096 |
| motherhood_hours_gap | 2010–19 | 51 | 3.0246 | 0.9058 | 1.4797 | 7.1255 |
| motherhood_hours_gap | 2015–24 | 51 | 2.7936 | 0.8964 | 1.4278 | 7.2183 |
| married_women_nilf | 1995–04 | 33 | 0.2696 | 0.0407 | 0.1867 | 0.3458 |
| married_women_nilf | 2000–09 | 48 | 0.2572 | 0.0452 | 0.1683 | 0.3466 |
| married_women_nilf | 2005–14 | 49 | 0.2474 | 0.0453 | 0.1535 | 0.3519 |
| married_women_nilf | 2010–19 | 51 | 0.243 | 0.0467 | 0.1509 | 0.3437 |
| married_women_nilf | 2015–24 | 51 | 0.2305 | 0.0452 | 0.1273 | 0.3173 |
| wife_earnings_share | 1995–04 | 33 | 0.3486 | 0.0225 | 0.2734 | 0.3926 |
| wife_earnings_share | 2000–09 | 48 | 0.3555 | 0.0253 | 0.2708 | 0.4169 |
| wife_earnings_share | 2005–14 | 49 | 0.3615 | 0.0255 | 0.2716 | 0.4226 |
| wife_earnings_share | 2010–19 | 51 | 0.365 | 0.0258 | 0.2736 | 0.421 |
| wife_earnings_share | 2015–24 | 51 | 0.3717 | 0.0256 | 0.2823 | 0.4397 |
| wife_earns_more | 1995–04 | 33 | 0.2517 | 0.0228 | 0.1776 | 0.3234 |
| wife_earns_more | 2000–09 | 48 | 0.2608 | 0.0242 | 0.1793 | 0.3325 |
| wife_earns_more | 2005–14 | 49 | 0.2762 | 0.0255 | 0.1894 | 0.3492 |
| wife_earns_more | 2010–19 | 51 | 0.2855 | 0.0273 | 0.1952 | 0.3564 |
| wife_earns_more | 2015–24 | 51 | 0.2928 | 0.0284 | 0.2018 | 0.3756 |
| gender_emp_gap | 1995–04 | 33 | 0.1284 | 0.0286 | 0.0589 | 0.2101 |
| gender_emp_gap | 2000–09 | 48 | 0.116 | 0.0288 | 0.0532 | 0.2116 |
| gender_emp_gap | 2005–14 | 49 | 0.098 | 0.0279 | 0.042 | 0.2047 |
| gender_emp_gap | 2010–19 | 51 | 0.0885 | 0.0268 | 0.0321 | 0.1938 |
| gender_emp_gap | 2015–24 | 51 | 0.0813 | 0.0267 | 0.0124 | 0.1782 |
| iat_sex_balanced | 2000–09 | 48 | 0.3679 | 0.0165 | 0.3118 | 0.4085 |
| iat_sex_balanced | 2005–14 | 49 | 0.3704 | 0.0147 | 0.3204 | 0.3934 |
| iat_sex_balanced | 2010–19 | 51 | 0.3552 | 0.0111 | 0.3267 | 0.3707 |
| iat_sex_balanced | 2015–24 | 51 | 0.332 | 0.012 | 0.3024 | 0.3627 |
| explicit_sex_balanced | 2000–09 | 48 | 1.5647 | 0.1215 | 1.2677 | 2.0061 |
| explicit_sex_balanced | 2005–14 | 49 | 1.6032 | 0.1172 | 1.3613 | 2.065 |
| explicit_sex_balanced | 2010–19 | 51 | 1.3077 | 0.1056 | 1.064 | 1.7277 |
| explicit_sex_balanced | 2015–24 | 51 | 0.9721 | 0.0743 | 0.8547 | 1.3105 |

### Survey sample sizes (unweighted respondents)

| source | window | states | respondents_total | respondents_median_per_state | respondents_min_per_state |
|---|---|---|---|---|---|
| ACS occupation (employed persons) | 1995–04 | 51 | 8626095 | 117238 | 27129 |
| ACS occupation (employed persons) | 2000–09 | 51 | 15573502 | 217485 | 41306 |
| ACS occupation (employed persons) | 2005–14 | 51 | 13897561 | 187510 | 28007 |
| ACS occupation (employed persons) | 2010–19 | 51 | 14383414 | 188541 | 28144 |
| ACS occupation (employed persons) | 2015–24 | 51 | 14889472 | 190654 | 27653 |
| ACS family (adults 25-54) | 1995–04 | 51 | 8003462 | 108826 | 25983 |
| ACS family (adults 25-54) | 2000–09 | 51 | 14013494 | 196128 | 36619 |
| ACS family (adults 25-54) | 2005–14 | 51 | 11946237 | 168668 | 21169 |
| ACS family (adults 25-54) | 2010–19 | 51 | 11789575 | 163548 | 20711 |
| ACS family (adults 25-54) | 2015–24 | 51 | 11605655 | 154065 | 19917 |
| Project Implicit IAT (US respondents) | 2000–09 | 51 | 211196 | 2449 | 339 |
| Project Implicit IAT (US respondents) | 2005–14 | 51 | 509289 | 5799 | 628 |
| Project Implicit IAT (US respondents) | 2010–19 | 51 | 996831 | 11607 | 1028 |
| Project Implicit IAT (US respondents) | 2015–24 | 51 | 1501880 | 18700 | 1637 |

### Text scores (RND, word fixed effects) per window

| category | window | states | words_in_set | words_per_state_median | mean_rnd | sd_rnd_across_states | median_ci_halfwidth |
|---|---|---|---|---|---|---|---|
| family_sphere | 1995–04 | 33 | 10 | 9.0 | 0.0106 | 0.0132 | 0.008 |
| family_sphere | 2000–09 | 48 | 10 | 10.0 | 0.009 | 0.0105 | 0.0066 |
| family_sphere | 2005–14 | 49 | 10 | 10.0 | 0.0098 | 0.0083 | 0.0062 |
| family_sphere | 2010–19 | 51 | 10 | 10.0 | 0.0104 | 0.0083 | 0.0057 |
| family_sphere | 2015–24 | 51 | 10 | 10.0 | 0.0109 | 0.0074 | 0.0061 |
| household | 1995–04 | 33 | 6 | 2.0 | -0.0052 | 0.0253 | 0.0066 |
| household | 2000–09 | 48 | 6 | 5.0 | -0.003 | 0.0169 | 0.0067 |
| household | 2005–14 | 49 | 6 | 6.0 | -0.0019 | 0.0129 | 0.0057 |
| household | 2010–19 | 51 | 6 | 6.0 | 0.0024 | 0.0089 | 0.0059 |
| household | 2015–24 | 51 | 6 | 6.0 | 0.0052 | 0.009 | 0.0056 |
| occupation | 1995–04 | 33 | 65 | 34.0 | -0.0093 | 0.0142 | 0.0041 |
| occupation | 2000–09 | 48 | 65 | 48.5 | -0.0122 | 0.0111 | 0.0036 |
| occupation | 2005–14 | 49 | 65 | 60.0 | -0.0101 | 0.0072 | 0.0033 |
| occupation | 2010–19 | 51 | 65 | 64.0 | -0.0065 | 0.0056 | 0.0032 |
| occupation | 2015–24 | 51 | 65 | 65.0 | -0.0049 | 0.0062 | 0.0029 |
