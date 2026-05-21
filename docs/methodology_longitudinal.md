# Methodology: Longitudinal Analysis of Gender Norms in Chinese Text

## 1. Overview

This study constructs time-varying measures of gender norm stereotyping in Chinese-language text corpora and validates them against nationally representative survey data on gender attitudes. We train Word2Vec embeddings on decade-length rolling windows of text, extract gender-stereotype dimensions using the Word Embedding Association Test (WEAT), and overlay the resulting indices with survey-based gender ideation scores from three independent survey programs spanning 1990--2023.

---

## 2. Text Data Sources

### 2.1 Google Ngram (Chinese Simplified)

We use the Chinese Simplified portion of the Google Books Ngram Corpus (version 3), which tabulates the frequency of five-word sequences across digitized books published from the 1940s onward. Each raw file is a gzip-compressed table in the format:

```
ngram    year1,match_count,volume_count    year2,...
```

We extract Chinese characters from each token via Unicode range filtering (`\u4e00`--`\u9fff`), discard single-character tokens, and retain only ngrams that yield at least two clean tokens after filtering. Frequency counts are not used as weights; each unique ngram contributes one line per year in the output corpus.

### 2.2 People's Daily (*Renmin Ribao*)

We use the full-text archive of the *People's Daily* (人民日报), the official newspaper of the Central Committee of the Chinese Communist Party. Source files are organized by decade and year, encoded in GB18030. Each article undergoes the following preprocessing: (i) removal of URLs, bracketed annotations, and non-Chinese characters; (ii) word segmentation via the jieba tokenizer with HMM enabled; (iii) removal of 35 high-frequency stopwords (的, 了, 在, 是, etc.); and (iv) exclusion of documents with fewer than five tokens after segmentation.

---

## 3. Corpus Construction

Both corpora are partitioned into overlapping time slices using a rolling-window scheme with a **10-year window** and a **5-year step**, spanning **1940--2020**. This yields 16 slices for Google Ngram (1940--1949, 1945--1954, ..., 2015--2020) and 17 for People's Daily (which includes a 2020--2020 slice). The overlapping design smooths year-to-year volatility while preserving sufficient temporal resolution to detect decadal shifts in language use.

For each time slice, all qualifying text from years falling within the window is pooled into a single corpus directory, segmented into files of manageable size (flushed at 10,000-line intervals).

---

## 4. Word Embedding Training

We train skip-gram Word2Vec models (Mikolov et al., 2013) independently on each time-slice corpus using the gensim implementation. All hyperparameters are held constant across slices and data sources to ensure comparability:

| Parameter | Value |
|-----------|-------|
| Architecture | Skip-gram (`sg=1`) |
| Vector dimensionality | 300 |
| Context window | 4 |
| Minimum word frequency | 50 |
| Negative samples | 15 |
| Training epochs | 5 |
| Random seed | 42 |

The minimum frequency threshold of 50 ensures that all retained words have sufficient co-occurrence statistics for reliable vector estimation. Each trained model is saved alongside a metadata file recording corpus statistics and training parameters.

---

## 5. Gender Norm Index Construction

### 5.1 Gender Axis

For each time-slice model, we construct a gender semantic axis following the method of Bolukbasi et al. (2016). The axis is defined as the normalized difference between the centroids of female-associated and male-associated seed words:

$$\mathbf{g} = \frac{\bar{\mathbf{v}}_{\text{female}} - \bar{\mathbf{v}}_{\text{male}}}{\|\bar{\mathbf{v}}_{\text{female}} - \bar{\mathbf{v}}_{\text{male}}\|}$$

where $\bar{\mathbf{v}}_{\text{female}}$ is the mean vector of all female seed words found in the model's vocabulary, and likewise for male. The seed lists comprise 22 female terms (女人, 女性, 女子, 女生, 女孩, 女士, 姑娘, 姐姐, 姐妹, 母亲, 妈妈, 女儿, 妻子, 老婆, etc.) and 20 male terms (男人, 男性, 男子, 男生, 男孩, 先生, 哥哥, 兄弟, 父亲, 爸爸, 儿子, 丈夫, 老公, etc.), selected to cover kinship, address, and relational terms in formal Chinese register.

### 5.2 Concept Word Lists

We measure gender stereotyping along three dimensions, each operationalized as a contrast between two concept categories:

**Work--Family.** Family/domestic words (20 terms: 家务, 做饭, 洗碗, 洗衣服, 拖地, 扫地, 打扫, 育儿, 照料, etc.) versus career/work words (23 terms: 工作, 上班, 加班, 出差, 公司, 同事, 升职, 晋升, 事业, 职场, etc.).

**Leadership.** Non-leadership positions (16 terms: 员工, 下属, 基层, 职员, 助理, 文员, 实习生, 职工, etc.) versus leadership positions (20 terms: 老板, 总裁, 董事长, 领导, 主管, 高管, 经理, 总监, 管理者, etc.).

**STEM.** Non-STEM fields (46 terms: 哲学, 经济, 法学, 文学, 历史, 艺术, 音乐, 设计, etc.) versus STEM fields (40 terms: 数学, 计算机, 物理, 化学, 生物, 工程, 人工智能, 软件, 土木, etc.).

All word lists use formal register appropriate to published books and newspaper text.

### 5.3 Projection and Effect Size

For each concept word $w$ present in a given time-slice model, we compute its cosine similarity with the gender axis:

$$\text{cos}(w, \mathbf{g}) = \frac{\mathbf{v}_w \cdot \mathbf{g}}{\|\mathbf{v}_w\|}$$

where $\mathbf{g}$ is unit-length by construction. Positive values indicate proximity to the female pole; negative values indicate proximity to the male pole.

We then compute the WEAT effect size (Caliskan et al., 2017) for each dimension as Cohen's $d$:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$$

where $\bar{x}_1$ and $\bar{x}_2$ are the mean cosine similarities of the two concept groups (e.g., family words vs. work words), and $s_p$ is the pooled standard deviation:

$$s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

A positive Cohen's $d$ in the Work--Family dimension, for instance, indicates that family-related words are more female-associated than work-related words in that time slice. We interpret $|d| \geq 0.2$ as a small effect, $|d| \geq 0.5$ as medium, and $|d| \geq 0.8$ as large, following conventional benchmarks (Cohen, 1988).

### 5.4 Cross-Period Comparability

Before computing effect sizes, we assess the comparability of projection distributions across time slices by calculating the coefficient of variation (CV) of per-slice means and standard deviations. If either CV exceeds 0.3, indicating substantial distributional drift across periods, projections are z-score standardized within each time slice before computing Cohen's $d$. This guards against conflating changes in the overall geometry of the embedding space with genuine shifts in gender associations.

### 5.5 Alternative metric: relative norm distance (Garg-WEAT)

As a complementary measure on the *same* trained models, we also compute Garg's relative norm distance (RND; Garg et al., 2018): for each concept word, the difference in distance to the male vs. female centroid, $\lVert v_w - c_{\text{male}}\rVert - \lVert v_w - c_{\text{female}}\rVert$ (positive = female-leaning). Words are grouped into the leadership, family, and science categories and oriented onto a single gender-ideation axis (higher = less traditional), with family reversed because its female association is the *traditional* view. Uncertainty is reported as both a with-replacement bootstrap (Garg's convention) and an 80% word-subsample band. This RND view does not replace the Cohen's $d$ index; the two are reported side by side. Full procedure: [`replication/garg_weat_per_category.md`](replication/garg_weat_per_category.md).

---

## 6. Survey Data

We validate the text-based indices against three nationally representative survey programs that include Likert-scale items on gender role attitudes. All items are recoded so that higher values indicate more traditional attitudes, then normalized to the unit interval $[0, 1]$.

### 6.1 ACWF Survey (Chinese Women's Status Survey)

The All-China Women's Federation conducted national surveys in 1990 ($n = 23{,}722$; 11 provinces), 2000 ($n = 19{,}283$; 30 provinces), and 2010 ($n = 26{,}021$; 31 provinces).

**1990 wave** (8 items, 5-point scale: 1 = strongly agree to 5 = disagree):

| Item | Statement | Coding |
|------|-----------|--------|
| w611 | Men should focus on society, women on family | Traditional: $(6 - x) / 4$ |
| w612 | Men are innately more capable than women | Traditional: $(6 - x) / 4$ |
| w613 | Women should avoid surpassing husband's social status | Traditional: $(6 - x) / 4$ |
| w614 | Husband's success is wife's success | Traditional: $(6 - x) / 4$ |
| w615 | Let child take mother's surname | Progressive: $(x - 1) / 4$ |
| w616 | Women haven't played a half-the-sky role | Traditional: $(6 - x) / 4$ |
| w617 | Men should handle external family affairs | Traditional: $(6 - x) / 4$ |
| w618 | Widow should leave property to ex-husband's family | Traditional: $(6 - x) / 4$ |

**2000 wave** (8 items, 4-point scale: 1 = strongly agree to 4 = strongly disagree):

Traditional items (6): men for society / women for family; men innately more capable; marrying well beats doing well; childless women incomplete; women should not surpass husbands; women's looks matter more than ability. Coded as $(5 - x) / 3$.

Progressive items (2): at least 30% of leaders should be women; men should share half the housework. Coded as $(x - 1) / 3$.

**2010 wave** (9 items, same 4-point scale; code 8 = "don't know" treated as missing):

Traditional items (5): men for society / women for family; breadwinning is men's job; husband's career more important; boys should be boys; marrying well beats doing well. Progressive items (4): women no less capable; men should share housework; leadership should be gender-equal; gender equality needs active promotion.

### 6.2 CFPS (China Family Panel Studies)

The CFPS, administered by Peking University's Institute of Social Science Survey, includes four gender attitude items in the 2014 ($n = 31{,}554$; 28 provinces) and 2020 ($n = 22{,}692$; 31 provinces) waves. The scale runs 1 = strongly disagree to 5 = strongly agree. Negative response codes ($-8$, $-2$, $-1$) are treated as missing.

| Item | Statement | Coding |
|------|-----------|--------|
| qm1101 | Men for career, women for family | Traditional: $(x - 1) / 4$ |
| qm1102 | Doing well is not as good as marrying well | Traditional: $(x - 1) / 4$ |
| qm1103 | A woman needs children to be complete | Traditional: $(x - 1) / 4$ |
| qm1104 | Men should share half the housework | Progressive: $(5 - x) / 4$ |

### 6.3 CGSS (Chinese General Social Survey)

The CGSS, conducted annually by Renmin University, provides the densest temporal coverage with eight waves: 2010, 2012, 2013, 2015, 2017, 2018, 2021, and 2023 (total $n = 86{,}359$; up to 31 provinces per wave). Five items are measured on a 5-point scale (1 = strongly disagree to 5 = strongly agree). Missing codes ($-8$, $-3$, $-2$, $-1$, 98, 99) are treated as missing.

| Item | Statement | Coding |
|------|-----------|--------|
| a421 | Men for career, women for family | Traditional: $(x - 1) / 4$ |
| a422 | Men are innately more capable than women | Traditional: $(x - 1) / 4$ |
| a423 | Doing well is not as good as marrying well | Traditional: $(x - 1) / 4$ |
| a424 | Women should be fired first in economic downturns | Traditional: $(x - 1) / 4$ |
| a425 | Couples should share housework equally | Progressive: $(5 - x) / 4$ |

### 6.4 Aggregation

For each respondent, the gender ideation score is the arithmetic mean of all non-missing normalized items. No imputation is applied. National-level scores are simple means of individual scores within each survey-year; province-level scores are analogously computed within province-year cells.

### 6.5 Cross-Survey Comparability

The three survey programs differ in item count (4--9), content coverage, scale anchoring, and sampling frame. We therefore treat absolute score levels as comparable only within each survey program. Cross-program comparisons are limited to directional trends and provincial rank-orderings.

---

## 7. Composite Visualization

We overlay the text-based and survey-based measures on a shared timeline (1940--2023) using dual y-axes. The left axis displays WEAT Cohen's $d$ for the three stereotype dimensions; the right axis displays the survey gender ideation index on the $[0, 1]$ scale. This visualization permits qualitative assessment of whether shifts in published-text gender associations track contemporaneous changes in population-level gender attitudes, without imposing a parametric relationship between the two measurement traditions.

---

## References

- Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings. *Advances in Neural Information Processing Systems*, 29.
- Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183--186.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *Advances in Neural Information Processing Systems*, 26.
