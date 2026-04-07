#!/usr/bin/env python3
"""
Process survey data to compute provincial gender ideation indices.

Produces two outputs:
  1. gender_ideation_by_province_year.csv  — province-level scores per survey-year
  2. gender_ideation_by_year.csv           — national-level scores per survey-year

All scores are normalized to [0, 1] where 1 = most traditional.

Datasets:
  - ACWF (中国妇女地位调查) 1990, 2000, 2010
  - CFPS 2014, 2020

Usage:
    python -m scripts.process_surveys [--output_dir=data/surveys/processed]
"""

import warnings
from pathlib import Path

import fire
import numpy as np
import pandas as pd

# Province code -> name mapping (GB/T 2260)
PROVINCE_CODES = {
    11: "Beijing", 12: "Tianjin", 13: "Hebei", 14: "Shanxi", 15: "Inner Mongolia",
    21: "Liaoning", 22: "Jilin", 23: "Heilongjiang",
    31: "Shanghai", 32: "Jiangsu", 33: "Zhejiang", 34: "Anhui", 35: "Fujian",
    36: "Jiangxi", 37: "Shandong",
    41: "Henan", 42: "Hubei", 43: "Hunan", 44: "Guangdong", 45: "Guangxi",
    46: "Hainan",
    50: "Chongqing", 51: "Sichuan", 52: "Guizhou", 53: "Yunnan", 54: "Tibet",
    61: "Shaanxi", 62: "Gansu", 63: "Qinghai", 64: "Ningxia", 65: "Xinjiang",
}


# Chinese province name -> code mapping (for CGSS 2018 which uses names)
PROVINCE_NAME_TO_CODE = {
    "北京市": 11, "天津市": 12, "河北省": 13, "山西省": 14, "内蒙古自治区": 15,
    "辽宁省": 21, "吉林省": 22, "黑龙江省": 23,
    "上海市": 31, "江苏省": 32, "浙江省": 33, "安徽省": 34, "福建省": 35,
    "江西省": 36, "山东省": 37,
    "河南省": 41, "湖北省": 42, "湖南省": 43, "广东省": 44, "广西壮族自治区": 45,
    "海南省": 46,
    "重庆市": 50, "四川省": 51, "贵州省": 52, "云南省": 53, "西藏自治区": 54,
    "陕西省": 61, "甘肃省": 62, "青海省": 63, "宁夏回族自治区": 64, "新疆维吾尔自治区": 65,
    # Aliases
    "深圳市": 44,  # Shenzhen → Guangdong
}


def _normalize(series, orig_min, orig_max):
    """Normalize score to [0, 1] range."""
    return (series - orig_min) / (orig_max - orig_min)


# ─── ACWF 1990 ───────────────────────────────────────────────────────────────

def process_acwf_1990():
    """Process 1990 ACWF survey.

    Items w611-w618: 1=strongly agree ... 5=disagree.
    All items except w615 are traditional statements.
    w615 (let child take mother's surname) is progressive.

    Scale: 1=非常同意, 2=同意, 3=无所谓, 4=不一定, 5=不同意. 0=未回答.
    """
    path = Path("data/surveys/中国妇女地位调查/1990/w1990_fixed.dta")
    if not path.exists():
        path = Path("data/surveys/中国妇女地位调查/1990/w1990.dta")

    items = [f"w61{i}" for i in range(1, 9)]
    # w611-w614, w616-w618: traditional statements (agree=traditional)
    # w615: progressive statement (agree=progressive)
    traditional_items = [c for c in items if c != "w615"]
    progressive_items = ["w615"]

    df = pd.read_stata(path, convert_categoricals=False, columns=["sheng"] + items)

    for col in items:
        df[col] = df[col].replace(0, np.nan)
        df.loc[~df[col].between(1, 5), col] = np.nan

    # Original scale: 1=strongly agree, 5=disagree
    # Traditional items: 1=agree=traditional → reverse so high=traditional
    for col in traditional_items:
        df[col] = _normalize(6 - df[col], 1, 5)

    # Progressive items: 1=agree=progressive(low trad), 5=disagree=traditional(high)
    for col in progressive_items:
        df[col] = _normalize(df[col], 1, 5)

    df["gender_ideation"] = df[items].mean(axis=1, skipna=True)
    df["province_code"] = df["sheng"].astype(int)
    df["year"] = 1990
    df["dataset"] = "ACWF"
    df["n_valid_items"] = df[items].notna().sum(axis=1)

    return df[["province_code", "year", "dataset", "gender_ideation", "n_valid_items"]].dropna(
        subset=["gender_ideation"]
    )


# ─── ACWF 2000 ───────────────────────────────────────────────────────────────

def process_acwf_2000():
    """Process 2000 ACWF survey.

    Items i3_a through i3_i: 1=非常同意, 2=比较同意, 3=不太同意, 4=很不同意.

    Traditional items: i3_a, b, c, d, e, g (agree=traditional)
    Progressive items: i3_h, i3_i (agree=progressive, reverse code)

    Config says use: i3_a, b, c, d, e, g, h(reverse), i(reverse)
    i3_f and i3_j are not in the config list — exclude them.
    """
    items_trad = ["i3_a", "i3_b", "i3_c", "i3_d", "i3_e", "i3_g"]
    items_prog = ["i3_h", "i3_i"]
    items = items_trad + items_prog

    df = pd.read_stata(
        "data/surveys/中国妇女地位调查/2000/w2000.dta",
        convert_categoricals=False,
        columns=["sheng"] + items,
    )

    for col in items:
        df.loc[~df[col].between(1, 4), col] = np.nan

    for col in items_trad:
        # 1=agree=traditional → reverse: 5-x. 1→4 (most trad), 4→1
        df[col] = _normalize(5 - df[col], 1, 4)

    for col in items_prog:
        # 1=agree=progressive → keep: 1=low trad, 4=high trad
        df[col] = _normalize(df[col], 1, 4)

    df["gender_ideation"] = df[items].mean(axis=1, skipna=True)
    df["province_code"] = df["sheng"].astype(int)
    df["year"] = 2000
    df["dataset"] = "ACWF"
    df["n_valid_items"] = df[items].notna().sum(axis=1)

    return df[["province_code", "year", "dataset", "gender_ideation", "n_valid_items"]].dropna(
        subset=["gender_ideation"]
    )


# ─── ACWF 2010 ───────────────────────────────────────────────────────────────

def process_acwf_2010():
    """Process 2010 ACWF survey.

    Items J2A-J2I: 1=非常同意, 2=比较同意, 3=不太同意, 4=很不同意. 8=说不清.

    Traditional items: J2B, J2C, J2D, J2F, J2G (agree=traditional)
    Progressive items: J2A, J2E, J2H, J2I (agree=progressive)
    """
    items_trad = ["J2B", "J2C", "J2D", "J2F", "J2G"]
    items_prog = ["J2A", "J2E", "J2H", "J2I"]
    items = items_trad + items_prog

    df = pd.read_stata(
        "data/surveys/中国妇女地位调查/2010/w2010.DTA",
        convert_categoricals=False,
        columns=["sheng"] + items,
    )

    for col in items:
        # 8 = don't know → missing
        df.loc[~df[col].between(1, 4), col] = np.nan

    for col in items_trad:
        df[col] = _normalize(5 - df[col], 1, 4)

    for col in items_prog:
        df[col] = _normalize(df[col], 1, 4)

    df["gender_ideation"] = df[items].mean(axis=1, skipna=True)
    df["province_code"] = df["sheng"].astype(int)
    df["year"] = 2010
    df["dataset"] = "ACWF"
    df["n_valid_items"] = df[items].notna().sum(axis=1)

    return df[["province_code", "year", "dataset", "gender_ideation", "n_valid_items"]].dropna(
        subset=["gender_ideation"]
    )


# ─── CFPS 2014 ────────────────────────────────────────────────────────────────

def process_cfps_2014():
    """Process CFPS 2014.

    Items qm1101-qm1104: 1=非常不同意, 5=非常同意.
    Negative values (-8, -2, -1) = missing.

    Traditional items: qm1101 (男主外女主内), qm1102 (干得好不如嫁得好),
                       qm1103 (女人应有孩子才完整)
    Progressive items: qm1104 (男人应承担一半家务)
    """
    items_trad = ["qm1101", "qm1102", "qm1103"]
    items_prog = ["qm1104"]
    items = items_trad + items_prog

    df = pd.read_stata(
        "data/surveys/CFPS/cfps2014_adult.dta",
        convert_categoricals=False,
        columns=["provcd14"] + items,
    )

    for col in items:
        df.loc[~df[col].between(1, 5), col] = np.nan

    for col in items_trad:
        # 5=strongly agree=traditional → keep: 5=high=traditional
        df[col] = _normalize(df[col], 1, 5)

    for col in items_prog:
        # 5=strongly agree=progressive → reverse: 6-x. 5→1, 1→5
        df[col] = _normalize(6 - df[col], 1, 5)

    df["gender_ideation"] = df[items].mean(axis=1, skipna=True)
    df["year"] = 2014
    df["dataset"] = "CFPS"
    df["n_valid_items"] = df[items].notna().sum(axis=1)

    # Drop invalid/missing province codes, then convert to int
    df = df[df["provcd14"].notna() & (df["provcd14"] > 0)]
    df["province_code"] = df["provcd14"].astype(int)

    return df[["province_code", "year", "dataset", "gender_ideation", "n_valid_items"]].dropna(
        subset=["gender_ideation"]
    )


# ─── CFPS 2020 ────────────────────────────────────────────────────────────────

def process_cfps_2020():
    """Process CFPS 2020. Same items as 2014."""
    items_trad = ["qm1101", "qm1102", "qm1103"]
    items_prog = ["qm1104"]
    items = items_trad + items_prog

    df = pd.read_stata(
        "data/surveys/CFPS/cfps2020_adult.dta",
        convert_categoricals=False,
        columns=["provcd20"] + items,
    )

    for col in items:
        df.loc[~df[col].between(1, 5), col] = np.nan

    for col in items_trad:
        df[col] = _normalize(df[col], 1, 5)

    for col in items_prog:
        df[col] = _normalize(6 - df[col], 1, 5)

    df["gender_ideation"] = df[items].mean(axis=1, skipna=True)
    df["year"] = 2020
    df["dataset"] = "CFPS"
    df["n_valid_items"] = df[items].notna().sum(axis=1)

    df = df[df["provcd20"].notna() & (df["provcd20"] > 0)]
    df["province_code"] = df["provcd20"].astype(int)

    return df[["province_code", "year", "dataset", "gender_ideation", "n_valid_items"]].dropna(
        subset=["gender_ideation"]
    )


# ─── CGSS ─────────────────────────────────────────────────────────────────────

def process_cgss():
    """Process all CGSS waves (2010-2023).

    Items a421-a425 (or A42_1-A42_5 in 2021):
        1=strongly disagree, 5=strongly agree.

    Traditional items: a421-a424 (agree=traditional)
    Progressive item:  a425 (agree=progressive, reverse code)

    Missing codes: -8, -3, -2, -1, 98, 99 → NaN
    Province: s41 (numeric code) in most years, 'provinces' (Chinese name) in 2018.
    """
    # Column name mapping per year
    year_configs = {
        2010: {"items_trad": ["a421", "a422", "a423", "a424"], "items_prog": ["a425"], "prov": "s41"},
        2012: {"items_trad": ["a421", "a422", "a423", "a424"], "items_prog": ["a425"], "prov": "s41"},
        2013: {"items_trad": ["a421", "a422", "a423", "a424"], "items_prog": ["a425"], "prov": "s41"},
        2015: {"items_trad": ["a421", "a422", "a423", "a424"], "items_prog": ["a425"], "prov": "s41"},
        2017: {"items_trad": ["a421", "a422", "a423", "a424"], "items_prog": ["a425"], "prov": "s41"},
        2018: {"items_trad": ["a421", "a422", "a423", "a424"], "items_prog": ["a425"], "prov": "provinces"},
        2021: {"items_trad": ["A42_1", "A42_2", "A42_3", "A42_4"], "items_prog": ["A42_5"], "prov": "s41"},
        2023: {"items_trad": ["a421", "a422", "a423", "a424"], "items_prog": ["a425"], "prov": "s41"},
    }

    all_dfs = []
    for year, cfg in year_configs.items():
        path = Path(f"data/surveys/CGSS/{year}/CGSS{year}.dta")
        if not path.exists():
            continue

        items = cfg["items_trad"] + cfg["items_prog"]
        prov_col = cfg["prov"]
        df = pd.read_stata(path, convert_categoricals=False, columns=[prov_col] + items)

        # Clean missing values: valid range is 1-5
        for col in items:
            df.loc[~df[col].between(1, 5), col] = np.nan

        # Traditional items: 5=agree=traditional → keep: (x-1)/4
        for col in cfg["items_trad"]:
            df[col] = _normalize(df[col], 1, 5)

        # Progressive item: 5=agree=progressive → reverse: (6-x-1)/4
        for col in cfg["items_prog"]:
            df[col] = _normalize(6 - df[col], 1, 5)

        df["gender_ideation"] = df[items].mean(axis=1, skipna=True)
        df["n_valid_items"] = df[items].notna().sum(axis=1)
        df["year"] = year
        df["dataset"] = "CGSS"

        # Province code
        if prov_col == "provinces":
            # 2018 uses Chinese province names
            df["province_code"] = df[prov_col].map(PROVINCE_NAME_TO_CODE)
        else:
            df["province_code"] = df[prov_col]

        # Drop invalid/missing province codes
        df = df[df["province_code"].notna() & (df["province_code"] > 0)]
        df["province_code"] = df["province_code"].astype(int)

        df = df[["province_code", "year", "dataset", "gender_ideation", "n_valid_items"]].dropna(
            subset=["gender_ideation"]
        )
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(individual_df):
    """Aggregate individual-level data to province-year and year level."""
    individual_df["province"] = individual_df["province_code"].map(PROVINCE_CODES)

    # Province-year level
    province_year = (
        individual_df.groupby(["dataset", "year", "province_code", "province"])
        .agg(
            gender_ideation_mean=("gender_ideation", "mean"),
            gender_ideation_sd=("gender_ideation", "std"),
            n_respondents=("gender_ideation", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "year", "province_code"])
    )

    # Year level (national aggregate)
    year_level = (
        individual_df.groupby(["dataset", "year"])
        .agg(
            gender_ideation_mean=("gender_ideation", "mean"),
            gender_ideation_sd=("gender_ideation", "std"),
            n_respondents=("gender_ideation", "count"),
            n_provinces=("province_code", "nunique"),
        )
        .reset_index()
        .sort_values(["dataset", "year"])
    )

    return province_year, year_level


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(output_dir="data/surveys/processed"):
    """Process all surveys and produce aggregated gender ideation CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processors = [
        ("ACWF 1990", process_acwf_1990),
        ("ACWF 2000", process_acwf_2000),
        ("ACWF 2010", process_acwf_2010),
        ("CFPS 2014", process_cfps_2014),
        ("CFPS 2020", process_cfps_2020),
        ("CGSS 2010-2023", process_cgss),
    ]

    all_individual = []
    for name, func in processors:
        print(f"Processing {name}...")
        df = func()
        print(f"  {len(df)} respondents, {df['province_code'].nunique()} provinces")
        print(f"  Mean ideation: {df['gender_ideation'].mean():.3f} (sd={df['gender_ideation'].std():.3f})")
        all_individual.append(df)

    combined = pd.concat(all_individual, ignore_index=True)
    print(f"\nTotal: {len(combined)} respondents across {combined['year'].nunique()} survey-years")

    province_year, year_level = aggregate(combined)

    # Save
    prov_path = output_dir / "gender_ideation_by_province_year.csv"
    year_path = output_dir / "gender_ideation_by_year.csv"
    individual_path = output_dir / "gender_ideation_individual.parquet"

    province_year.to_csv(prov_path, index=False, float_format="%.4f")
    year_level.to_csv(year_path, index=False, float_format="%.4f")
    combined.to_parquet(individual_path, index=False)

    print(f"\nSaved:")
    print(f"  {prov_path}")
    print(f"  {year_path}")
    print(f"  {individual_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("National-level gender ideation (0=progressive, 1=traditional):")
    print(f"{'='*70}")
    for _, row in year_level.iterrows():
        print(f"  {row['dataset']} {int(row['year'])}: "
              f"mean={row['gender_ideation_mean']:.3f} "
              f"(sd={row['gender_ideation_sd']:.3f}, "
              f"n={int(row['n_respondents'])}, "
              f"provinces={int(row['n_provinces'])})")


if __name__ == "__main__":
    fire.Fire(main)
