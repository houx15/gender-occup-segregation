from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    base = Path(__file__).resolve().parent
    output_dir = base / "distribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base / "provincial_cleaned.csv")

    df["emp_diff_m_f"] = df["emp_m_2020"] - df["emp_f_2020"]
    df["eduy_diff_m_f"] = df["eduy_m_gt25_2020"] - df["eduy_f_gt25_2020"]

    plots = [
        ("gdp_2024", "gdp_2024_kde.pdf", "GDP 2024"),
        ("avg_income_2024", "avg_income_2024_kde.pdf", "Average Income 2024"),
        ("emp_2020", "emp_2020_kde.pdf", "Employment Rate 2020"),
        ("eduy_gt25_2020", "eduy_gt25_2020_kde.pdf", "Education Years (25+) 2020"),
        ("emp_diff_m_f", "emp_diff_m_f_kde.pdf", "Employment Rate (Male - Female)"),
        ("eduy_diff_m_f", "eduy_diff_m_f_kde.pdf", "Education Years (Male - Female)"),
    ]

    sns.set_theme(style="whitegrid")

    for col, filename, title in plots:
        series = df[col].dropna()
        if series.empty:
            continue
        plt.figure(figsize=(6, 4))
        sns.kdeplot(series, fill=True)
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_dir / filename, format="pdf")
        plt.close()

    print(f"Saved KDE plots to {output_dir}")


if __name__ == "__main__":
    main()
