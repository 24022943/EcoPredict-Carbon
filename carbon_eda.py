"""
carbon_eda.py
EDA for EcoPredict Carbon.
Run:
    python carbon_eda.py
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from carbon_utils import load_carbon_catalogue, add_model_features, TARGET_COL

OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_carbon_catalogue("carbon_catalogue.csv")
    df = add_model_features(df)
    print("Data shape:", df.shape)
    print(df[["year", "country", "industry_group", TARGET_COL]].head())

    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "pcf_min": float(df[TARGET_COL].min()),
        "pcf_median": float(df[TARGET_COL].median()),
        "pcf_mean": float(df[TARGET_COL].mean()),
        "pcf_max": float(df[TARGET_COL].max()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "n_countries": int(df["country"].nunique()),
        "n_industry_groups": int(df["industry_group"].nunique()),
    }
    pd.Series(summary).to_csv(TAB / "eda_summary.csv", header=["value"])
    (TAB / "eda_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 1. Raw PCF histogram: shows right skew / outliers
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df[TARGET_COL], bins=60, color="#10b981")
    plt.title("Phân phối PCF thang tuyến tính - bị lệch phải do outlier")
    plt.xlabel("PCF (kg CO2e)")
    savefig("eda_pcf_hist_linear.png")

    # 2. Log PCF distribution: preferred
    plt.figure(figsize=(8, 4.5))
    sns.histplot(np.log1p(df[TARGET_COL]), bins=45, kde=True, color="#047857")
    plt.title("Phân phối log(PCF + 1) - đọc rõ hơn khi PCF lệch phải")
    plt.xlabel("log(PCF + 1)")
    savefig("eda_pcf_hist_log.png")

    # 3. Boxplot by top industry groups on log scale
    top_groups = df["industry_group"].value_counts().head(8).index
    dfg = df[df["industry_group"].isin(top_groups)].copy()
    dfg["log_pcf"] = np.log1p(dfg[TARGET_COL])
    plt.figure(figsize=(10, 5.2))
    sns.boxplot(data=dfg, y="industry_group", x="log_pcf", color="#a7f3d0")
    plt.title("PCF theo nhóm ngành GICS (log-scale)")
    plt.ylabel("")
    plt.xlabel("log(PCF + 1)")
    savefig("eda_boxplot_industry_logpcf.png")

    # 4. Missing values
    miss = df.isna().mean().sort_values(ascending=False).head(15) * 100
    miss.to_csv(TAB / "missing_values_top15.csv")
    plt.figure(figsize=(8.5, 4.8))
    sns.barplot(x=miss.values, y=miss.index, color="#64748b")
    plt.title("Top cột có tỷ lệ thiếu dữ liệu cao")
    plt.xlabel("Tỷ lệ thiếu (%)")
    plt.ylabel("")
    savefig("eda_missing_values.png")

    # 5. Top industry groups by samples
    counts = df["industry_group"].value_counts().head(10)
    counts.to_csv(TAB / "top_industry_groups.csv")
    plt.figure(figsize=(8.5, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#10b981")
    plt.title("Top nhóm ngành theo số mẫu")
    plt.xlabel("Số mẫu")
    plt.ylabel("")
    savefig("eda_top_industry_groups.png")

    # 6. Time trend: median and mean PCF by year (log y)
    time_stats = df.groupby("year")[TARGET_COL].agg(["count", "median", "mean"]).reset_index()
    time_stats.to_csv(TAB / "pcf_by_year.csv", index=False)
    plt.figure(figsize=(7.5, 4.3))
    plt.plot(time_stats["year"], time_stats["median"], marker="o", label="Median PCF")
    plt.plot(time_stats["year"], time_stats["mean"], marker="o", label="Mean PCF")
    plt.yscale("log")
    plt.title("Xu hướng PCF theo năm báo cáo (trục log)")
    plt.xlabel("Năm")
    plt.ylabel("PCF (kg CO2e, log-scale)")
    plt.legend()
    savefig("eda_pcf_time_trend.png")

    # 7. Lifecycle stage average stacked bar
    stages = ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]
    stage_mean = df[stages].mean().rename({
        "upstream_frac": "Upstream",
        "operations_frac": "Operations",
        "downstream_frac": "Downstream",
        "transport_frac": "Transport",
        "end_of_life_frac": "End-of-life",
    })
    stage_mean.to_csv(TAB / "lifecycle_stage_mean.csv")
    plt.figure(figsize=(8, 2.2))
    left = 0
    colors = ["#047857", "#10b981", "#86efac", "#64748b", "#cbd5e1"]
    for (name, val), color in zip(stage_mean.items(), colors):
        plt.barh(["Average"], [val * 100], left=left, color=color, label=name)
        left += val * 100
    plt.title("Tỷ trọng phát thải vòng đời trung bình")
    plt.xlabel("Tỷ trọng (%)")
    plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.35), loc="upper center")
    savefig("eda_lifecycle_stacked_bar.png")

    # 8. Correlation heatmap among numeric engineered features
    num_cols = ["year", "product_weight_log", "upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "lifecycle_balance_std", TARGET_COL]
    corr = df[num_cols].corr(numeric_only=True)
    corr.to_csv(TAB / "correlation_matrix.csv")
    plt.figure(figsize=(8.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Greens", center=0)
    plt.title("Heatmap tương quan giữa các biến số chính")
    savefig("eda_correlation_heatmap.png")

    print("✅ EDA completed. Outputs saved to outputs/figures and outputs/tables.")


if __name__ == "__main__":
    main()
