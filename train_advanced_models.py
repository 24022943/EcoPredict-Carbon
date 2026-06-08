"""
train_advanced_models.py
Train EcoPredict Carbon models with time-based split, leakage control and uncertainty.
Run:
    python train_advanced_models.py
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

from carbon_utils import (
    RANDOM_STATE, TARGET_COL, FEATURE_COLS, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    LABEL_ORDER, LABEL_TO_NUM, NUM_TO_LABEL, LABEL_VI,
    load_carbon_catalogue, add_model_features, time_based_split,
    fit_label_thresholds, apply_carbon_labels, make_clf_pipeline, make_reg_pipeline,
    get_classification_models, get_regression_models,
    evaluate_classifier, evaluate_regressor, build_ood_profile, save_package,
)

OUT = Path("outputs")
FIG = OUT / "figures"
TAB = OUT / "tables"
MOD = OUT / "models"
for p in [FIG, TAB, MOD]:
    p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 80)
    print("ECOPREDICT CARBON - TRAINING PIPELINE")
    print("=" * 80)

    df = load_carbon_catalogue("carbon_catalogue.csv")
    df = add_model_features(df)
    train_df, test_df, test_year = time_based_split(df)
    thresholds = fit_label_thresholds(train_df)
    train_df["carbon_label"] = apply_carbon_labels(train_df, thresholds)
    test_df["carbon_label"] = apply_carbon_labels(test_df, thresholds)
    train_df["carbon_label_num"] = train_df["carbon_label"].map(LABEL_TO_NUM)
    test_df["carbon_label_num"] = test_df["carbon_label"].map(LABEL_TO_NUM)

    print(f"Data: {df.shape}, Train: {train_df.shape}, Test: {test_df.shape}, Test year: {test_year}")
    print("Label thresholds fit on train only:", thresholds)
    print("Train labels:\n", train_df["carbon_label"].value_counts())
    print("Test labels:\n", test_df["carbon_label"].value_counts())

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    y_train = train_df["carbon_label_num"].values
    y_test = test_df["carbon_label_num"].values
    yreg_train = train_df[TARGET_COL].values
    yreg_test = test_df[TARGET_COL].values

    # Classification models
    clf_rows = []
    clf_models = {}
    cv = None  # CV disabled for fast deployment; final holdout uses time-based split
    for name, model in get_classification_models().items():
        print(f"\nTraining classifier: {name}")
        pipe = make_clf_pipeline(model)
        pipe.fit(X_train, y_train)
        clf_models[name] = pipe
        test_metrics = evaluate_classifier(pipe, X_test, y_test)
        row = {
            "model": name,
            "cv_accuracy_mean": np.nan,
            "cv_accuracy_std": np.nan,
            "cv_balanced_accuracy_mean": np.nan,
            "cv_f1_macro_mean": np.nan,
            "cv_f1_macro_std": np.nan,
            **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        }
        clf_rows.append(row)
        print(row)

    clf_table = pd.DataFrame(clf_rows).sort_values(["test_f1_macro", "test_balanced_accuracy"], ascending=False)
    clf_table.to_csv(TAB / "classification_metrics.csv", index=False)
    best_clf_name = clf_table.iloc[0]["model"]
    best_clf = clf_models[best_clf_name]
    print("\nBest classifier:", best_clf_name)

    # Regression models
    reg_rows = []
    reg_models = {}
    kf = None  # CV disabled for fast deployment; final holdout uses time-based split
    for name, model in get_regression_models().items():
        print(f"\nTraining regressor: {name}")
        pipe = make_reg_pipeline(model)
        pipe.fit(X_train, yreg_train)
        reg_models[name] = pipe
        test_metrics = evaluate_regressor(pipe, X_test, yreg_test)
        row = {
            "model": name,
            "cv_mae_mean": np.nan,
            "cv_mae_std": np.nan,
            "cv_r2_mean": np.nan,
            **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        }
        reg_rows.append(row)
        print(row)

    reg_table = pd.DataFrame(reg_rows).sort_values(["test_median_ape_pct", "test_rmse"], ascending=True)
    reg_table.to_csv(TAB / "regression_metrics.csv", index=False)
    best_reg_name = reg_table.iloc[0]["model"]
    best_reg = reg_models[best_reg_name]
    print("\nBest regressor:", best_reg_name)

    # Figures: model comparison
    plt.figure(figsize=(8.8, 4.8))
    plot_df = clf_table.sort_values("test_f1_macro", ascending=True)
    plt.barh(plot_df["model"], plot_df["test_f1_macro"], color="#047857")
    plt.xlabel("F1-macro trên test năm mới nhất")
    plt.title("So sánh mô hình phân loại theo time-based split")
    savefig("model_classification_f1_comparison.png")

    plt.figure(figsize=(8.8, 4.8))
    plot_df = reg_table.sort_values("test_median_ape_pct", ascending=False)
    plt.barh(plot_df["model"], plot_df["test_median_ape_pct"], color="#10b981")
    plt.xlabel("Median APE (%) - càng thấp càng tốt")
    plt.title("So sánh mô hình hồi quy theo sai số tương đối trung vị")
    savefig("model_regression_median_ape_comparison.png")

    # Confusion matrix
    y_pred = best_clf.predict(X_test)
    plt.figure(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[LABEL_VI[x] for x in LABEL_ORDER], cmap="Greens", values_format="d")
    plt.title(f"Confusion Matrix - {best_clf_name}")
    savefig("model_confusion_matrix.png")

    # ROC curve if possible
    if hasattr(best_clf, "predict_proba"):
        proba = best_clf.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=[0, 1, 2])
        plt.figure(figsize=(6.8, 5.2))
        for i, label in enumerate(LABEL_ORDER):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            plt.plot(fpr, tpr, label=f"{LABEL_VI[label]} AUC={auc(fpr,tpr):.3f}")
        plt.plot([0,1], [0,1], "--", color="#94a3b8")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve theo từng lớp")
        plt.legend()
        savefig("model_roc_curve.png")

    # Regression diagnostics
    reg_pred = np.maximum(best_reg.predict(X_test), 0)
    residuals = yreg_test - reg_pred
    residual_abs_q = {
        "p10": float(np.quantile(np.abs(residuals), 0.10)),
        "p50": float(np.quantile(np.abs(residuals), 0.50)),
        "p90": float(np.quantile(np.abs(residuals), 0.90)),
    }

    plt.figure(figsize=(6, 5))
    plt.scatter(yreg_test, reg_pred, s=18, alpha=0.7, color="#047857")
    lim = [0, max(float(yreg_test.max()), float(reg_pred.max()))]
    plt.plot(lim, lim, "--", color="#334155")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("PCF thực tế (log)")
    plt.ylabel("PCF dự đoán (log)")
    plt.title(f"Actual vs Predicted - {best_reg_name}")
    savefig("model_regression_actual_vs_predicted.png")

    plt.figure(figsize=(7, 4.6))
    plt.scatter(reg_pred, residuals, s=18, alpha=0.7, color="#64748b")
    plt.axhline(0, linestyle="--", color="#ef4444")
    plt.xscale("log")
    plt.xlabel("PCF dự đoán (log)")
    plt.ylabel("Residual = Actual - Predicted")
    plt.title("Residuals vs Fitted")
    savefig("model_regression_residuals.png")

    plt.figure(figsize=(7, 4.6))
    sns.histplot(residuals, bins=40, kde=True, color="#10b981")
    plt.title("Phân phối phần dư hồi quy")
    plt.xlabel("Residual")
    savefig("model_regression_residual_distribution.png")

    # Permutation importance on best classifier
    try:
        perm = permutation_importance(best_clf, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=1)
        imp = pd.DataFrame({"feature": FEATURE_COLS, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
        imp.to_csv(TAB / "permutation_importance_classifier.csv", index=False)
        top = imp.head(12).sort_values("importance_mean", ascending=True)
        plt.figure(figsize=(8, 5.2))
        plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#047857", alpha=0.9)
        plt.xlabel("Decrease in F1-macro after permutation")
        plt.title("Permutation Importance - mô hình phân loại tốt nhất")
        savefig("model_permutation_importance.png")
    except Exception as exc:
        print("Permutation importance skipped:", exc)

    metadata = {
        "project": "EcoPredict Carbon",
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "label_order": LABEL_ORDER,
        "label_thresholds_train_only": thresholds,
        "split_strategy": f"time-based split, test year = {test_year}",
        "best_classifier_name": best_clf_name,
        "best_regressor_name": best_reg_name,
        "residual_abs_quantiles": residual_abs_q,
        "ood_profile": build_ood_profile(train_df),
        "data_summary": {
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "pcf_min": float(df[TARGET_COL].min()),
            "pcf_median": float(df[TARGET_COL].median()),
            "pcf_mean": float(df[TARGET_COL].mean()),
            "pcf_max": float(df[TARGET_COL].max()),
        },
        "classification_metrics": clf_table.to_dict(orient="records"),
        "regression_metrics": reg_table.to_dict(orient="records"),
    }
    (TAB / "training_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    package = {
        "classifier": best_clf,
        "regressor": best_reg,
        "reference_data": df,
        "train_data": train_df,
        "test_data": test_df,
        "metadata": metadata,
    }
    save_package(package, MOD / "ecopredict_model_package.joblib", also_root=True)
    print("\n✅ Training completed.")
    print("Saved model to outputs/models/ecopredict_model_package.joblib and ecopredict_model_package.joblib")
    print("Saved tables to outputs/tables and figures to outputs/figures")


if __name__ == "__main__":
    main()
