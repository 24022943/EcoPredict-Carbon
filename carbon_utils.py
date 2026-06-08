"""
carbon_utils.py
Core utilities for EcoPredict Carbon.

Mục tiêu:
- Đọc và chuẩn hóa dữ liệu Carbon Catalogue.
- Tạo đặc trưng ML có kiểm soát data leakage.
- Huấn luyện/đánh giá các mô hình phân loại và hồi quy.
- Hỗ trợ LCA bottom-up, uncertainty, OOD check và giải thích cục bộ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import math
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

RANDOM_STATE = 42
DATA_PATH = Path("carbon_catalogue.csv")
MODEL_PATH = Path("outputs/models/ecopredict_model_package.joblib")
ROOT_MODEL_PATH = Path("ecopredict_model_package.joblib")
TARGET_COL = "pcf_kg_co2e"
LABEL_ORDER = ["Low", "Medium", "High"]
LABEL_TO_NUM = {v: i for i, v in enumerate(LABEL_ORDER)}
NUM_TO_LABEL = {i: v for v, i in LABEL_TO_NUM.items()}
LABEL_VI = {"Low": "Thấp", "Medium": "Trung bình", "High": "Cao"}

RAW_TO_CANONICAL = {
    "Year of reporting": "year",
    "*Stage-level CO2e available": "stage_level_available",
    "Product name (and functional unit)": "product_name",
    "Product detail": "product_detail",
    "Company": "company",
    "Country (where company is incorporated)": "country",
    "Company's GICS Industry Group": "industry_group",
    "Company's GICS Industry": "industry",
    "*Company's sector": "company_sector",
    "Product weight (kg)": "product_weight_kg",
    "*Source for product weight": "weight_source",
    "Product's carbon footprint (PCF, kg CO2e)": TARGET_COL,
    "*Carbon intensity": "carbon_intensity",
    "Protocol used for PCF": "protocol",
    "Relative change in PCF vs previous": "relative_change_pcf",
    "Company-reported reason for change": "change_reason_text",
    "*Change reason category": "change_reason_category",
    "*%Upstream estimated from %Operations": "upstream_estimated_from_operations",
    "*Upstream CO2e (fraction of total PCF)": "upstream_frac",
    "*Operations CO2e (fraction of total PCF)": "operations_frac",
    "*Downstream CO2e (fraction of total PCF)": "downstream_frac",
    "*Transport CO2e (fraction of total PCF)": "transport_frac",
    "*EndOfLife CO2e (fraction of total PCF)": "end_of_life_frac",
    "*Adjustments to raw data (if any)": "raw_adjustments",
}

# Feature groups used by the final model. carbon_intensity is intentionally excluded to avoid target leakage.
NUMERIC_FEATURES = [
    "year", "year_offset", "future_year_gap",
    "product_weight_kg", "product_weight_log",
    "upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac",
    "lifecycle_fraction_sum", "lifecycle_balance_std", "lifecycle_max_share", "lifecycle_min_share",
    "upstream_x_weight_log", "operations_x_weight_log", "downstream_x_weight_log",
    "transport_x_weight_log", "product_name_length", "product_detail_length",
    "is_weight_estimated", "has_stage_data", "has_transport_data", "has_eol_data",
]
CATEGORICAL_FEATURES = [
    "country", "industry_group", "industry", "company_sector",
    "stage_level_available", "weight_source", "protocol_simple",
    "dominant_stage", "weight_category",
]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURE_NAME_VI = {
    "year": "Năm báo cáo",
    "year_offset": "Khoảng cách so với năm đầu dữ liệu",
    "future_year_gap": "Khoảng cách năm ngoài dữ liệu huấn luyện",
    "product_weight_kg": "Khối lượng sản phẩm",
    "product_weight_log": "Khối lượng sản phẩm (log)",
    "upstream_frac": "Tỷ trọng phát thải đầu vào",
    "operations_frac": "Tỷ trọng phát thải sản xuất/vận hành",
    "downstream_frac": "Tỷ trọng phát thải đầu ra",
    "transport_frac": "Tỷ trọng vận chuyển",
    "end_of_life_frac": "Tỷ trọng cuối vòng đời",
    "lifecycle_fraction_sum": "Tổng tỷ trọng vòng đời",
    "lifecycle_balance_std": "Độ lệch giữa các giai đoạn vòng đời",
    "lifecycle_max_share": "Tỷ trọng vòng đời lớn nhất",
    "lifecycle_min_share": "Tỷ trọng vòng đời nhỏ nhất",
    "upstream_x_weight_log": "Tương tác đầu vào và khối lượng",
    "operations_x_weight_log": "Tương tác vận hành và khối lượng",
    "downstream_x_weight_log": "Tương tác đầu ra và khối lượng",
    "transport_x_weight_log": "Tương tác vận chuyển và khối lượng",
    "product_name_length": "Độ dài tên sản phẩm",
    "product_detail_length": "Độ dài mô tả sản phẩm",
    "is_weight_estimated": "Khối lượng có tính ước lượng",
    "has_stage_data": "Có dữ liệu theo giai đoạn vòng đời",
    "has_transport_data": "Có dữ liệu vận chuyển",
    "has_eol_data": "Có dữ liệu cuối vòng đời",
    "country": "Quốc gia",
    "industry_group": "Nhóm ngành GICS",
    "industry": "Ngành sản phẩm",
    "company_sector": "Sector công ty",
    "stage_level_available": "Có dữ liệu stage-level",
    "weight_source": "Nguồn khối lượng",
    "protocol_simple": "Chuẩn PCF",
    "dominant_stage": "Giai đoạn chiếm tỷ trọng lớn nhất",
    "weight_category": "Nhóm khối lượng",
}

# Approximate emission factors for demo LCA bottom-up. In a real ISO system these must be sourced/versioned.
DEFAULT_EMISSION_FACTORS = pd.DataFrame([
    {"activity_group": "Material", "activity_name": "Steel", "unit": "kg", "emission_factor": 1.90, "source": "Demo factor", "quality": "Medium"},
    {"activity_group": "Material", "activity_name": "Aluminium", "unit": "kg", "emission_factor": 8.60, "source": "Demo factor", "quality": "Medium"},
    {"activity_group": "Material", "activity_name": "Plastic", "unit": "kg", "emission_factor": 2.50, "source": "Demo factor", "quality": "Medium"},
    {"activity_group": "Material", "activity_name": "Paper/Cardboard", "unit": "kg", "emission_factor": 0.90, "source": "Demo factor", "quality": "Medium"},
    {"activity_group": "Energy", "activity_name": "Electricity", "unit": "kWh", "emission_factor": 0.45, "source": "Demo factor", "quality": "Medium"},
    {"activity_group": "Transport", "activity_name": "Truck freight", "unit": "ton-km", "emission_factor": 0.12, "source": "Demo factor", "quality": "Low"},
])


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    last_error = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise last_error


def parse_numeric(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"n/a", "na", "not reported", "unknown", "-"}:
        return np.nan
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_fraction(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v / 100.0 if v > 1.5 else v
    s = str(x).strip()
    if s == "" or s.lower() in {"n/a", "na", "not reported", "unknown", "-"}:
        return np.nan
    if "included" in s.lower() or "not reported" in s.lower():
        return np.nan
    pct = "%" in s
    s = s.replace("%", "").replace(",", "")
    try:
        v = float(s)
        return v / 100.0 if pct or v > 1.5 else v
    except ValueError:
        return np.nan


def clean_text_value(x: Any, default: str = "Unknown") -> str:
    if pd.isna(x):
        return default
    s = str(x).strip()
    if s == "" or s.lower() in {"n/a", "na", "none", "nan", "not reported"}:
        return default
    return s


def simplify_protocol(x: Any) -> str:
    s = clean_text_value(x, "Unknown")
    sl = s.lower()
    if "ghg" in sl:
        return "GHG Protocol"
    if "iso" in sl:
        return "ISO"
    if "pas" in sl:
        return "PAS 2050"
    if "not" in sl or "unknown" in sl:
        return "Unknown"
    return "Other"


def weight_category(w: float) -> str:
    if pd.isna(w) or w <= 0:
        return "Unknown"
    if w < 1:
        return "Very light (<1 kg)"
    if w < 10:
        return "Light (1-10 kg)"
    if w < 100:
        return "Medium (10-100 kg)"
    if w < 1000:
        return "Heavy (100-1000 kg)"
    return "Very heavy (>1000 kg)"


def load_carbon_catalogue(path: str | Path = DATA_PATH) -> pd.DataFrame:
    raw = read_csv_safely(path)
    df = raw.rename(columns={c: RAW_TO_CANONICAL.get(c, c) for c in raw.columns}).copy()

    required = ["year", TARGET_COL, "product_weight_kg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"File CSV thiếu cột bắt buộc sau khi chuẩn hóa: {missing}")

    for col in ["year", "product_weight_kg", TARGET_COL, "carbon_intensity"]:
        if col in df.columns:
            df[col] = df[col].map(parse_numeric)
    for col in ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]:
        if col in df.columns:
            df[col] = df[col].map(parse_fraction)
        else:
            df[col] = np.nan

    text_cols = [
        "stage_level_available", "product_name", "product_detail", "company", "country",
        "industry_group", "industry", "company_sector", "weight_source", "protocol",
        "upstream_estimated_from_operations", "change_reason_category", "relative_change_pcf",
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].map(clean_text_value)

    df = df.dropna(subset=[TARGET_COL]).copy()
    df = df[df[TARGET_COL] > 0].copy()
    df["product_weight_kg"] = df["product_weight_kg"].fillna(df["product_weight_kg"].median())
    df["year"] = df["year"].fillna(df["year"].median()).astype(int)
    df["protocol_simple"] = df["protocol"].map(simplify_protocol)

    return df.reset_index(drop=True)


def add_model_features(df: pd.DataFrame, reference_min_year: int | None = None, max_train_year: int | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in ["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Normalize main lifecycle stages if users enter percentages or non-100 totals.
    main = ["upstream_frac", "operations_frac", "downstream_frac"]
    s = out[main].sum(axis=1).replace(0, np.nan)
    out.loc[s.notna(), main] = out.loc[s.notna(), main].div(s[s.notna()], axis=0)

    out["product_weight_kg"] = pd.to_numeric(out.get("product_weight_kg"), errors="coerce").fillna(0.0).clip(lower=0.0001)
    out["product_weight_log"] = np.log1p(out["product_weight_kg"])
    out["year"] = pd.to_numeric(out.get("year"), errors="coerce").fillna(out.get("year", pd.Series([2015])).median()).astype(int)
    min_year = int(reference_min_year if reference_min_year is not None else out["year"].min())
    max_year = int(max_train_year if max_train_year is not None else out["year"].max())
    out["year_offset"] = out["year"] - min_year
    out["future_year_gap"] = np.maximum(out["year"] - max_year, 0)

    stages = out[["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]]
    out["lifecycle_fraction_sum"] = stages.sum(axis=1)
    out["lifecycle_balance_std"] = stages.std(axis=1)
    out["lifecycle_max_share"] = stages.max(axis=1)
    out["lifecycle_min_share"] = stages.min(axis=1)
    stage_names = ["Upstream", "Operations", "Downstream", "Transport", "End-of-life"]
    out["dominant_stage"] = stages.values.argmax(axis=1)
    out["dominant_stage"] = out["dominant_stage"].map(lambda i: stage_names[int(i)] if pd.notna(i) else "Unknown")

    out["upstream_x_weight_log"] = out["upstream_frac"] * out["product_weight_log"]
    out["operations_x_weight_log"] = out["operations_frac"] * out["product_weight_log"]
    out["downstream_x_weight_log"] = out["downstream_frac"] * out["product_weight_log"]
    out["transport_x_weight_log"] = out["transport_frac"] * out["product_weight_log"]

    out["product_name"] = out.get("product_name", "Unknown")
    out["product_detail"] = out.get("product_detail", "")
    out["product_name_length"] = out["product_name"].astype(str).str.len()
    out["product_detail_length"] = out["product_detail"].astype(str).str.len()
    out["is_weight_estimated"] = out.get("weight_source", "Unknown").astype(str).str.lower().str.contains("estim|external|calculated", regex=True).astype(int)
    out["has_stage_data"] = out.get("stage_level_available", "Unknown").astype(str).str.lower().str.startswith("yes").astype(int)
    out["has_transport_data"] = (out["transport_frac"] > 0).astype(int)
    out["has_eol_data"] = (out["end_of_life_frac"] > 0).astype(int)
    out["weight_category"] = out["product_weight_kg"].map(weight_category)
    out["protocol_simple"] = out.get("protocol_simple", out.get("protocol", "Unknown")).map(simplify_protocol)

    for c in CATEGORICAL_FEATURES:
        if c not in out.columns:
            out[c] = "Unknown"
        out[c] = out[c].map(clean_text_value)
    for c in NUMERIC_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def time_based_split(df: pd.DataFrame, year_col: str = "year") -> tuple[pd.DataFrame, pd.DataFrame, int]:
    years = sorted(df[year_col].dropna().unique())
    if len(years) >= 3:
        test_year = int(years[-1])
        train = df[df[year_col] < test_year].copy()
        test = df[df[year_col] == test_year].copy()
        if len(test) >= max(50, int(len(df) * 0.12)):
            return train, test, test_year
    # fallback: last 20% by year then original order
    d = df.sort_values(year_col).copy()
    cut = int(len(d) * 0.8)
    return d.iloc[:cut].copy(), d.iloc[cut:].copy(), int(d.iloc[cut][year_col])


def fit_label_thresholds(train_df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, float]:
    q25 = float(train_df[target_col].quantile(0.25))
    q75 = float(train_df[target_col].quantile(0.75))
    return {"q25": q25, "q75": q75}


def apply_carbon_labels(df: pd.DataFrame, thresholds: dict[str, float], target_col: str = TARGET_COL) -> pd.Series:
    q25, q75 = thresholds["q25"], thresholds["q75"]
    labels = np.where(df[target_col] <= q25, "Low", np.where(df[target_col] <= q75, "Medium", "High"))
    return pd.Series(labels, index=df.index)


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)


def get_classification_models() -> dict[str, Any]:
    # Chọn các mô hình chạy ổn định trên Streamlit Cloud/Colab.
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=400, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=None, min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
    }


def get_regression_models() -> dict[str, Any]:
    # Hồi quy dùng log-target để giảm ảnh hưởng outlier PCF rất lớn.
    return {
        "Dummy Mean": DummyRegressor(strategy="mean"),
        "Ridge log-target": TransformedTargetRegressor(regressor=Ridge(alpha=2.0), func=np.log1p, inverse_func=np.expm1),
        "Random Forest Regressor": TransformedTargetRegressor(regressor=RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1), func=np.log1p, inverse_func=np.expm1),
        "Extra Trees Regressor": TransformedTargetRegressor(regressor=ExtraTreesRegressor(n_estimators=100, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1), func=np.log1p, inverse_func=np.expm1),
    }


def make_clf_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def make_reg_pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def evaluate_classifier(model: Pipeline, X: pd.DataFrame, y_num: np.ndarray) -> dict[str, float]:
    pred = model.predict(X)
    if isinstance(pred[0], str):
        pred_num = np.array([LABEL_TO_NUM[p] for p in pred])
    else:
        pred_num = pred.astype(int)
    out = {
        "accuracy": accuracy_score(y_num, pred_num),
        "balanced_accuracy": balanced_accuracy_score(y_num, pred_num),
        "f1_macro": f1_score(y_num, pred_num, average="macro", zero_division=0),
        "precision_macro": precision_score(y_num, pred_num, average="macro", zero_division=0),
        "recall_macro": recall_score(y_num, pred_num, average="macro", zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            out["roc_auc_ovr"] = roc_auc_score(y_num, proba, multi_class="ovr")
        except Exception:
            out["roc_auc_ovr"] = np.nan
    else:
        out["roc_auc_ovr"] = np.nan
    return out


def safe_mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def median_ape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.median(np.abs((y_true - y_pred) / denom)) * 100)


def evaluate_regressor(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    pred = np.maximum(model.predict(X), 0)
    rmse = math.sqrt(mean_squared_error(y, pred))
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": rmse,
        "r2": r2_score(y, pred),
        "mape_pct": safe_mape(y, pred),
        "median_ape_pct": median_ape(y, pred),
    }


def build_ood_profile(train_df: pd.DataFrame) -> dict[str, Any]:
    numeric_stats = {}
    for c in NUMERIC_FEATURES:
        s = pd.to_numeric(train_df[c], errors="coerce")
        numeric_stats[c] = {"mean": float(s.mean()), "std": float(s.std() if s.std() > 1e-9 else 1.0), "min": float(s.min()), "max": float(s.max())}
    cat_levels = {c: sorted(train_df[c].astype(str).dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    return {"numeric_stats": numeric_stats, "cat_levels": cat_levels, "min_year": int(train_df["year"].min()), "max_year": int(train_df["year"].max())}


def check_ood(input_df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    max_z = 0.0
    for c, st in profile.get("numeric_stats", {}).items():
        if c not in input_df.columns:
            continue
        v = float(pd.to_numeric(input_df[c], errors="coerce").iloc[0])
        z = abs((v - st["mean"]) / max(st["std"], 1e-9))
        max_z = max(max_z, z)
        if v < st["min"] or v > st["max"]:
            warnings.append(f"{FEATURE_NAME_VI.get(c,c)} nằm ngoài khoảng dữ liệu huấn luyện ({st['min']:.2f}–{st['max']:.2f}).")
    unseen = []
    for c, levels in profile.get("cat_levels", {}).items():
        if c in input_df.columns:
            v = str(input_df[c].iloc[0])
            if v not in set(levels):
                unseen.append(FEATURE_NAME_VI.get(c, c))
    if unseen:
        warnings.append("Có giá trị phân loại chưa từng xuất hiện trong train: " + ", ".join(unseen[:5]))
    confidence = "Cao"
    if max_z > 4 or len(warnings) >= 3:
        confidence = "Thấp"
    elif max_z > 2.5 or warnings:
        confidence = "Trung bình"
    return {"warnings": warnings, "max_z": max_z, "unseen_count": len(unseen), "confidence": confidence}


def calculate_lca_bottom_up(inventory: pd.DataFrame) -> dict[str, Any]:
    """Calculate approximate PCF = sum(activity_amount * emission_factor)."""
    if inventory is None or len(inventory) == 0:
        return {"total_pcf": 0.0, "by_group": pd.DataFrame(), "detail": pd.DataFrame()}
    inv = inventory.copy()
    for col in ["amount", "emission_factor"]:
        inv[col] = pd.to_numeric(inv.get(col, 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    inv["co2e"] = inv["amount"] * inv["emission_factor"]
    inv["activity_group"] = inv.get("activity_group", "Other").astype(str).fillna("Other")
    by_group = inv.groupby("activity_group", as_index=False)["co2e"].sum().sort_values("co2e", ascending=False)
    return {"total_pcf": float(inv["co2e"].sum()), "by_group": by_group, "detail": inv}


def hybrid_pcf_estimate(ml_pcf: float, lca_pcf: float, lca_weight: float = 0.35) -> float:
    if lca_pcf and lca_pcf > 0:
        return float((1 - lca_weight) * ml_pcf + lca_weight * lca_pcf)
    return float(ml_pcf)


def scenario_projection(base_pcf: float, years: list[int], renewable_gain_pct: float = 0, material_reduction_pct: float = 0, logistics_gain_pct: float = 0) -> pd.DataFrame:
    """Simple scenario forecast for decision support, not ISO certification."""
    rows = []
    start = min(years)
    for y in years:
        horizon = max(y - start, 0)
        # gradual technology improvement assumed from user levers
        reduction = (renewable_gain_pct * 0.35 + material_reduction_pct * 0.45 + logistics_gain_pct * 0.20) / 100.0
        trend = 1 - min(reduction * (1 + horizon / 10), 0.85)
        rows.append({"year": y, "projected_pcf": max(base_pcf * trend, 0), "reduction_pct": (1 - trend) * 100})
    return pd.DataFrame(rows)


def build_input_row(reference_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    row = {}
    # sensible defaults from medians/modes
    for c in ["year", "product_weight_kg", "upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]:
        row[c] = kwargs.get(c, float(pd.to_numeric(reference_df.get(c, pd.Series([0])), errors="coerce").median()))
    for c in ["product_name", "product_detail", "country", "industry_group", "industry", "company_sector", "stage_level_available", "weight_source", "protocol", "protocol_simple", "upstream_estimated_from_operations"]:
        if c in kwargs:
            row[c] = kwargs[c]
        elif c in reference_df.columns:
            row[c] = reference_df[c].mode().iloc[0] if not reference_df[c].mode().empty else "Unknown"
        else:
            row[c] = "Unknown"
    df = pd.DataFrame([row])
    return add_model_features(df, reference_min_year=int(reference_df["year"].min()), max_train_year=int(reference_df["year"].max()))


def predict_with_package(package: dict[str, Any], input_row: pd.DataFrame) -> dict[str, Any]:
    feature_cols = package["metadata"]["feature_cols"]
    clf = package["classifier"]
    reg = package["regressor"]
    X = input_row[feature_cols]
    pred_num = int(clf.predict(X)[0])
    pred_label = NUM_TO_LABEL.get(pred_num, str(pred_num))
    proba = clf.predict_proba(X)[0] if hasattr(clf, "predict_proba") else np.eye(3)[pred_num]
    pcf = float(max(reg.predict(X)[0], 0))
    residual_q = package["metadata"].get("residual_abs_quantiles", {"p10": 0, "p90": 0})
    q10 = max(pcf - float(residual_q.get("p90", 0)), 0)
    q90 = pcf + float(residual_q.get("p90", 0))
    return {"label": pred_label, "label_vi": LABEL_VI.get(pred_label, pred_label), "proba": proba, "pcf": pcf, "p10": q10, "p90": q90}


def get_local_factor_impact(package: dict[str, Any], input_row: pd.DataFrame, n_top: int = 6) -> pd.DataFrame:
    """Simple local perturbation explanation on classifier probability for predicted class."""
    clf = package["classifier"]
    meta = package["metadata"]
    feature_cols = meta["feature_cols"]
    base_X = input_row[feature_cols].copy()
    base_proba = clf.predict_proba(base_X)[0]
    pred_idx = int(np.argmax(base_proba))
    impacts = []

    # numeric perturbation: +10% or +0.1 depending magnitude
    for c in NUMERIC_FEATURES:
        if c not in base_X.columns:
            continue
        test = base_X.copy()
        v = float(test[c].iloc[0]) if pd.notna(test[c].iloc[0]) else 0.0
        delta = max(abs(v) * 0.10, 0.05)
        test[c] = v + delta
        try:
            new_proba = clf.predict_proba(test)[0][pred_idx]
            impacts.append({"feature": c, "feature_vi": FEATURE_NAME_VI.get(c, c), "impact": float(new_proba - base_proba[pred_idx])})
        except Exception:
            pass

    # categorical perturbation: set to most frequent alternative if available
    cat_levels = meta.get("ood_profile", {}).get("cat_levels", {})
    for c in CATEGORICAL_FEATURES:
        if c not in base_X.columns:
            continue
        cur = str(base_X[c].iloc[0])
        alts = [v for v in cat_levels.get(c, []) if v != cur]
        if not alts:
            continue
        test = base_X.copy()
        test[c] = alts[0]
        try:
            new_proba = clf.predict_proba(test)[0][pred_idx]
            impacts.append({"feature": c, "feature_vi": FEATURE_NAME_VI.get(c, c), "impact": float(new_proba - base_proba[pred_idx])})
        except Exception:
            pass

    out = pd.DataFrame(impacts)
    if out.empty:
        return out
    out["abs_impact"] = out["impact"].abs()
    out["direction_vi"] = np.where(out["impact"] >= 0, "Tăng xu hướng nhãn dự báo", "Giảm xu hướng nhãn dự báo")
    return out.sort_values("abs_impact", ascending=False).head(n_top).reset_index(drop=True)


def save_package(package: dict[str, Any], path: str | Path = MODEL_PATH, also_root: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, path)
    if also_root:
        joblib.dump(package, ROOT_MODEL_PATH)


def load_package(path: str | Path | None = None) -> dict[str, Any]:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates += [ROOT_MODEL_PATH, MODEL_PATH, Path("outputs/models/ecopredict_model_package.joblib")]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError("Không tìm thấy ecopredict_model_package.joblib. Hãy chạy train_advanced_models.py trước.")


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"
