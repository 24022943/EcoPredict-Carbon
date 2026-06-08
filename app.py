"""
app.py
EcoPredict Carbon Streamlit web app.
Run:
    python -m streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import subprocess
import sys
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from carbon_utils import (
    DATA_PATH, TARGET_COL, LABEL_ORDER, LABEL_TO_NUM, LABEL_VI,
    FEATURE_NAME_VI, DEFAULT_EMISSION_FACTORS,
    load_carbon_catalogue, add_model_features, load_package, build_input_row,
    predict_with_package, calculate_lca_bottom_up, hybrid_pcf_estimate, scenario_projection,
    check_ood, get_local_factor_impact, fmt_num,
)

st.set_page_config(
    page_title="EcoPredict Carbon",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root{--primary:#047857;--accent:#10b981;--dark:#063b2d;--bg:#f4fbf7;--muted:#64748b;--line:#dbe7e0;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#f7fbf9 0%,#edf7f2 100%);} 
[data-testid="stSidebar"]{background:linear-gradient(180deg,#063b2d 0%,#0f5132 100%);} 
[data-testid="stSidebar"] *{color:#f7fff9 !important;}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] div[data-baseweb="input"] input,
[data-testid="stSidebar"] div[data-baseweb="base-input"] input,
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] > div{color:#111827 !important;-webkit-text-fill-color:#111827 !important;}
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="input"] > div,
[data-testid="stSidebar"] div[data-baseweb="base-input"]{background:#ffffff !important;border-radius:14px !important;}
div[data-baseweb="popover"],div[data-baseweb="popover"] *,div[role="listbox"],div[role="option"]{color:#111827 !important;-webkit-text-fill-color:#111827 !important;}
.hero-shell{position:relative;overflow:hidden;margin-bottom:24px;padding:34px 36px 28px 36px;border-radius:32px;background:linear-gradient(180deg,#f4fbf7 0%,#edf7f2 100%);border:1px solid #bce5cf;box-shadow:0 18px 42px rgba(15,81,50,.08);} 
.hero-shell::after{content:"";position:absolute;right:-70px;bottom:-100px;width:360px;height:360px;border-radius:50%;background:radial-gradient(circle,rgba(16,185,129,.18),rgba(16,185,129,.10) 55%,transparent 56%);} 
.hero-badge{display:inline-flex;align-items:center;gap:10px;padding:14px 24px;border-radius:999px;background:#d9ece4;color:#0f5132;font-size:18px;font-weight:850;margin-bottom:24px;} 
.hero-title{position:relative;z-index:1;font-size:42px;line-height:1.16;font-weight:900;color:#0f172a;margin:0 0 16px 0;letter-spacing:-1px;max-width:1050px;} 
.hero-subtitle{position:relative;z-index:1;max-width:1100px;font-size:18px;line-height:1.7;color:#667085;margin-bottom:24px;} 
.hero-chip-row{display:flex;flex-wrap:wrap;gap:16px;position:relative;z-index:1;} .hero-chip{display:inline-flex;align-items:center;padding:14px 24px;border-radius:999px;background:#d6efe3;color:#0f5132;font-size:16px;font-weight:800;} 
.card{background:white;padding:24px;border-radius:26px;box-shadow:0 12px 32px rgba(15,81,50,.07);border:1px solid #e6efe9;margin-bottom:18px;} 
.section-title{font-size:24px;font-weight:900;color:#0f172a;margin-bottom:8px;} .card-subtitle{font-size:15px;color:#667085;margin-bottom:14px;} 
.kpi-card{background:white;border-radius:24px;padding:24px;box-shadow:0 12px 28px rgba(15,81,50,.07);border:1px solid #e8f2ed;min-height:130px;} 
.kpi-title{font-size:14px;color:#64748b;font-weight:800;margin-bottom:8px;} .kpi-value{font-size:38px;font-weight:900;margin:0;letter-spacing:-1px;color:#0f172a;} .kpi-note{font-size:13px;color:#64748b;margin-top:8px;line-height:1.45;} 
.kpi-green{background:radial-gradient(circle at 90% 20%,rgba(16,185,129,.30),transparent 30%),linear-gradient(135deg,#047857 0%,#0f5132 100%);color:white;border:none;} .kpi-green .kpi-title,.kpi-green .kpi-value,.kpi-green .kpi-note{color:white;} 
.success-box{padding:16px 18px;border-radius:18px;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;line-height:1.65;} .warning-box{padding:16px 18px;border-radius:18px;background:#fff7ed;color:#9a3412;border:1px solid #fed7aa;line-height:1.65;} 
.insight-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:8px 0 14px 0;} .insight-card{background:#f8fbfa;border:1px solid #e5ece8;border-radius:18px;padding:16px 18px;} .insight-label{font-size:13px;color:#64748b;font-weight:800;margin-bottom:6px;} .insight-value{font-size:24px;color:#111827;font-weight:900;line-height:1.15;} .insight-note{font-size:13px;color:#64748b;margin-top:4px;line-height:1.45;} 
.badge{display:inline-block;padding:8px 12px;border-radius:999px;font-size:13px;font-weight:800;background:#dcfce7;color:#166534;margin-right:6px;} 
.small-muted{font-size:13px;color:#64748b;} 
@media(max-width:900px){.hero-title{font-size:32px}.insight-grid{grid-template-columns:1fr}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def render_hero() -> None:
    st.markdown(
        """
        <div class='hero-shell'>
            <div class='hero-badge'>🌱 EcoPredict Carbon • VI / EN</div>
            <div class='hero-title'>EcoPredict Carbon – Hệ thống dự báo phát thải carbon của sản phẩm</div>
            <div class='hero-subtitle'>Product Carbon Footprint prediction and emission-level classification based on Carbon Catalogue data</div>
            <div class='hero-chip-row'>
                <div class='hero-chip'>Machine Learning</div>
                <div class='hero-chip'>Carbon Catalogue</div>
                <div class='hero-chip'>PCF Prediction</div>
                <div class='hero-chip'>Eco-Tech Dashboard</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data_cached(path: str) -> pd.DataFrame:
    return add_model_features(load_carbon_catalogue(path))


@st.cache_resource(show_spinner=False)
def load_model_cached() -> dict[str, Any] | None:
    for p in [Path("ecopredict_model_package.joblib"), Path("outputs/models/ecopredict_model_package.joblib")]:
        if p.exists():
            return load_package(p)
    return None


def bootstrap() -> tuple[pd.DataFrame, dict[str, Any] | None]:
    data = load_data_cached(str(DATA_PATH))
    package = load_model_cached()
    return data, package


def train_now_button() -> None:
    st.warning("Chưa có model package. Hãy chạy `python train_advanced_models.py` trước, hoặc bấm nút dưới để huấn luyện ngay trên môi trường hiện tại.")
    if st.button("Huấn luyện model ngay", type="primary"):
        with st.spinner("Đang huấn luyện mô hình, vui lòng chờ..."):
            result = subprocess.run([sys.executable, "train_advanced_models.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Huấn luyện xong. Hãy refresh app.")
                st.code(result.stdout[-2000:])
            else:
                st.error("Huấn luyện lỗi")
                st.code(result.stderr[-4000:])


def plot_probability_bar(proba: np.ndarray) -> go.Figure:
    labels = [LABEL_VI[x] for x in LABEL_ORDER]
    colors = ["#047857", "#f59e0b", "#ef4444"]
    fig = go.Figure(go.Bar(x=labels, y=proba * 100, marker_color=colors, text=[f"{p*100:.1f}%" for p in proba], textposition="outside", showlegend=False))
    fig.update_layout(height=310, margin=dict(l=30,r=30,t=10,b=40), yaxis_title="Xác suất (%)", xaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), yaxis=dict(range=[0, max(100, float(proba.max()*115))], gridcolor="#e5e7eb"))
    return fig


def plot_lifecycle_stacked(up: float, op: float, down: float, transport: float = 0.0, eol: float = 0.0) -> go.Figure:
    names = ["Upstream", "Operations", "Downstream", "Transport", "End-of-life"]
    vals = np.array([up, op, down, transport, eol], dtype=float)
    vals = vals / vals.sum() if vals.sum() > 0 else vals
    colors = ["#047857", "#10b981", "#86efac", "#64748b", "#cbd5e1"]
    fig = go.Figure()
    left = 0
    for name, val, color in zip(names, vals, colors):
        fig.add_trace(go.Bar(y=["Tỷ trọng"], x=[val*100], orientation="h", name=name, marker_color=color, text=[f"{val*100:.1f}%"], textposition="inside"))
        left += val
    fig.update_layout(barmode="stack", height=220, margin=dict(l=20,r=20,t=10,b=40), xaxis_title="Tỷ trọng (%)", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), legend=dict(orientation="h", y=-0.3), xaxis=dict(range=[0,100], gridcolor="#e5e7eb"))
    return fig


def plot_benchmark(pred_pcf: float, industry_median: float, global_median: float) -> go.Figure:
    labels = ["Sản phẩm của bạn", "Trung vị ngành", "Trung vị toàn bộ"]
    values = [pred_pcf, industry_median, global_median]
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker_color=["#047857", "#64748b", "#cbd5e1"], text=[fmt_num(v) for v in values], textposition="outside", showlegend=False))
    fig.update_layout(height=310, margin=dict(l=135,r=70,t=8,b=40), xaxis_title="kg CO₂e", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827", size=13), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(categoryorder="array", categoryarray=labels[::-1]))
    return fig


def plot_factor_impact(impact_df: pd.DataFrame) -> go.Figure:
    if impact_df.empty:
        return go.Figure()
    d = impact_df.copy().sort_values("impact", ascending=True)
    labels = d["feature_vi"].map(lambda s: s if len(str(s)) <= 30 else str(s)[:27] + "...")
    colors = np.where(d["impact"] >= 0, "#047857", "#f97316")
    max_abs = max(float(d["impact"].abs().max()), 1e-6)
    fig = go.Figure(go.Bar(y=labels, x=d["impact"], orientation="h", marker_color=colors, text=[f"{v:+.3f}" for v in d["impact"]], textposition="outside", cliponaxis=False, hovertemplate="%{customdata}<br>Tác động: %{x:.4f}<extra></extra>", customdata=d["feature_vi"], showlegend=False))
    fig.add_vline(x=0, line_color="#94a3b8", line_dash="dash")
    fig.update_layout(height=400, margin=dict(l=170,r=85,t=10,b=45), xaxis_title="Thay đổi xác suất nhãn dự báo khi tăng/đổi yếu tố", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827", size=13), xaxis=dict(range=[-max_abs*1.45, max_abs*1.45], gridcolor="#e5e7eb"), yaxis=dict(automargin=True))
    return fig


def plot_scenario(df_scn: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df_scn["year"], y=df_scn["projected_pcf"], mode="lines+markers", line=dict(color="#047857", width=3), marker=dict(size=9), name="Scenario PCF"))
    fig.update_layout(height=300, margin=dict(l=40,r=30,t=10,b=40), xaxis_title="Năm", yaxis_title="PCF dự kiến (kg CO₂e)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"))
    return fig


def benchmark_stats(data: pd.DataFrame, pred_pcf: float, industry_group: str) -> dict[str, Any]:
    subset = data[data["industry_group"].eq(industry_group)]
    if subset.empty:
        subset = data
    industry_median = float(subset[TARGET_COL].median())
    global_median = float(data[TARGET_COL].median())
    percentile = float((subset[TARGET_COL] <= pred_pcf).mean() * 100)
    ratio = pred_pcf / max(industry_median, 1e-9)
    diff = pred_pcf - industry_median
    if ratio < 0.8:
        conclusion = "PCF dự báo thấp hơn đáng kể so với trung vị ngành. Đây là tín hiệu tích cực, nhưng vẫn cần đối chiếu với dữ liệu LCA thực tế trước khi dùng như chứng nhận xanh."
    elif ratio <= 1.2:
        conclusion = "PCF dự báo nằm gần vùng trung vị ngành. Sản phẩm không quá khác biệt so với nhóm tham chiếu, nên cần phân tích yếu tố ảnh hưởng để tìm cơ hội giảm phát thải."
    else:
        conclusion = "PCF dự báo cao hơn trung vị ngành. Nên rà soát vật liệu, năng lượng, vận chuyển và tỷ trọng vòng đời để tìm điểm tối ưu."
    return {"industry_median": industry_median, "global_median": global_median, "percentile": percentile, "ratio": ratio, "diff": diff, "n": int(len(subset)), "conclusion": conclusion}


def prediction_page(data: pd.DataFrame, package: dict[str, Any]) -> None:
    render_hero()
    meta = package["metadata"]
    min_year = int(data["year"].min())
    max_year = int(data["year"].max())

    with st.sidebar:
        st.markdown("## 🌱 EcoPredict Carbon")
        st.caption("Nhập thông tin sản phẩm để dự báo PCF và phân loại mức phát thải.")
        st.markdown("---")
        year = st.slider("Năm báo cáo / kịch bản", min_year, 2035, max_year)
        weight = st.number_input("Khối lượng sản phẩm (kg)", min_value=0.0001, value=float(data["product_weight_kg"].median()), step=0.1)
        countries = sorted(data["country"].dropna().unique().tolist()) + ["Other"]
        country = st.selectbox("Quốc gia", countries, index=min(0, len(countries)-1))
        groups = sorted(data["industry_group"].dropna().unique().tolist()) + ["Other"]
        industry_group = st.selectbox("Nhóm ngành GICS", groups)
        inds = sorted(data.loc[data["industry_group"].eq(industry_group), "industry"].dropna().unique().tolist()) or sorted(data["industry"].dropna().unique().tolist())
        industry = st.selectbox("Ngành sản phẩm", inds + ["Other"])
        sectors = sorted(data["company_sector"].dropna().unique().tolist()) + ["Other"]
        sector = st.selectbox("Sector", sectors)
        protocol = st.selectbox("Chuẩn PCF", sorted(data["protocol_simple"].dropna().unique().tolist()) + ["Other"])
        weight_source = st.selectbox("Nguồn khối lượng", sorted(data["weight_source"].dropna().unique().tolist())[:30] + ["Other"])
        st.markdown("### Tỷ trọng vòng đời")
        upstream = st.slider("Upstream (%)", 0, 100, 45)
        operations = st.slider("Operations (%)", 0, 100, 35)
        downstream = st.slider("Downstream (%)", 0, 100, 20)
        transport = st.slider("Transport (%)", 0, 100, 5)
        eol = st.slider("End-of-life (%)", 0, 100, 0)

    if year > max_year:
        st.markdown(f"<div class='warning-box'>Năm {year} nằm ngoài phạm vi dữ liệu huấn luyện ({min_year}–{max_year}). Kết quả là dự báo kịch bản tham khảo, không phải chứng nhận LCA/ISO chính thức.</div>", unsafe_allow_html=True)

    total = max(upstream + operations + downstream, 1)
    up_frac, op_frac, down_frac = upstream/total, operations/total, downstream/total

    input_row = build_input_row(
        data, year=year, product_weight_kg=weight, country=country, industry_group=industry_group,
        industry=industry, company_sector=sector, protocol=protocol, protocol_simple=protocol,
        weight_source=weight_source, stage_level_available="Yes",
        upstream_frac=up_frac, operations_frac=op_frac, downstream_frac=down_frac,
        transport_frac=transport/100, end_of_life_frac=eol/100,
        product_name="User product", product_detail="User scenario input",
    )
    pred = predict_with_package(package, input_row)

    with st.expander("LCA bottom-up inventory editor (không bắt buộc)", expanded=False):
        st.caption("PCF bottom-up = Σ(activity data × emission factor). Hệ số dưới đây là demo; khi làm ISO cần nguồn hệ số chính thức và có version.")
        default_inv = DEFAULT_EMISSION_FACTORS.copy()
        default_inv["amount"] = [weight*0.6, 0, weight*0.2, weight*0.1, 50, weight*0.2]
        inv = st.data_editor(default_inv[["activity_group", "activity_name", "unit", "amount", "emission_factor", "source", "quality"]], num_rows="dynamic", use_container_width=True)
        lca = calculate_lca_bottom_up(inv)
        st.write(f"PCF bottom-up ước tính: **{fmt_num(lca['total_pcf'])} kg CO₂e**")
    lca_pcf = lca["total_pcf"] if "lca" in locals() else 0.0
    hybrid = hybrid_pcf_estimate(pred["pcf"], lca_pcf)

    ood = check_ood(input_row, meta.get("ood_profile", {}))
    if ood["warnings"]:
        st.markdown("<div class='warning-box'><b>Cảnh báo ngoài miền dữ liệu:</b><br>" + "<br>".join(ood["warnings"][:4]) + f"<br>Độ tin cậy tương đối: <b>{ood['confidence']}</b></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi-card kpi-green'><div class='kpi-title'>Phân loại carbon</div><div class='kpi-value'>{pred['label_vi']}</div><div class='kpi-note'>Mô hình: {meta.get('best_classifier_name','')}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>PCF ML ước lượng</div><div class='kpi-value'>{fmt_num(pred['pcf'])}</div><div class='kpi-note'>kg CO₂e; khoảng P10–P90: {fmt_num(pred['p10'])}–{fmt_num(pred['p90'])}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>PCF Hybrid</div><div class='kpi-value'>{fmt_num(hybrid)}</div><div class='kpi-note'>Kết hợp ML và LCA bottom-up nếu có inventory.</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Độ tin cậy dữ liệu</div><div class='kpi-value'>{ood['confidence']}</div><div class='kpi-note'>Dựa trên OOD check và phạm vi dữ liệu huấn luyện.</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Diễn giải kết quả</div>", unsafe_allow_html=True)
    st.markdown("<div class='success-box'>Hệ thống sử dụng mô hình học máy để ước lượng PCF và phân loại mức phát thải. Kết quả là ước tính hỗ trợ ra quyết định, không thay thế đánh giá LCA/ISO chính thức nếu chưa có kiểm định chuyên gia và dữ liệu kiểm chứng.</div></div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='section-title'>Xác suất phân loại</div><div class='card-subtitle'>Xác suất mô hình xếp sản phẩm vào từng mức phát thải.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_probability_bar(pred["proba"]), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='section-title'>Tỷ trọng vòng đời</div><div class='card-subtitle'>Stacked bar giúp so sánh tỷ trọng rõ hơn biểu đồ bánh rán.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_lifecycle_stacked(up_frac, op_frac, down_frac, transport/100, eol/100), use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>So sánh PCF với ngành</div><div class='card-subtitle'>Benchmark chỉ dựa trên Carbon Catalogue, dùng để tham khảo tương đối.</div>", unsafe_allow_html=True)
    stats = benchmark_stats(data, hybrid, industry_group)
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(plot_benchmark(hybrid, stats["industry_median"], stats["global_median"]), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        st.markdown(f"""
        <div class='insight-grid'>
          <div class='insight-card'><div class='insight-label'>Vị trí trong nhóm ngành</div><div class='insight-value'>{stats['percentile']:.1f}%</div><div class='insight-note'>Tỷ lệ mẫu cùng ngành có PCF thấp hơn hoặc bằng sản phẩm này.</div></div>
          <div class='insight-card'><div class='insight-label'>Tỷ lệ so với trung vị ngành</div><div class='insight-value'>{stats['ratio']:.2f}x</div><div class='insight-note'>Nhỏ hơn 1 là thấp hơn benchmark ngành.</div></div>
          <div class='insight-card'><div class='insight-label'>Số mẫu tham chiếu</div><div class='insight-value'>{stats['n']}</div><div class='insight-note'>Số mẫu cùng nhóm ngành trong dữ liệu.</div></div>
          <div class='insight-card'><div class='insight-label'>Chênh lệch tuyệt đối</div><div class='insight-value'>{fmt_num(stats['diff'])}</div><div class='insight-note'>kg CO₂e so với trung vị ngành.</div></div>
        </div>
        <div class='success-box'>{stats['conclusion']}</div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Scenario forecasting 2026–2035</div><div class='card-subtitle'>Mô phỏng tác động nếu tăng điện tái tạo, giảm vật liệu hoặc cải thiện logistics.</div>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1: ren = st.slider("Tăng tỷ lệ điện tái tạo (%)", 0, 100, 20)
    with s2: mat_red = st.slider("Giảm khối lượng/vật liệu (%)", 0, 80, 10)
    with s3: logi = st.slider("Cải thiện vận chuyển (%)", 0, 80, 10)
    scn = scenario_projection(hybrid, [2026, 2027, 2030, 2035], ren, mat_red, logi)
    st.plotly_chart(plot_scenario(scn), use_container_width=True, config=PLOTLY_CONFIG)
    st.dataframe(scn.assign(projected_pcf=scn["projected_pcf"].round(2), reduction_pct=scn["reduction_pct"].round(1)), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'><div class='section-title'>Giải thích yếu tố ảnh hưởng</div><div class='card-subtitle'>Top 6 yếu tố cục bộ ảnh hưởng đến nhãn dự báo, không hiểu là quan hệ nhân quả tuyệt đối.</div>", unsafe_allow_html=True)
    impact = get_local_factor_impact(package, input_row, n_top=6)
    if impact.empty:
        st.info("Không tạo được giải thích cục bộ cho dự báo này.")
    else:
        top_names = ", ".join(impact["feature_vi"].head(3).tolist())
        st.markdown(f"<div class='success-box'>Các yếu tố ảnh hưởng mạnh nhất trong dự báo hiện tại gồm: {top_names}. Thanh xanh làm tăng xác suất nhãn hiện tại, thanh cam kéo dự báo theo chiều ngược lại.</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_factor_impact(impact), use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("Bảng chi tiết dùng để xem đầy đủ tên yếu tố nếu nhãn biểu đồ đã được rút gọn.")
        st.dataframe(impact[["feature_vi", "impact", "direction_vi"]].rename(columns={"feature_vi":"Tên yếu tố", "impact":"Mức tác động", "direction_vi":"Chiều tác động"}), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def data_page(data: pd.DataFrame) -> None:
    render_hero()
    st.markdown("<div class='card'><div class='section-title'>Tổng quan dữ liệu Carbon Catalogue</div><div class='card-subtitle'>PCF có phân phối lệch phải mạnh, vì vậy hệ thống ưu tiên biểu đồ log-scale.</div>", unsafe_allow_html=True)
    k1,k2,k3,k4,k5 = st.columns(5)
    stats = {
        "Số mẫu": len(data), "Số cột": data.shape[1], "PCF thấp nhất": data[TARGET_COL].min(),
        "PCF trung vị": data[TARGET_COL].median(), "PCF cao nhất": data[TARGET_COL].max()
    }
    for col, (lab, val) in zip([k1,k2,k3,k4,k5], stats.items()):
        with col:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>{lab}</div><div class='kpi-value'>{fmt_num(float(val)) if isinstance(val,(int,float,np.integer,np.floating)) else val}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><div class='section-title'>Phân phối PCF theo thang log</div>", unsafe_allow_html=True)
        fig = px.histogram(data, x=np.log1p(data[TARGET_COL]), nbins=45, labels={"x":"log(PCF + 1)"})
        fig.update_traces(marker_color="#047857")
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("<div class='small-muted'>Do PCF có một số giá trị rất lớn, histogram tuyến tính sẽ làm dữ liệu dồn vào một cột. Log-scale giúp nhìn rõ cấu trúc phân phối hơn.</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><div class='section-title'>Top nhóm ngành theo số mẫu</div>", unsafe_allow_html=True)
        counts = data["industry_group"].value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(y=counts.index, x=counts.values, orientation="h", marker_color="#10b981", text=counts.values, textposition="outside"))
        fig.update_layout(height=360, margin=dict(l=170,r=40,t=5,b=35), xaxis_title="Số mẫu", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='card'><div class='section-title'>Xu hướng PCF theo năm</div>", unsafe_allow_html=True)
        t = data.groupby("year")[TARGET_COL].agg(["median", "mean", "count"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t["year"], y=t["median"], mode="lines+markers", name="Median", line=dict(color="#047857", width=3)))
        fig.add_trace(go.Scatter(x=t["year"], y=t["mean"], mode="lines+markers", name="Mean", line=dict(color="#64748b", width=3)))
        fig.update_layout(height=350, yaxis_type="log", yaxis_title="PCF kg CO₂e (log)", xaxis_title="Năm", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#111827"), xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card'><div class='section-title'>Tỷ trọng vòng đời trung bình</div>", unsafe_allow_html=True)
        vals = data[["upstream_frac", "operations_frac", "downstream_frac", "transport_frac", "end_of_life_frac"]].mean().fillna(0)
        fig = plot_lifecycle_stacked(vals["upstream_frac"], vals["operations_frac"], vals["downstream_frac"], vals["transport_frac"], vals["end_of_life_frac"])
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)


def lca_iso_page(data: pd.DataFrame) -> None:
    render_hero()
    st.markdown("<div class='card'><div class='section-title'>LCA/ISO Decision-support Layer</div><div class='card-subtitle'>Phần này mô phỏng cấu trúc cần có nếu phát triển thành hệ thống hỗ trợ LCA theo ISO 14040/14044/14067.</div>", unsafe_allow_html=True)
    st.markdown("<div class='warning-box'>Lưu ý: hệ thống hiện là prototype học thuật. Kết quả chưa phải chứng nhận ISO/EPD chính thức vì chưa có cơ sở emission factors được kiểm định, audit trail đầy đủ và review bởi bên thứ ba.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        st.markdown("<div class='card'><div class='section-title'>Goal & Scope</div>", unsafe_allow_html=True)
        st.text_input("Mục tiêu nghiên cứu", "Ước tính PCF và so sánh kịch bản giảm phát thải")
        st.text_input("Functional unit", "1 sản phẩm / 1 đơn vị chức năng")
        st.selectbox("System boundary", ["Cradle-to-gate", "Cradle-to-grave", "Gate-to-gate"])
        st.selectbox("Allocation method", ["Mass allocation", "Economic allocation", "Energy allocation", "Not applicable"])
        st.text_area("Cut-off criteria", "Loại các dòng vật liệu/hoạt động có đóng góp rất nhỏ nếu thiếu dữ liệu đáng tin cậy.")
        st.markdown("</div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><div class='section-title'>Data Quality Rating</div>", unsafe_allow_html=True)
        temporal = st.slider("Temporal representativeness", 1, 5, 3)
        geo = st.slider("Geographical representativeness", 1, 5, 3)
        tech = st.slider("Technological representativeness", 1, 5, 3)
        complete = st.slider("Completeness", 1, 5, 3)
        reliab = st.slider("Reliability", 1, 5, 3)
        score = np.mean([temporal, geo, tech, complete, reliab])
        st.metric("Điểm chất lượng dữ liệu", f"{score:.1f}/5")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'><div class='section-title'>Audit trail cần lưu khi triển khai thật</div>", unsafe_allow_html=True)
    st.markdown("""
    - Input sản phẩm, inventory, emission factor và nguồn factor.
    - Phiên bản model, phiên bản dữ liệu, ngày chạy, người chạy.
    - Kết quả PCF, khoảng bất định, cảnh báo OOD và nhận xét reviewer.
    - Trạng thái dự án: draft / review / approved.
    """)
    st.markdown("</div>", unsafe_allow_html=True)


def advanced_page(package: dict[str, Any] | None) -> None:
    render_hero()
    if package is None:
        train_now_button(); return
    meta = package["metadata"]
    st.markdown("<div class='card'><div class='section-title'>Thông tin mô hình và thuật toán</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <span class='badge'>Best Classifier: {meta.get('best_classifier_name')}</span>
    <span class='badge'>Best Regressor: {meta.get('best_regressor_name')}</span>
    <span class='badge'>{meta.get('split_strategy')}</span>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Metric phân loại", "Metric hồi quy", "Kiểm soát chất lượng ML"])
    with tab1:
        st.dataframe(pd.DataFrame(meta.get("classification_metrics", [])), use_container_width=True)
        for img in ["outputs/figures/model_classification_f1_comparison.png", "outputs/figures/model_confusion_matrix.png", "outputs/figures/model_roc_curve.png"]:
            if Path(img).exists(): st.image(img)
    with tab2:
        st.dataframe(pd.DataFrame(meta.get("regression_metrics", [])), use_container_width=True)
        for img in ["outputs/figures/model_regression_median_ape_comparison.png", "outputs/figures/model_regression_actual_vs_predicted.png", "outputs/figures/model_regression_residuals.png"]:
            if Path(img).exists(): st.image(img)
    with tab3:
        st.markdown("""
        - Nhãn Low/Medium/High được tạo bằng Q25/Q75 trên train set, tránh label leakage.
        - Không dùng `carbon_intensity` làm feature vì biến này có quan hệ trực tiếp với PCF.
        - Split theo thời gian: train trên năm cũ, test trên năm mới hơn để mô phỏng dự báo tương lai.
        - Có OOD check và uncertainty interval để tránh hiểu sai một giá trị dự báo đơn lẻ.
        - ML là lớp hỗ trợ dự báo, còn LCA/ISO đầy đủ cần bottom-up inventory × emission factors và audit trail.
        """)
        if Path("outputs/figures/model_permutation_importance.png").exists(): st.image("outputs/figures/model_permutation_importance.png")


def guide_page() -> None:
    render_hero()
    st.markdown("<div class='card'><div class='section-title'>Hướng dẫn sử dụng</div>", unsafe_allow_html=True)
    st.markdown("""
    1. Vào trang **Dự báo**, nhập thông tin sản phẩm và tỷ trọng vòng đời.
    2. Xem kết quả PCF ML, PCF Hybrid, phân loại Low/Medium/High và khoảng bất định.
    3. Dùng phần **So sánh PCF với ngành** để hiểu sản phẩm thấp/cao hơn benchmark tương đối.
    4. Dùng **Scenario forecasting** để mô phỏng kịch bản 2026–2035.
    5. Xem **Giải thích yếu tố ảnh hưởng** để biết yếu tố nào đang làm tăng/giảm xu hướng dự báo.
    6. Kết quả chỉ là ước tính hỗ trợ quyết định, chưa thay thế báo cáo LCA/ISO chính thức.
    """)
    st.code("python carbon_eda.py\npython train_advanced_models.py\npython -m streamlit run app.py", language="powershell")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    data, package = bootstrap()
    with st.sidebar:
        page = st.radio("", ["Dự báo", "Dữ liệu", "LCA/ISO nâng cao", "Thông tin nâng cao", "Hướng dẫn"], index=0)
    if page == "Dự báo":
        if package is None:
            render_hero(); train_now_button()
        else:
            prediction_page(data, package)
    elif page == "Dữ liệu":
        data_page(data)
    elif page == "LCA/ISO nâng cao":
        lca_iso_page(data)
    elif page == "Thông tin nâng cao":
        advanced_page(package)
    else:
        guide_page()


if __name__ == "__main__":
    main()
