"""
ISO 9283 Repeatability & Accuracy Analyzer — Web App
=====================================================
Streamlit tabanlı web uygulaması.
CSV yükleme veya manuel tablo girişi desteklenir.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
#  Quality thresholds (mm)  — ISO 9283
# ─────────────────────────────────────────────
THRESHOLDS = {
    "Iyi":     0.01,
    "Kabul":   0.02,
    "Sinirli": 0.05,
}

GRADE_COLORS = {
    "Iyi":     "#2d9e4b",
    "Kabul":   "#c89000",
    "Sinirli": "#c04400",
    "Kotu":    "#cc1e1e",
}
GRADE_COLORS_LIGHT = {
    "Iyi":     "#d6f5db",
    "Kabul":   "#fff4cc",
    "Sinirli": "#ffe8d1",
    "Kotu":    "#ffd6d6",
}
GRADE_LABELS = {
    "Iyi":     "İyi (Yüksek Performans)",
    "Kabul":   "Kabul Edilebilir",
    "Sinirli": "Sınırlı",
    "Kotu":    "Uygun Değil",
}


def rate_quality(value: float) -> str:
    if value <= THRESHOLDS["Iyi"]:
        return "Iyi"
    elif value <= THRESHOLDS["Kabul"]:
        return "Kabul"
    elif value <= THRESHOLDS["Sinirli"]:
        return "Sinirli"
    else:
        return "Kotu"


def grade_label(g: str) -> str:
    return GRADE_LABELS.get(g, g)


# ─────────────────────────────────────────────
#  ISO 9283 Core Calculations
# ─────────────────────────────────────────────

def normalize_to_set1(df_raw):
    df = df_raw.copy()
    numeric_cols = [c for c in df.columns if c != "Set"]
    reference = df[numeric_cols].iloc[0]
    df[numeric_cols] = df[numeric_cols] - reference
    return df


def parse_poses(df_norm):
    pose_data = {}
    numeric_cols = [c for c in df_norm.columns if c != "Set"]
    pose_ids = sorted(set(c.split("_")[0] for c in numeric_cols))
    for pid in pose_ids:
        axes = sorted([c for c in numeric_cols if c.startswith(pid + "_")])
        if len(axes) >= 3:
            sub = df_norm[axes].copy()
            sub.columns = [a.split("_")[1] for a in axes]
            pose_data[pid] = sub
    return pose_data


def compute_rp(pose_df):
    mean_pos = pose_df.mean()
    l_i   = np.sqrt(((pose_df - mean_pos) ** 2).sum(axis=1))
    l_bar = l_i.mean()
    S_l   = l_i.std(ddof=1)
    RP    = l_bar + 3 * S_l
    return {
        "mean_A": float(mean_pos.iloc[0]),
        "mean_B": float(mean_pos.iloc[1]),
        "mean_C": float(mean_pos.iloc[2]),
        "l_bar":  float(l_bar),
        "S_l":    float(S_l),
        "RP":     float(RP),
        "l_i":    l_i,
    }


def compute_ap(pose_df, cmd_A=0.0, cmd_B=0.0, cmd_C=0.0):
    mean_pos = pose_df.mean()
    return float(np.sqrt(
        (mean_pos.iloc[0] - cmd_A) ** 2 +
        (mean_pos.iloc[1] - cmd_B) ** 2 +
        (mean_pos.iloc[2] - cmd_C) ** 2
    ))


def axis_repeatability(pose_df):
    out = {}
    for col in pose_df.columns:
        v = pose_df[col]
        out[col] = {
            "mean":    float(v.mean()),
            "std":     float(v.std(ddof=1)),
            "RP_axis": float(v.std(ddof=1) * 3),
            "min":     float(v.min()),
            "max":     float(v.max()),
            "range":   float(v.max() - v.min()),
        }
    return out


# ─────────────────────────────────────────────
#  Plotly Chart Helpers
# ─────────────────────────────────────────────

def _circle_xy(cx, cy, r, n=120):
    theta = np.linspace(0, 2 * np.pi, n)
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def make_bullseye_plotly(pid, pose_df, res):
    A       = pose_df.iloc[:, 0].values
    B       = pose_df.iloc[:, 1].values
    mA, mB  = res["mean_A"], res["mean_B"]
    RP, AP  = res["RP"], res["AP"]
    ax_names = list(pose_df.columns)
    n_sets  = len(A)

    all_r = np.sqrt(A ** 2 + B ** 2)
    r_max = max(
        all_r.max() * 1.45 if len(all_r) > 0 else 0.01,
        (np.sqrt(mA ** 2 + mB ** 2) + RP) * 1.35,
        THRESHOLDS["Sinirli"] * 1.6,
        1e-5,
    )

    fig = go.Figure()

    # Zone fills: draw outside-in so inner zones win
    # Kotu zone = plot_bgcolor (red background)
    for zone, r in [
        ("Sinirli", THRESHOLDS["Sinirli"]),
        ("Kabul",   THRESHOLDS["Kabul"]),
        ("Iyi",     THRESHOLDS["Iyi"]),
    ]:
        cx, cy = _circle_xy(0, 0, r)
        fig.add_trace(go.Scatter(
            x=np.append(cx, cx[0]),
            y=np.append(cy, cy[0]),
            fill="toself",
            fillcolor=GRADE_COLORS_LIGHT[zone],
            line=dict(color=GRADE_COLORS[zone], width=1.8, dash="dash"),
            name=f"{grade_label(zone)} ≤ {r:.2f} mm",
            hoverinfo="skip",
            mode="lines",
        ))

    # Trail line between measurement points
    fig.add_trace(go.Scatter(
        x=A, y=B,
        mode="lines",
        line=dict(color="rgba(136,136,136,0.25)", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Measurement points (color = set index)
    fig.add_trace(go.Scatter(
        x=A, y=B,
        mode="markers",
        marker=dict(
            size=11,
            color=list(range(n_sets)),
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Set #", thickness=12, len=0.6),
            line=dict(color="white", width=1),
        ),
        text=[
            f"Set {i + 1}<br>Δ{ax_names[0]} = {a:.5f} mm<br>Δ{ax_names[1]} = {b:.5f} mm"
            for i, (a, b) in enumerate(zip(A, B))
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Ölçüm Noktaları",
    ))

    # RP circle centered on mean
    rx, ry = _circle_xy(mA, mB, RP)
    fig.add_trace(go.Scatter(
        x=np.append(rx, rx[0]),
        y=np.append(ry, ry[0]),
        mode="lines",
        line=dict(color="#6a1de0", width=2.5),
        fill="toself",
        fillcolor="rgba(106,29,224,0.06)",
        name=f"RP = {RP:.5f} mm",
        hoverinfo="skip",
    ))

    # AP arrow: origin → mean
    if abs(mA) > 1e-9 or abs(mB) > 1e-9:
        fig.add_annotation(
            x=mA, y=mB, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor="#1565C0",
            showarrow=True,
            text="",
        )
        fig.add_annotation(
            x=mA * 0.5, y=mB * 0.5,
            text=f"AP = {AP:.4f} mm",
            showarrow=False,
            font=dict(color="#1565C0", size=11, family="monospace"),
            bgcolor="white",
            bordercolor="#1565C0",
            borderwidth=1,
            borderpad=3,
        )

    # Mean marker
    fig.add_trace(go.Scatter(
        x=[mA], y=[mB],
        mode="markers",
        marker=dict(symbol="cross", size=16, color="#1565C0",
                    line=dict(width=3, color="#1565C0")),
        name=f"Ort. ({mA:.4f}, {mB:.4f})",
        hovertemplate=(
            f"Ortalama<br>Δ{ax_names[0]} = {mA:.5f} mm"
            f"<br>Δ{ax_names[1]} = {mB:.5f} mm<extra></extra>"
        ),
    ))

    # Target origin
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="markers",
        marker=dict(symbol="x", size=18, color="#cc1e1e",
                    line=dict(width=3, color="#cc1e1e")),
        name="Hedef (0, 0)",
        hoverinfo="skip",
    ))

    # Crosshairs
    fig.add_hline(y=0, line_color="#888888", line_width=0.7, opacity=0.6)
    fig.add_vline(x=0, line_color="#888888", line_width=0.7, opacity=0.6)

    ap_g = rate_quality(AP)
    rp_g = rate_quality(RP)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{pid} — Bullseye Hedef Grafiği  ({n_sets} ölçüm)</b><br>"
                f"<sup>AP = {AP:.5f} mm → {grade_label(ap_g)}"
                f"  &nbsp;|&nbsp;  RP = {RP:.5f} mm → {grade_label(rp_g)}</sup>"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(
            title=f"Δ{ax_names[0]} (mm)",
            scaleanchor="y", scaleratio=1,
            range=[-r_max, r_max],
            gridcolor="#dddddd",
        ),
        yaxis=dict(
            title=f"Δ{ax_names[1]} (mm)",
            range=[-r_max, r_max],
            gridcolor="#dddddd",
        ),
        plot_bgcolor=GRADE_COLORS_LIGHT["Kotu"],
        paper_bgcolor="white",
        height=640,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.28,
            xanchor="center", x=0.5,
            font=dict(size=10),
        ),
    )
    return fig


def make_summary_plotly(pid, pose_df, res):
    sets     = list(range(1, len(pose_df) + 1))
    ax_names = list(pose_df.columns)
    l_i      = res["l_i"]
    colors   = ["#00b4d8", "#06d6a0", "#ef476f"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"{pid} — Eksen Sapması (Set bazlı)",
            f"{pid} — L_i Uzaklığı + RP Sınırı",
        ],
    )

    for i, col in enumerate(ax_names):
        fig.add_trace(go.Scatter(
            x=sets, y=pose_df[col],
            name=f"Δ{col}",
            line=dict(color=colors[i % len(colors)], width=2),
            mode="lines+markers",
            marker=dict(size=7),
        ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4, row=1, col=1)

    fig.add_trace(go.Bar(
        x=sets, y=l_i,
        name="l_i",
        marker_color="rgba(180,190,254,0.7)",
    ), row=1, col=2)

    fig.add_hline(
        y=res["l_bar"], line_color="#06d6a0", line_width=2,
        annotation_text=f"l̄ = {res['l_bar']:.5f} mm",
        annotation_font_color="#06d6a0",
        row=1, col=2,
    )
    fig.add_hline(
        y=res["RP"], line_dash="dash", line_color="#ef476f", line_width=2,
        annotation_text=f"RP = {res['RP']:.5f} mm",
        annotation_font_color="#ef476f",
        row=1, col=2,
    )

    fig.update_layout(
        height=420,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#313244",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#313244", bordercolor="#585b70", font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor="#45475a", title_text="Set")
    fig.update_yaxes(gridcolor="#45475a", title_text="mm")
    return fig


def make_boxplot_plotly(pid, pose_df):
    colors = ["#89dceb", "#a6e3a1", "#f38ba8"]
    fig    = go.Figure()

    for i, col in enumerate(pose_df.columns):
        fig.add_trace(go.Box(
            y=pose_df[col],
            name=f"Δ{col}",
            marker_color=colors[i % len(colors)],
            boxmean="sd",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.35)
    fig.update_layout(
        title=f"{pid} — Eksen Dağılımı (Box Plot)",
        yaxis_title="Delta (mm)",
        height=420,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#313244",
        font=dict(color="#cdd6f4"),
    )
    fig.update_yaxes(gridcolor="#45475a")
    return fig


def make_quality_plotly(results):
    pose_ids = list(results.keys())
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Doğruluk (AP)", "Tekrarlanabilirlik (RP)"],
    )

    for col_idx, metric in enumerate(["AP", "RP"], start=1):
        values     = [results[pid][metric] for pid in pose_ids]
        bar_colors = [GRADE_COLORS[rate_quality(v)] for v in values]
        grades     = [grade_label(rate_quality(v)) for v in values]

        fig.add_trace(go.Bar(
            x=pose_ids,
            y=values,
            marker_color=bar_colors,
            text=[f"{v:.5f} mm<br>{g}" for v, g in zip(values, grades)],
            textposition="auto",
            name=metric,
            showlegend=False,
            hovertemplate="%{x}<br>" + metric + " = %{y:.6f} mm<br>%{text}<extra></extra>",
        ), row=1, col=col_idx)

        for thr_name, thr_val in THRESHOLDS.items():
            fig.add_hline(
                y=thr_val,
                line_dash="dot",
                line_color=GRADE_COLORS[thr_name],
                opacity=0.8,
                annotation_text=thr_name,
                annotation_font_size=9,
                annotation_font_color=GRADE_COLORS[thr_name],
                row=1, col=col_idx,
            )

    fig.update_layout(
        title="ISO 9283 — Kalite Özet Raporu",
        height=440,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#313244",
        font=dict(color="#cdd6f4"),
    )
    fig.update_xaxes(gridcolor="#45475a")
    fig.update_yaxes(gridcolor="#45475a", title_text="mm")
    return fig


# ─────────────────────────────────────────────
#  Streamlit Page Setup
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ISO 9283 RP & AP Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Arka plan ── */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main { background-color: #1e1e2e !important; }
    [data-testid="stSidebar"] { background-color: #181825 !important; }
    [data-testid="stSidebarContent"] { background-color: #181825 !important; }

    /* ── Başlıklar & metin ── */
    h1, h2, h3, h4 { color: #cba6f7 !important; }
    p, li, span, div { color: #cdd6f4; }
    label, .stRadio label, .stRadio div[role="radiogroup"] label {
        color: #cdd6f4 !important;
    }

    /* ── Sekmeler ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #181825;
        border-radius: 8px 8px 0 0;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a6adc8 !important;
        font-weight: 600;
        background-color: #313244 !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 6px 16px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #cba6f7 !important;
        background-color: #45475a !important;
        border-bottom: 3px solid #cba6f7 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1e1e2e;
        border: 1px solid #313244;
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 12px;
    }

    /* ── Butonlar — tüm türler ── */
    button[kind="primary"],
    [data-testid="baseButton-primary"],
    .stButton > button[kind="primary"] {
        background-color: #cba6f7 !important;
        color: #1e1e2e !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    button[kind="secondary"],
    [data-testid="baseButton-secondary"],
    .stButton > button {
        background-color: #313244 !important;
        color: #cdd6f4 !important;
        border: 1px solid #585b70 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    button[kind="secondary"]:hover,
    .stButton > button:hover {
        background-color: #45475a !important;
        border-color: #cba6f7 !important;
        color: #cba6f7 !important;
    }
    /* İndirme butonu */
    [data-testid="stDownloadButton"] > button,
    .stDownloadButton > button {
        background-color: #a6e3a1 !important;
        color: #1e1e2e !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background-color: #7ccc76 !important;
    }

    /* ── Metrik kartlar ── */
    div[data-testid="stMetric"] {
        background-color: #313244 !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
        border-left: 3px solid #cba6f7 !important;
    }
    div[data-testid="stMetricValue"] > div { color: #cdd6f4 !important; }
    div[data-testid="stMetricDelta"] > div {
        color: #a6e3a1 !important;
        font-size: 0.82rem !important;
    }

    /* ── Giriş alanları ── */
    [data-testid="stFileUploader"] {
        background-color: #313244 !important;
        border-radius: 8px !important;
        border: 1px dashed #585b70 !important;
    }
    [data-testid="stFileUploader"] label { color: #cdd6f4 !important; }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background-color: #313244 !important;
        color: #cdd6f4 !important;
        border: 1px solid #585b70 !important;
        border-radius: 6px !important;
    }

    /* ── Data editor / tablo ── */
    [data-testid="stDataFrame"],
    .stDataFrame { background-color: #313244 !important; }
    [data-testid="data-grid-canvas"] { background-color: #313244 !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background-color: #313244 !important;
        border: 1px solid #45475a !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary { color: #cdd6f4 !important; }

    /* ── Divider ── */
    hr { border-color: #45475a !important; }

    /* ── Uyarı/bilgi kutuları ── */
    .stAlert { border-radius: 8px !important; }
    [data-testid="stNotification"] { border-radius: 8px !important; }

    /* ── Radio buton ── */
    [data-testid="stRadio"] label { color: #cdd6f4 !important; }
    [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
        border-color: #cba6f7 !important;
    }

    /* ── Sidebar başlık & yazı ── */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #cba6f7 !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #cdd6f4 !important; }
    [data-testid="stSidebar"] hr  { border-color: #45475a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────

st.title("🎯 ISO 9283 — RP & AP Analyzer")
st.caption("Repeatability & Accuracy Position Analysis  |  ISO 9283 Standard")

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Veri Girişi")
    input_mode = st.radio(
        "Yöntem:",
        ["📁 CSV Yükle", "✏️ Manuel Giriş"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("📐 Kalite Eşikleri (ISO 9283)")
    for key, val in THRESHOLDS.items():
        c = GRADE_COLORS[key]
        st.markdown(
            f'<span style="color:{c}; font-weight:bold;">■</span> '
            f'<span style="color:#cdd6f4">{grade_label(key)}: ≤ {val} mm</span>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<span style="color:{GRADE_COLORS["Kotu"]}; font-weight:bold;">■</span> '
        f'<span style="color:#cdd6f4">{grade_label("Kotu")}: > {THRESHOLDS["Sinirli"]} mm</span>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

# ─────────────────────────────────────────────
#  Data Input
# ─────────────────────────────────────────────

if input_mode == "📁 CSV Yükle":
    uploaded = st.file_uploader(
        "CSV Dosyası Seç",
        type=["csv"],
        help="Format: Set, P1_A, P1_B, P1_C  (birden fazla pose: P2_A, P2_B, P2_C, ...)",
        label_visibility="collapsed",
    )
    if uploaded:
        try:
            st.session_state.df_raw = pd.read_csv(uploaded)
            st.success(f"✅ Yüklendi: **{uploaded.name}** | {len(st.session_state.df_raw)} set")
        except Exception as e:
            st.error(f"❌ CSV okunamadı: {e}")

    with st.expander("📋 CSV Format & Örnek Dosya", expanded=(st.session_state.df_raw is None)):
        st.markdown("""
**Kolon formatı:** `Set, P1_A, P1_B, P1_C`  
Birden fazla pose için: `P2_A, P2_B, P2_C` ekleyin.  
**Set 1** referans noktasıdır — tüm değerler otomatik normalize edilir.
        """)
        st.code("Set,P1_A,P1_B,P1_C\n1,0.000,0.000,0.000\n2,0.002,0.001,0.003\n3,-0.001,0.002,0.001")
        ex_df = pd.DataFrame({
            "Set":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "P1_A": [0.000,  0.0021, -0.0013,  0.0031,  0.0012,
                    -0.0008,  0.0025,  0.0009, -0.0015,  0.0018],
            "P1_B": [0.000,  0.0012,  0.0023, -0.0011,  0.0031,
                     0.0008, -0.0019,  0.0024,  0.0007, -0.0013],
            "P1_C": [0.000,  0.0031,  0.0011,  0.0021, -0.0012,
                     0.0019,  0.0007, -0.0022,  0.0015,  0.0011],
        })
        st.download_button(
            "⬇️ Örnek CSV İndir",
            ex_df.to_csv(index=False).encode("utf-8"),
            "ornek_veri.csv",
            "text/csv",
        )

else:  # Manuel Giriş
    st.subheader("✏️ Manuel Veri Girişi")
    c1, c2 = st.columns(2)
    with c1:
        n_sets  = st.number_input("Kaç Set?",  min_value=2, max_value=100, value=5, step=1)
    with c2:
        n_poses = st.number_input("Kaç Pose?", min_value=1, max_value=10,  value=1, step=1)

    data_cols = ["Set"] + [
        f"P{p}_{ax}" for p in range(1, n_poses + 1) for ax in ["A", "B", "C"]
    ]

    # Rebuild default table only when shape changes
    shape_key = (int(n_sets), int(n_poses))
    if "manual_shape" not in st.session_state or st.session_state.manual_shape != shape_key:
        st.session_state.manual_shape = shape_key
        default_data = {"Set": list(range(1, n_sets + 1))}
        for c in data_cols[1:]:
            default_data[c] = [0.0] * n_sets
        st.session_state.manual_df = pd.DataFrame(default_data)

    edited = st.data_editor(
        st.session_state.manual_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            c: st.column_config.NumberColumn(c, format="%.6f", step=0.000001)
            for c in data_cols[1:]
        },
        key="manual_editor",
    )

    if st.button("▶️ Analiz Et", type="primary", use_container_width=True):
        st.session_state.df_raw    = edited.copy()
        st.session_state.manual_df = edited.copy()

# ─────────────────────────────────────────────
#  Analysis & Results
# ─────────────────────────────────────────────

df_raw = st.session_state.df_raw

if df_raw is not None:
    try:
        df_norm   = normalize_to_set1(df_raw)
        pose_data = parse_poses(df_norm)

        if not pose_data:
            st.error(
                "❌ Geçerli pose bulunamadı. "
                "Kolon formatını kontrol edin: Set, P1_A, P1_B, P1_C ..."
            )
            st.stop()

        results = {}
        for pid, pose_df in pose_data.items():
            res        = compute_rp(pose_df)
            res["AP"]  = compute_ap(pose_df)
            results[pid] = res

        # ── KPI Cards ──────────────────────────────
        st.divider()
        st.subheader("📊 ISO 9283 Sonuçlar")

        metric_cols = st.columns(len(results) * 2)
        for i, (pid, res) in enumerate(results.items()):
            ap_g = rate_quality(res["AP"])
            rp_g = rate_quality(res["RP"])
            with metric_cols[i * 2]:
                st.metric(
                    label=f"{pid} — Doğruluk (AP)",
                    value=f"{res['AP']:.5f} mm",
                    delta=grade_label(ap_g),
                )
            with metric_cols[i * 2 + 1]:
                st.metric(
                    label=f"{pid} — Tekrarlanabilirlik (RP)",
                    value=f"{res['RP']:.5f} mm",
                    delta=grade_label(rp_g),
                )

        # ── Chart Tabs ─────────────────────────────
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Bullseye",
            "📈 Eksen Sapması",
            "📦 Box Plot",
            "🏆 Kalite Özeti",
            "📋 Tablolar",
        ])

        with tab1:
            for pid, pose_df in pose_data.items():
                st.plotly_chart(
                    make_bullseye_plotly(pid, pose_df, results[pid]),
                    use_container_width=True,
                )

        with tab2:
            for pid, pose_df in pose_data.items():
                st.plotly_chart(
                    make_summary_plotly(pid, pose_df, results[pid]),
                    use_container_width=True,
                )

        with tab3:
            for pid, pose_df in pose_data.items():
                st.plotly_chart(
                    make_boxplot_plotly(pid, pose_df),
                    use_container_width=True,
                )

        with tab4:
            st.plotly_chart(
                make_quality_plotly(results),
                use_container_width=True,
            )

        with tab5:
            st.subheader("Ham Veri")
            st.dataframe(df_raw, use_container_width=True)

            st.subheader("Normalize Veri (Set 1 = 0)")
            st.dataframe(df_norm.round(6), use_container_width=True)

            st.subheader("AP / RP Sonuçları")
            rp_rows = []
            for pid, res in results.items():
                ap_g = rate_quality(res["AP"])
                rp_g = rate_quality(res["RP"])
                rp_rows.append({
                    "Pose":       pid,
                    "AP (mm)":    round(res["AP"],    6),
                    "AP Kalite":  grade_label(ap_g),
                    "RP (mm)":    round(res["RP"],    6),
                    "RP Kalite":  grade_label(rp_g),
                    "l-bar (mm)": round(res["l_bar"], 6),
                    "S_l (mm)":   round(res["S_l"],   6),
                    "Mean_A":     round(res["mean_A"], 6),
                    "Mean_B":     round(res["mean_B"], 6),
                    "Mean_C":     round(res["mean_C"], 6),
                })
            st.dataframe(pd.DataFrame(rp_rows), use_container_width=True)

            st.subheader("Eksen İstatistikleri")
            ax_rows = []
            for pid, pose_df in pose_data.items():
                for ax_n, s in axis_repeatability(pose_df).items():
                    ax_rows.append({
                        "Pose": pid,
                        "Eksen": ax_n,
                        **{k: round(v, 6) for k, v in s.items()},
                    })
            st.dataframe(pd.DataFrame(ax_rows), use_container_width=True)

        # ── Excel Export ────────────────────────────
        st.divider()
        st.subheader("📥 Rapor İndir")

        rp_exp, ax_exp = [], []
        for pid, res in results.items():
            rp_exp.append({
                "Pose":       pid,
                "AP (mm)":    res["AP"],
                "AP Kalite":  grade_label(rate_quality(res["AP"])),
                "RP (mm)":    res["RP"],
                "RP Kalite":  grade_label(rate_quality(res["RP"])),
                "l_bar":      res["l_bar"],
                "S_l":        res["S_l"],
                "Mean_A":     res["mean_A"],
                "Mean_B":     res["mean_B"],
                "Mean_C":     res["mean_C"],
            })
            for ax_n, s in axis_repeatability(pose_data[pid]).items():
                ax_exp.append({"Pose": pid, "Eksen": ax_n, **s})

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_raw.to_excel(writer, sheet_name="Ham Veri",         index=False)
            df_norm.round(6).to_excel(writer, sheet_name="Normalize",  index=False)
            pd.DataFrame(rp_exp).to_excel(writer, sheet_name="ISO9283 Sonuclar", index=False)
            pd.DataFrame(ax_exp).to_excel(writer, sheet_name="Eksen Istatistik", index=False)

        st.download_button(
            label="⬇️ Excel Raporu İndir (.xlsx)",
            data=buf.getvalue(),
            file_name="iso9283_analiz.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"❌ Analiz hatası: {e}")
        st.exception(e)

else:
    # Welcome screen
    st.info("👈 Sol panelden **CSV yükle** veya **manuel giriş** yaparak analizi başlat.")

    with st.expander("📖 Nasıl Kullanılır?", expanded=True):
        st.markdown("""
### 📁 CSV Yöntemi
1. Sol panelden **CSV Yükle** seçeneğini seç
2. Dosyayı yükle
3. Grafikler ve tablolar otomatik oluşur

### ✏️ Manuel Giriş Yöntemi
1. Sol panelden **Manuel Giriş** seçeneğini seç
2. Set ve pose sayısını belirle
3. Tabloya değerleri gir (telefondan da girebilirsin)
4. **Analiz Et** butonuna bas

### CSV Format
```
Set,P1_A,P1_B,P1_C
1,0.000,0.000,0.000
2,0.002,0.001,0.003
3,-0.001,0.002,0.001
```
        """)
