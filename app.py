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
    # Coerce every non-Set column to float64; columns that can't convert become NaN
    candidate_cols = [c for c in df.columns if c != "Set"]
    for c in candidate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Keep only columns that actually contain numeric data
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    reference = df[numeric_cols].iloc[0].to_numpy(dtype="float64")
    df[numeric_cols] = df[numeric_cols].to_numpy(dtype="float64") - reference
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


def make_comparison_bar_plotly(results_a, results_b, label_a, label_b):
    """Grouped bar chart comparing AP and RP for two datasets."""
    poses = sorted(set(list(results_a.keys()) + list(results_b.keys())))
    ap_a = [results_a[p]["AP"] if p in results_a else None for p in poses]
    ap_b = [results_b[p]["AP"] if p in results_b else None for p in poses]
    rp_a = [results_a[p]["RP"] if p in results_a else None for p in poses]
    rp_b = [results_b[p]["RP"] if p in results_b else None for p in poses]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Doğruluk (AP): {label_a} vs {label_b}",
            f"Tekrarlanabilirlik (RP): {label_a} vs {label_b}",
        ],
    )

    for col_idx, (vals_a, vals_b, metric) in enumerate(
        [(ap_a, ap_b, "AP"), (rp_a, rp_b, "RP")], start=1
    ):
        fig.add_trace(go.Bar(
            x=poses, y=vals_a,
            name=f"{metric} — {label_a}",
            marker_color="#89b4fa",
            text=[f"{v:.5f}" if v is not None else "" for v in vals_a],
            textposition="auto",
            showlegend=(col_idx == 1),
            legendgroup="a",
            hovertemplate="%{x}<br>" + metric + f" ({label_a}) = %{{y:.6f}} mm<extra></extra>",
        ), row=1, col=col_idx)

        fig.add_trace(go.Bar(
            x=poses, y=vals_b,
            name=f"{metric} — {label_b}",
            marker_color="#f38ba8",
            text=[f"{v:.5f}" if v is not None else "" for v in vals_b],
            textposition="auto",
            showlegend=(col_idx == 1),
            legendgroup="b",
            hovertemplate="%{x}<br>" + metric + f" ({label_b}) = %{{y:.6f}} mm<extra></extra>",
        ), row=1, col=col_idx)

        for thr_name, thr_val in THRESHOLDS.items():
            fig.add_hline(
                y=thr_val, line_dash="dot",
                line_color=GRADE_COLORS[thr_name], opacity=0.8,
                annotation_text=thr_name, annotation_font_size=9,
                annotation_font_color=GRADE_COLORS[thr_name],
                row=1, col=col_idx,
            )

    fig.update_layout(
        title=f"ISO 9283 Karşılaştırma: {label_a} vs {label_b}",
        barmode="group", height=450,
        paper_bgcolor="#1e1e2e", plot_bgcolor="#313244",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#313244"),
    )
    fig.update_xaxes(gridcolor="#45475a")
    fig.update_yaxes(gridcolor="#45475a", title_text="mm")
    return fig


def make_comparison_bullseye_plotly(pid, pose_df_a, res_a, pose_df_b, res_b, label_a, label_b):
    """Overlaid bullseye showing two datasets on the same pose chart."""
    A_a = pose_df_a.iloc[:, 0].values
    B_a = pose_df_a.iloc[:, 1].values
    A_b = pose_df_b.iloc[:, 0].values
    B_b = pose_df_b.iloc[:, 1].values
    ax_names = list(pose_df_a.columns)

    all_r = np.concatenate([
        np.sqrt(A_a ** 2 + B_a ** 2),
        np.sqrt(A_b ** 2 + B_b ** 2),
    ])
    r_max = max(
        all_r.max() * 1.45 if len(all_r) > 0 else 0.01,
        THRESHOLDS["Sinirli"] * 1.6,
        1e-5,
    )

    fig = go.Figure()

    for zone, r in [
        ("Sinirli", THRESHOLDS["Sinirli"]),
        ("Kabul",   THRESHOLDS["Kabul"]),
        ("Iyi",     THRESHOLDS["Iyi"]),
    ]:
        cx, cy = _circle_xy(0, 0, r)
        fig.add_trace(go.Scatter(
            x=np.append(cx, cx[0]), y=np.append(cy, cy[0]),
            fill="toself", fillcolor=GRADE_COLORS_LIGHT[zone],
            line=dict(color=GRADE_COLORS[zone], width=1.5, dash="dash"),
            name=f"{grade_label(zone)} ≤ {r:.2f} mm",
            hoverinfo="skip", mode="lines",
        ))

    fig.add_trace(go.Scatter(
        x=A_a, y=B_a, mode="markers+lines",
        marker=dict(size=9, color="#89b4fa", symbol="circle",
                    line=dict(color="white", width=1)),
        line=dict(color="rgba(137,180,250,0.3)", width=1),
        name=label_a,
        text=[
            f"{label_a} Set {i+1}<br>Δ{ax_names[0]}={a:.5f}<br>Δ{ax_names[1]}={b:.5f}"
            for i, (a, b) in enumerate(zip(A_a, B_a))
        ],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=A_b, y=B_b, mode="markers+lines",
        marker=dict(size=9, color="#f38ba8", symbol="diamond",
                    line=dict(color="white", width=1)),
        line=dict(color="rgba(243,139,168,0.3)", width=1),
        name=label_b,
        text=[
            f"{label_b} Set {i+1}<br>Δ{ax_names[0]}={a:.5f}<br>Δ{ax_names[1]}={b:.5f}"
            for i, (a, b) in enumerate(zip(A_b, B_b))
        ],
        hovertemplate="%{text}<extra></extra>",
    ))

    for mA, mB, RP, color, lbl in [
        (res_a["mean_A"], res_a["mean_B"], res_a["RP"], "#89b4fa", label_a),
        (res_b["mean_A"], res_b["mean_B"], res_b["RP"], "#f38ba8", label_b),
    ]:
        cx, cy = _circle_xy(mA, mB, RP)
        fig.add_trace(go.Scatter(
            x=np.append(cx, cx[0]), y=np.append(cy, cy[0]),
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            name=f"RP ({lbl}) = {RP:.5f} mm",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[mA], y=[mB], mode="markers",
            marker=dict(symbol="cross", size=12, color=color),
            name=f"Ort. ({lbl})",
            hovertemplate=(
                f"{lbl} ort.: Δ{ax_names[0]}={mA:.5f}, Δ{ax_names[1]}={mB:.5f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"{pid} — Bullseye Karşılaştırma: {label_a} vs {label_b}",
        xaxis=dict(title=f"Δ{ax_names[0]} (mm)", scaleanchor="y",
                   gridcolor="#45475a", zeroline=True, zerolinecolor="#585b70"),
        yaxis=dict(title=f"Δ{ax_names[1]} (mm)",
                   gridcolor="#45475a", zeroline=True, zerolinecolor="#585b70"),
        xaxis_range=[-r_max, r_max],
        yaxis_range=[-r_max, r_max],
        height=520,
        paper_bgcolor="#1e1e2e", plot_bgcolor="#313244",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#313244", font=dict(size=10)),
    )
    return fig


def generate_pdf_report(df_raw, df_norm, pose_data, results):
    """
    Matplotlib PdfPages ile çok sayfalı PDF raporu üretir.
    Sayfa 1: Kapak + özet tablo
    Sayfa 2: Bullseye (tüm pose'lar)
    Sayfa 3: Eksen sapması + L_i
    Sayfa 4: Box plot
    Sayfa 5: Kalite özeti
    Sayfa 6: Eksen istatistikleri tablosu
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    import datetime as _dt

    _RING_BORDER = {
        "Iyi": "#1a7a32", "Kabul": "#9e6d00",
        "Sinirli": "#9e3600", "Kotu": "#a01010",
    }

    def _style_ax(ax):
        ax.set_facecolor("#313244")
        ax.tick_params(colors="#cdd6f4", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#585b70")
        ax.grid(color="#45475a", linewidth=0.5, alpha=0.5)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # ── Sayfa 1: Kapak + özet tablo ───────────────────────────
        fig, ax = plt.subplots(figsize=(11.7, 8.3), facecolor="#1e1e2e")
        ax.set_facecolor("#1e1e2e")
        ax.axis("off")
        ax.text(0.5, 0.93, "ISO 9283 — RP & AP Analiz Raporu",
                ha="center", va="center", fontsize=22, fontweight="bold",
                color="#cba6f7", transform=ax.transAxes)
        ax.text(0.5, 0.86,
                f"Tarih: {_dt.datetime.now().strftime('%d.%m.%Y %H:%M')}   |   "
                f"{len(df_raw)} set   |   {len(results)} pose",
                ha="center", va="center", fontsize=11,
                color="#a6adc8", transform=ax.transAxes)
        col_labels = ["Pose", "AP (mm)", "AP Kalite", "RP (mm)", "RP Kalite",
                      "l-bar (mm)", "S_l (mm)"]
        rows = []
        for pid, res in results.items():
            rows.append([
                pid,
                f"{res['AP']:.6f}",
                grade_label(rate_quality(res['AP'])),
                f"{res['RP']:.6f}",
                grade_label(rate_quality(res['RP'])),
                f"{res['l_bar']:.6f}",
                f"{res['S_l']:.6f}",
            ])
        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       loc="center", cellLoc="center",
                       bbox=[0.05, 0.25, 0.90, 0.52])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#45475a")
                cell.set_text_props(color="#cba6f7", fontweight="bold")
            else:
                cell.set_facecolor("#313244")
                if c in (2, 4):
                    try:
                        met_val = float(rows[r - 1][c - 1])
                        g = rate_quality(met_val)
                        cell.set_facecolor(GRADE_COLORS_LIGHT[g])
                        cell.set_text_props(color=GRADE_COLORS[g], fontweight="bold")
                    except Exception:
                        cell.set_text_props(color="#cdd6f4")
                else:
                    cell.set_text_props(color="#cdd6f4")
            cell.set_edgecolor("#585b70")
        thr_line = (
            f"İyi ≤ {THRESHOLDS['Iyi']:.2f} mm   |   "
            f"Kabul Edilebilir ≤ {THRESHOLDS['Kabul']:.2f} mm   |   "
            f"Sınırlı ≤ {THRESHOLDS['Sinirli']:.2f} mm   |   "
            f"Uygun Değil > {THRESHOLDS['Sinirli']:.2f} mm"
        )
        ax.text(0.5, 0.17, thr_line, ha="center", va="center",
                fontsize=8.5, color="#585b70", transform=ax.transAxes)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Sayfa 2: Bullseye ─────────────────────────────────────
        n = len(pose_data)
        fig, axlist = plt.subplots(1, n, figsize=(7.5 * n, 7.5), facecolor="#f5f5f5")
        if n == 1:
            axlist = [axlist]
        fig.suptitle("ISO 9283  —  Bullseye Hedef Grafiği  (ΔA  vs  ΔB)",
                     color="#1e1e2e", fontsize=13, fontweight="bold")
        for bax, (pid, pose_df) in zip(axlist, pose_data.items()):
            res = results[pid]
            A, B = pose_df.iloc[:, 0].values, pose_df.iloc[:, 1].values
            mA, mB, RP, AP = res["mean_A"], res["mean_B"], res["RP"], res["AP"]
            ax_names = list(pose_df.columns)
            n_sets = len(A)
            all_r = np.sqrt(A**2 + B**2)
            r_max = max(all_r.max() * 1.45 if len(all_r) else 0.01,
                        (np.sqrt(mA**2 + mB**2) + RP) * 1.35,
                        THRESHOLDS["Sinirli"] * 1.6, 1e-5)
            bax.set_aspect("equal")
            bax.set_facecolor(GRADE_COLORS_LIGHT["Kotu"])
            for gc in ["Sinirli", "Kabul", "Iyi"]:
                bax.add_patch(plt.Circle((0, 0), THRESHOLDS[gc],
                                         color=GRADE_COLORS_LIGHT[gc], zorder=1, linewidth=0))
            for gc in ["Sinirli", "Kabul", "Iyi"]:
                r = THRESHOLDS[gc]
                bax.add_patch(plt.Circle((0, 0), r, color=_RING_BORDER[gc],
                                          fill=False, linewidth=1.8, linestyle="--",
                                          zorder=2, alpha=0.75))
                ang = np.radians(42)
                frac = 0.68 if gc != "Iyi" else 0.55
                lx, ly = r * np.cos(ang) * frac, r * np.sin(ang) * frac
                short = {"Iyi": "İyi", "Kabul": "Kabul", "Sinirli": "Sınırlı"}
                bax.text(lx, ly, f"{short[gc]}\n≤{r:.2f} mm",
                         fontsize=6.5, color=_RING_BORDER[gc],
                         ha="left", va="bottom", zorder=6, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.18", fc="white", alpha=0.80, ec="none"))
            bax.axhline(0, color="#bbbbbb", lw=0.7, zorder=2, alpha=0.8)
            bax.axvline(0, color="#bbbbbb", lw=0.7, zorder=2, alpha=0.8)
            sc = bax.scatter(A, B, c=np.arange(n_sets), cmap="plasma", s=55, zorder=7,
                             alpha=0.55, edgecolors="white", linewidths=0.7,
                             vmin=0, vmax=n_sets - 1)
            bax.plot(A, B, color="#888888", lw=0.35, alpha=0.22, zorder=6)
            plt.colorbar(sc, ax=bax, pad=0.02, fraction=0.034, shrink=0.85).set_label("Set #", color="#333333", fontsize=8)
            if abs(mA) > 1e-9 or abs(mB) > 1e-9:
                bax.annotate("", xy=(mA, mB), xytext=(0, 0),
                             arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2.2, mutation_scale=14), zorder=8)
                bax.text(mA * 0.5, mB * 0.5, f"AP={AP:.4f} mm",
                         color="#1565C0", fontsize=7.5, zorder=9, ha="center", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.88, ec="#1565C0", lw=0.8))
            bax.add_patch(plt.Circle((mA, mB), RP, color="#6a1de0",
                                      fill=False, linewidth=2.4, linestyle="-", zorder=8))
            bax.add_patch(plt.Circle((mA, mB), RP, color="#6a1de0", fill=True, alpha=0.06, zorder=7))
            bax.text(mA + RP * 0.72, mB + RP * 0.72, f"RP={RP:.4f} mm",
                     color="#6a1de0", fontsize=7.5, zorder=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.88, ec="#6a1de0", lw=0.8))
            bax.scatter(mA, mB, marker="+", s=280, color="#1565C0", linewidths=3.2, zorder=10)
            bax.scatter(0, 0, marker="x", s=170, color="#cc1e1e", linewidths=2.8, zorder=10)
            ap_g, rp_g = rate_quality(AP), rate_quality(RP)
            fcs = {"Iyi": "#edfaef", "Kabul": "#fffaed", "Sinirli": "#fff3eb", "Kotu": "#fff0f0"}
            badge = f"AP = {AP:.5f} mm  →  {grade_label(ap_g)}\nRP = {RP:.5f} mm  →  {grade_label(rp_g)}"
            bax.text(0.02, 0.98, badge, transform=bax.transAxes, fontsize=8.5, color="#1e1e2e",
                     va="top", ha="left", family="monospace", zorder=12,
                     bbox=dict(boxstyle="round,pad=0.5", fc=fcs.get(rp_g, "white"),
                               alpha=0.95, ec=GRADE_COLORS[rp_g], lw=2.2))
            bax.set_xlim(-r_max, r_max)
            bax.set_ylim(-r_max, r_max)
            bax.set_title(f"{pid}  —  Bullseye  ({n_sets} ölçüm)", color="#1e1e2e", fontsize=11, fontweight="bold")
            bax.set_xlabel(f"Δ{ax_names[0]} (mm)", color="#333333")
            bax.set_ylabel(f"Δ{ax_names[1]} (mm)", color="#333333")
            bax.tick_params(colors="#333333", labelsize=8)
            for sp in bax.spines.values():
                sp.set_edgecolor("#cccccc")
            bax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Sayfa 3: Eksen sapması + L_i ─────────────────────────
        pt_colors = ["#89dceb", "#a6e3a1", "#f38ba8"]
        fig2 = plt.figure(figsize=(14, 5 * n), facecolor="#1e1e2e")
        fig2.suptitle("ISO 9283  —  Eksen Sapması & Uzaklık Analizi",
                      fontsize=13, color="white", fontweight="bold", y=0.99)
        gs = gridspec.GridSpec(n, 2, figure=fig2, hspace=0.55, wspace=0.35)
        for row_idx, (pid, pose_df) in enumerate(pose_data.items()):
            res = results[pid]
            sets = np.arange(1, len(pose_df) + 1)
            ax_names = list(pose_df.columns)
            ax1 = fig2.add_subplot(gs[row_idx, 0])
            for i, col in enumerate(ax_names):
                ax1.plot(sets, pose_df[col], color=pt_colors[i], lw=1.3,
                         label=f"Delta-{col}", alpha=0.9)
            ax1.axhline(0, color="white", lw=0.6, ls="--", alpha=0.35)
            ax1.set_title(f"{pid}  —  Eksen Sapması", color="white", fontsize=10)
            ax1.set_xlabel("Set", color="#cdd6f4")
            ax1.set_ylabel("Delta (mm)", color="#cdd6f4")
            ax1.legend(fontsize=8, facecolor="#313244", labelcolor="white")
            _style_ax(ax1)
            ax2 = fig2.add_subplot(gs[row_idx, 1])
            l_i = res["l_i"]
            ax2.bar(sets, l_i, color="#b4befe", alpha=0.70, label="l_i")
            ax2.axhline(res["l_bar"], color="#a6e3a1", lw=1.8, ls="-",
                        label=f"l-bar = {res['l_bar']:.5f}")
            ax2.axhline(res["RP"], color="#f38ba8", lw=2.0, ls="--",
                        label=f"RP = {res['RP']:.5f}")
            ax2.set_title(f"{pid}  —  L_i + RP sınırı", color="white", fontsize=10)
            ax2.set_xlabel("Set", color="#cdd6f4")
            ax2.set_ylabel("l_i (mm)", color="#cdd6f4")
            ax2.legend(fontsize=8, facecolor="#313244", labelcolor="white")
            _style_ax(ax2)
        fig2.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig2, facecolor=fig2.get_facecolor())
        plt.close(fig2)

        # ── Sayfa 4: Box plot ────────────────────────────────────
        box_colors = ["#89dceb", "#a6e3a1", "#f38ba8"]
        fig3, axes3 = plt.subplots(1, n, figsize=(6 * n, 5), facecolor="#1e1e2e")
        if n == 1:
            axes3 = [axes3]
        fig3.suptitle("ISO 9283  —  Eksen Bazlı Box Plot",
                      fontsize=13, color="white", fontweight="bold")
        for bax3, (pid, pose_df) in zip(axes3, pose_data.items()):
            bp = bax3.boxplot(
                [pose_df.iloc[:, i] for i in range(len(pose_df.columns))],
                tick_labels=list(pose_df.columns),
                patch_artist=True,
                medianprops=dict(color="#f38ba8", linewidth=2),
                whiskerprops=dict(color="#cdd6f4"),
                capprops=dict(color="#cdd6f4"),
                flierprops=dict(markerfacecolor="#fab387", marker="o", ms=4),
            )
            for patch, col in zip(bp["boxes"], box_colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.72)
            bax3.axhline(0, color="white", ls="--", lw=0.8, alpha=0.4)
            bax3.set_title(f"{pid}  —  Eksen Dağılımı", color="white", fontsize=11)
            bax3.set_ylabel("Delta (mm)", color="#cdd6f4")
            _style_ax(bax3)
        fig3.tight_layout()
        pdf.savefig(fig3, facecolor=fig3.get_facecolor())
        plt.close(fig3)

        # ── Sayfa 5: Kalite özeti ─────────────────────────────────
        pose_ids = list(results.keys())
        n_p = len(pose_ids)
        fig4, axes4 = plt.subplots(1, 2,
                                   figsize=(max(9, n_p * 3.5), max(4, n_p * 1.4)),
                                   facecolor="#1e1e2e")
        fig4.suptitle("ISO 9283  —  Kalite Özet Raporu",
                      color="white", fontsize=13, fontweight="bold")
        bar_max = THRESHOLDS["Sinirli"] * 1.6
        for col_idx, (metric, title) in enumerate(
                [("AP", "Doğruluk  (AP)"), ("RP", "Tekrarlanabilirlik  (RP)")]):
            ax4 = axes4[col_idx]
            ax4.set_facecolor("#181825")
            ax4.set_xlim(0, 1)
            ax4.set_ylim(-0.6, n_p - 0.4)
            ax4.axis("off")
            ax4.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
            for i, pid in enumerate(pose_ids):
                val = results[pid][metric]
                grade = rate_quality(val)
                color = GRADE_COLORS[grade]
                y = n_p - 1 - i
                ax4.add_patch(mpatches.FancyBboxPatch(
                    (0.01, y - 0.40), 0.98, 0.75, boxstyle="round,pad=0.02",
                    fc=color, alpha=0.12, ec=color, lw=1.2, transform=ax4.transData))
                ax4.add_patch(mpatches.FancyBboxPatch(
                    (0.03, y - 0.17), 0.63, 0.26, boxstyle="round,pad=0.01",
                    fc="#313244", alpha=0.9, ec="#585b70", lw=0.5, transform=ax4.transData))
                frac = min(val / bar_max, 1.0)
                if frac > 0.005:
                    ax4.add_patch(mpatches.FancyBboxPatch(
                        (0.03, y - 0.17), 0.63 * frac, 0.26, boxstyle="round,pad=0.01",
                        fc=color, alpha=0.78, ec="none", transform=ax4.transData))
                ax4.add_patch(mpatches.FancyBboxPatch(
                    (0.70, y - 0.34), 0.27, 0.63, boxstyle="round,pad=0.02",
                    fc=color, alpha=0.88, ec="none", transform=ax4.transData))
                ax4.text(0.835, y + 0.00, grade_label(grade),
                         ha="center", va="center", fontsize=10.5, fontweight="bold",
                         color="#1e1e2e", transform=ax4.transData)
                ax4.text(0.055, y + 0.18, pid, ha="left", va="center",
                         fontsize=10, fontweight="bold", color="white", transform=ax4.transData)
                ax4.text(0.055, y - 0.01, f"{val:.6f} mm", ha="left", va="center",
                         fontsize=8.5, color="#cdd6f4", family="monospace",
                         transform=ax4.transData)
            thr_txt = (f"İyi ≤ {THRESHOLDS['Iyi']:.2f} mm   |   "
                       f"Kabul Edilebilir ≤ {THRESHOLDS['Kabul']:.2f} mm   |   "
                       f"Sınırlı ≤ {THRESHOLDS['Sinirli']:.2f} mm   |   "
                       f"Uygun Değil > {THRESHOLDS['Sinirli']:.2f} mm")
            ax4.text(0.5, -0.55, thr_txt, ha="center", va="center",
                     fontsize=6.8, color="#444444", transform=ax4.transData)
        fig4.tight_layout(rect=[0, 0.02, 1, 0.93])
        pdf.savefig(fig4, facecolor=fig4.get_facecolor())
        plt.close(fig4)

        # ── Sayfa 6: Eksen istatistikleri tablosu ────────────────
        ax_rows = []
        for pid, pose_df in pose_data.items():
            for ax_n, s in axis_repeatability(pose_df).items():
                ax_rows.append({
                    "Pose": pid, "Eksen": ax_n,
                    "Mean(mm)": round(s["mean"], 6),
                    "Std(mm)": round(s["std"], 6),
                    "3σ-RP": round(s["RP_axis"], 6),
                    "Min": round(s["min"], 6),
                    "Max": round(s["max"], 6),
                    "Range": round(s["range"], 6),
                })
        ax_df = pd.DataFrame(ax_rows)
        fig5_h = max(4.0, (len(ax_df) + 1) * 0.4)
        fig5, ax5 = plt.subplots(figsize=(11.7, fig5_h), facecolor="#1e1e2e")
        ax5.set_facecolor("#1e1e2e")
        ax5.axis("off")
        ax5.text(0.5, 1.01, "Eksen İstatistikleri",
                 ha="center", va="bottom", fontsize=13, color="#cba6f7",
                 fontweight="bold", transform=ax5.transAxes)
        if len(ax_df) > 0:
            tbl5 = ax5.table(cellText=ax_df.values.tolist(),
                             colLabels=list(ax_df.columns),
                             loc="center", cellLoc="center")
            tbl5.auto_set_font_size(False)
            tbl5.set_fontsize(8.5)
            for (r, c), cell in tbl5.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#45475a")
                    cell.set_text_props(color="#cba6f7", fontweight="bold")
                else:
                    cell.set_facecolor("#313244")
                    cell.set_text_props(color="#cdd6f4")
                cell.set_edgecolor("#585b70")
        pdf.savefig(fig5, facecolor=fig5.get_facecolor())
        plt.close(fig5)

    buf.seek(0)
    return buf.getvalue()


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
        ["📁 CSV Yükle", "✏️ Manuel Giriş", "🔀 Karşılaştırma"],
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
if "backup_csv" not in st.session_state:
    st.session_state.backup_csv = None  # son manuel verinin CSV yedeği
if "df_raw_b" not in st.session_state:
    st.session_state.df_raw_b = None
if "label_a" not in st.session_state:
    st.session_state.label_a = "Veri Seti A"
if "label_b" not in st.session_state:
    st.session_state.label_b = "Veri Seti B"

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

elif input_mode == "✏️ Manuel Giriş":
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

    # ── Yedekten geri yükle ──
    with st.expander("↩ Yedekten Geri Yükle", expanded=False):
        restore_file = st.file_uploader(
            "Daha önce indirdiğin CSV yedeğini seç",
            type=["csv"],
            key="restore_uploader",
            label_visibility="collapsed",
        )
        if restore_file:
            try:
                restored = pd.read_csv(restore_file)
                st.session_state.manual_df    = restored
                st.session_state.manual_shape = (len(restored), (len(restored.columns) - 1) // 3)
                st.success(f"✅ Yedek yüklendi: {len(restored)} set, {len(restored.columns)-1} kolon")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Yedek yüklenemedi: {e}")

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

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("▶️ Analiz Et", type="primary", use_container_width=True):
            st.session_state.df_raw    = edited.copy()
            st.session_state.manual_df = edited.copy()
    with btn_col2:
        import datetime as _dt
        _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="💾 Veriyi Yedekle (CSV İndir)",
            data=edited.to_csv(index=False).encode("utf-8"),
            file_name=f"manuel_veri_yedek_{_ts}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Net kesilirse veya sayfa kapanırsa bu CSV'yi tekrar yükleyebilirsin",
        )


else:  # Karşılaştırma modu
    st.subheader("🔀 İki Veri Seti Karşılaştırma")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Veri Seti A")
        st.session_state.label_a = st.text_input(
            "İsim (A):", value=st.session_state.label_a, key="label_a_input"
        )
        up_a = st.file_uploader(
            "CSV A", type=["csv"], key="comp_upload_a",
            label_visibility="collapsed",
        )
        if up_a:
            try:
                st.session_state.df_raw = pd.read_csv(up_a)
                st.success(f"✅ A yüklendi: **{up_a.name}** | {len(st.session_state.df_raw)} set")
            except Exception as e:
                st.error(f"❌ {e}")
        elif st.session_state.df_raw is not None:
            st.info("✅ A veri seti hazır")
    with col_b:
        st.markdown("### Veri Seti B")
        st.session_state.label_b = st.text_input(
            "İsim (B):", value=st.session_state.label_b, key="label_b_input"
        )
        up_b = st.file_uploader(
            "CSV B", type=["csv"], key="comp_upload_b",
            label_visibility="collapsed",
        )
        if up_b:
            try:
                st.session_state.df_raw_b = pd.read_csv(up_b)
                st.success(f"✅ B yüklendi: **{up_b.name}** | {len(st.session_state.df_raw_b)} set")
            except Exception as e:
                st.error(f"❌ {e}")
        elif st.session_state.df_raw_b is not None:
            st.info("✅ B veri seti hazır")

# ─────────────────────────────────────────────
#  Analysis & Results
# ─────────────────────────────────────────────

df_raw   = st.session_state.df_raw
df_raw_b = st.session_state.get("df_raw_b")

if input_mode == "🔀 Karşılaştırma":
    _la  = st.session_state.get("label_a", "Veri Seti A")
    _lb  = st.session_state.get("label_b", "Veri Seti B")
    _raw_b = st.session_state.get("df_raw_b")
    if df_raw is not None and _raw_b is not None:
        try:
            df_norm_a   = normalize_to_set1(df_raw)
            df_norm_b   = normalize_to_set1(_raw_b)
            pose_data_a = parse_poses(df_norm_a)
            pose_data_b = parse_poses(df_norm_b)
            if not pose_data_a or not pose_data_b:
                st.error("❌ Geçerli pose bulunamadı. Kolon formatını kontrol edin.")
                st.stop()

            results_a, results_b = {}, {}
            for pid, pose_df in pose_data_a.items():
                r = compute_rp(pose_df); r["AP"] = compute_ap(pose_df); results_a[pid] = r
            for pid, pose_df in pose_data_b.items():
                r = compute_rp(pose_df); r["AP"] = compute_ap(pose_df); results_b[pid] = r

            all_poses = sorted(set(list(results_a.keys()) + list(results_b.keys())))
            st.divider()
            st.subheader(f"📊 Karşılaştırma: {_la} vs {_lb}")
            for pname in all_poses:
                with st.expander(f"📍 {pname}", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    r_a = results_a.get(pname)
                    r_b = results_b.get(pname)
                    with c1:
                        v = r_a["AP"] if r_a else None
                        st.metric(f"AP — {_la}", f"{v:.5f} mm" if v is not None else "—",
                                  grade_label(rate_quality(v)) if v is not None else "")
                    with c2:
                        v = r_b["AP"] if r_b else None
                        st.metric(f"AP — {_lb}", f"{v:.5f} mm" if v is not None else "—",
                                  grade_label(rate_quality(v)) if v is not None else "")
                    with c3:
                        v = r_a["RP"] if r_a else None
                        st.metric(f"RP — {_la}", f"{v:.5f} mm" if v is not None else "—",
                                  grade_label(rate_quality(v)) if v is not None else "")
                    with c4:
                        v = r_b["RP"] if r_b else None
                        st.metric(f"RP — {_lb}", f"{v:.5f} mm" if v is not None else "—",
                                  grade_label(rate_quality(v)) if v is not None else "")

            cmp_tab1, cmp_tab2, cmp_tab3 = st.tabs([
                "📊 AP & RP Karşılaştırma",
                "🎯 Bullseye Karşılaştırma",
                "📋 Delta Tablosu",
            ])
            with cmp_tab1:
                st.plotly_chart(
                    make_comparison_bar_plotly(results_a, results_b, _la, _lb),
                    use_container_width=True,
                )
            with cmp_tab2:
                common_poses = sorted(set(pose_data_a.keys()) & set(pose_data_b.keys()))
                if not common_poses:
                    st.warning("İki veri setinde eşleşen pose bulunamadı.")
                else:
                    for pid in common_poses:
                        st.plotly_chart(
                            make_comparison_bullseye_plotly(
                                pid,
                                pose_data_a[pid], results_a[pid],
                                pose_data_b[pid], results_b[pid],
                                _la, _lb,
                            ),
                            use_container_width=True,
                        )
                only_a = sorted(set(pose_data_a.keys()) - set(pose_data_b.keys()))
                only_b = sorted(set(pose_data_b.keys()) - set(pose_data_a.keys()))
                if only_a:
                    st.info(f"Sadece {_la}’da bulunan pose’lar: {', '.join(only_a)}")
                if only_b:
                    st.info(f"Sadece {_lb}’de bulunan pose’lar: {', '.join(only_b)}")
            with cmp_tab3:
                delta_rows = []
                for pname in all_poses:
                    r_a = results_a.get(pname)
                    r_b = results_b.get(pname)
                    ap_a = r_a["AP"] if r_a else None
                    ap_b = r_b["AP"] if r_b else None
                    rp_a = r_a["RP"] if r_a else None
                    rp_b = r_b["RP"] if r_b else None
                    delta_rows.append({
                        "Pose":             pname,
                        f"AP {_la} (mm)":   round(ap_a, 6) if ap_a is not None else None,
                        f"AP {_lb} (mm)":   round(ap_b, 6) if ap_b is not None else None,
                        "ΔAP (B−A) mm":  round(ap_b - ap_a, 6) if (ap_a is not None and ap_b is not None) else None,
                        f"RP {_la} (mm)":   round(rp_a, 6) if rp_a is not None else None,
                        f"RP {_lb} (mm)":   round(rp_b, 6) if rp_b is not None else None,
                        "ΔRP (B−A) mm":  round(rp_b - rp_a, 6) if (rp_a is not None and rp_b is not None) else None,
                    })
                st.dataframe(pd.DataFrame(delta_rows), use_container_width=True)
                st.caption("Δ < 0 → B daha iyi  |  Δ > 0 → A daha iyi")
        except Exception as e:
            st.error(f"❌ Karşılaştırma hatası: {e}")
            st.exception(e)
    elif df_raw is None and _raw_b is None:
        st.info("👈 Sol panelden **iki CSV dosyası** yükleyin.")
    else:
        missing = "B" if df_raw is not None else "A"
        st.warning(f"⚠️ Veri Seti **{missing}** henüz yüklenmedi — yukarıdan CSV yükleyin.")

elif df_raw is not None:
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

        import datetime as _dt
        _ts2 = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
        with dl_col1:
            st.download_button(
                label="⬇️ Excel Raporu İndir (.xlsx)",
                data=buf.getvalue(),
                file_name=f"iso9283_analiz_{_ts2}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="💾 Ham Veri Yedeği (.csv)",
                data=df_raw.to_csv(index=False).encode("utf-8"),
                file_name=f"ham_veri_yedek_{_ts2}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Bu CSV'yi daha sonra tekrar yükleyerek analizi sıfırdan yapabilirsin",
            )
        with dl_col3:
            _html_bytes = generate_html_report(
                df_raw, df_norm, pose_data, results,
                label=f"{len(pose_data)} pose, {len(df_raw)} set",
            ).encode("utf-8")
            st.download_button(
                label="📄 HTML Rapor İndir",
                data=_html_bytes,
                file_name=f"iso9283_rapor_{_ts2}.html",
                mime="text/html",
                use_container_width=True,
                help="Tüm grafikler ve tablolar dahil tam rapor. Tarayıcıdan PDF olarak da yazdırabilirsin.",
            )
        with dl_col4:
            _pdf_bytes = generate_pdf_report(df_raw, df_norm, pose_data, results)
            st.download_button(
                label="📑 PDF Rapor İndir",
                data=_pdf_bytes,
                file_name=f"iso9283_rapor_{_ts2}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Bullseye, eksen sapması, box plot, kalite özeti dahil 6 sayfalık PDF raporu",
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

### 🔀 Karşılaştırma Yöntemi
1. Sol panelden **Karşılaştırma** seçeneğini seç
2. İki ayrı CSV dosyası yükle (A ve B)
3. Her veri seti için bir isim ver (örn: "Kalibrasyon Öncesi", "Kalibrasyon Sonrası")
4. AP, RP karşılaştırma grafikleri ve delta tablosu otomatik çıkar

### CSV Format
```
Set,P1_A,P1_B,P1_C
1,0.000,0.000,0.000
2,0.002,0.001,0.003
3,-0.001,0.002,0.001
```
        """)
