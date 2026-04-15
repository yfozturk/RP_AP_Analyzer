"""
ISO 9283 Repeatability & Accuracy Analyzer
============================================
CSV Format:  Set, P1_A, P1_B, P1_C, P2_A, ...
Set 1 = sifir referans noktasi. Her deger Set-1'den cikarilir.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import os
import json
import datetime
import threading

# ─────────────────────────────────────────────
#  Backup ayarları
# ─────────────────────────────────────────────
BACKUP_DIR  = os.path.join(os.path.expanduser("~"), "iso9283_backups")
BACKUP_FILE = os.path.join(BACKUP_DIR, "manual_backup.json")
AUTO_SAVE_INTERVAL_SEC = 30   # otomatik kayıt her 30 saniyede bir

os.makedirs(BACKUP_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Quality thresholds (mm)  — ISO 9283
# ─────────────────────────────────────────────
THRESHOLDS = {
    "Iyi":     0.01,   # Good       ≤0.01 mm
    "Kabul":   0.02,   # Acceptable  0.01–0.02 mm
    "Sinirli": 0.05,   # Limited     0.02–0.05 mm
}                      # Bad  >0.05 mm  → "Kotu"

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
    "Iyi":     "İyi  (Yüksek Performans)",
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
    l_i  = np.sqrt(((pose_df - mean_pos) ** 2).sum(axis=1))
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
#  Shared style helper
# ─────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor("#313244")
    ax.tick_params(colors="#cdd6f4", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#585b70")
    ax.grid(color="#45475a", linewidth=0.5, alpha=0.5)


# ═════════════════════════════════════════════
#  PLOT 1 — BULLSEYE / DARTBOARD
# ═════════════════════════════════════════════
_RING_BORDER = {
    "Iyi":     "#1a7a32",
    "Kabul":   "#9e6d00",
    "Sinirli": "#9e3600",
    "Kotu":    "#a01010",
}

def make_bullseye_figure(pose_data, results):
    n = len(pose_data)
    fig, axlist = plt.subplots(1, n, figsize=(7.5 * n, 7.5), facecolor="#f5f5f5")
    if n == 1:
        axlist = [axlist]

    fig.suptitle("ISO 9283  —  Bullseye Hedef Grafiği  (ΔA  vs  ΔB)",
                 color="#1e1e2e", fontsize=13, fontweight="bold")

    for ax, (pid, pose_df) in zip(axlist, pose_data.items()):
        res      = results[pid]
        A        = pose_df.iloc[:, 0].values
        B        = pose_df.iloc[:, 1].values
        mA, mB   = res["mean_A"], res["mean_B"]
        RP, AP   = res["RP"], res["AP"]
        ax_names = list(pose_df.columns)
        n_sets   = len(A)

        # Dynamic radius: fits all points + RP ring comfortably
        all_r = np.sqrt(A**2 + B**2)
        r_max = max(
            all_r.max() * 1.45,
            (np.sqrt(mA**2 + mB**2) + RP) * 1.35,
            THRESHOLDS["Sinirli"] * 1.6,
            1e-5
        )

        ax.set_aspect("equal")

        # ── Zone fills — outside-in so inner wins ──
        ax.set_facecolor(GRADE_COLORS_LIGHT["Kotu"])  # outermost background
        for gc in ["Sinirli", "Kabul", "Iyi"]:
            ax.add_patch(plt.Circle(
                (0, 0), THRESHOLDS[gc],
                color=GRADE_COLORS_LIGHT[gc],
                zorder=1, linewidth=0
            ))

        # ── Zone border rings + inline labels ──
        for gc in ["Sinirli", "Kabul", "Iyi"]:
            r = THRESHOLDS[gc]
            ax.add_patch(plt.Circle(
                (0, 0), r, color=_RING_BORDER[gc],
                fill=False, linewidth=1.8,
                linestyle="--", zorder=2, alpha=0.75
            ))
            # label placed at ~42° inside the ring
            ang      = np.radians(42)
            frac     = 0.68 if gc != "Iyi" else 0.55
            lx, ly   = r * np.cos(ang) * frac, r * np.sin(ang) * frac
            short    = {"Iyi": "İyi", "Kabul": "Kabul", "Sinirli": "Sınırlı"}
            ax.text(lx, ly,
                    f"{short[gc]}\n≤{r:.2f} mm",
                    fontsize=6.5, color=_RING_BORDER[gc],
                    ha="left", va="bottom", zorder=6, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18",
                              fc="white", alpha=0.80, ec="none"))

        # ── Crosshairs ──
        ax.axhline(0, color="#bbbbbb", lw=0.7, zorder=2, alpha=0.8)
        ax.axvline(0, color="#bbbbbb", lw=0.7, zorder=2, alpha=0.8)

        # ── All measurement points (opacity → density effect) ──
        sc = ax.scatter(
            A, B,
            c=np.arange(n_sets), cmap="plasma",
            s=55, zorder=7,
            alpha=0.55,          # overlapping points appear denser
            edgecolors="white",
            linewidths=0.7,
            vmin=0, vmax=n_sets - 1
        )
        # Faint trail line connecting measurements in order
        ax.plot(A, B, color="#888888", lw=0.35, alpha=0.22,
                zorder=6, linestyle="-")

        cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.034, shrink=0.85)
        cbar.set_label("Set #", color="#333333", fontsize=8)
        cbar.ax.tick_params(colors="#333333", labelsize=7)

        # ── AP arrow: origin → mean ──
        if abs(mA) > 1e-9 or abs(mB) > 1e-9:
            ax.annotate("",
                        xy=(mA, mB), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->",
                                        color="#1565C0",
                                        lw=2.2, mutation_scale=14),
                        zorder=8)
            ax.text(mA * 0.5, mB * 0.5,
                    f"AP={AP:.4f} mm",
                    color="#1565C0", fontsize=7.5, zorder=9,
                    ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", alpha=0.88,
                              ec="#1565C0", lw=0.8))

        # ── RP circle  (centered on mean) ──
        ax.add_patch(plt.Circle(
            (mA, mB), RP,
            color="#6a1de0", fill=False,
            linewidth=2.4, linestyle="-", zorder=8
        ))
        # RP hatch fill for visibility
        ax.add_patch(plt.Circle(
            (mA, mB), RP,
            color="#6a1de0", fill=True,
            alpha=0.06, zorder=7
        ))
        ax.text(mA + RP * 0.72, mB + RP * 0.72,
                f"RP={RP:.4f} mm",
                color="#6a1de0", fontsize=7.5, zorder=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", alpha=0.88,
                          ec="#6a1de0", lw=0.8))

        # ── Mean marker ──
        ax.scatter(mA, mB, marker="+", s=280,
                   color="#1565C0", linewidths=3.2, zorder=10)
        ax.text(mA, mB - r_max * 0.07,
                f"Ort. ({mA:.4f}, {mB:.4f})",
                color="#1565C0", fontsize=6.5,
                ha="center", va="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", alpha=0.88,
                          ec="#1565C0", lw=0.6))

        # ── Target (origin) ──
        ax.scatter(0, 0, marker="x", s=170, color="#cc1e1e",
                   linewidths=2.8, zorder=10)
        ax.text(0, -r_max * 0.07, "Hedef (0,0)",
                color="#cc1e1e", fontsize=6.5,
                ha="center", va="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", alpha=0.88,
                          ec="#cc1e1e", lw=0.6))

        # ── Quality badge (top-left) ──
        ap_g  = rate_quality(AP)
        rp_g  = rate_quality(RP)
        fcs   = {"Iyi": "#edfaef", "Kabul": "#fffaed",
                 "Sinirli": "#fff3eb", "Kotu": "#fff0f0"}
        badge = (f"AP = {AP:.5f} mm  →  {grade_label(ap_g)}\n"
                 f"RP = {RP:.5f} mm  →  {grade_label(rp_g)}")
        ax.text(0.02, 0.98, badge,
                transform=ax.transAxes,
                fontsize=8.5, color="#1e1e2e",
                va="top", ha="left", family="monospace",
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.5",
                          fc=fcs.get(rp_g, "white"),
                          alpha=0.95,
                          ec=GRADE_COLORS[rp_g], lw=2.2))

        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_title(f"{pid}  —  Bullseye  ({n_sets} ölçüm noktası)",
                     color="#1e1e2e", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Δ{ax_names[0]} (mm)", color="#333333")
        ax.set_ylabel(f"Δ{ax_names[1]} (mm)", color="#333333")
        ax.tick_params(colors="#333333", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
        ax.grid(color="#dddddd", linewidth=0.5, alpha=0.7)

        # ── Legend ──
        legend_patches = [
            mpatches.Patch(fc=GRADE_COLORS_LIGHT["Iyi"],
                           ec=_RING_BORDER["Iyi"], lw=1.5,
                           label=f"İyi  (Yüksek Perf.)      ≤ {THRESHOLDS['Iyi']:.2f} mm"),
            mpatches.Patch(fc=GRADE_COLORS_LIGHT["Kabul"],
                           ec=_RING_BORDER["Kabul"], lw=1.5,
                           label=f"Kabul Edilebilir          ≤ {THRESHOLDS['Kabul']:.2f} mm"),
            mpatches.Patch(fc=GRADE_COLORS_LIGHT["Sinirli"],
                           ec=_RING_BORDER["Sinirli"], lw=1.5,
                           label=f"Sınırlı                   ≤ {THRESHOLDS['Sinirli']:.2f} mm"),
            mpatches.Patch(fc=GRADE_COLORS_LIGHT["Kotu"],
                           ec=_RING_BORDER["Kotu"], lw=1.5,
                           label=f"Uygun Değil               > {THRESHOLDS['Sinirli']:.2f} mm"),
            mpatches.Patch(fc="#e8e0ff", ec="#6a1de0", lw=1.5,
                           label=f"RP Çemberi  = {RP:.5f} mm"),
        ]
        ax.legend(handles=legend_patches, fontsize=7.5,
                  facecolor="white", edgecolor="#cccccc",
                  labelcolor="#1e1e2e",
                  loc="lower right", framealpha=0.95)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ═════════════════════════════════════════════
#  PLOT 2 — Axis deviation + l_i bar
# ═════════════════════════════════════════════
def make_summary_figure(pose_data, results):
    n   = len(pose_data)
    fig = plt.figure(figsize=(14, 5 * n), facecolor="#1e1e2e")
    fig.suptitle("ISO 9283  —  Eksen Sapmasi & Uzaklik Analizi",
                 fontsize=13, color="white", fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.55, wspace=0.35)
    pt_colors = ["#89dceb", "#a6e3a1", "#f38ba8"]

    for row, (pid, pose_df) in enumerate(pose_data.items()):
        res      = results[pid]
        sets     = np.arange(1, len(pose_df) + 1)
        ax_names = list(pose_df.columns)

        ax1 = fig.add_subplot(gs[row, 0])
        for i, col in enumerate(ax_names):
            ax1.plot(sets, pose_df[col], color=pt_colors[i],
                     lw=1.3, label=f"Delta-{col}", alpha=0.9)
        ax1.axhline(0, color="white", lw=0.6, ls="--", alpha=0.35)
        ax1.set_title(f"{pid}  —  Eksen Sapmasi (Set bazli)",
                      color="white", fontsize=10)
        ax1.set_xlabel("Set", color="#cdd6f4")
        ax1.set_ylabel("Delta (mm)", color="#cdd6f4")
        ax1.legend(fontsize=8, facecolor="#313244", labelcolor="white")
        _style_ax(ax1)

        ax2 = fig.add_subplot(gs[row, 1])
        l_i = res["l_i"]
        ax2.bar(sets, l_i, color="#b4befe", alpha=0.70, label="l_i")
        ax2.axhline(res["l_bar"], color="#a6e3a1", lw=1.8,
                    ls="-",  label=f"l-bar = {res['l_bar']:.5f}")
        ax2.axhline(res["RP"], color="#f38ba8", lw=2.0,
                    ls="--", label=f"RP    = {res['RP']:.5f}")
        ax2.set_title(f"{pid}  —  L_i Uzakligi + RP siniri",
                      color="white", fontsize=10)
        ax2.set_xlabel("Set", color="#cdd6f4")
        ax2.set_ylabel("l_i (mm)", color="#cdd6f4")
        ax2.legend(fontsize=8, facecolor="#313244", labelcolor="white")
        _style_ax(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ═════════════════════════════════════════════
#  PLOT 3 — Box plots
# ═════════════════════════════════════════════
def make_boxplot_figure(pose_data):
    n    = len(pose_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor="#1e1e2e")
    if n == 1:
        axes = [axes]
    fig.suptitle("ISO 9283  —  Eksen Bazli Box Plot",
                 fontsize=13, color="white", fontweight="bold")

    box_colors = ["#89dceb", "#a6e3a1", "#f38ba8"]
    for ax, (pid, pose_df) in zip(axes, pose_data.items()):
        bp = ax.boxplot(
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
        ax.axhline(0, color="white", ls="--", lw=0.8, alpha=0.4)
        ax.set_title(f"{pid}  —  Eksen Dagilimi",
                     color="white", fontsize=11)
        ax.set_ylabel("Delta (mm)", color="#cdd6f4")
        _style_ax(ax)

    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════
#  PLOT 4 — Quality dashboard
# ═════════════════════════════════════════════
def make_quality_figure(results):
    pose_ids = list(results.keys())
    n = len(pose_ids)

    fig, axes = plt.subplots(1, 2, figsize=(max(9, n * 3.5), max(4, n * 1.4)),
                              facecolor="#1e1e2e")
    fig.suptitle("ISO 9283  —  Kalite Ozet Raporu",
                 color="white", fontsize=13, fontweight="bold")

    bar_max = THRESHOLDS["Sinirli"] * 1.6

    for col_idx, (metric, title) in enumerate(
            [("AP", "Dogruluk  (AP)"), ("RP", "Tekrarlanabilirlik  (RP)")]):
        ax = axes[col_idx]
        ax.set_facecolor("#181825")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, n - 0.4)
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)

        for i, pid in enumerate(pose_ids):
            val   = results[pid][metric]
            grade = rate_quality(val)
            color = GRADE_COLORS[grade]
            y     = n - 1 - i

            # Row background
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.01, y - 0.40), 0.98, 0.75,
                boxstyle="round,pad=0.02",
                fc=color, alpha=0.12, ec=color, lw=1.2,
                transform=ax.transData))

            # Progress bar background
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.03, y - 0.17), 0.63, 0.26,
                boxstyle="round,pad=0.01",
                fc="#313244", alpha=0.9, ec="#585b70", lw=0.5,
                transform=ax.transData))

            # Progress bar fill
            frac = min(val / bar_max, 1.0)
            if frac > 0.005:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (0.03, y - 0.17), 0.63 * frac, 0.26,
                    boxstyle="round,pad=0.01",
                    fc=color, alpha=0.78, ec="none",
                    transform=ax.transData))

            # Grade badge (right side)
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.70, y - 0.34), 0.27, 0.63,
                boxstyle="round,pad=0.02",
                fc=color, alpha=0.88, ec="none",
                transform=ax.transData))
            ax.text(0.835, y + 0.00, grade_label(grade),
                    ha="center", va="center",
                    fontsize=10.5, fontweight="bold",
                    color="#1e1e2e", transform=ax.transData)

            # Pose name
            ax.text(0.055, y + 0.18, pid,
                    ha="left", va="center",
                    fontsize=10, fontweight="bold",
                    color="white", transform=ax.transData)

            # Value
            ax.text(0.055, y - 0.01, f"{val:.6f} mm",
                    ha="left", va="center",
                    fontsize=8.5, color="#cdd6f4",
                    family="monospace", transform=ax.transData)

        # Threshold legend at bottom
        thr_txt = (f"İyi ≤ {THRESHOLDS['Iyi']:.2f} mm   |   "
                   f"Kabul Edilebilir ≤ {THRESHOLDS['Kabul']:.2f} mm   |   "
                   f"Sınırlı ≤ {THRESHOLDS['Sinirli']:.2f} mm   |   "
                   f"Uygun Değil > {THRESHOLDS['Sinirli']:.2f} mm")
        ax.text(0.5, -0.55, thr_txt,
                ha="center", va="center", fontsize=6.8,
                color="#444444", transform=ax.transData)

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return fig


# ═════════════════════════════════════════════
#  GUI
# ═════════════════════════════════════════════
class ISO9283App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ISO 9283  —  RP & AP Analyzer")
        self.geometry("1300x860")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)

        self.df_raw    = None
        self.df_norm   = None
        self.pose_data = None
        self.results   = None
        self._auto_save_job = None

        self._build_ui()
        self._start_auto_save()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ─────────────────────────────────────
    def _build_ui(self):
        # TOP BAR
        top = tk.Frame(self, bg="#181825", pady=8)
        top.pack(fill="x")

        tk.Label(top, text="ISO 9283  RP & AP Analyzer",
                 font=("Segoe UI", 14, "bold"),
                 bg="#181825", fg="#cba6f7").pack(side="left", padx=16)

        bb = {"fg": "#1e1e2e", "font": ("Segoe UI", 10, "bold"),
              "relief": "flat", "padx": 12, "pady": 4, "cursor": "hand2"}

        tk.Button(top, text="Load CSV",     command=self.load_csv,
                  bg="#89b4fa", **bb).pack(side="left", padx=5)
        tk.Button(top, text="Manuel Giris", command=lambda: self.nb.select(3),
                  bg="#94e2d5", **bb).pack(side="left", padx=5)
        tk.Button(top, text="Run Analysis", command=self.run_analysis,
                  bg="#a6e3a1", **bb).pack(side="left", padx=5)
        tk.Button(top, text="Bullseye",     command=self.show_bullseye,
                  bg="#cba6f7", **bb).pack(side="left", padx=5)
        tk.Button(top, text="Show Plots",   command=self.show_plots,
                  bg="#f38ba8", **bb).pack(side="left", padx=5)
        tk.Button(top, text="Export Excel", command=self.export_excel,
                  bg="#fab387", **bb).pack(side="left", padx=5)
        tk.Button(top, text="📄 PDF Rapor", command=self._export_pdf_report,
                  bg="#cba6f7", **bb).pack(side="left", padx=5)

        self.status_var = tk.StringVar(value="Dosya yuklenmedi.")
        tk.Label(top, textvariable=self.status_var,
                 bg="#181825", fg="#a6adc8",
                 font=("Segoe UI", 9)).pack(side="right", padx=16)

        # NOTEBOOK
        nb_style = ttk.Style()
        nb_style.theme_use("clam")
        nb_style.configure("D.TNotebook",
                            background="#1e1e2e", borderwidth=0)
        nb_style.configure("D.TNotebook.Tab",
                            background="#313244", foreground="#cdd6f4",
                            padding=[10, 4], font=("Segoe UI", 9, "bold"))
        nb_style.map("D.TNotebook.Tab",
                     background=[("selected", "#45475a")],
                     foreground=[("selected", "#cba6f7")])

        self.nb = ttk.Notebook(self, style="D.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # Tab 1 — Raw / Normalized data
        tab_data = tk.Frame(self.nb, bg="#1e1e2e")
        self.nb.add(tab_data, text="  Veri  ")
        pw = ttk.PanedWindow(tab_data, orient="horizontal")
        pw.pack(fill="both", expand=True)

        lf = tk.Frame(pw, bg="#1e1e2e")
        pw.add(lf, weight=1)
        tk.Label(lf, text="Ham Veri", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=4, pady=(4, 1))
        self.tree_raw = self._make_tv(lf)

        rf = tk.Frame(pw, bg="#1e1e2e")
        pw.add(rf, weight=1)
        tk.Label(rf, text="Normalize  (Set 1 = 0)", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=4, pady=(4, 1))
        self.tree_norm = self._make_tv(rf)

        # Tab 2 — ISO Results
        tab_res = tk.Frame(self.nb, bg="#1e1e2e")
        self.nb.add(tab_res, text="  ISO 9283 Sonuclar  ")
        pv = ttk.PanedWindow(tab_res, orient="vertical")
        pv.pack(fill="both", expand=True)

        tf = tk.Frame(pv, bg="#1e1e2e")
        pv.add(tf, weight=1)
        tk.Label(tf, text="AP / RP  +  Kalite Notu",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=4, pady=(4, 1))
        self.tree_results = self._make_tv(tf, 10)

        bf = tk.Frame(pv, bg="#1e1e2e")
        pv.add(bf, weight=1)
        tk.Label(bf, text="Eksen Istatistikleri",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=4, pady=(4, 1))
        self.tree_axis = self._make_tv(bf, 10)

        # Tab 3 — Quality chart (embedded)
        tab_qual = tk.Frame(self.nb, bg="#1e1e2e")
        self.nb.add(tab_qual, text="  Kalite Ozeti  ")
        self.qual_frame = tab_qual

        # Tab 4 — Manuel Giriş
        self._man_entries = []
        self._man_cols    = []
        tab_manual = tk.Frame(self.nb, bg="#1e1e2e")
        self.nb.add(tab_manual, text="  Manuel Giris  ")
        self._build_manual_tab(tab_manual)

    def _make_tv(self, parent, height=8):
        frm = tk.Frame(parent, bg="#1e1e2e")
        frm.pack(fill="both", expand=True, padx=4, pady=2)

        s = ttk.Style()
        s.configure("D.Treeview",
                     background="#313244", foreground="#cdd6f4",
                     fieldbackground="#313244", rowheight=22,
                     font=("Consolas", 9))
        s.configure("D.Treeview.Heading",
                     background="#45475a", foreground="#cdd6f4",
                     font=("Segoe UI", 9, "bold"))
        s.map("D.Treeview", background=[("selected", "#585b70")])

        tv  = ttk.Treeview(frm, style="D.Treeview", height=height, show="headings")
        vsb = ttk.Scrollbar(frm, orient="vertical",   command=tv.yview)
        hsb = ttk.Scrollbar(frm, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)
        return tv

    # ── Load CSV ───────────────────────────────
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="CSV Dosyasi Sec",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            self.df_raw    = pd.read_csv(path)
            self.df_norm   = normalize_to_set1(self.df_raw)
            self.pose_data = parse_poses(self.df_norm)
            self._fill(self.tree_raw,  self.df_raw)
            self._fill(self.tree_norm, self.df_norm.round(5))
            self.results = None
            self.status_var.set(
                f"Yuklendi: {os.path.basename(path)}  |  "
                f"{len(self.df_raw)} set  |  {len(self.pose_data)} pose"
            )
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya yuklenemedi:\n{e}")

    # ── Run Analysis ───────────────────────────
    def run_analysis(self):
        if self.pose_data is None:
            messagebox.showwarning("Veri Yok", "Once CSV yukleyin veya Manuel Giris yapin.")
            return

        self.results = {}
        rp_rows, ax_rows = [], []

        for pid, pose_df in self.pose_data.items():
            res = compute_rp(pose_df)
            ap  = compute_ap(pose_df)
            res["AP"] = ap
            self.results[pid] = res

            ap_g = rate_quality(ap)
            rp_g = rate_quality(res["RP"])
            rp_rows.append({
                "Pose":      pid,
                "AP (mm)":   round(ap,           6),
                "AP Kalite": grade_label(ap_g),
                "RP (mm)":   round(res["RP"],     6),
                "RP Kalite": grade_label(rp_g),
                "l-bar(mm)": round(res["l_bar"],  6),
                "S_l (mm)":  round(res["S_l"],    6),
                "Mean_A":    round(res["mean_A"],  6),
                "Mean_B":    round(res["mean_B"],  6),
                "Mean_C":    round(res["mean_C"],  6),
            })

            for ax_n, st in axis_repeatability(pose_df).items():
                ax_rows.append({
                    "Pose":      pid,
                    "Eksen":     ax_n,
                    "Mean(mm)":  round(st["mean"],    6),
                    "Std (mm)":  round(st["std"],     6),
                    "3sig-RP":   round(st["RP_axis"], 6),
                    "Min":       round(st["min"],     6),
                    "Max":       round(st["max"],     6),
                    "Range":     round(st["range"],   6),
                })

        self._fill(self.tree_results, pd.DataFrame(rp_rows))
        self._fill(self.tree_axis,    pd.DataFrame(ax_rows))
        self._embed_quality()
        self.status_var.set(
            f"Analiz tamamlandi  —  {len(self.results)} pose  (ISO 9283)"
        )
        self.nb.select(1)

    def _embed_quality(self):
        for w in self.qual_frame.winfo_children():
            w.destroy()
        fig = make_quality_figure(self.results)
        c   = FigureCanvasTkAgg(fig, master=self.qual_frame)
        c.draw()
        c.get_tk_widget().pack(fill="both", expand=True)

    # ── Bullseye ───────────────────────────────
    def show_bullseye(self):
        if self.results is None:
            messagebox.showwarning("Sonuc Yok", "Once Run Analysis yapin.")
            return
        win = tk.Toplevel(self)
        win.title("ISO 9283  —  Bullseye Dart Plot")
        win.configure(bg="#1e1e2e")
        fig = make_bullseye_figure(self.pose_data, self.results)
        c   = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        NavigationToolbar2Tk(c, win).pack(side="bottom", fill="x")
        c.get_tk_widget().pack(fill="both", expand=True)

    # ── Show Plots ─────────────────────────────
    def show_plots(self):
        if self.results is None:
            messagebox.showwarning("Sonuc Yok", "Once Run Analysis yapin.")
            return
        for fig, title in [
            (make_summary_figure(self.pose_data, self.results),
             "ISO 9283  —  Eksen Sapmasi & L_i"),
            (make_boxplot_figure(self.pose_data),
             "ISO 9283  —  Box Plot"),
        ]:
            win = tk.Toplevel(self)
            win.title(title)
            win.configure(bg="#1e1e2e")
            c = FigureCanvasTkAgg(fig, master=win)
            c.draw()
            NavigationToolbar2Tk(c, win).pack(side="bottom", fill="x")
            c.get_tk_widget().pack(fill="both", expand=True)

    # ── Export Excel ───────────────────────────
    def export_excel(self):
        if self.results is None:
            messagebox.showwarning("Sonuc Yok", "Once Run Analysis yapin.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            title="Sonuclari Kaydet"
        )
        if not path:
            return

        rp_rows, ax_rows = [], []
        for pid, res in self.results.items():
            rp_rows.append({
                "Pose":      pid,
                "AP (mm)":   res["AP"],
                "AP Kalite": grade_label(rate_quality(res["AP"])),
                "RP (mm)":   res["RP"],
                "RP Kalite": grade_label(rate_quality(res["RP"])),
                "l_bar":     res["l_bar"],
                "S_l":       res["S_l"],
                "Mean_A":    res["mean_A"],
                "Mean_B":    res["mean_B"],
                "Mean_C":    res["mean_C"],
            })
            for ax_n, st in axis_repeatability(self.pose_data[pid]).items():
                ax_rows.append({"Pose": pid, "Eksen": ax_n, **st})

        with pd.ExcelWriter(path, engine="openpyxl") as w:
            self.df_raw.to_excel(           w, sheet_name="Ham Veri",         index=False)
            self.df_norm.round(6).to_excel( w, sheet_name="Normalize",        index=False)
            pd.DataFrame(rp_rows).to_excel( w, sheet_name="ISO9283 Sonuclar", index=False)
            pd.DataFrame(ax_rows).to_excel( w, sheet_name="Eksen Istatistik", index=False)

        messagebox.showinfo("Kaydedildi", f"Dosya kaydedildi:\n{path}")
        self.status_var.set(f"Disari aktarildi -> {os.path.basename(path)}")

    # ── Manuel Giriş ─────────────────────────────
    def _build_manual_tab(self, parent):
        ctrl = tk.Frame(parent, bg="#1e1e2e", pady=8)
        ctrl.pack(fill="x", padx=10)

        tk.Label(ctrl, text="Set Sayisi:", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 4))
        self.man_sets_var = tk.IntVar(value=5)
        tk.Spinbox(ctrl, from_=2, to=100, textvariable=self.man_sets_var, width=6,
                   bg="#313244", fg="#cdd6f4", font=("Consolas", 10),
                   buttonbackground="#45475a", relief="flat").pack(side="left", padx=(0, 16))

        tk.Label(ctrl, text="Pose Sayisi:", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 4))
        self.man_poses_var = tk.IntVar(value=1)
        tk.Spinbox(ctrl, from_=1, to=10, textvariable=self.man_poses_var, width=6,
                   bg="#313244", fg="#cdd6f4", font=("Consolas", 10),
                   buttonbackground="#45475a", relief="flat").pack(side="left", padx=(0, 16))

        bb = {"fg": "#1e1e2e", "font": ("Segoe UI", 10, "bold"),
              "relief": "flat", "padx": 12, "pady": 4, "cursor": "hand2"}
        tk.Button(ctrl, text="Tabloyu Olustur", command=self._rebuild_manual_table,
                  bg="#89b4fa", **bb).pack(side="left", padx=5)
        tk.Button(ctrl, text="\u25b6  Analiz Et", command=self._run_manual_analysis,
                  bg="#a6e3a1", **bb).pack(side="left", padx=5)
        tk.Button(ctrl, text="\U0001f4be  Yedekle", command=self._save_manual_backup,
                  bg="#fab387", **bb).pack(side="left", padx=5)
        tk.Button(ctrl, text="\u21a9  Yedek Y\u00fckle", command=self._restore_manual_backup,
                  bg="#cba6f7", **bb).pack(side="left", padx=5)
        tk.Button(ctrl, text="Temizle", command=self._clear_manual_table,
                  bg="#f38ba8", **bb).pack(side="left", padx=5)
        self._backup_status = tk.Label(ctrl, text="",
                 bg="#1e1e2e", fg="#a6e3a1",
                 font=("Segoe UI", 8, "italic"))
        self._backup_status.pack(side="left", padx=10)
        tk.Label(ctrl, text="  Tab/Enter \u2192 sonraki h\u00fccre",
                 bg="#1e1e2e", fg="#6c7086",
                 font=("Segoe UI", 9, "italic")).pack(side="left", padx=10)

        cf = tk.Frame(parent, bg="#1e1e2e")
        cf.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self._man_canvas = tk.Canvas(cf, bg="#1e1e2e", highlightthickness=0)
        vsb = ttk.Scrollbar(cf, orient="vertical",   command=self._man_canvas.yview)
        hsb = ttk.Scrollbar(cf, orient="horizontal", command=self._man_canvas.xview)
        self._man_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._man_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        self._man_inner = tk.Frame(self._man_canvas, bg="#1e1e2e")
        self._man_canvas.create_window((0, 0), window=self._man_inner, anchor="nw")
        self._man_inner.bind(
            "<Configure>",
            lambda e: self._man_canvas.configure(
                scrollregion=self._man_canvas.bbox("all")
            ),
        )
        self._man_canvas.bind(
            "<MouseWheel>",
            lambda e: self._man_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )
        self._rebuild_manual_table()

    def _rebuild_manual_table(self):
        for w in self._man_inner.winfo_children():
            w.destroy()
        self._man_entries = []

        n_sets  = int(self.man_sets_var.get())
        n_poses = int(self.man_poses_var.get())
        self._man_cols = ["Set"] + [
            f"P{p}_{ax}"
            for p in range(1, n_poses + 1)
            for ax in ["A", "B", "C"]
        ]

        HDR_BG  = "#45475a"
        HDR_FG  = "#cdd6f4"
        CELL_BG = "#313244"
        CELL_FG = "#cdd6f4"
        SET_BG  = "#181825"
        EW      = 11

        for j, col in enumerate(self._man_cols):
            tk.Label(
                self._man_inner, text=col,
                bg=HDR_BG, fg=HDR_FG,
                font=("Segoe UI", 9, "bold"),
                width=EW, anchor="center",
                relief="flat", padx=2, pady=5,
            ).grid(row=0, column=j, padx=1, pady=1, sticky="nsew")

        for i in range(n_sets):
            row_ents = []
            for j, col in enumerate(self._man_cols):
                if col == "Set":
                    e = tk.Entry(
                        self._man_inner, width=EW,
                        bg=SET_BG, fg="#89b4fa",
                        font=("Consolas", 9), justify="center",
                        relief="flat", bd=1,
                    )
                    e.insert(0, str(i + 1))
                    e.configure(state="readonly")
                else:
                    e = tk.Entry(
                        self._man_inner, width=EW,
                        bg=CELL_BG, fg=CELL_FG,
                        font=("Consolas", 9), justify="center",
                        relief="flat", bd=1,
                        insertbackground="#cdd6f4",
                        selectbackground="#585b70",
                    )
                    e.insert(0, "0.000000")
                    e.bind("<FocusIn>",
                           lambda ev: ev.widget.select_range(0, "end"))
                    e.bind("<Tab>",
                           lambda ev, r=i, c=j: (self._man_nav(r, c, +1), "break")[1])
                    e.bind("<Return>",
                           lambda ev, r=i, c=j: (self._man_nav(r, c, +1), "break")[1])
                    e.bind("<Shift-Tab>",
                           lambda ev, r=i, c=j: (self._man_nav(r, c, -1), "break")[1])
                    e.bind("<MouseWheel>",
                           lambda ev: self._man_canvas.yview_scroll(
                               int(-1 * (ev.delta / 120)), "units"))
                e.grid(row=i + 1, column=j, padx=1, pady=1, sticky="nsew")
                row_ents.append(e)
            self._man_entries.append(row_ents)

        for j in range(len(self._man_cols)):
            self._man_inner.columnconfigure(j, weight=1)
        self._man_canvas.update_idletasks()
        self._man_canvas.configure(scrollregion=self._man_canvas.bbox("all"))

    def _man_nav(self, row, col, direction):
        """Move focus to next/prev editable cell (skips Set column at index 0)."""
        n_rows  = len(self._man_entries)
        n_edit  = len(self._man_cols) - 1
        flat    = row * n_edit + (col - 1) + direction
        flat    = max(0, min(flat, n_rows * n_edit - 1))
        new_row = flat // n_edit
        new_col = flat %  n_edit + 1
        try:
            self._man_entries[new_row][new_col].focus_set()
            self._man_entries[new_row][new_col].select_range(0, "end")
        except IndexError:
            pass

    def _clear_manual_table(self):
        for row in self._man_entries:
            for j, e in enumerate(row):
                if self._man_cols[j] != "Set":
                    e.delete(0, "end")
                    e.insert(0, "0.000000")

    # ── Backup / Restore ───────────────────────
    def _collect_manual_data(self):
        """Tablodaki mevcut değerleri dict olarak döner."""
        if not self._man_entries or not self._man_cols:
            return None
        data = {}
        for j, col in enumerate(self._man_cols):
            vals = []
            for i, row_ents in enumerate(self._man_entries):
                txt = row_ents[j].get().strip().replace(",", ".")
                vals.append(i + 1 if col == "Set" else txt)
            data[col] = vals
        return data

    def _save_manual_backup(self, silent=False):
        """Manuel tabloyu JSON dosyasına yedekler."""
        data = self._collect_manual_data()
        if data is None:
            return
        payload = {
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_sets":   int(self.man_sets_var.get()),
            "n_poses":  int(self.man_poses_var.get()),
            "cols":     self._man_cols,
            "data":     data,
        }
        try:
            with open(BACKUP_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            if hasattr(self, "_backup_status"):
                self._backup_status.config(text=f"\u2714 Yedeklendi {ts}")
            if not silent:
                self.status_var.set(f"Yedeklendi  →  {BACKUP_FILE}")
        except Exception as e:
            if not silent:
                messagebox.showerror("Yedek Hatasi", f"Yedek kaydedilemedi:\n{e}")

    def _restore_manual_backup(self):
        """Yedek JSON dosyasından tabloyu geri yükler."""
        if not os.path.exists(BACKUP_FILE):
            messagebox.showinfo("Yedek Yok", "Henüz kaydedilmiş yedek bulunamadı.")
            return
        try:
            with open(BACKUP_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            saved_at = payload.get("saved_at", "?")
            if not messagebox.askyesno(
                "Yedeği Yükle",
                f"Kaydedilme zamanı: {saved_at}\n\nBu yedek yüklensin mi?"
            ):
                return
            self.man_sets_var.set(payload["n_sets"])
            self.man_poses_var.set(payload["n_poses"])
            self._rebuild_manual_table()
            cols = payload["cols"]
            data = payload["data"]
            for i, row_ents in enumerate(self._man_entries):
                for j, col in enumerate(self._man_cols):
                    if col == "Set" or col not in data:
                        continue
                    if i < len(data[col]):
                        e = row_ents[j]
                        e.delete(0, "end")
                        e.insert(0, str(data[col][i]))
            self.status_var.set(f"Yedek yüklendi  ←  {saved_at}")
            if hasattr(self, "_backup_status"):
                self._backup_status.config(text=f"\u21a9 Yüklendi: {saved_at}")
        except Exception as e:
            messagebox.showerror("Yükleme Hatasi", f"Yedek yüklenemedi:\n{e}")

    def _start_auto_save(self):
        """Her 30 saniyede manuel tabloyu otomatik yedekler."""
        self._save_manual_backup(silent=True)
        self._auto_save_job = self.after(
            AUTO_SAVE_INTERVAL_SEC * 1000, self._start_auto_save
        )

    def _on_close(self):
        """Pencere kapatılırken son yedeği alır."""
        self._save_manual_backup(silent=True)
        # Analiz sonuçları varsa otomatik Excel yedek
        if self.results is not None and self.df_raw is not None:
            ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(BACKUP_DIR, f"analiz_yedek_{ts}.xlsx")
            try:
                rp_rows, ax_rows = [], []
                for pid, res in self.results.items():
                    rp_rows.append({
                        "Pose": pid,
                        "AP (mm)": res["AP"],
                        "AP Kalite": grade_label(rate_quality(res["AP"])),
                        "RP (mm)": res["RP"],
                        "RP Kalite": grade_label(rate_quality(res["RP"])),
                        "l_bar": res["l_bar"], "S_l": res["S_l"],
                        "Mean_A": res["mean_A"], "Mean_B": res["mean_B"],
                        "Mean_C": res["mean_C"],
                    })
                    for ax_n, st in axis_repeatability(self.pose_data[pid]).items():
                        ax_rows.append({"Pose": pid, "Eksen": ax_n, **st})
                with pd.ExcelWriter(path, engine="openpyxl") as w:
                    self.df_raw.to_excel(w, sheet_name="Ham Veri", index=False)
                    self.df_norm.round(6).to_excel(w, sheet_name="Normalize", index=False)
                    pd.DataFrame(rp_rows).to_excel(w, sheet_name="ISO9283 Sonuclar", index=False)
                    pd.DataFrame(ax_rows).to_excel(w, sheet_name="Eksen Istatistik", index=False)
            except Exception:
                pass
        if self._auto_save_job:
            self.after_cancel(self._auto_save_job)
        self.destroy()

    def _run_manual_analysis(self):
        if not self._man_entries:
            messagebox.showwarning("Tablo Bos", "Once 'Tabloyu Olustur' butonuna basin.")
            return
        data = {}
        for j, col in enumerate(self._man_cols):
            vals = []
            for i, row_ents in enumerate(self._man_entries):
                txt = row_ents[j].get().strip().replace(",", ".")
                if col == "Set":
                    vals.append(i + 1)
                else:
                    try:
                        vals.append(float(txt))
                    except ValueError:
                        messagebox.showerror(
                            "Girdi Hatasi",
                            f"Satir {i+1}, Kolon '{col}':  gecersiz deger  \u2192  '{txt}'"
                        )
                        return
            data[col] = vals
        try:
            self.df_raw    = pd.DataFrame(data)
            self.df_norm   = normalize_to_set1(self.df_raw)
            self.pose_data = parse_poses(self.df_norm)
            self._fill(self.tree_raw,  self.df_raw)
            self._fill(self.tree_norm, self.df_norm.round(5))
            self.results   = None
            self.status_var.set(
                f"Manuel Giris  |  {len(self.df_raw)} set  |  "
                f"{len(self.pose_data)} pose"
            )
            self.run_analysis()
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz yapilamadi:\n{e}")

    # ── Treeview fill ──────────────────────────
    # ── PDF Rapor ────────────────────────────────
    def _export_pdf_report(self):
        if self.results is None:
            messagebox.showwarning("Sonuç Yok", "Once Run Analysis yapın.")
            return
        from matplotlib.backends.backend_pdf import PdfPages
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=f"iso9283_rapor_{ts}.pdf",
            title="PDF Raporu Kaydet",
        )
        if not path:
            return
        try:
            with PdfPages(path) as pdf:
                # ─ Sayfa 1: Başlık + Özet Tablosu ─────────────────
                fig, ax = plt.subplots(figsize=(11.7, 8.3), facecolor="#1e1e2e")
                ax.set_facecolor("#1e1e2e")
                ax.axis("off")

                ax.text(0.5, 0.93,
                        "ISO 9283 — RP & AP Analiz Raporu",
                        ha="center", va="center", fontsize=22,
                        fontweight="bold", color="#cba6f7",
                        transform=ax.transAxes)
                ax.text(0.5, 0.86,
                        f"Tarih: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}   |   "
                        f"{len(self.df_raw)} set   |   {len(self.results)} pose",
                        ha="center", va="center", fontsize=11,
                        color="#a6adc8", transform=ax.transAxes)

                # summary table
                col_labels = ["Pose", "AP (mm)", "AP Kalite", "RP (mm)", "RP Kalite",
                              "l-bar (mm)", "S_l (mm)"]
                rows = []
                for pid, res in self.results.items():
                    rows.append([
                        pid,
                        f"{res['AP']:.6f}",
                        grade_label(rate_quality(res['AP'])),
                        f"{res['RP']:.6f}",
                        grade_label(rate_quality(res['RP'])),
                        f"{res['l_bar']:.6f}",
                        f"{res['S_l']:.6f}",
                    ])

                tbl = ax.table(
                    cellText=rows, colLabels=col_labels,
                    loc="center", cellLoc="center",
                    bbox=[0.05, 0.25, 0.90, 0.52],
                )
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(10)
                for (r, c), cell in tbl.get_celld().items():
                    if r == 0:
                        cell.set_facecolor("#45475a")
                        cell.set_text_props(color="#cba6f7", fontweight="bold")
                    else:
                        cell.set_facecolor("#313244")
                        # colour-code quality cells
                        if c in (2, 4):
                            val_str = rows[r - 1][c - 1]
                            met     = rows[r - 1][c - (1 if c == 2 else 3)]
                            try:
                                g = rate_quality(float(met))
                                cell.set_facecolor(GRADE_COLORS_LIGHT[g])
                                cell.set_text_props(color=GRADE_COLORS[g],
                                                    fontweight="bold")
                            except Exception:
                                cell.set_text_props(color="#cdd6f4")
                        else:
                            cell.set_text_props(color="#cdd6f4")
                    cell.set_edgecolor("#585b70")

                # ISO threshold legend at bottom
                thr_line = (
                    f"İyi ≤ {THRESHOLDS['Iyi']:.2f} mm   |   "
                    f"Kabul Edilebilir ≤ {THRESHOLDS['Kabul']:.2f} mm   |   "
                    f"Sınırlı ≤ {THRESHOLDS['Sinirli']:.2f} mm   |   "
                    f"Uygun Değil > {THRESHOLDS['Sinirli']:.2f} mm"
                )
                ax.text(0.5, 0.17, thr_line,
                        ha="center", va="center", fontsize=8.5,
                        color="#585b70", transform=ax.transAxes)

                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

                # ─ Sayfa 2: Bullseye ─────────────────────────
                fig = make_bullseye_figure(self.pose_data, self.results)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

                # ─ Sayfa 3: Eksen Sapması + L_i ─────────────────
                fig = make_summary_figure(self.pose_data, self.results)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

                # ─ Sayfa 4: Box Plot ─────────────────────────
                fig = make_boxplot_figure(self.pose_data)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

                # ─ Sayfa 5: Kalite Özeti ─────────────────────
                fig = make_quality_figure(self.results)
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

                # ─ Sayfa 6: Veri Tabloları ────────────────────
                ax_rows = []
                for pid, pose_df in self.pose_data.items():
                    for ax_n, st in axis_repeatability(pose_df).items():
                        ax_rows.append({
                            "Pose": pid, "Eksen": ax_n,
                            "Mean(mm)": round(st["mean"], 6),
                            "Std(mm)": round(st["std"], 6),
                            "3σ-RP": round(st["RP_axis"], 6),
                            "Min": round(st["min"], 6),
                            "Max": round(st["max"], 6),
                            "Range": round(st["range"], 6),
                        })
                ax_df = pd.DataFrame(ax_rows)
                n_rows = min(len(ax_df) + 1, 30)
                fig_h  = max(4.0, n_rows * 0.38)
                fig, ax = plt.subplots(figsize=(11.7, fig_h), facecolor="#1e1e2e")
                ax.set_facecolor("#1e1e2e")
                ax.axis("off")
                ax.text(0.5, 1.01, "Eksen İstatistikleri",
                        ha="center", va="bottom", fontsize=13,
                        color="#cba6f7", fontweight="bold",
                        transform=ax.transAxes)
                if len(ax_df) > 0:
                    tbl2 = ax.table(
                        cellText=ax_df.values.tolist(),
                        colLabels=list(ax_df.columns),
                        loc="center", cellLoc="center",
                    )
                    tbl2.auto_set_font_size(False)
                    tbl2.set_fontsize(8.5)
                    for (r, c), cell in tbl2.get_celld().items():
                        if r == 0:
                            cell.set_facecolor("#45475a")
                            cell.set_text_props(color="#cba6f7", fontweight="bold")
                        else:
                            cell.set_facecolor("#313244")
                            cell.set_text_props(color="#cdd6f4")
                        cell.set_edgecolor("#585b70")
                pdf.savefig(fig, facecolor=fig.get_facecolor())
                plt.close(fig)

            messagebox.showinfo("PDF Kaydedildi",
                                f"Rapor kaydedildi ({len(self.results)} pose, 6 sayfa):\n{path}")
            self.status_var.set(
                f"PDF raporu — {os.path.basename(path)}"
            )
        except Exception as e:
            messagebox.showerror("Hata", f"PDF oluşturulamadı:\n{e}")

    def _fill(self, tree, df):
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=max(80, len(str(col)) * 9), anchor="center")
        for _, row in df.iterrows():
            tree.insert("", "end", values=[str(v) for v in row])


# ═════════════════════════════════════════════
if __name__ == "__main__":
    app = ISO9283App()
    app.mainloop()
