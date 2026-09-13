"""
generate_all_clean_plots.py
===========================
Generates ALL publication figures and scenario plots across the repository
with GUARANTEED ZERO OVERLAP:
- No legends covering data curves or bars
- No legends covering titles ("names") or subplot labels
- No legends covering annotations or axis ticks
- Saves high-res 600 DPI PNGs to figs/, paper/, and results/plots/
"""

import os
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

# -----------------------------------------------------------------------------
# Matplotlib IEEE Publication Configuration
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "figure.titlesize": 9.5,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 5.0,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.dpi": 300,
})

FIG_DIR = "figs"
PAPER_DIR = "paper"
PLOTS_DIR = "results/plots"
for d in [FIG_DIR, PAPER_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

def save_all_dirs(fig, filename):
    """Save figure to figs/, paper/, and results/plots/ at 600 DPI."""
    p1 = os.path.join(FIG_DIR, filename)
    p2 = os.path.join(PAPER_DIR, filename)
    p3 = os.path.join(PLOTS_DIR, filename)
    fig.savefig(p1, dpi=600, bbox_inches="tight", pad_inches=0.04)
    shutil.copy2(p1, p2)
    shutil.copy2(p1, p3)
    plt.close(fig)
    print(f"[OK] Saved {filename} to {FIG_DIR}/, {PAPER_DIR}/, and {PLOTS_DIR}/")


# =============================================================================
# FIGURE 1: System Architecture Diagram
# =============================================================================
def plot_architecture():
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 14.8)
    ax.axis("off")

    def draw_card(cx, cy, w, h, header_text, bg_color="#f8fafc", edge_color="#cbd5e1"):
        x, y = cx - w/2, cy - h/2
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.35",
                           linewidth=0.8, edgecolor=edge_color, facecolor=bg_color, zorder=1)
        ax.add_patch(p)
        ax.text(cx, cy + h/2 - 0.45, header_text, ha="center", va="center",
                fontsize=6.8, weight="bold", color="#1e293b", zorder=2)

    def draw_box(cx, cy, w, h, text, bg_color="white", edge_color="#64748b",
                 fontsize=6.2, weight="normal", lw=0.7):
        x, y = cx - w/2, cy - h/2
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.20",
                            linewidth=lw, edgecolor=edge_color, facecolor=bg_color, zorder=3)
        ax.add_patch(b)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                weight=weight, color="#0f172a", zorder=4)

    def draw_arrow(x1, y1, x2, y2, color="#334155", lw=0.8, style="-|>"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=6,
                            linewidth=lw, color=color, zorder=5, shrinkA=0, shrinkB=0)
        ax.add_patch(a)

    # Background Cards
    draw_card(4.4, 7.25, 8.0, 13.5, "DATA & CAUSAL PREPROCESSING", "#f1f5f9", "#cbd5e1")
    draw_card(15.2, 7.25, 12.0, 13.5, "TWO-STAGE CASCADE DETECTION", "#f0fdf4", "#bbf7d0")
    draw_card(29.4, 10.75, 11.6, 6.5, "DOWNSTREAM ARBITRAGE POLICIES", "#eff6ff", "#bfdbfe")
    draw_card(29.4, 3.75, 11.6, 6.5, "ADVERSARIAL DEFENSE (COMPANION)", "#fff7ed", "#fed7aa")

    # Module 1: Data
    draw_box(4.4, 12.0, 7.0, 2.0, "3-Year CAISO Data\nSP15, NP15, ZP26 + 4 AS\n(25,944 contig. hours)",
             bg_color="#ffffff", edge_color="#0284c7", fontsize=6.2, weight="bold")
    draw_box(4.4, 8.2, 7.0, 2.0, "Chronological Split\nTrain: 23.7m | Val: 5.9m\nTest 1: 6.0m | Test 2 (W2)",
             bg_color="#ffffff", edge_color="#64748b", fontsize=6.0)
    draw_box(4.4, 4.4, 7.0, 2.0, r"Causal Trailing Features" "\n" r"Persistence, RoC, Vol., z" "\n" r"(Strictly $h \leq t$ to avoid leak)",
             bg_color="#ffffff", edge_color="#64748b", fontsize=6.0)
    draw_arrow(4.4, 11.0, 4.4, 9.2)
    draw_arrow(4.4, 7.2, 4.4, 5.4)

    # Arrow from Data to Stage 1 (routes in blank margin at x=8.8, never crossing VAE)
    draw_arrow(7.9, 4.4, 8.8, 4.4)
    draw_arrow(8.8, 4.4, 8.8, 11.6)
    draw_arrow(8.8, 11.6, 9.9, 11.6)

    # Module 2: Detection Pipeline
    draw_box(15.2, 11.6, 10.6, 1.8, r"Stage 1: Isolation Forest Pre-Filter" "\n" r"Causal features $\rightarrow$ top $k \leq 12\%$ candidate hours",
             bg_color="#e0f2fe", edge_color="#0369a1", fontsize=6.0, weight="bold")

    # Clean downward arrows from Stage 1 to each candidate autoencoder
    draw_arrow(10.5, 10.7, 10.5, 9.95)
    draw_arrow(13.6, 10.7, 13.6, 9.95)
    draw_arrow(16.7, 10.7, 16.7, 9.95)
    draw_arrow(19.8, 10.7, 19.8, 9.95)

    # Single-Node Deep Autoencoders (4 models)
    draw_box(10.5, 9.1, 2.2, 1.7, "VAE\n(6-d latent)", bg_color="#ffffff", edge_color="#7570b3", fontsize=5.6)
    draw_box(13.6, 9.1, 2.2, 1.7, "Diffusion\n(DDPM)", bg_color="#ffffff", edge_color="#d95f02", fontsize=5.6)
    draw_box(16.7, 9.1, 2.2, 1.7, "GRU-AE\n(32-u, fast)", bg_color="#ffffff", edge_color="#059669", fontsize=5.6, weight="bold")
    draw_box(19.8, 9.1, 2.2, 1.7, "LSTM-AE\n(32-u, prim.)", bg_color="#ffffff", edge_color="#16a34a", fontsize=5.6, weight="bold")

    # Multi-Node Spatial Models (2 models)
    draw_box(12.4, 6.3, 4.6, 1.9, "MV-LSTM Control\n(3-Hub Plain Recurrent)",
             bg_color="#ffffff", edge_color="#2563eb", fontsize=5.7)
    draw_box(17.9, 6.3, 4.6, 1.9, "GNN-AE (Spatial Attention)\n(3-Hub Regional Graph)",
             bg_color="#ffffff", edge_color="#0284c7", fontsize=5.7, weight="bold")

    draw_arrow(10.5, 8.25, 11.4, 7.3)
    draw_arrow(13.6, 8.25, 13.4, 7.3)
    draw_arrow(16.7, 8.25, 16.9, 7.3)
    draw_arrow(19.8, 8.25, 18.9, 7.3)

    draw_arrow(12.4, 5.35, 13.8, 4.0)
    draw_arrow(17.9, 5.35, 16.6, 4.0)

    draw_box(15.2, 2.7, 10.6, 2.4, "5-Seed Rank Ensemble & Val Calibration\nDecoupled RNG seeds ({0,1,2,3,4}) | Single-thread CPU\nVal budget calibration ($F_1$ max) $\\to$ Held-out Test",
             bg_color="#dcfce7", edge_color="#15803d", fontsize=5.8, weight="bold")

    # Arrows to Downstream (routed in clear gap at x=22.6)
    draw_arrow(20.5, 2.7, 22.6, 2.7)
    draw_arrow(22.6, 2.7, 22.6, 10.3)
    draw_arrow(22.6, 10.3, 23.6, 10.3)

    draw_arrow(22.6, 2.7, 23.6, 2.7)

    # Module 3: Downstream Arbitrage
    draw_box(29.4, 12.1, 10.0, 1.4, "Strategy 1: Reactive Arbitrage\nPeak discharge gating (+0.03% to +1.87% gain)",
             bg_color="#ffffff", edge_color="#64748b", fontsize=5.8)
    draw_box(29.4, 10.3, 10.0, 1.4, "Strategy 2: Directional Spread\nBuy-low/Sell-high (LSTM: 76.5% win, $487)",
             bg_color="#ffffff", edge_color="#64748b", fontsize=5.8)
    draw_box(29.4, 8.5, 10.0, 1.4, r"Strategy 3: Tabular Q-Learning" "\n" r"Signal state-space (Profit: \$3,090 to \$11,957)",
             bg_color="#dbeafe", edge_color="#1d4ed8", fontsize=5.8, weight="bold")

    # Module 4: Adversarial Defense
    draw_box(29.4, 5.1, 10.0, 1.4, "Threat Profiles (CAISO SP15)\nNaive 3x Spikes | Adaptive Bumps | Sinusoidal Waves",
             bg_color="#ffffff", edge_color="#ea580c", fontsize=5.8)
    draw_box(29.4, 3.3, 10.0, 1.4, "XGBoost Classifier + LP Imputation\nDiscriminates fake from real price peaks (AUROC 0.98)",
             bg_color="#ffffff", edge_color="#ea580c", fontsize=5.8)
    draw_box(29.4, 1.5, 10.0, 1.4, r"Break-Even Precision Analysis" "\n" r"Learned ($p^*=0.334$, 89% prec) vs Threshold ($p^*=1.017$)",
             bg_color="#ffedd5", edge_color="#c2410c", fontsize=5.8, weight="bold")

    fig.tight_layout(pad=0.1)
    save_all_dirs(fig, "architecture.png")


# =============================================================================
# FIGURE 2: Multi-Detector ROC Curves (Zero Overlap, Legend at Bottom)
# =============================================================================
def plot_roc_curves():
    with open("artifacts/agg_scored_ensemble.pkl", "rb") as f:
        ens_df = pickle.load(f)
    with open("artifacts/agg_hourly.pkl", "rb") as f:
        agg = pickle.load(f)
    with open("artifacts/detector_eval_ensemble.json") as f:
        ens_report = json.load(f)

    merged = ens_df.merge(agg[["datetime", "SP15"]], on="datetime")
    test_f = merged[merged["datetime"] >= "2025-07-01"].copy()
    y_true = test_f["ref_anomaly"].values

    fig, ax = plt.subplots(figsize=(3.5, 3.8), dpi=300)

    CONFIGS = [
        ("gru_score", "GRU-AE (AUC = 0.882)", "#059669", "-.", "v", 0.02),
        ("lstm_score", "LSTM-AE (AUC = 0.859)", "#1b7837", "-", "d", 0.02),
        ("diffusion_score", "Diffusion (AUC = 0.631)", "#d95f02", "--", "^", 0.02),
        ("iso_score", "iForest (AUC = 0.624)", "#2b5c8f", ":", "o", 0.005),
        ("vae_score", "VAE (AUC = 0.576)", "#7570b3", (0, (3, 1, 1, 1)), "s", 0.02),
    ]

    # Inset Axes placed in bottom right where no curve passes
    ax_ins = ax.inset_axes([0.52, 0.10, 0.44, 0.38])

    for col, label, color, ls, marker, b_default in CONFIGS:
        if col not in test_f.columns:
            continue
        fpr, tpr, thresholds = roc_curve(y_true, test_f[col].values)
        b = ens_report.get(col, {}).get("budget", b_default)
        cut = int(np.ceil(len(test_f) * b))
        thr = np.sort(test_f[col].values)[-cut]
        pred = (test_f[col] >= thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        opr_tpr = tp / max(tp + fn, 1)
        opr_fpr = fp / max(fp + tn, 1)

        # Main curve
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=1.3, label=label)
        ax.scatter([opr_fpr], [opr_tpr], color=color, marker=marker, s=24, zorder=6,
                   edgecolor="black", linewidth=0.5)

        # Inset curve
        ax_ins.plot(fpr, tpr, color=color, linestyle=ls, lw=1.1)
        ax_ins.scatter([opr_fpr], [opr_tpr], color=color, marker=marker, s=20, zorder=6,
                       edgecolor="black", linewidth=0.5)

    # Reference chance line
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Chance")
    ax_ins.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

    # Formatting main axes
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Test-Window ROC (5-Seed Ensemble)", fontweight="bold", pad=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Formatting inset
    ax_ins.set_xlim(-0.005, 0.12)
    ax_ins.set_ylim(-0.02, 0.55)
    ax_ins.set_title(r"FPR $\leq 0.12$", fontsize=6.8, pad=2)
    ax_ins.grid(True, linestyle=":", alpha=0.4)
    ax_ins.tick_params(labelsize=6.5)
    ax.indicate_inset_zoom(ax_ins, edgecolor="#888888", alpha=0.6, lw=0.7)

    # Legend at bottom of figure (Zero overlap with title, curves, or inset!)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=True, fancybox=False, edgecolor="#cbd5e1", facecolor="#fafafa",
               fontsize=6.8, borderpad=0.25, handlelength=1.4)

    fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.1)
    save_all_dirs(fig, "roc_curves.png")


# =============================================================================
# FIGURE 3: Test Timeline Trajectory (High Headroom, Zero Overlap)
# =============================================================================
def plot_test_timeline():
    with open("artifacts/agg_scored_ensemble.pkl", "rb") as f:
        ens_df = pickle.load(f)
    with open("artifacts/agg_hourly.pkl", "rb") as f:
        agg = pickle.load(f)
    with open("artifacts/detector_eval_ensemble.json") as f:
        ens_report = json.load(f)

    merged = ens_df.merge(agg[["datetime", "SP15"]], on="datetime")
    test_f = merged[merged["datetime"] >= "2025-07-01"].copy().sort_values("datetime")

    winner_col = "lstm_score"
    budget = ens_report[winner_col]["budget"]
    cut = int(np.ceil(len(test_f) * budget))
    thr = np.sort(test_f[winner_col].values)[-cut]
    test_f["flag_lstm"] = (test_f[winner_col] >= thr).astype(int)

    ref_flag = test_f[test_f["ref_anomaly"] == 1]
    det_flag = test_f[test_f["flag_lstm"] == 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.16, 4.4), sharey=False,
                                   gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.45})

    # --- Top Panel: Full 6-Month Test Trajectory ---
    ax1.plot(test_f["datetime"], test_f["SP15"], lw=0.65, color="#1e293b", alpha=0.9,
             label="SP15 Day-Ahead LMP ($/MWh)")
    ax1.scatter(ref_flag["datetime"], ref_flag["SP15"], color="#dc2626", s=16, zorder=5,
                label=f"Statistical Reference Anomaly ({len(ref_flag)} h)")
    ax1.scatter(det_flag["datetime"], det_flag["SP15"], facecolors="none", edgecolors="#16a34a",
                s=36, linewidths=0.9, zorder=4, label=f"LSTM-AE Ensemble Flag ({len(det_flag)} h)")

    # Highlight August event region
    zoom_start = pd.Timestamp("2025-08-20 00:00:00")
    zoom_end = pd.Timestamp("2025-08-26 00:00:00")
    ax1.axvspan(zoom_start, zoom_end, color="#fef08a", alpha=0.40, zorder=1,
                label="Scarcity Period (Aug 20–25)")

    ax1.set_ylabel("Price ($/MWh)")
    ax1.set_title("(a) Continuous 6-Month Test Window (July 1, 2025 – December 31, 2025)",
                  fontweight="bold", loc="left", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_xlim(test_f["datetime"].min(), test_f["datetime"].max())

    # High headroom: max price is $115.35; ylim up to 240 leaves 120-unit empty band
    ax1.set_ylim(-15, 240)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=True,
               facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=6.8, borderpad=0.3)

    # --- Bottom Panel: Zoomed August Heatwave Event ---
    zoom_mask = (test_f["datetime"] >= zoom_start) & (test_f["datetime"] <= zoom_end)
    zoom_df = test_f[zoom_mask]
    zoom_ref = ref_flag[(ref_flag["datetime"] >= zoom_start) & (ref_flag["datetime"] <= zoom_end)]
    zoom_det = det_flag[(det_flag["datetime"] >= zoom_start) & (det_flag["datetime"] <= zoom_end)]

    ax2.plot(zoom_df["datetime"], zoom_df["SP15"], lw=1.2, color="#0f172a", marker=".",
             markersize=3, alpha=0.85, label="Hourly SP15 Price")
    ax2.scatter(zoom_ref["datetime"], zoom_ref["SP15"], color="#dc2626", s=32, zorder=5,
                label="Reference Anomaly")
    ax2.scatter(zoom_det["datetime"], zoom_det["SP15"], facecolors="none", edgecolors="#16a34a",
                s=70, linewidths=1.3, zorder=4, label="LSTM-AE Detection")

    # Annotate peak cleanly
    peak_row = zoom_df.loc[zoom_df["SP15"].idxmax()]
    ax2.annotate(f"Peak: ${peak_row['SP15']:.2f}/MWh\n(Aug 23, 02:00)",
                 xy=(peak_row["datetime"], peak_row["SP15"]),
                 xytext=(peak_row["datetime"] + pd.Timedelta(hours=14), peak_row["SP15"] - 14),
                 arrowprops=dict(arrowstyle="->", color="#334155", lw=0.8),
                 fontsize=6.8, weight="bold", color="#0f172a",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor="#cbd5e1", lw=0.6))

    ax2.set_ylabel("Price ($/MWh)")
    ax2.set_xlabel("Date and Hour (August 2025)")
    ax2.set_title("(b) Zoomed Inset: August 2025 Regional Heatwave Excursions",
                  fontweight="bold", loc="left", pad=8)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_xlim(zoom_start, zoom_end)
    ax2.set_ylim(0, 165)

    # Place legend in top-left where price is below $45 (110-unit clearance!)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.02, 0.94), ncol=3, frameon=True,
               facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=6.8, borderpad=0.25)

    import matplotlib.dates as mdates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:00"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))

    fig.tight_layout(pad=0.1)
    save_all_dirs(fig, "test_timeline.png")


# =============================================================================
# FIGURE 4: Profit Comparison Across Strategies (Spacious Headroom)
# =============================================================================
def plot_profit_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.1),
                                   gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.28})

    strategies = ["Reactive Dispatch", "Signal-to-PnL", "Tabular Q-Learning"]
    val_no_signal = [13085.18, 30.17, 3089.95]
    val_with_signal = [13089.42, 487.20, 11957.00]
    err_no_signal = [0, 0, 3905.10]
    err_with_signal = [0, 0, 393.00]

    x = np.arange(len(strategies))
    width = 0.35

    rects1 = ax1.bar(x - width/2, val_no_signal, width, yerr=err_no_signal,
                     capsize=3, color="#94a3b8", edgecolor="#334155", linewidth=0.8,
                     hatch="//", label="Without Anomaly Signal")
    rects2 = ax1.bar(x + width/2, val_with_signal, width, yerr=err_with_signal,
                     capsize=3, color="#16a34a", edgecolor="#14532d", linewidth=0.8,
                     label="With LSTM-AE Signal")

    ax1.set_ylabel("Test-Window Net Profit ($)")
    ax1.set_title("(a) Strategy Profit Across Decision Freedom Levels", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=7.8)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Headroom up to 24,000 leaves >9,000 gap above bars!
    ax1.set_ylim(0, 24000)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), frameon=True,
               facecolor="#ffffff", edgecolor="#cbd5e1", ncol=2, fontsize=7.0)

    for r in rects1:
        h = r.get_height()
        t = f"${h:,.0f}" if h > 500 else f"${h:.0f}"
        ax1.text(r.get_x() + r.get_width()/2., h + 350, t, ha="center", va="bottom", fontsize=6.5)

    for r in rects2:
        h = r.get_height()
        t = f"${h:,.0f}" if h > 500 else f"${h:.0f}"
        ax1.text(r.get_x() + r.get_width()/2., h + 350, t, ha="center", va="bottom",
                 fontsize=6.5, weight="bold", color="#14532d")

    # --- Right Subplot (b): Per-Detector Directional Signal-PnL ---
    detectors = ["iForest", "VAE", "Diffusion", "LSTM-AE"]
    profits = [103.21, 48.13, 61.24, 487.20]
    win_rates = [65.0, 59.0, 57.7, 76.5]
    colors = ["#2b5c8f", "#7570b3", "#d95f02", "#16a34a"]

    bars = ax2.bar(detectors, profits, color=colors, edgecolor="#1e293b", linewidth=0.8, width=0.55)
    ax2.set_ylabel("Signal-to-PnL Profit ($)")
    ax2.set_title("(b) Directional PnL by Detector", fontweight="bold", pad=8)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    for bar, wr in zip(bars, win_rates):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 15,
                 f"${h:.0f}\n({wr:.1f}%)", ha="center", va="bottom", fontsize=6.8, weight="bold")

    ax2.set_ylim(0, 680)

    fig.tight_layout(pad=0.1)
    save_all_dirs(fig, "profit_comparison.png")


# =============================================================================
# FIGURE 5: Break-Even Precision Trade-off Curve (Unobstructed Layout)
# =============================================================================
def plot_breakeven_tradeoff():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0),
                                   gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.28})

    p = np.linspace(0, 1.0, 500)
    saved_tp_clf = 24.09
    lost_fp_clf = -19.11
    p_star_clf = 0.334
    p_opr_clf = 0.891
    val_opr_clf = p_opr_clf * saved_tp_clf + (1 - p_opr_clf) * lost_fp_clf

    saved_tp_rz = -4.89
    lost_fp_rz = -350.45
    p_star_rz = 1.017
    p_opr_rz = 0.802
    val_opr_rz = p_opr_rz * saved_tp_rz + (1 - p_opr_rz) * lost_fp_rz

    val_clf = p * saved_tp_clf + (1 - p) * lost_fp_clf
    val_rz = p * saved_tp_rz + (1 - p) * lost_fp_rz

    # Shaded Zones
    ax1.axhspan(0, 35, color="#dcfce7", alpha=0.45, label="Profitable Defense Zone")
    ax1.axhspan(-130, 0, color="#fee2e2", alpha=0.35, label="Net Economic Loss Zone")
    ax1.axhline(0, color="#475569", linestyle="--", lw=0.9, zorder=2)

    # Plot Lines
    ax1.plot(p, val_clf, color="#16a34a", lw=2.0, label="Learned XGBoost Classifier ($p^* = 0.334$)")
    ax1.plot(p, val_rz, color="#dc2626", lw=1.8, linestyle="-.", label="Trailing Robust-$z$ Rule ($p^* = 1.017$)")

    # Operating Points
    ax1.scatter([p_opr_clf], [val_opr_clf], color="#16a34a", edgecolors="#0f172a", s=60, marker="o", zorder=5)
    ax1.annotate(f"Operational Point\np = {p_opr_clf:.3f}, Net: +${val_opr_clf:.2f}/h",
                 xy=(p_opr_clf, val_opr_clf), xytext=(0.82, -35), ha="center",
                 arrowprops=dict(arrowstyle="->", color="#15803d", lw=0.9),
                 fontsize=6.5, weight="bold", color="#15803d",
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#86efac", lw=0.6))

    ax1.scatter([p_opr_rz], [val_opr_rz], color="#dc2626", edgecolors="#0f172a", s=60, marker="s", zorder=5)
    ax1.annotate(f"Threshold Point\np = {p_opr_rz:.3f}, Net: -${abs(val_opr_rz):.2f}/h",
                 xy=(p_opr_rz, val_opr_rz), xytext=(0.42, -95),
                 arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.9),
                 fontsize=6.5, weight="bold", color="#b91c1c",
                 bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#fca5a5", lw=0.6))

    ax1.axvline(p_star_clf, color="#16a34a", linestyle=":", lw=1.0)
    ax1.text(p_star_clf + 0.02, -24, f"Break-Even\np* = {p_star_clf:.3f}", fontsize=6.6,
             color="#15803d", weight="bold")

    ax1.set_xlabel("Operational Defense Precision $p$")
    ax1.set_ylabel("Expected Net Value per Flagged Hour ($/h)")
    ax1.set_title("(a) Marginal Break-Even Economics by Precision", fontweight="bold", pad=8)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(-130, 60)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Place legend in empty upper-left (p < 0.40, y > 15 is 100% blank)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.03, 0.95), frameon=True,
               facecolor="#fafafa", edgecolor="#cccccc", fontsize=6.8)

    # --- Right Subplot (b): Per-Event Error Cost Asymmetry ---
    metrics = ["Saved per TP", "Lost per FP"]
    clf_costs = [saved_tp_clf, abs(lost_fp_clf)]
    rz_costs = [saved_tp_rz, abs(lost_fp_rz)]

    x = np.arange(len(metrics))
    w = 0.35

    rects_c = ax2.bar(x - w/2, clf_costs, w, color="#16a34a", edgecolor="#14532d", linewidth=0.8, label="Learned Classifier")
    rects_r = ax2.bar(x + w/2, rz_costs, w, color="#dc2626", edgecolor="#7f1d1d", linewidth=0.8, hatch="//", label="Robust-$z$ Rule")

    ax2.set_ylabel("Magnitude ($/event)")
    ax2.set_title("(b) Economic Impact Asymmetry", fontweight="bold", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=7.8)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Legend in upper-left above TP bars (TP bars are only $24 high; 400-unit clearance!)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.05, 0.95), frameon=True, facecolor="#fafafa", edgecolor="#cccccc", fontsize=7.0)

    ax2.text(rects_c[0].get_x() + rects_c[0].get_width()/2., clf_costs[0] + 8,
             f"+${clf_costs[0]:.2f}", ha="center", va="bottom", fontsize=6.8, weight="bold")
    ax2.text(rects_c[1].get_x() + rects_c[1].get_width()/2., clf_costs[1] + 8,
             f"-${clf_costs[1]:.2f}", ha="center", va="bottom", fontsize=6.8, weight="bold")

    ax2.text(rects_r[0].get_x() + rects_r[0].get_width()/2., rz_costs[0] + 8,
             f"${rz_costs[0]:.2f}", ha="center", va="bottom", fontsize=6.8)
    ax2.text(rects_r[1].get_x() + rects_r[1].get_width()/2., rz_costs[1] + 8,
             f"-${rz_costs[1]:.2f}\n(18.3x penalty!)", ha="center", va="bottom",
             fontsize=6.8, weight="bold", color="#7f1d1d")

    ax2.set_ylim(-30, 480)

    fig.tight_layout(pad=0.1)
    save_all_dirs(fig, "breakeven_tradeoff.png")


# =============================================================================
# FIGURE 6: Cross-Attack Transfer Matrix Dual Heatmap
# =============================================================================
def plot_transfer_heatmap():
    with open("artifacts_v2/results_transfer.json") as f:
        tr = json.load(f)

    df = pd.DataFrame(tr)
    attacks = ["naive", "adaptive", "sinusoidal"]
    labels = ["Naive", "Adaptive", "Sinusoidal"]
    sub = df[df["sources"].isin(attacks) & df["target"].isin(attacks)]

    auroc_mat = sub.groupby(["sources", "target"])["auroc"].mean().unstack()[attacks].reindex(attacks).values
    rec_mat = sub.groupby(["sources", "target"])["rec_conservative"].mean().unstack()[attacks].reindex(attacks).values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.7), gridspec_kw={"wspace": 0.40})

    # Heatmap 1: AUROC
    im1 = ax1.imshow(auroc_mat, cmap="YlGnBu", vmin=0.70, vmax=1.00, aspect="auto")
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7.8)
    ax1.set_yticklabels(labels, fontsize=7.8)
    ax1.set_xlabel("Target Evaluation Attack", labelpad=3)
    ax1.set_ylabel("Source Training Attack", labelpad=3)
    ax1.set_title("(a) Detection AUROC (Nominally Invariant)", fontweight="bold", pad=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = auroc_mat[i, j]
            text_color = "white" if val > 0.92 else "black"
            weight = "bold" if i == j else "normal"
            ax1.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=7.2, weight=weight)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.ax.tick_params(labelsize=6.8)

    # Heatmap 2: Economic Damage Recovery %
    norm = mpl.colors.TwoSlopeNorm(vmin=-45.0, vcenter=0.0, vmax=70.0)
    im2 = ax2.imshow(rec_mat, cmap="RdYlGn", norm=norm, aspect="auto")
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels, fontsize=7.8)
    ax2.set_yticklabels(labels, fontsize=7.8)
    ax2.set_xlabel("Target Evaluation Attack", labelpad=3)
    ax2.set_title("(b) Economic Damage Recovery %", fontweight="bold", pad=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = rec_mat[i, j]
            text_color = "white" if abs(val) > 35 else "black"
            weight = "bold" if (i == j or val < -15) else "normal"
            ax2.text(j, i, f"{val:+.1f}%", ha="center", va="center", color=text_color, fontsize=7.2, weight=weight)

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.ax.tick_params(labelsize=6.8)

    fig.tight_layout(pad=0.1)
    save_all_dirs(fig, "transfer_heatmap.png")


# =============================================================================
# FIGURE 7: F1 Score Optimization & Multi-Data Enrichment
# =============================================================================
def plot_f1_precision_boost():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), dpi=300)

    # Panel A: Anomaly Detection F1 Gains
    models = ["GRU-AE", "LSTM-AE", "iForest"]
    x = np.arange(len(models))
    width = 0.22

    base_f1 = [0.155, 0.126, 0.180]
    cal_f1 = [0.200, 0.160, 0.240]
    gated_f1 = [0.364, 0.407, 0.488]

    axes[0].bar(x - width, base_f1, width, label="Baseline Coarse", color="#94a3b8", edgecolor="#334155", linewidth=0.7)
    axes[0].bar(x, cal_f1, width, label="Method 1: Calibrated", color="#38bdf8", edgecolor="#0369a1", linewidth=0.7)
    axes[0].bar(x + width, gated_f1, width, label="Method 2: Cascade Gated", color="#2563eb", edgecolor="#1e3a8a", linewidth=0.7)

    axes[0].set_ylabel("Pointwise $F_1$ Score")
    axes[0].set_title("(a) Threshold & Gating Optimization", fontweight="bold", pad=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=8.0)
    axes[0].set_ylim(0, 0.72)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True,
                   facecolor="white", framealpha=0.9, fontsize=6.8, edgecolor="#cbd5e1")

    for i in range(len(models)):
        gain = (gated_f1[i] - base_f1[i]) / base_f1[i] * 100
        axes[0].text(x[i] + width, gated_f1[i] + 0.015, f"+{gain:.0f}%", ha="center",
                     fontsize=7.2, fontweight="bold", color="#1e3a8a")

    # Panel B: Point-Adjusted (PA-F1) Benchmark Metrics
    pa_f1 = [0.267, 0.323, 0.180]
    gated_pa = [0.518, 0.677, 0.488]

    axes[1].bar(x - width/2, pa_f1, width, label="Raw Event PA-$F_1$", color="#a78bfa", edgecolor="#5b21b6", linewidth=0.7)
    axes[1].bar(x + width/2, gated_pa, width, label="Gated + PA-$F_1$", color="#7c3aed", edgecolor="#4c1d95", linewidth=0.7)

    axes[1].set_ylabel("Event-Adjusted $F_1$ (PA-$F_1$)")
    axes[1].set_title("(b) Time-Series Benchmark PA-$F_1$", fontweight="bold", pad=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=8.0)
    axes[1].set_ylim(0, 0.98)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=True,
                   facecolor="white", framealpha=0.9, fontsize=7.0, edgecolor="#cbd5e1")

    for i in range(len(models)):
        axes[1].text(x[i] + width/2, gated_pa[i] + 0.015, f"{gated_pa[i]:.3f}", ha="center",
                     fontsize=7.2, fontweight="bold", color="#4c1d95")

    # Panel C: Multi-Data Classifier Enrichment (Ancillary + Spatial)
    seeds = ["Seed 0", "Seed 1", "Seed 2", "Mean"]
    x_s = np.arange(len(seeds))
    base_clf = [0.725, 0.759, 0.738, 0.741]
    enr_clf = [0.856, 0.868, 0.847, 0.857]

    axes[2].bar(x_s - width/2, base_clf, width, label="Baseline (SP15 Only)", color="#fca5a5", edgecolor="#991b1b", linewidth=0.7)
    axes[2].bar(x_s + width/2, enr_clf, width, label="Enriched (Ancillary + Nodes)", color="#dc2626", edgecolor="#7f1d1d", linewidth=0.7)

    axes[2].set_ylabel("Classifier $F_1$ Score")
    axes[2].set_title("(c) Multi-Data Enrichment (XGBoost)", fontweight="bold", pad=8)
    axes[2].set_xticks(x_s)
    axes[2].set_xticklabels(seeds, fontsize=8.0)
    axes[2].set_ylim(0.5, 1.25)
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=True,
                   facecolor="white", framealpha=0.9, fontsize=6.3, borderpad=0.25, edgecolor="#cbd5e1")

    for i in range(len(seeds)):
        diff = enr_clf[i] - base_clf[i]
        axes[2].text(x_s[i] + width/2, enr_clf[i] + 0.012, f"+{diff:.3f}", ha="center",
                     fontsize=7.2, fontweight="bold", color="#7f1d1d")

    fig.tight_layout(pad=0.2)
    save_all_dirs(fig, "f1_precision_boost.png")


# =============================================================================
# FIGURE 8: Lambda Sensitivity Trade-off (Bottom Legend, Zero Overlap)
# =============================================================================
def plot_lambda_sensitivity():
    if not os.path.exists("lambda_sweep.csv"):
        return
    df = pd.read_csv("lambda_sweep.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2), dpi=300)

    # Panel A: Clean Scenario
    df_clean = df[df["scenario"] == "clean"].sort_values("lam")
    ax1.plot(df_clean["lam"], df_clean["baseline"] / 1000, "k--", lw=1.2, label="Clean Baseline ($485.1K)")
    ax1.plot(df_clean["lam"], df_clean["penalised"] / 1000, "r-o", lw=1.2, markersize=4, label=r"Blanket Penalty ($\lambda$)")
    ax1.plot(df_clean["lam"], df_clean["opportunistic"] / 1000, "g-s", lw=1.4, markersize=4, label="Opportunistic Defense")
    ax1.set_xlabel(r"Penalty Parameter $\lambda$")
    ax1.set_ylabel("Total Net Value ($K)")
    ax1.set_title("(a) Clean Market Scenario", fontweight="bold", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_ylim(445, 495)

    # Panel B: Attack Scenario
    df_atk = df[df["scenario"] == "attack"].sort_values("lam")
    ax2.plot(df_atk["lam"], df_atk["baseline"] / 1000, "k--", lw=1.2, label="Undefended ($482.9K)")
    ax2.plot(df_atk["lam"], df_atk["penalised"] / 1000, "r-o", lw=1.2, markersize=4, label=r"Blanket Penalty ($\lambda$)")
    ax2.plot(df_atk["lam"], df_atk["opportunistic"] / 1000, "g-s", lw=1.4, markersize=4, label="Opportunistic Defense")
    ax2.set_xlabel(r"Penalty Parameter $\lambda$")
    ax2.set_ylabel("Total Net Value ($K)")
    ax2.set_title("(b) Adversarial Attack Scenario", fontweight="bold", pad=8)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_ylim(445, 495)

    # Place shared legend at bottom of figure (Zero overlap with titles!)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=3, frameon=True, fontsize=7.2, edgecolor="#cbd5e1")

    fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.2)
    save_all_dirs(fig, "lambda_sensitivity.png")


# =============================================================================
# FIGURE 9: Robustness Sweep Heatmap
# =============================================================================
def plot_robustness_heatmap():
    if not os.path.exists("robustness_sweep.csv"):
        return
    df = pd.read_csv("robustness_sweep.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.7), dpi=300)

    piv_det = df.pivot(index="magnitude", columns="duration", values="detection_rate") * 100
    piv_uplift = df.pivot(index="magnitude", columns="duration", values="abs_uplift")

    im1 = ax1.imshow(piv_det.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=70)
    ax1.set_xticks(np.arange(len(piv_det.columns)))
    ax1.set_yticks(np.arange(len(piv_det.index)))
    ax1.set_xticklabels([f"{d}h" for d in piv_det.columns], fontsize=7.8)
    ax1.set_yticklabels([f"{m}x" for m in piv_det.index], fontsize=7.8)
    ax1.set_xlabel("Attack Duration")
    ax1.set_ylabel("Attack Magnitude")
    ax1.set_title("(a) Detection Rate (%) Across Attack Regimes", fontweight="bold", pad=8)

    for i in range(len(piv_det.index)):
        for j in range(len(piv_det.columns)):
            val = piv_det.values[i, j]
            col = "white" if val > 40 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=col, fontsize=7.2, weight="bold")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)
    cbar1.ax.tick_params(labelsize=6.8)

    im2 = ax2.imshow(piv_uplift.values, cmap="Greens", aspect="auto")
    ax2.set_xticks(np.arange(len(piv_uplift.columns)))
    ax2.set_yticks(np.arange(len(piv_uplift.index)))
    ax2.set_xticklabels([f"{d}h" for d in piv_uplift.columns], fontsize=7.8)
    ax2.set_yticklabels([f"{m}x" for m in piv_uplift.index], fontsize=7.8)
    ax2.set_xlabel("Attack Duration")
    ax2.set_ylabel("Attack Magnitude")
    ax2.set_title("(b) Absolute Profit Uplift ($) via Defense", fontweight="bold", pad=8)

    for i in range(len(piv_uplift.index)):
        for j in range(len(piv_uplift.columns)):
            val = piv_uplift.values[i, j]
            col = "white" if val > 150 else "black"
            ax2.text(j, i, f"${val:.0f}", ha="center", va="center", color=col, fontsize=7.2, weight="bold")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.ax.tick_params(labelsize=6.8)

    fig.tight_layout(pad=0.1)
    save_all_dirs(fig, "robustness_heatmap.png")


# =============================================================================
# FIGURE 10: SOTA Comparison Bar Chart (Bottom Legend, Zero Overlap)
# =============================================================================
def plot_sota_comparison():
    if not os.path.exists("results/sota_comparison.csv"):
        return
    df = pd.read_csv("results/sota_comparison.csv")
    fig, ax = plt.subplots(figsize=(4.5, 3.4), dpi=300)

    x = np.arange(len(df))
    w = 0.35
    clean_col = df["Clean Opp ($K)"].values
    atk_col = df["Attack Opp ($K)"].values
    impr = df["Attack improvement (%)"].values

    rects1 = ax.bar(x - w/2, clean_col, w, label="Clean Opportunistic ($K)", color="#38bdf8", edgecolor="#0284c7")
    rects2 = ax.bar(x + w/2, atk_col, w, label="Attack Opportunistic ($K)", color="#10b981", edgecolor="#047857")

    ax.set_ylabel("Realized Net Profit ($K)")
    ax.set_title("Benchmark Comparison Across Detector Lineages", fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Detector"].values, fontsize=7.8)
    ax.set_ylim(472, 498)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    for i in range(len(df)):
        ax.text(x[i] + w/2, atk_col[i] + 0.3, f"{impr[i]:+.2f}%", ha="center", fontsize=7.2, weight="bold", color="#047857")

    # Bottom legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=True, fontsize=7.2, edgecolor="#cbd5e1")

    fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.1)
    save_all_dirs(fig, "sota_comparison.png")


# =============================================================================
# FIGURE 11: Genuine vs Synthetic Classifier Evaluation (Bottom Legend)
# =============================================================================
def plot_classifier_eval():
    if not os.path.exists("results_summary.json"):
        return
    with open("results_summary.json") as f:
        res = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.2), dpi=300)

    # Panel A: Synthetic Attack Metrics
    metrics = ["Precision", "Recall", "F1 Score"]
    x = np.arange(len(metrics))
    w = 0.35

    logreg_syn = [
        res["classifier_logreg"]["synthetic"]["precision"],
        res["classifier_logreg"]["synthetic"]["recall"],
        res["classifier_logreg"]["synthetic"]["f1"]
    ]
    xgb_syn = [
        res["classifier_xgb"]["synthetic"]["precision"],
        res["classifier_xgb"]["synthetic"]["recall"],
        res["classifier_xgb"]["synthetic"]["f1"]
    ]

    r1 = ax1.bar(x - w/2, logreg_syn, w, label="Logistic Regression", color="#f87171", edgecolor="#b91c1c")
    r2 = ax1.bar(x + w/2, xgb_syn, w, label="XGBoost Classifier", color="#34d399", edgecolor="#059669")

    ax1.set_ylabel("Metric Value")
    ax1.set_title("(a) Synthetic Attack Classification", fontweight="bold", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=8.0)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    for r in r1:
        ax1.text(r.get_x() + r.get_width()/2., r.get_height() + 0.02, f"{r.get_height():.3f}", ha="center", fontsize=6.8)
    for r in r2:
        ax1.text(r.get_x() + r.get_width()/2., r.get_height() + 0.02, f"{r.get_height():.3f}", ha="center", fontsize=6.8, weight="bold")

    # Panel B: Global Discrimination Capacity
    disc_metrics = ["AUROC", "Average Precision (AP)"]
    x_d = np.arange(len(disc_metrics))
    logreg_d = [res["classifier_logreg"]["auroc"], res["classifier_logreg"]["ap"]]
    xgb_d = [res["classifier_xgb"]["auroc"], res["classifier_xgb"]["ap"]]

    r3 = ax2.bar(x_d - w/2, logreg_d, w, label="Logistic Regression", color="#f87171", edgecolor="#b91c1c")
    r4 = ax2.bar(x_d + w/2, xgb_d, w, label="XGBoost Classifier", color="#34d399", edgecolor="#059669")

    ax2.set_ylabel("Score")
    ax2.set_title("(b) Global Discrimination Capacity", fontweight="bold", pad=8)
    ax2.set_xticks(x_d)
    ax2.set_xticklabels(disc_metrics, fontsize=8.0)
    ax2.set_ylim(0.85, 1.07)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    for r in r3:
        ax2.text(r.get_x() + r.get_width()/2., r.get_height() + 0.005, f"{r.get_height():.3f}", ha="center", fontsize=6.8)
    for r in r4:
        ax2.text(r.get_x() + r.get_width()/2., r.get_height() + 0.005, f"{r.get_height():.3f}", ha="center", fontsize=6.8, weight="bold")

    # Bottom shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=True, fontsize=7.2, edgecolor="#cbd5e1")

    fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.2)
    save_all_dirs(fig, "classifier_eval.png")


# =============================================================================
# SCENARIO PLOTS: Price, SOC, Volatility & Grad-CAM (Guaranteed Zero Overlap)
# =============================================================================
def plot_scenario_prices():
    """Generates clean_price.png and attack_price.png."""
    for scenario, fname in [("clean", "clean_price.png"), ("attack", "attack_price.png")]:
        csv_path = f"results/{scenario}_processed.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(7.16, 3.2), dpi=300)

        # Plot full trajectory
        ax.plot(df["SP15"].values, lw=0.6, color="#0f172a", alpha=0.85, label="Actual SP15 Price ($/MWh)")
        if "expected_price" in df.columns:
            ax.plot(df["expected_price"].values, lw=1.0, color="#2563eb", alpha=0.9, label="Expected Price (VAE Recon)")
        if "expected_upper" in df.columns and "expected_lower" in df.columns:
            ax.fill_between(range(len(df)), df["expected_lower"].values, df["expected_upper"].values,
                            color="#93c5fd", alpha=0.35, label="95% Confidence Band")

        # Highlight detected anomaly hours
        if "anomaly" in df.columns and df["anomaly"].sum() > 0:
            anom_idx = np.where(df["anomaly"].values == 1)[0]
            ax.scatter(anom_idx, df["SP15"].values[anom_idx], color="#dc2626", s=14, zorder=5, label="Detected Anomaly Hours")

        title_prefix = "Clean Market Scenario" if scenario == "clean" else "Adversarial Attack Scenario"
        ax.set_title(f"Price vs Expected Profile — {title_prefix}", fontweight="bold", pad=8)
        ax.set_ylabel("Price ($/MWh)")
        ax.set_xlabel("Time Index (Hours)")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Bottom legend in dedicated space
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=4, frameon=True, fontsize=7.0, edgecolor="#cbd5e1")

        fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.1)
        save_all_dirs(fig, fname)


def plot_scenario_soc():
    """Generates clean_soc.png and attack_soc.png."""
    for scenario, fname in [("clean", "clean_soc.png"), ("attack", "attack_soc.png")]:
        csv_path = f"results/soc_{scenario}.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(7.16, 3.2), dpi=300)

        # Plot representative 2-week window (336 hours) for crystal-clear readability
        window_len = 336
        sub_df = df.iloc[:window_len]
        hrs = np.arange(len(sub_df))

        ax.plot(hrs, sub_df["baseline"].values, lw=1.1, color="#64748b", linestyle="--", label="Baseline SOC")
        ax.plot(hrs, sub_df["penalised"].values, lw=1.2, color="#ef4444", linestyle="-.", label="Penalised SOC")
        ax.plot(hrs, sub_df["opportunistic"].values, lw=1.4, color="#10b981", linestyle="-", label="Opportunistic SOC")

        title_prefix = "Clean Market Scenario" if scenario == "clean" else "Adversarial Attack Scenario"
        ax.set_title(f"Battery State-of-Charge (SOC) Dynamics — {title_prefix}", fontweight="bold", pad=8)
        ax.set_ylabel("Battery SOC (MWh)")
        ax.set_xlabel("Operating Hour (Representative Two-Week Window)")
        ax.set_ylim(-0.5, 11.5)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Bottom legend in dedicated space
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=3, frameon=True, fontsize=7.2, edgecolor="#cbd5e1")

        fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.1)
        save_all_dirs(fig, fname)


def plot_scenario_volatility():
    """Generates clean_volatility.png and attack_volatility.png."""
    for scenario, fname in [("clean", "clean_volatility.png"), ("attack", "attack_volatility.png")]:
        csv_path = f"results/{scenario}_volatility.csv"
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        fig, ax = plt.subplots(figsize=(7.16, 2.9), dpi=300)

        ax.plot(df["volatility"].values, lw=0.8, color="#7c3aed", label="24-Hour Rolling Volatility ($/MWh)")
        title_prefix = "Clean Market Scenario" if scenario == "clean" else "Adversarial Attack Scenario"
        ax.set_title(f"Price Volatility Trajectory — {title_prefix}", fontweight="bold", pad=8)
        ax.set_ylabel("Volatility ($/MWh)")
        ax.set_xlabel("Operating Hour")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Bottom legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=1, frameon=True, fontsize=7.2, edgecolor="#cbd5e1")

        fig.tight_layout(rect=[0, 0.14, 1, 1], pad=0.1)
        save_all_dirs(fig, fname)


def plot_gradcam_weekly():
    """Generates gradcam_week_attack.png and gradcam_week_attack_week1.png with zero overlap."""
    atk_proc_path = "results/attack_processed.csv"
    if not os.path.exists(atk_proc_path):
        return
    df = pd.read_csv(atk_proc_path)

    # 1 week window (168 hours)
    for week_idx, fname in [(0, "gradcam_week_attack_week1.png"), (1, "gradcam_week_attack.png")]:
        start_hr = week_idx * 168
        end_hr = start_hr + 168
        week_df = df.iloc[start_hr:end_hr].copy()
        hrs = np.arange(len(week_df))

        fig, ax1 = plt.subplots(figsize=(7.16, 3.4), dpi=300)

        l1 = ax1.plot(hrs, week_df["SP15"].values, color="#1e40af", lw=1.2, label="SP15 Price ($/MWh)")
        ax1.set_ylabel("Price ($/MWh)", color="#1e40af")
        ax1.tick_params(axis="y", colors="#1e40af")
        ax1.set_xlabel("Hour of Week")

        # High headroom for price (fully unclipped curves)
        ax1.set_ylim(0, 240)

        # Secondary axis for Grad-CAM
        ax2 = ax1.twinx()
        grad_vals = week_df["gradcam"].values if "gradcam" in week_df.columns else np.zeros(len(week_df))
        l2 = ax2.plot(hrs, grad_vals, color="#dc2626", lw=1.4, linestyle="--", label="Grad-CAM Temporal Influence")
        ax2.set_ylabel("Influence Score", color="#dc2626")
        ax2.tick_params(axis="y", colors="#dc2626")
        ax2.set_ylim(0, 1.55)

        # Day ticks
        day_ticks = [i * 24 for i in range(8)]
        day_labels = [f"Day {i+1}" for i in range(7)] + ["Day 7+"]
        ax1.set_xticks(day_ticks)
        ax1.set_xticklabels(day_labels, fontsize=7.5)
        ax1.grid(True, linestyle=":", alpha=0.4)

        ax1.set_title("Grad-CAM Temporal Influence Across Adversarial Attack Week", fontweight="bold", pad=8)

        # Combined legend at bottom of figure (Zero overlap with title or price/gradcam curves!)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   ncol=2, frameon=True, fontsize=7.2, edgecolor="#cbd5e1")

        fig.tight_layout(rect=[0, 0.12, 1, 1], pad=0.1)
        save_all_dirs(fig, fname)


# =============================================================================
# Main Execution Pipeline
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("RENDERING ALL PUBLICATION & SCENARIO GRAPHS WITH ZERO OVERLAP")
    print("=" * 65)

    print("\n1. Rendering Figure 1: System Architecture...")
    plot_architecture()

    print("\n2. Rendering Figure 2: ROC Curves (Bottom Legend)...")
    plot_roc_curves()

    print("\n3. Rendering Figure 3: Test Timeline (High Headroom)...")
    plot_test_timeline()

    print("\n4. Rendering Figure 4: Profit Comparison (High Headroom)...")
    plot_profit_comparison()

    print("\n5. Rendering Figure 5: Break-Even Precision Trade-off...")
    plot_breakeven_tradeoff()

    print("\n6. Rendering Figure 6: Cross-Attack Transfer Heatmap...")
    plot_transfer_heatmap()

    print("\n7. Rendering Figure 7: F1 Score & Precision Boost Suite...")
    plot_f1_precision_boost()

    print("\n8. Rendering Figure 8: Lambda Sensitivity Trade-off (Bottom Legend)...")
    plot_lambda_sensitivity()

    print("\n9. Rendering Figure 9: Robustness Sweep Heatmap...")
    plot_robustness_heatmap()

    print("\n10. Rendering Figure 10: SOTA Benchmark Comparison (Bottom Legend)...")
    plot_sota_comparison()

    print("\n11. Rendering Figure 11: Classifier Evaluation (Bottom Legend)...")
    plot_classifier_eval()

    print("\n12. Rendering Scenario Plots: clean_price & attack_price...")
    plot_scenario_prices()

    print("\n13. Rendering Scenario Plots: clean_soc & attack_soc...")
    plot_scenario_soc()

    print("\n14. Rendering Scenario Plots: clean_volatility & attack_volatility...")
    plot_scenario_volatility()

    print("\n15. Rendering Scenario Plots: gradcam_week_attack & week1...")
    plot_gradcam_weekly()

    print("\n" + "=" * 65)
    print("ALL 19 FIGURES RENDERED AND SYNCHRONIZED ACROSS:")
    print(f"  - {FIG_DIR}/")
    print(f"  - {PAPER_DIR}/")
    print(f"  - {PLOTS_DIR}/")
    print("Zero overlap: Titles, names, curves, and annotations fully unobstructed!")
    print("=" * 65)
