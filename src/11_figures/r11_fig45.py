#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig4 | 协方差层脆弱性  +  Fig5 | 阈值反推与两个"不可判"门"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_PY, C_R, C_NEU, C_BAD, C_GY = "#3D6BA8", "#C23531", "#177CB0", "#9D2933", "#999999"
C_OK = "#3F6B3F"
OUT = "/Users/taozhu/my researches/retyping"; T = "/tmp/gyn_retyping"
sys.path.insert(0, T)
from r5_geomcheck import check
def J(p): return json.load(open(f"{T}/{p}"))

bv = pd.read_csv(f"{T}/eeec_batch/batch_verify.csv")
pl = J("platform/platform_results.json")
g2, g3, gf = J("g2_refined.json"), J("g3_power.json"), J("g2_final.json")

SHORT = {"S0 原始 log2": "S0 raw", "S1 逐基因 z（全局）": "S1 glob-z",
         "S2 批次内逐基因 z（上限）": "S2 batch-z", "S3 逐样本分位数标准化": "S3 quant",
         "S4 逐样本秩→逆正态": "S4 rank", "ComBat 规范（sva）": "ComBat",
         "ComBat mean-only": "ComBat-M", "removeBatchEffect（limma）": "limma",
         "S5 Harmony（嵌入层）": "Harmony"}
LONG = {"S0 原始 log2": "S0 raw log2", "S1 逐基因 z（全局）": "S1 global z",
        "S2 批次内逐基因 z（上限）": "S2 within-batch z", "S3 逐样本分位数标准化": "S3 per-sample quantile",
        "S4 逐样本秩→逆正态": "S4 rank-INT", "ComBat 规范（sva）": "ComBat (sva)",
        "ComBat mean-only": "ComBat mean-only", "removeBatchEffect（limma）": "limma rmBE",
        "S5 Harmony（嵌入层）": "Harmony"}
KIND = {"S0 raw": "none", "S1 glob-z": "per-gene", "S2 batch-z": "per-gene",
        "S3 quant": "per-sample", "S4 rank": "per-sample", "ComBat": "per-gene",
        "ComBat-M": "per-gene", "limma": "per-gene", "Harmony": "covariance"}
COL = {"none": C_GY, "per-gene": C_NEU, "per-sample": C_R, "covariance": C_OK}
bv["sh"] = bv["arm"].map(SHORT); bv["lg"] = bv["arm"].map(LONG)

# ═══════════════════════════════════════════════════════════════════
# Figure 4
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(6.94, 5.52))
gs = fig.add_gridspec(2, 3, hspace=0.78, wspace=0.52, left=0.075, right=0.975, top=0.905, bottom=0.185)

# --- a ---
ax = fig.add_subplot(gs[0, 0:2])
x = np.arange(len(bv)); w = 0.36
ax.bar(x - w/2, bv["centroid_AUC"], w, color=C_NEU, edgecolor="white", label="per-gene mean separation")
ax.bar(x + w/2, bv["LR_AUC_abs"], w, color=C_BAD, edgecolor="white", label="total separability")
ax.axhline(0.5, color="#666666", lw=1.0, ls=":")
ax.text(0.52, 0.53, "chance", fontsize=8, color="#666666", ha="left", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(bv["sh"], rotation=52, ha="right", fontsize=8)
ax.set_ylabel("AUC"); ax.set_ylim(0, 1.72); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_title("a   Batch effect: mean is removable, structure is not", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.4, columnspacing=1.0,
          borderpad=0.1, fontsize=8)
for xi, (c, l) in enumerate(zip(bv["centroid_AUC"], bv["LR_AUC_abs"])):
    ax.text(xi - w/2, 0.055, f"{c:.2f}", ha="center", fontsize=8, color="white")
    ax.text(xi + w/2, l + 0.026, f"{l:.2f}", ha="center", fontsize=8, color=C_BAD)
ax.text(0.985, 0.87, "limma / ComBat reach 0.50 exactly;\ntotal separability stays 1.00",
        transform=ax.transAxes, fontsize=8, color=C_BAD, ha="right", va="top", linespacing=1.35)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- b ---
ax = fig.add_subplot(gs[0, 2])
ax.barh(np.arange(len(bv)), bv["knn_mix"], 0.70,
        color=[COL[KIND[s]] for s in bv["sh"]], edgecolor="white")
ax.axvline(0.5, color="#666666", lw=1.0, ls=":")
ax.text(0.30, -0.95, "full mixing", fontsize=8, color="#666666", ha="center", va="center")
for yi, v in zip(np.arange(len(bv)), bv["knn_mix"]):
    ax.text(v + 0.010, yi, f"{v:.3f}", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(bv))); ax.set_yticklabels(bv["lg"], fontsize=8)
ax.invert_yaxis()
ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_xlim(0, 0.62)
ax.set_xlabel("kNN batch mixing")
ax.set_title("b   Neighbourhood mixing", loc="left", pad=6)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- c ---
ax = fig.add_subplot(gs[1, 0])
lad = pl["ladder"]
names = ["same\ncohort", "vs\nEEEC", "vs\nEEEC-E", "vs\nEEEC-L"]
vals = [lad[i]["spearman"] for i in range(4)]
ax.bar(range(4), vals, 0.62, color=[C_OK, C_PY, C_PY, C_PY], edgecolor="white")
for xi, v in enumerate(vals):
    ax.text(xi, v + 0.018, f"{v:.3f}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("Δ concordance (Spearman)"); ax.set_ylim(0, 1.06); ax.set_yticks([0, 0.5, 1.0])
ax.set_title("c   Δ transfers across platforms", loc="left", pad=6)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- d ---
ax = fig.add_subplot(gs[1, 1])
bins = pl["bins"]; xs = np.arange(len(bins))
ax.plot(xs, [b["spearman"] for b in bins], color=C_PY, marker="o", lw=1.6, ms=6, label="Spearman ρ")
ax.plot(xs, [b["sign_conc"] for b in bins], color=C_BAD, marker="s", lw=1.6, ms=6, ls="--",
        label="sign concordance")
for xi, b in zip(xs, bins):
    ax.text(xi, b["spearman"] - 0.082, f"{b['spearman']:.3f}", ha="center", fontsize=8, color=C_PY)
    ax.text(xi, b["sign_conc"] + 0.042, f"{b['sign_conc']*100:.1f}%", ha="center", fontsize=8, color=C_BAD)
ax.set_xticks(xs); ax.set_xticklabels(["Q1\nweakest", "Q2", "Q3", "Q4\nstrongest"], fontsize=8)
ax.set_xlabel("Effect-size quartile of |Δ|"); ax.set_ylabel("Concordance")
ax.set_ylim(0.16, 1.16); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
ax.set_title("d   Weak effects do not transfer", loc="left", pad=6)
ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderpad=0.1)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- e ---
ax = fig.add_subplot(gs[1, 2])
net = pl["network"]
first = [b["spearman"] for b in bins]
ax.scatter(np.zeros(len(first)), first, s=26, color=C_NEU, alpha=0.7, edgecolors="none", zorder=3)
ax.scatter([1], [net["network_spearman"]], s=80, color=C_BAD, marker="D", zorder=3, linewidths=0)
ax.plot([-0.30, 1.42], [0.8473, 0.8473], color=C_GY, lw=1.0, ls="--", zorder=1)
ax.text(1.42, 0.872, "cross-platform Δ", fontsize=8, color=C_GY, ha="right", va="bottom")
ax.set_xticks([0, 1]); ax.set_xticklabels(["first-order\n(effect size)", "second-order\n(network)"], fontsize=8)
ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel("Cross-platform Spearman")
ax.set_title("e   Second-order decays 65%", loc="left", pad=6)
ax.text(1.0, net["network_spearman"] + 0.040, f"{net['network_spearman']:.3f}",
        ha="center", fontsize=8, color=C_BAD, fontweight="bold")
ax.text(0.0, 0.945, f"{np.mean(first):.2f}\n(mean)", ha="center", fontsize=8, color=C_NEU, linespacing=1.3)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig.suptitle("Figure 4 |  The covariance layer is the fragile layer",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig4")
fig.savefig(f"{OUT}/Fig4_covariance_fragility.png", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig4_covariance_fragility.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig4_covariance_fragility.eps", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# Figure 5
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(6.94, 6.05))
fig.subplots_adjust(left=0.215, right=0.965, top=0.905, bottom=0.115, hspace=0.62, wspace=0.42)

# --- a 噪声 SD 分解 ---
ax = axs[0, 0]
sd = [0.0643, 0.0975, 0.2795, 0.3146]
nm = ["Hyperparameter K", "Held-out cohort", "Method pair (all)", "Method pair (relevant)"]
cols = [C_GY, C_NEU, C_PY, C_BAD]
ax.barh(np.arange(4), sd, 0.66, color=cols, edgecolor="white")
for yi, v in zip(np.arange(4), sd):
    ax.text(v + 0.008, yi, f"{v:.4f}", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(4)); ax.set_yticklabels(nm, fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 0.40)
ax.set_xlabel("SD of paired ΔARI")
ax.set_title("a   Noise floor must be measured", loc="left", pad=6)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- b G2 所需样本量 ---
ax = axs[0, 1]
req = g2["required_n"]
rel = [r["rel"] * 100 for r in req]
nn = [r["n_relevant"] for r in req]
ax.plot(rel, nn, color=C_PY, marker="o", lw=1.7, ms=6, zorder=3)
for xx, yy in zip(rel, nn):
    ax.text(xx, yy * 1.16, f"{int(yy)}", ha="center", fontsize=8, color=C_PY)
ax.axhline(22, color=C_BAD, lw=1.3, ls="--", zorder=2)
ax.text(31.5, 26, "current design\nn = 22", fontsize=8, color=C_BAD, ha="right",
        va="bottom", linespacing=1.3)
ax.set_yscale("log"); ax.set_ylim(15, 4200); ax.set_xlim(2.5, 33)
ax.set_yticks([20, 50, 100, 500, 1000, 2000])
ax.set_yticklabels(["20", "50", "100", "500", "1000", "2000"])
ax.set_xticks([5, 10, 15, 20, 25, 30])
ax.set_xlabel("Required improvement over baseline (%)")
ax.set_ylabel("Records of evaluation needed")
ax.set_title("b   G2: +10% needs 479 records", loc="left", pad=6)
ax.axvspan(4, 10, color="#F4E8E8", zorder=0)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- c G3 split-half vs null ---
ax = axs[1, 0]
KS = ["2", "3", "4"]; COH = ["TCGA", "V1", "V2"]
x = np.arange(len(KS)); w = 0.26
for i, (c, col) in enumerate(zip(COH, [C_BAD, C_PY, C_NEU])):
    v = [g3["split_half"][c][k] for k in KS]
    ax.bar(x + (i - 1) * w, v, w, color=col, edgecolor="white", label=f"{c} (n={g3['n'][c]})")
ax.axhline(0.10, color=C_OK, lw=1.3, ls="--")
ax.text(2.44, 0.108, "substantive floor 0.10", fontsize=8, color=C_OK, ha="right", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([f"K = {k}" for k in KS], fontsize=8)
ax.set_ylabel("Within-cohort split-half ARI"); ax.set_ylim(-0.10, 0.20)
ax.set_title("c   G3: reproducibility ceiling ≈ 0", loc="left", pad=6)
_all = [g3["split_half"][c][k] for c in COH for k in KS]
ax.text(0.5, 0.05, f"all cohorts × all K:\n{min(_all):+.3f} to {max(_all):+.3f}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#333333", linespacing=1.35)
ax.legend(loc="upper left", frameon=False, ncol=3, handletextpad=0.3, columnspacing=0.8,
          borderpad=0.1, fontsize=8)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- d G3 退化 ---
ax = axs[1, 1]
M5 = np.array([[g3["bootstrap_ceiling"][c][k]["maxfrac_base"] for k in KS] for c in COH])
im = ax.imshow(M5, cmap="OrRd", aspect="auto", vmin=0.4, vmax=1.0)
for i in range(M5.shape[0]):
    for j in range(M5.shape[1]):
        ax.text(j, i, f"{M5[i,j]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if M5[i, j] > 0.86 else "#222222")
ax.set_xticks(range(3)); ax.set_xticklabels([f"K = {k}" for k in KS], fontsize=8)
ax.set_yticks(range(3)); ax.set_yticklabels(COH, fontsize=8)
ax.set_title("d   G3: every ceiling is degenerate\nblack outline: max cluster fraction > 0.70", loc="left", pad=6, fontsize=8.5)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035); cb.ax.tick_params(labelsize=8)
for i in range(M5.shape[0]):
    for j in range(M5.shape[1]):
        if M5[i, j] > 0.70:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.5))

fig.suptitle("Figure 5 |  Thresholds are derived from noise, not assumed",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig5")
fig.savefig(f"{OUT}/Fig5_thresholds_undecidable.png", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig5_thresholds_undecidable.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig5_thresholds_undecidable.eps", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Fig4 + Fig5 done")
