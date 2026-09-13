#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6 | EEEC group-level semantic decoding（数据自证）"""
import sys, json
sys.path.insert(0, "/tmp/gyn_retyping")
from figstyle import *
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

meta = pd.read_csv(f"{T}/eeec_batch/sample_meta.csv")
pc = pd.read_csv(f"{T}/eeec_batch/pc_assoc.csv")
mat = pd.read_csv(f"{T}/eeec_batch/matrix_log2.csv", index_col=0)
meta = meta[meta["col"].isin(mat.columns)].reset_index(drop=True)
sub = mat[meta["col"].tolist()]
fams = sorted(meta["family"].unique())
print("[probe] families:", fams, "| counts:", meta["family"].value_counts().to_dict())
print("[probe] batch:", meta["batch"].value_counts().to_dict(), "| cond:", meta["cond"].value_counts().to_dict())

fig, axs = plt.subplots(2, 2, figsize=(7.087, 5.88))
fig.subplots_adjust(left=0.165, right=0.960, top=0.905, bottom=0.155, hspace=0.68, wspace=0.62)

# a: 组均值谱相关矩阵
ax = axs[0, 0]
gm = pd.DataFrame({g: sub.loc[:, meta.loc[meta.family == g, "col"].tolist()].mean(axis=1) for g in fams})
R = gm.corr(method="spearman")
im = ax.imshow(R.values, cmap="RdYlBu_r", vmin=0.55, vmax=1.0, aspect="auto")
for i in range(len(fams)):
    for j in range(len(fams)):
        ax.text(j, i, f"{R.values[i,j]:.3f}", ha="center", va="center", fontsize=8, color="#222222")
ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=8)
ax.set_title("a   Group-mean spectra", loc="left", pad=6)
cbar(im, ax)

# b: PC 与条件/批次
ax = axs[0, 1]
n = min(8, len(pc)); x = np.arange(n)
ax.bar(x - 0.19, pc["AUROC_vs_cond"][:n], 0.38, color=C_PY, edgecolor="white", label="vs condition")
ax.bar(x + 0.19, pc["AUROC_vs_batch"][:n], 0.38, color=C_GY, edgecolor="white", label="vs batch")
ax.axhline(0.5, color="#666666", lw=1.0, ls=":")
ax.set_xticks(x); ax.set_xticklabels([f"PC{i+1}" for i in range(n)], fontsize=8)
ax.set_ylim(0.4, 1.12); ax.set_yticks([0.5, 0.75, 1.0]); ax.set_ylabel("AUROC")
ax.set_title("b   PC1 recovers the condition axis", loc="left", pad=6)
ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.4, columnspacing=0.9, fontsize=8)
ax.text(0.03, 0.66, f"PC1: cond {pc['AUROC_vs_cond'][0]:.3f}\n      batch {pc['AUROC_vs_batch'][0]:.3f}",
        transform=ax.transAxes, fontsize=8, va="top", color=C_BAD, linespacing=1.45)
nospines(ax)

# c: 组内 vs 跨组样本相关
ax = axs[1, 0]
gg = meta["family"].tolist(); cols = meta["col"].tolist()
Cz = np.corrcoef(np.asarray(sub.values, dtype=float).T)
within, cross = [], []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        (within if gg[i] == gg[j] else cross).append(Cz[i, j])
bp = ax.boxplot([within, cross], widths=0.5, patch_artist=True,
                boxprops=dict(facecolor=C_PY, alpha=0.72, linewidth=0.7),
                medianprops=dict(color=C_BAD, lw=1.4), whiskerprops=dict(lw=0.7),
                capprops=dict(lw=0.7),
                flierprops=dict(marker=".", ms=2.5, markerfacecolor=C_GY, markeredgecolor="none"))
ax.set_xticks([1, 2]); ax.set_xticklabels(["within\ngroup", "across\ngroups"], fontsize=8)
ax.set_ylabel("Sample–sample correlation")
ax.set_title("c   Group structure is real", loc="left", pad=6)
ax.text(0.97, 0.05, f"within {np.mean(within):.3f}\nacross {np.mean(cross):.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#333333", linespacing=1.45)
nospines(ax)

# d: 方差解释
ax = axs[1, 1]
ve = pc["var_exp"][:n] * 100
ax.bar(x, ve, 0.62, color=[C_BAD if i == 0 else C_GY for i in range(n)], edgecolor="white")
for xi, v in zip(x, ve):
    ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(x); ax.set_xticklabels([f"PC{i+1}" for i in range(n)], fontsize=8)
ax.set_ylabel("Variance explained (%)"); ax.set_ylim(0, max(ve) * 1.30)
ax.set_title("d   One axis dominates", loc="left", pad=6)
ax.text(0.97, 0.95, "decoded from the data alone:\nno per-sample phenotype key\nwas publicly available",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#333333", linespacing=1.45)
nospines(ax)

fig.suptitle("Figure S6 |  EEEC group-level semantic decoding",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS6_eeec_semantics", "S6")
print("S6 done")
