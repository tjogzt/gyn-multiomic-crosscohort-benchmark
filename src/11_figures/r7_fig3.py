#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig3 | 跨队列方法排序不可分辨"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C_PY, C_R, C_NEU, C_BAD, C_GY = "#3D6BA8", "#C23531", "#177CB0", "#9D2933", "#999999"
EN_METHOD = {
    "1_SNF": "SNF", "2_NMF(concat)": "NMF", "3_KMeans(concat)": "KMeans", "4_PCA(concat)": "PCA",
    "5_谱嵌入(concat)": "Spectral", "6_共联共识": "Co-assoc.", "7_MCCA-lite": "MCCA",
    "8_MOFA-lite": "MOFA", "2_intNMF等价": "intNMF*", "6_MOFA2": "MOFA2",
    "5_iClusterPlus": "iCluster", "3_MCIA等价(MFA)": "MCIA*", "4_mixOmics": "mixOmics",
}
def EN(s):
    s = str(s)
    return EN_METHOD.get(s, s.split("_", 1)[1].replace("(concat)", "").replace("-lite", "") if "_" in s else s)
OUT = "/Users/taozhu/my researches/retyping"; T = "/tmp/gyn_retyping"
sys.path.insert(0, T)
from r5_geomcheck import check
def J(p): return json.load(open(f"{T}/{p}"))

bs, bm, pa = J("bench_summary.json"), J("bench_merged.json"), J("pac_agg.json")
flip = pd.read_csv(f"{T}/g2_champ_flip.csv")
K = "4"
P = pd.DataFrame(J("bench_matrix.json")); P["ari_k"] = P["ari"].map(lambda d: d.get(K) if isinstance(d, dict) else None)
R = pd.read_csv(f"{T}/bench_r_matrix.csv"); R["ari_k"] = R[f"ari_{K}"]
norm = lambda s: "+".join(sorted(str(s).replace("+", " ").split()))
for Df in (P, R):
    Df["sub"] = Df["subset"].map(norm)
    Df["pairn"] = Df["pair"].astype(str).str.replace("×", "x").str.strip()
    Df["dom"] = Df["domain"].map(lambda s: "A" if "CPTAC" in str(s) else "B")
key = ["dom", "sub", "pairn"]
Pa = P.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("PY")
Ra = R.dropna(subset=["ari_k"]).groupby(key)["ari_k"].mean().rename("R")
cm = pd.concat([Pa, Ra], axis=1).dropna()
rho, pv = spearmanr(cm["PY"], cm["R"])
snf_py = P[P["method"].astype(str).str.contains("SNF")]["ari_k"].dropna()
snf_r = R[R["method"].astype(str).str.contains("SNF")]["ari_k"].dropna()
degen = P[P["method"].astype(str).str.contains("SNF")]["degen"].map(
    lambda d: d.get("4") if isinstance(d, dict) else None).dropna()
print(f"[核] 网格 n={len(cm)}  Spearman={rho:.4f} p={pv:.3e}  (报告值 0.4182 / 2.5e-3)")
print(f"[核] SNF Python mean={snf_py.mean():.4f}  R mean={snf_r.mean():.4f}  退化率={degen.mean():.3f}")

fig = plt.figure(figsize=(6.94, 6.28))
gs = fig.add_gridspec(2, 6, hspace=0.60, wspace=1.65, left=0.115, right=0.965, top=0.915, bottom=0.085)

# --- a Python 侧排名 ---
ax = fig.add_subplot(gs[0, 0:3])
py = bs["tab1"]; meth = sorted(py, key=lambda m: py[m]["all"], reverse=True)
y = np.arange(len(meth))
ax.barh(y, [py[m]["all"] for m in meth], 0.68, color=C_PY, edgecolor="white")
for yi, m in zip(y, meth):
    ax.text(py[m]["all"] + 0.010, yi, f"{py[m]['all']:.3f}", va="center", fontsize=8, color=C_PY)
ax.set_yticks(y); ax.set_yticklabels([EN(m) for m in meth], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 0.60)
ax.set_xlabel("Cross-cohort transfer ARI")
ax.set_title("a   Python side (8 self-implemented)", loc="left", pad=6)
ax.text(0.415, 3.10, "SNF is degenerate:\nmax cluster 0.811,\n16% of records",
        fontsize=8, color=C_BAD, ha="left", va="center", linespacing=1.3)
ax.annotate("", xy=(0.045, 6.25), xytext=(0.395, 3.75), fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_BAD, lw=0.9, shrinkA=1, shrinkB=2))
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- b R 侧排名 ---
ax = fig.add_subplot(gs[0, 3:6])
rk = bm["r_ranking"]; rm = sorted(rk, key=lambda m: rk[m], reverse=True)
ax.barh(np.arange(len(rm)), [rk[m] for m in rm], 0.68, color=C_R, edgecolor="white")
for yi, m in zip(np.arange(len(rm)), rm):
    ax.text(rk[m] + 0.010, yi, f"{rk[m]:.3f}", va="center", fontsize=8, color=C_R)
ax.set_yticks(np.arange(len(rm))); ax.set_yticklabels([EN(m) for m in rm], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 0.56)
ax.set_xticks([0, 0.2, 0.4])
ax.set_xlabel("Cross-cohort transfer ARI")
ax.set_title("b   R side (6 official packages)", loc="left", pad=6)
ax.text(0.36, 4.35, "SNF is NOT near-zero\nonce the official\nimplementation is used",
        fontsize=8, color="#3F6B3F", ha="left", va="center", linespacing=1.3)
ax.annotate("", xy=(0.215, 5.0), xytext=(0.345, 4.35), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#3F6B3F", lw=0.9, shrinkA=1, shrinkB=3))
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- c 冠军翻转 ---
ax = fig.add_subplot(gs[1, 0:2])
cnt = flip.groupby("n_unique_winner").size().reindex([1, 2, 3], fill_value=0)
ax.bar(cnt.index, cnt.values, 0.6, color=[C_NEU, C_BAD, C_BAD], edgecolor="white")
for xi, vv in zip(cnt.index, cnt.values):
    if vv: ax.text(xi, vv + 0.35, str(int(vv)), ha="center", fontsize=8, color="#333333")
incons = int((~flip["consistent"]).sum())
ax.set_xticks([1, 2, 3]); ax.set_xlabel("Distinct winners")
ax.set_ylabel("Units (of 22)"); ax.set_ylim(0, 12)
ax.set_title("c   Champion flips", loc="left", pad=6)
ax.text(0.97, 0.94, f"{incons}/{len(flip)} inconsistent\n({incons/len(flip)*100:.0f}%)",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=C_BAD,
        fontweight="bold", linespacing=1.3)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- d 跨实现一致性（真实网格）---
ax = fig.add_subplot(gs[1, 2:4])
ax.plot([0, 0.75], [0, 0.75], color="#BBBBBB", lw=1.0, ls="--", zorder=1)
ax.scatter(cm["PY"], cm["R"], s=20, color=C_NEU, alpha=0.75, edgecolors="none", zorder=2)
ax.set_xlim(0, 0.75); ax.set_ylim(0, 0.75)
ax.set_xlabel("Python ARI"); ax.set_ylabel("R ARI")
ax.set_title("d   Cross-implementation", loc="left", pad=6)
ax.text(0.05, 0.95, f"n = {len(cm)} grids\nSpearman = {rho:.3f}\np = {pv:.4f}",
        transform=ax.transAxes, fontsize=8, va="top", color="#333333", linespacing=1.35)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- e PAC vs 迁移 ---
ax = fig.add_subplot(gs[1, 4:6])
p1 = pa["tab1"]
for m in p1:
    is_snf = "SNF" in m
    col = C_BAD if is_snf else (C_PY if "共联共识" in m else C_GY)
    ax.scatter(p1[m]["all"], bs["tab1"][m]["all"], s=62, color=col,
               marker="*" if is_snf else "o", zorder=3, linewidths=0)
ax.set_xlabel("PAC (instability)"); ax.set_ylabel("Cross-cohort ARI")
ax.set_title("e   Stability predicts transfer", loc="left", pad=6)
ax.text(0.97, 0.95, "Spearman −0.158\n(p = 0.0015)", transform=ax.transAxes, ha="right",
        va="top", fontsize=8, color="#333333", linespacing=1.35)
ax.text(0.03, 0.06, "SNF (red star): lowest PAC\nbut a degenerate solution",
        transform=ax.transAxes, fontsize=8, color=C_BAD, ha="left", va="bottom", linespacing=1.3)
ax.set_xlim(0.10, 0.72); ax.set_ylim(0, 0.52)
ax.set_xticks([0.2, 0.4, 0.6])
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig.suptitle("Figure 3 |  Cross-cohort method ranking is not resolvable",
             x=0.012, y=0.977, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig3")
fig.savefig(f"{OUT}/Fig3_method_not_resolvable.png", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig3_method_not_resolvable.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig3_method_not_resolvable.eps", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Fig3 done")
