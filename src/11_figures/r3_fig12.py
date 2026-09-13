#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig1 + Fig2（修订版：全部字号 ≥8pt、消除标签碰撞与越界）
配色语义（全局统一）：
  C_PY  石青 #3D6BA8 = Python 自实现工具链     C_R   朱砂 #C23531 = R 规范实现工具链
  C_NEU 靛青 #177CB0 = 中性/参照               C_BAD 胭脂 #9D2933 = 失效/退化/不可判
  C_GY  灰   #999999 = 辅助
"""
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
EN_METHOD = {
    "1_SNF": "SNF", "2_NMF(concat)": "NMF", "3_KMeans(concat)": "KMeans",
    "4_PCA(concat)": "PCA", "5_谱嵌入(concat)": "Spectral", "6_共联共识": "Co-assoc.",
    "7_MCCA-lite": "MCCA", "8_MOFA-lite": "MOFA",
    "2_intNMF等价": "intNMF*", "6_MOFA2": "MOFA2", "5_iClusterPlus": "iCluster",
    "3_MCIA等价(MFA)": "MCIA*", "4_mixOmics": "mixOmics",
}
def EN(s):
    s = str(s)
    if s in EN_METHOD: return EN_METHOD[s]
    return s.split("_", 1)[1].replace("(concat)", "").replace("-lite", "") if "_" in s else s

OUT = "/Users/taozhu/my researches/retyping"; T = "/tmp/gyn_retyping"
sys.path.insert(0, T)
from r5_geomcheck import check
def J(p): return json.load(open(f"{T}/{p}"))
def cbar(im, ax, label=None):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.ax.tick_params(labelsize=8)
    return cb

av  = pd.DataFrame(J("bench_availability.json"))
ac  = J("align_concordance.json")
b3c = J("bridge_3cohort.json")
b3  = J("b3_anchors.json")
bs  = J("bench_summary.json")
bm  = J("bench_merged.json")

# ═══════════════════════════════════════════════════════════════════
# Figure 1
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(6.94, 6.14))
fig.subplots_adjust(left=0.145, right=0.955, top=0.915, bottom=0.095, hspace=0.60, wspace=0.52)

# --- a ---
ax = axs[0, 0]
av2 = av.copy(); av2["nL"] = av2["subset"].map(lambda s: len(str(s).split("+")))
piv = av2.groupby(["domain", "cohort", "nL"])["n"].max().reset_index()
rowlab = [("B_EC四层", "TCGA"), ("B_EC四层", "ind"), ("B_EC四层", "dis"),
          ("A_CPTAC三队列", "ind"), ("A_CPTAC三队列", "dis"), ("A_CPTAC三队列", "OV")]
disp = {"B_EC四层": "EC", "A_CPTAC三队列": "CPTAC"}
names = [f"{disp[d]} {c}" for d, c in rowlab]
M = np.full((len(rowlab), 4), np.nan)
for i, (d, c) in enumerate(rowlab):
    for _, r in piv[(piv.domain == d) & (piv.cohort == c)].iterrows():
        M[i, int(r["nL"]) - 1] = r["n"]
im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=560)
for i in range(M.shape[0]):
    for j in range(4):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{int(M[i,j])}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 330 else "#222222")
        else:
            ax.text(j, i, "–", ha="center", va="center", fontsize=8.5, color=C_GY)
ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"])
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Integrated layers"); ax.set_ylabel("")
ax.set_title("a   Availability collapses with layers", loc="left", pad=6)
ax.axhline(2.5, color="white", lw=1.8)
for i, (d, c) in enumerate(rowlab):
    if d == "B_EC四层":
        v = M[i]; eff = v[np.isfinite(v)]
        if len(eff) > 1:
            ax.text(3.72, i, f"−{(1-eff[-1]/eff[0])*100:.0f}%", va="center", ha="left",
                    fontsize=8, color=C_BAD, fontweight="bold")
ax.set_xlim(-0.5, 4.02)
cbar(im, ax)

# --- b ---
ax = axs[0, 1]
rows = []
for pair, v in ac["表达"].items(): rows.append(("mRNA", v["rho"], pair))
for pair, v in ac["拷贝数"].items(): rows.append(("CNA", v["rho"], pair))
for pair, v in ac["甲基化"].items(): rows.append(("Methyl.\n(CPTAC)", v["rho"], pair))
for pair, v in b3c["raw"].items(): rows.append(("Protein\n(raw)", v["rho"], pair))
for pair, v in b3c["delta"].items(): rows.append(("Protein\nΔ(T−N)", v["rho"], pair))
D = pd.DataFrame(rows, columns=["layer", "rho", "pair"])
D["inner"] = D["pair"].str.contains("CPTAC-")
order = ["mRNA", "CNA", "Methyl.\n(CPTAC)", "Protein\n(raw)", "Protein\nΔ(T−N)"]
for i, lay in enumerate(order):
    sub = D[D.layer == lay]
    ax.plot([sub["rho"].min(), sub["rho"].max()], [i, i], color="#CCCCCC", lw=1.6, zorder=1)
    inn = sub["inner"].values
    ax.scatter(sub.loc[inn, "rho"], np.full(inn.sum(), i), s=22, color=C_NEU, marker="o",
               zorder=3, linewidths=0)
    ax.scatter(sub.loc[~inn, "rho"], np.full((~inn).sum(), i), s=24, color=C_BAD, marker="s",
               zorder=3, linewidths=0)
ax.axvline(0, color="#555555", lw=0.8, ls=":")
ax.axvspan(0.4, 1.10, color="#E8F0E8", zorder=0)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
ax.invert_yaxis(); ax.set_xlim(-0.40, 1.14)
ax.set_xticks([-0.3, 0, 0.3, 0.6, 0.9])
ax.set_xlabel("Cross-cohort Spearman ρ")
ax.set_title("b   Comparability differs by layer", loc="left", pad=6)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_NEU, ms=5.5, label="within CPTAC"),
                   Line2D([], [], marker="s", ls="", color=C_BAD, ms=5.5, label="TCGA ↔ CPTAC")],
          loc="lower left", frameon=False, handletextpad=0.3, borderpad=0.1)
ax.text(0.75, 3.72, "usable (ρ ≥ 0.4)", fontsize=8, color="#3F6B3F", ha="center", va="center")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- c ---
ax = axs[1, 0]
pr = [("dis × ov", "dis × ov", "Δdis × Δov"), ("dis × ind", "dis × ind", "Δdis × Δind"),
      ("ov × ind", "ov × ind", "Δov × Δind")]
labs = [p[0] for p in pr]
raw = [b3c["raw"][p[1]]["rho"] for p in pr]
dlt = [b3c["delta"][p[2]]["rho"] for p in pr]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, raw, w, color=C_GY, edgecolor="white", label="raw TMT ratio")
ax.bar(x + w/2, dlt, w, color=C_PY, edgecolor="white", label="Δ(T−N)")
for xi, v in zip(x - w/2, raw):
    ax.text(xi, v + (0.030 if v >= 0 else -0.088), f"{v:.3f}", ha="center", fontsize=8,
            color="#555555", va="bottom" if v >= 0 else "top")
for xi, v in zip(x + w/2, dlt):
    ax.text(xi, v + 0.030, f"{v:.3f}", ha="center", fontsize=8, color=C_PY,
            fontweight="bold", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("Spearman ρ (protein)"); ax.set_ylim(-0.52, 0.92)
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
ax.set_title("c   Δ(T−N) restores comparability", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderpad=0.1)
ax.text(2.02, -0.40, "ov × ind:\n−0.066 → 0.412", fontsize=8, color=C_BAD,
        ha="center", va="center", linespacing=1.30)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- d ---
ax = axs[1, 1]
anch = [("Discovery", b3["dis_raw_Prot_RNA"]["rho"], b3["dis_dProt_dRNA"]["rho"]),
        ("Independ.", b3["ind_raw_Prot_RNA"]["rho"], b3["ind_dProt_dRNA"]["rho"])]
x = np.arange(2)
ax.bar(x - w/2, [a[1] for a in anch], w, color=C_GY, edgecolor="white", label="raw")
ax.bar(x + w/2, [a[2] for a in anch], w, color=C_PY, edgecolor="white", label="Δ(T−N)")
for xi, a in zip(x - w/2, anch):
    ax.text(xi, a[1] + (0.032 if a[1] >= 0 else -0.090), f"{a[1]:.3f}", ha="center", fontsize=8,
            color="#555555", va="bottom" if a[1] >= 0 else "top")
for xi, a in zip(x + w/2, anch):
    ax.text(xi, a[2] + 0.032, f"{a[2]:.3f}", ha="center", fontsize=8, color=C_PY,
            fontweight="bold", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
ax.axhspan(0.40, 0.62, color="#E8F0E8", zorder=0)
ax.text(1.66, 0.51, "literature\ntypical\n0.40–0.62", fontsize=8, color="#3F6B3F",
        va="center", ha="center")
ax.set_xticks(x); ax.set_xticklabels([a[0] for a in anch], fontsize=8)
ax.set_ylabel("Spearman ρ (mRNA × protein)")
ax.set_ylim(-0.22, 0.86); ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlim(-0.52, 1.96)
ax.set_title("d   Biological anchor validates Δ", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderpad=0.1)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig.suptitle("Figure 1 |  Cohort availability and cross-cohort comparability",
             x=0.012, y=0.977, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig1")
fig.savefig(f"{OUT}/Fig1_availability_comparability.png", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig1_availability_comparability.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig1_availability_comparability.eps", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# Figure 2
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(6.94, 6.14))
fig.subplots_adjust(left=0.105, right=0.955, top=0.915, bottom=0.115, hspace=0.58, wspace=0.56)

PY = {d: [bs["tab3"][k][i]["ari"] for i in range(len(bs["tab3"][k]))]
      for d, k in [("A", "A_CPTAC三队列"), ("B", "B_EC四层")]}
RL = {"A": [bm["r_layer_effect"]["A|1"], bm["r_layer_effect"]["A|2"], bm["r_layer_effect"]["A|3"]],
      "B": [bm["r_layer_effect"]["B|1"], bm["r_layer_effect"]["B|2"], bm["r_layer_effect"]["B|3"],
            bm["r_layer_effect"]["B|4"]]}

for pi, (dom, ax) in enumerate(zip(["A", "B"], axs[0])):
    series = [(PY[dom], C_PY, "o", "-", "Python (8 methods)", 0.024, "bottom"),
              (RL[dom], C_R, "s", "--", "R (6 packages)", -0.030, "top")]
    xs = [np.arange(1, len(v) + 1) for v, *_ in series]
    for (vals, col, mk, ls, lab, dy, va) in series:
        xs = np.arange(1, len(vals) + 1)
        ax.plot(xs, vals, color=col, marker=mk, ls=ls, lw=1.6, ms=6, label=lab, zorder=3)
        for xi, vv in zip(xs, vals):
            ax.text(xi, vv + dy, f"{vv:.3f}", ha="center", fontsize=8, color=col, va=va)
    ax.set_xticks(np.arange(1, 5)); ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("Integrated layers"); ax.set_ylabel("Cross-cohort transfer ARI")
    ttl = "a" if dom == "A" else "b"
    nm = "CPTAC three-cohort" if dom == "A" else "TCGA + CPTAC-UCEC"
    ax.set_title(f"{ttl}   Domain {dom} ({nm})", loc="left", pad=6)
    ax.set_ylim(0, 0.66)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.3)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- c ---
ax = axs[1, 0]
for (c, col, mk, off) in [("TCGA", C_BAD, "o", 7), ("ind", C_PY, "s", 6), ("dis", C_NEU, "^", -11)]:
    i = rowlab.index(("B_EC四层", c))
    v = [M[i, j] if np.isfinite(M[i, j]) else np.nan for j in range(4)]
    ax.plot(np.arange(1, 5), v, color=col, marker=mk, lw=1.6, ms=6, label=f"EC {c}", zorder=3)
    for xi, vv in zip(np.arange(1, 5), v):
        if np.isfinite(vv):
            ax.annotate(f"{int(vv)}", xy=(xi, vv), xytext=(0, off), textcoords="offset points",
                        ha="center", fontsize=8, color=col)
ax.set_xticks(np.arange(1, 5)); ax.set_xticklabels(["1", "2", "3", "4"])
ax.set_xlabel("Integrated layers"); ax.set_ylabel("Samples with all layers")
ax.set_ylim(0, 660); ax.set_yticks([0, 200, 400, 600])
ax.set_title("c   Sample intersection collapse", loc="left", pad=6)
ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.1)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

# --- d ---
ax = axs[1, 1]
tb = pd.DataFrame(J("bench_table_B.json"))
ABBR = {"mRNA": "R", "miRNA": "m", "CNA": "C", "meth": "M"}
def compact(s):
    return "+".join(ABBR.get(x, x) for x in s.split("+"))
cols_ord = sorted(tb.columns, key=lambda s: (len(s.split("+")), s))
H = tb[cols_ord].T.values.astype(float)
im = ax.imshow(H, cmap="RdYlBu_r", aspect="auto", vmin=-0.05, vmax=0.85)
ax.set_xticks(range(len(tb.index)))
ax.set_xticklabels([EN(m).replace("Spectral", "Spec").replace("Co-assoc.", "CoAssoc") for m in tb.index],
                   rotation=65, ha="right", fontsize=8)
ax.set_yticks(range(len(cols_ord)))
ax.set_yticklabels([f"{len(s.split('+'))}L  {compact(s)}" for s in cols_ord], fontsize=8)
ax.set_title("d   Layer subset × method (Domain B)\nR = mRNA   m = miRNA   C = CNA   M = meth",
             loc="left", pad=6, fontsize=8.5)
cbar(im, ax)
best = np.nanargmax(H, axis=1)
for i, j in enumerate(best):
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.4))

fig.suptitle("Figure 2 |  More omics layers do not improve cross-cohort transfer",
             x=0.012, y=0.977, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig2")
fig.savefig(f"{OUT}/Fig2_layer_negative_marginal.png", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig2_layer_negative_marginal.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(f"{OUT}/Fig2_layer_negative_marginal.eps", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Done.")
