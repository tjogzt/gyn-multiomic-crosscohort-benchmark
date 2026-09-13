#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S8 | 数据可得性态势（队列 × 层 × 规模）"""
import sys, json
sys.path.insert(0, "/tmp/gyn_retyping")
from figstyle import *
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

st = pd.DataFrame(J("align_stage1.json"))
c5 = J("ucec_core5.json"); ns = J("ucec_core_nsmp.json")
av = pd.DataFrame(J("bench_availability.json"))
print("[probe] stage1 cols:", list(st.columns)); print(st.head(6).to_string(index=False))
print("[probe] core5:", (len(c5) if isinstance(c5, list) else c5), "| nsmp:", ns)

st["cohort_s"] = st["cohort"].astype(str).str.replace("TCGA-UCEC", "TCGA", regex=False) \
    .str.replace("CPTAC-UCEC-", "", regex=False).str.replace("CPTAC-OV", "OV", regex=False)
st["layer_s"] = st["layer"].astype(str).str.replace("甲基化450K", "Methyl450K", regex=False) \
    .str.replace("(EB++Adjust)", "", regex=False).str.replace("RNAseq", "RNA", regex=False)

fig, axs = plt.subplots(2, 2, figsize=(7.087, 5.74))
fig.subplots_adjust(left=0.155, right=0.960, top=0.905, bottom=0.155, hspace=0.70, wspace=0.62)

# a: 样本 × 特征 规模图
ax = axs[0, 0]
mk = {"symbol": "o", "混合/其它": "s", "探针": "^"}
for kind, g in st.groupby("id_kind"):
    ax.scatter(g["samples"], g["genes"], s=44, label=kind,
               marker=mk.get(kind, "s"), color=C_PY if "symbol" in kind else C_BAD,
               alpha=0.85, zorder=3, linewidths=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([200, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["200", "500", "1k", "2k", "5k", "10k"])
ax.set_yticks([100, 1000, 10000, 100000]); ax.set_yticklabels(["100", "1k", "10k", "100k"])
ax.set_xlim(120, 20000); ax.set_ylim(50, 2.2e6)
ax.set_xlabel("Samples"); ax.set_ylabel("Features")
ax.set_title("a   Data scale spans four orders", loc="left", pad=6)
ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=0.3)
_hi = st.nlargest(1, "genes").iloc[0]
_lo = st.nsmallest(1, "genes").iloc[0]
ax.annotate(f"{_hi['cohort_s'][:6]} {_hi['layer_s'][:9]}", (_hi["samples"], _hi["genes"]),
            textcoords="offset points", xytext=(7, -3), fontsize=8, color="#333333")
ax.annotate(f"{_lo['cohort_s'][:6]} {_lo['layer_s'][:6]}", (_lo["samples"], _lo["genes"]),
            textcoords="offset points", xytext=(7, 4), fontsize=8, color="#333333")
nospines(ax)

# b: 数据体积
ax = axs[0, 1]
g = st.groupby("cohort_s")["mb"].sum().sort_values()
ax.barh(np.arange(len(g)), g.values, 0.64, color=C_NEU, edgecolor="white")
for yi, v in zip(np.arange(len(g)), g.values):
    ax.text(v + 12, yi, f"{v:,.0f} MB", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.index, fontsize=8)
ax.set_xlim(0, max(g.values) * 1.34); ax.set_xlabel("On-disk volume (MB)")
ax.set_title("b   Storage per cohort", loc="left", pad=6)
nospines(ax)

# c: 核心集与 NSMP 塌缩
ax = axs[1, 0]
av2 = av.copy(); av2["nL"] = av2["subset"].map(lambda s: len(str(s).split("+")))
lab, val = [], []
for (d, c) in [("B_EC四层", "TCGA"), ("B_EC四层", "ind"), ("B_EC四层", "dis")]:
    v = av2[(av2.domain == d) & (av2.cohort == c)].groupby("nL")["n"].max()
    lab.append(f"EC {c}"); val.append(int(v.get(4, 0)))
core = len(c5) if isinstance(c5, list) else 0
nsmp = len((ns or {}).get("nsmp_core", [])) if isinstance(ns, dict) else 0
print(f"[probe] core={core} nsmp={nsmp}")
val.append(core); lab.append("TCGA\n5-layer")
val.append(nsmp); lab.append("TCGA\nNSMP")
cols = [C_PY, C_PY, C_PY, C_NEU, C_BAD]
ax.bar(range(len(val)), val, 0.62, color=cols, edgecolor="white")
for xi, v in enumerate(val):
    ax.text(xi, v + 8, str(v), ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(len(val))); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("Samples"); ax.set_ylim(0, max(val) * 1.28)
ax.set_title("c   Analysis-ready sample sets", loc="left", pad=6)
nospines(ax)

# d: 关键事实
ax = axs[1, 1]
ax.axis("off")
ax.text(0.0, 0.99, "Availability summary", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax.text(0.0, 0.87,
        "•  Five public cohorts, 4 omics layers\n"
        "    in the common cross-cohort space\n\n"
        "•  TCGA↔CPTAC sample-ID overlap = 0\n"
        "    (independent validation, not re-split)\n\n"
        "•  Protein usable only as Δ(T−N)\n\n"
        "•  TCGA RPPA ↔ CPTAC overlap = 0\n\n"
        "•  EEEC: 684.8 GB archive → 68.6 MB\n"
        "    via remote central-directory parsing\n"
        "    (0.010% of the original)",
        transform=ax.transAxes, fontsize=8, va="top", linespacing=1.30)

fig.suptitle("Figure S8 |  Data availability and scale",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS8_availability_landscape", "S8")
print("S8 done")
