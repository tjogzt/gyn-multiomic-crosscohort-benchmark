#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""u4：S7 可验证版本——4 面板全部从原始数据/已验证产物重推"""
import sys, json, gzip
sys.path.insert(0, "/tmp/gyn_retyping")
from figstyle import *
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

D = "/Volumes/tjogzt4T/data/xena"; UC = f"{D}/UCEC"; PC = f"{D}/tcgapancan"
def cols(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", errors="replace") as f:
        h = f.readline().rstrip("\n").split("\t")
    return {c.strip('"')[:15].upper() for c in h[1:]
            if c.strip('"').upper().startswith("TCGA-") and c.strip('"').endswith("-01")}

LAY = {"mRNA": cols(f"{PC}/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"),
       "miRNA": cols(f"{UC}/TCGA.UCEC.sampleMap_miRNA_HiSeq_gene.gz"),
       "Methylation": cols(f"{UC}/TCGA.UCEC.sampleMap_HumanMethylation450.gz"),
       "RPPA": cols(f"{UC}/TCGA.UCEC.sampleMap_RPPA.gz"),
       "CNA": cols(f"{UC}/TCGA.UCEC.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz")}
cm = pd.read_csv(f"{UC}/TCGA.UCEC.sampleMap_UCEC_clinicalMatrix", sep="\t", low_memory=False)
UCEC = {s[:15].upper() for s in cm["sampleID"].astype(str) if s.endswith("-01")}
print(f"UCEC 样本宇宙（临床表 -01）= {len(UCEC)}")
LAY = {k: (v & UCEC) for k, v in LAY.items()}
print("限定 UCEC 后的层样本数:", {k: len(v) for k, v in LAY.items()})
CORE = set.intersection(*LAY.values())
print("五层核心集 =", len(CORE), "（既有记录 306）")

# 留一
lo = []
for drop in LAY:
    use = {k: v for k, v in LAY.items() if k != drop}
    inter = set.intersection(*use.values())
    lo.append(dict(dropped=drop, n_all=len(inter), gain=len(inter) - len(CORE)))
LO = pd.DataFrame(lo).sort_values("gain", ascending=False)
print("\n留一（全体样本）："); print(LO.to_string(index=False))

# 生存/随访（对 306 核心集）
sv = pd.read_csv(f"{UC}/survival_UCEC_survival.txt", sep="\t", low_memory=False)
sv["pat15"] = sv["sample"].astype(str).str[:15].str.upper()
sv = sv.drop_duplicates("pat15").set_index("pat15")
inC = sv.loc[sorted(set(sv.index) & CORE)]
os_e = pd.to_numeric(inC["OS"], errors="coerce"); os_t = pd.to_numeric(inC["OS.time"], errors="coerce")
ok = os_e.notna() & os_t.notna()
os_e, os_t = os_e[ok], os_t[ok]
print(f"\n核心集 306 中有生存数据 = {len(os_e)}")
print(f"  OS 事件 = {int(os_e.sum())} ({os_e.mean()*100:.1f}%)")
print(f"  随访中位 = {os_t.median()/365.25:.2f} 年")
print(f"  随访 <3 年占比 = {(os_t < 1095.75).mean()*100:.1f}%")

# G3 已验证数值
g3 = J("g3_power.json")
NS = {"TCGA": g3["n"]["TCGA"], "V1": g3["n"]["V1"], "V2": g3["n"]["V2"]}

fig, axs = plt.subplots(2, 2, figsize=(7.087, 5.88))
fig.subplots_adjust(left=0.185, right=0.965, top=0.905, bottom=0.145, hspace=0.68, wspace=0.56)

# a: 层样本数与交集
ax = axs[0, 0]
nm = list(LAY) + ["5-layer\nintersection"]
vl = [len(LAY[k]) for k in LAY] + [len(CORE)]
ax.barh(np.arange(6), vl, 0.66,
        color=[C_GY] * 5 + [C_BAD], edgecolor="white")
for yi, v in zip(np.arange(6), vl):
    ax.text(v + 8, yi, str(v), va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(6)); ax.set_yticklabels(nm, fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 640); ax.set_xlabel("Primary-tumour samples")
ax.set_title("a   Five-way intersection", loc="left", pad=6)
ax.text(0.030, 0.085, "set by miRNA\n+ methylation", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8, color="white", linespacing=1.35)
nospines(ax)

# b: 留一增益
ax = axs[0, 1]
ax.barh(np.arange(len(LO)), LO["gain"], 0.64,
        color=[C_BAD if g == LO["gain"].max() else C_PY for g in LO["gain"]], edgecolor="white")
for yi, (g, n) in enumerate(zip(LO["gain"], LO["n_all"])):
    ax.text(g + 3, yi, f"+{g}  →  {n}", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(LO))); ax.set_yticklabels(["drop " + d for d in LO["dropped"]], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, max(LO["gain"]) * 1.75)
ax.set_xlabel("Samples gained (all primary tumours)")
ax.set_title("b   Cost of each omitted layer", loc="left", pad=6)
nospines(ax)

# c: 队列规模级联（G3 已验证数值）
ax = axs[1, 0]
lab = ["5-layer\ncore set", "NSMP\n(TCGA)", "NSMP\n(V2)", "NSMP\n(V1)"]
val = [len(CORE), NS["TCGA"], NS["V2"], NS["V1"]]
ax.bar(range(4), val, 0.62, color=[C_NEU, C_BAD, C_PY, C_PY], edgecolor="white")
for xi, v in zip(range(4), val):
    ax.text(xi, v + 9, str(v), ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("Samples"); ax.set_ylim(0, 380)
ax.set_title("c   Size of the analysis sets", loc="left", pad=6)
nospines(ax)

# d: 生存数据可得性
ax = axs[1, 1]
ax.axis("off")
ax.text(0.0, 0.99, "Why no survival endpoint", transform=ax.transAxes, fontsize=9,
        fontweight="bold", va="top")
ax.text(0.0, 0.87,
        f"•  5-layer core set: n = {len(CORE)}\n"
        f"•  With survival data: n = {len(os_e)}\n"
        f"•  OS events: {int(os_e.sum())} ({os_e.mean()*100:.1f}%)\n"
        f"•  Median follow-up: {os_t.median()/365.25:.2f} years\n"
        f"•  Follow-up < 3 years: {(os_t < 1095.75).mean()*100:.0f}%",
        transform=ax.transAxes, fontsize=8, va="top", linespacing=1.42)
ax.text(0.0, 0.42, "→  Event count far below what a\n     multivariate model needs.",
        transform=ax.transAxes, fontsize=8, va="top", color="#333333", linespacing=1.34)
ax.text(0.0, 0.02, "No survival or prognostic endpoint\nis reported anywhere in this work.",
        transform=ax.transAxes, fontsize=8, va="bottom", color=C_BAD, linespacing=1.30)

fig.suptitle("Figure S7 |  Sample availability and the limits it imposes",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS7_availability_limits", "S7")

json.dump(dict(layer_n={k: len(v) for k, v in LAY.items()}, core=len(CORE),
               leave_one_out=LO.to_dict("records"),
               survival=dict(n_surv=int(len(os_e)), os_events=int(os_e.sum()),
                             median_fu_years=round(float(os_t.median() / 365.25), 2),
                             pct_under_3y=round(float((os_t < 1095.75).mean() * 100), 1))),
          open(f"{T}/s7_final.json", "w"), ensure_ascii=False, indent=1)
print("\nS7 done")
