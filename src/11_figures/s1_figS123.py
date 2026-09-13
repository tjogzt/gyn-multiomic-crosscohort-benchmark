#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1–S3 补充图（英文，统一 figstyle）"""
import sys, json
sys.path.insert(0, "/tmp/gyn_retyping")
from figstyle import *
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# ══════════════ S1 | Layer harmonisation & numeric domains ══════════════
ad = J("align_domains.json"); aid = J("align_ids.json")
apm = J("align_probe_map.json"); adups = J("align_dups.json")
fig, axs = plt.subplots(2, 2, figsize=(7.087, 5.88))
fig.subplots_adjust(left=0.145, right=0.965, top=0.905, bottom=0.135, hspace=0.70, wspace=0.75)

# a: ID 体系普查
ax = axs[0, 0]
KIND_EN = {"Entrez数字": "Entrez ID", "symbol样": "symbol-like", "450K探针": "450K probe",
           "其它/位点ID": "other / locus ID", "miRNA名": "miRNA name", "GISTIC峰": "GISTIC peak"}
sel = ["TCGA mRNA(EB++)", "TCGA mRNA(Hugo)", "TCGA 450K", "TCGA RPPA", "TCGA CNA"]
kinds = ["symbol-like", "Entrez ID", "450K probe", "other / locus ID", "miRNA name"]
cols = [C_PY, C_BAD, C_NEU, C_GY, C_OK]
M = np.zeros((len(sel), len(kinds)))
for i, k in enumerate(sel):
    cen = aid[k]["census"]
    tot = sum(cen.values())
    for j, kk in enumerate(kinds):
        zh = [z for z, e in KIND_EN.items() if e == kk]
        M[i, j] = sum(cen.get(z, 0) for z in zh) / tot * 100
left = np.zeros(len(sel))
for j, kk in enumerate(kinds):
    ax.barh(np.arange(len(sel)), M[:, j], 0.66, left=left, color=cols[j], edgecolor="white", label=kk)
    left += M[:, j]
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(["mRNA (EB++)", "mRNA (Hugo)", "Methylation 450K", "RPPA", "CNA"], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of identifiers (%)")
ax.set_title("a   ID systems are not interchangeable", loc="left", pad=6)
ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=0.3, labelspacing=0.28)
nospines(ax)

# b: probe → gene 映射
ax = axs[0, 1]
pv = [apm["ucec_probes"], apm["mapped"], apm["genes"]]
lb = ["450K probes\nin UCEC", "mapped to\ngene symbol", "unique\ngenes"]
ax.bar(range(3), pv, 0.60, color=[C_GY, C_PY, C_NEU], edgecolor="white")
for xi, v in enumerate(pv):
    ax.text(xi, v * 1.10, f"{v:,}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(3)); ax.set_xticklabels(lb, fontsize=8)
ax.set_yscale("log"); ax.set_ylim(1e3, 4e6)
ax.set_yticks([1e3, 1e4, 1e5, 1e6]); ax.set_yticklabels(["1k", "10k", "100k", "1M"])
ax.set_ylabel("Count (log scale)")
ax.set_title("b   Probe→gene mapping loses 29%", loc="left", pad=6)
ax.text(1.0, 1.35e6, f"{apm['mapped']/apm['ucec_probes']*100:.1f}% mapped",
        ha="center", fontsize=8, color=C_OK, fontweight="bold")
nospines(ax)

# c: 数值域
ax = axs[1, 0]
rows = [r for r in ad if r["layer"] in ("mRNA", "甲基化450K", "RPPA", "CNA", "RNAseq")][:9]
SHORT_COH = {"TCGA-UCEC": "TCGA", "CPTAC-UCEC-ind": "ind", "CPTAC-UCEC-dis": "dis",
             "CPTAC-OV": "OV", "CPTAC-UCEC-OV": "OV", "CPTAC-OV-pro": "OV"}
SHORT_LAY = {"mRNA": "mRNA", "甲基化450K": "Meth450K", "RPPA": "RPPA", "CNA": "CNA",
             "RNAseq": "RNA", "Proteome": "Prot", "Phospho": "Phos", "Acetyl": "Ac", "Meth": "Meth", "miRNA": "miR"}
labs, lo, hi, mid = [], [], [], []
for r in rows:
    labs.append(f"{SHORT_COH.get(r['cohort'], r['cohort'][:8])} {SHORT_LAY.get(r['layer'], r['layer'][:8])}")
    lo.append(r["q05"]); hi.append(r["q95"]); mid.append(r["q50"])
y = np.arange(len(labs))
ax.hlines(y, lo, hi, color=C_NEU, lw=3.2, alpha=0.55)
ax.plot(mid, y, "o", color=C_BAD, ms=5, zorder=3)
ax.axvline(0, color="#555555", lw=0.8, ls=":")
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("Value (5th / 50th / 95th percentile)")
ax.set_xticks([0, 10, 20])
ax.set_title("c   Numeric domains differ across layers", loc="left", pad=6)
nospines(ax)

# d: 重复 ID 普查
ax = axs[1, 1]
dk = list(adups.keys()); dv = [adups[k]["dup"] for k in dk]
ax.barh(np.arange(len(dk)), [max(v, 0) for v in dv], 0.66,
        color=[C_OK if v == 0 else C_BAD for v in dv], edgecolor="white")
for yi, v in zip(np.arange(len(dk)), dv):
    ax.text(max(v, 0) + 0.02, yi, f"{int(v)}", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(dk))); ax.set_yticklabels(dk, fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 1.4); ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel("Duplicate gene identifiers")
ax.set_title("d   No duplicate IDs anywhere", loc="left", pad=6)
ax.text(0.98, 0.06, "all audits return 0", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=C_OK, fontweight="bold")
nospines(ax)

fig.suptitle("Figure S1 |  Layer harmonisation and numeric domains",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS1_harmonisation_domains", "S1")

# ══════════════ S2 | Protein layer bridge ══════════════
b3c = J("bridge_3cohort.json"); bd = J("bridge_diag.json"); b3 = J("b3_anchors.json")
ac = J("align_concordance.json")
fig, axs = plt.subplots(2, 2, figsize=(7.087, 6.02))
fig.subplots_adjust(left=0.135, right=0.965, top=0.905, bottom=0.135, hspace=0.66, wspace=0.50)

# a: raw vs Δ
ax = axs[0, 0]
pk = [("dis × ov", "dis × ov", "Δdis × Δov"), ("dis × ind", "dis × ind", "Δdis × Δind"),
      ("ov × ind", "ov × ind", "Δov × Δind")]
x = np.arange(3); w = 0.36
raw = [b3c["raw"][p[1]]["rho"] for p in pk]; dlt = [b3c["delta"][p[2]]["rho"] for p in pk]
ax.bar(x - w/2, raw, w, color=C_GY, edgecolor="white", label="raw TMT ratio")
ax.bar(x + w/2, dlt, w, color=C_PY, edgecolor="white", label="Δ(T−N)")
for xi, v in zip(x - w/2, raw):
    ax.text(xi, v + (0.028 if v >= 0 else -0.082), f"{v:.3f}", ha="center", fontsize=8,
            color="#555555", va="bottom" if v >= 0 else "top")
for xi, v in zip(x + w/2, dlt):
    ax.text(xi, v + 0.028, f"{v:.3f}", ha="center", fontsize=8, color=C_PY, fontweight="bold", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pk], fontsize=8)
ax.set_ylabel("Spearman ρ (protein)"); ax.set_ylim(-0.38, 0.90); ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
ax.set_title("a   Δ(T−N) across three cohorts", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderpad=0.1)
nospines(ax)

# b: 修剪检验
ax = axs[0, 1]
lb = ["raw\n(all genes)", "Δ(T−N)\n(all genes)", "raw\n(trimmed 80%)"]
vv = [bd["raw_dis_T_ov_T"]["rho"], bd["delta_dis_ov"]["rho"], bd["trim_dis_ov"]["rho"]]
ax.bar(range(3), vv, 0.60, color=[C_GY, C_PY, C_NEU], edgecolor="white")
for xi, v in zip(range(3), vv):
    ax.text(xi, v + 0.020, f"{v:.3f}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(3)); ax.set_xticklabels(lb, fontsize=8)
ax.set_ylabel("Spearman ρ (dis × ov)"); ax.set_ylim(0, 0.60)
ax.set_title("b   Trimming cannot explain it", loc="left", pad=6)
nospines(ax)

# c: 生物学锚
ax = axs[1, 0]
anc = [("Discovery", b3["dis_raw_Prot_RNA"]["rho"], b3["dis_dProt_dRNA"]["rho"]),
       ("Independent", b3["ind_raw_Prot_RNA"]["rho"], b3["ind_dProt_dRNA"]["rho"])]
x = np.arange(2)
ax.bar(x - w/2, [a[1] for a in anc], w, color=C_GY, edgecolor="white", label="raw")
ax.bar(x + w/2, [a[2] for a in anc], w, color=C_PY, edgecolor="white", label="Δ(T−N)")
for xi, a in zip(x - w/2, anc):
    ax.text(xi, a[1] + (0.030 if a[1] >= 0 else -0.085), f"{a[1]:.3f}", ha="center", fontsize=8,
            color="#555555", va="bottom" if a[1] >= 0 else "top")
for xi, a in zip(x + w/2, anc):
    ax.text(xi, a[2] + 0.030, f"{a[2]:.3f}", ha="center", fontsize=8, color=C_PY, fontweight="bold", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
ax.axhspan(0.40, 0.62, color=C_SPAN, zorder=0)
ax.text(1.62, 0.51, "literature\ntypical", fontsize=8, color=C_OK, va="center", ha="center", linespacing=1.3)
ax.set_xticks(x); ax.set_xticklabels([a[0] for a in anc], fontsize=8)
ax.set_ylabel("Spearman ρ (mRNA × protein)"); ax.set_ylim(-0.22, 0.86)
ax.set_xlim(-0.52, 1.92); ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_title("c   mRNA–protein anchor", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.1)
nospines(ax)

# d: 甲基化粒度
ax = axs[1, 1]
mm = [("CPTAC\ninternal", ac["甲基化"]["CPTAC-ind|CPTAC-dis"]["rho"], C_PY),
      ("TCGA vs\nCPTAC-ind", b3["meth_TCGA_ind"]["rho"], C_BAD),
      ("TCGA vs\nCPTAC-dis", b3["meth_TCGA_dis"]["rho"], C_BAD)]
ax.bar(range(3), [m[1] for m in mm], 0.60, color=[m[2] for m in mm], edgecolor="white")
for xi, m in enumerate(mm):
    ax.text(xi, m[1] + 0.022, f"{m[1]:.3f}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(3)); ax.set_xticklabels([m[0] for m in mm], fontsize=8)
ax.axhline(0.4, color=C_OK, lw=1.2, ls="--")
ax.text(2.45, 0.418, "usability floor 0.4", fontsize=8, color=C_OK, ha="right", va="bottom")
ax.set_ylabel("Spearman ρ (methylation)"); ax.set_ylim(0, 0.86)
ax.set_title("d   Methylation degrades across platforms", loc="left", pad=6)
nospines(ax)

fig.suptitle("Figure S2 |  Protein bridge diagnosis and layer-specific degradation",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS2_protein_bridge", "S2")

# ══════════════ S3 | Alignment strategies ══════════════
ar = J("ari_agg.json")
fig, axs = plt.subplots(2, 2, figsize=(7.087, 7.20))
fig.subplots_adjust(left=0.195, right=0.955, top=0.925, bottom=0.125, hspace=0.80, wspace=0.62)

# a: 层 × 策略 热图
ax = axs[0, 0]
agg = pd.DataFrame(ar["agg_K4"]).T
order = [c for c in agg.columns]
REN = {"S0_基线": "S0 base", "S1_分位数标准化": "S1 quant", "S2_Δ参考校正": "S2 Δ-ref",
       "S3_ComBat(池化)": "S3 ComBat"}
H = agg.values.astype(float)
im = ax.imshow(np.nan_to_num(H, nan=0.0), cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=0.80)
ax.set_xticks(range(len(order))); ax.set_xticklabels([REN.get(c, c) for c in order], rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(agg.index))); ax.set_yticklabels([EN_LAYER.get(i, i) for i in agg.index], fontsize=8)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if np.isnan(H[i, j]): ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#666666")
        else: ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center", fontsize=8, color="#222222")
ax.set_title("a   Layer × alignment strategy (K=4)", loc="left", pad=6)
cbar(im, ax)

# b: S1 相对 S0 的逐对变化
ax = axs[0, 1]
det = pd.DataFrame(ar["detail"])
det["d"] = det["S1_分位数标准化"] - det["S0_基线"]
det = det.sort_values("d")
y = np.arange(len(det))
ax.barh(y, det["d"], 0.72, color=[C_R if v < 0 else C_PY for v in det["d"]], edgecolor="white")
ax.axvline(0, color="#444444", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels([f"{r.layer[:4]} {r.pair}" for r in det.itertuples()], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("ΔARI (S1 quantile − S0 baseline)")
ax.set_title("b   Quantile normalisation is not reliable", loc="left", pad=6)
d = ar["direction"]
ax.text(0.97, 0.06, f"better {d['better']} | worse {d['worse']} | same {d['same']}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C_BAD, fontweight="bold")
nospines(ax)

# c: 恒等性
ax = axs[1, 0]
idn = [("S3 ComBat ≡ S0", 19), ("S2 Δ-ref ≡ S0", 4), ("MCCA ≡ MOFA", 50)]
ax.barh(np.arange(3), [i[1] for i in idn], 0.62, color=C_NEU, edgecolor="white")
for yi, i in enumerate(idn):
    ax.text(i[1] + 0.8, yi, f"{i[1]} records", va="center", fontsize=8, color="#333333")
ax.set_yticks(range(3)); ax.set_yticklabels([i[0] for i in idn], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 78); ax.set_xlabel("Records with identical ARI")
ax.set_title("c   Three exact identities", loc="left", pad=6)
ax.text(0.97, 0.30, "max pairwise\ndifference = 0.000000", transform=ax.transAxes,
        ha="right", va="center", fontsize=8, color=C_OK, fontweight="bold", linespacing=1.35)
nospines(ax)

# d: 结论
ax = axs[1, 1]
ax.axis("off")
ax.text(0.0, 0.99, "G4 verdict", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax.text(0.0, 0.90,
        "•  ComBat and Δ-reference correction are\n"
        "    absorbed by the subsequent z-scoring\n"
        "    → identical to baseline, by construction\n\n"
        "•  Quantile normalisation flips sign\n"
        "    depending on layer and cohort pair\n\n"
        "•  No strategy improves protein transfer\n\n"
        "→  G4 = NOT PASSED",
        transform=ax.transAxes, fontsize=8, va="top", linespacing=1.28)
ax.text(0.0, 0.02, "Only single-axis affine corrections can be\nabsorbed — exactly what z-scoring removes.",
        transform=ax.transAxes, fontsize=8, va="bottom", linespacing=1.40, color=C_BAD)

fig.suptitle("Figure S3 |  Cross-cohort alignment strategies do not help",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS3_alignment_strategies", "S3")
print("S1-S3 done")
