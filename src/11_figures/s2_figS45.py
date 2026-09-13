#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4–S5 补充图 + S6 数据探查"""
import sys, json, os
sys.path.insert(0, "/tmp/gyn_retyping")
from figstyle import *
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

pa = J("pac_agg.json")
print("[probe] pac_agg keys:", list(pa))
for k, v in pa.items():
    if k == "tab1": continue
    print(f"   {k}: {type(v).__name__} {json.dumps(v, ensure_ascii=False)[:200]}")

# ══════════════ S4 | Clustering stability (PAC) ══════════════
fig, axs = plt.subplots(2, 2, figsize=(7.087, 6.02))
fig.subplots_adjust(left=0.135, right=0.965, top=0.905, bottom=0.135, hspace=0.66, wspace=0.52)

# a: 方法 PAC
ax = axs[0, 0]
p1 = pa["tab1"]
ms = sorted(p1, key=lambda m: p1[m]["all"])
ax.barh(np.arange(len(ms)), [p1[m]["all"] for m in ms], 0.68,
        color=[C_BAD if "SNF" in m else (C_PY if "共联共识" in m else C_GY) for m in ms], edgecolor="white")
for yi, m in zip(np.arange(len(ms)), ms):
    ax.text(p1[m]["all"] + 0.010, yi, f"{p1[m]['all']:.3f}", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(ms))); ax.set_yticklabels([EN(m) for m in ms], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 0.78); ax.set_xlabel("PAC (lower = more stable)")
ax.set_title("a   Method stability", loc="left", pad=6)
nospines(ax)

# b: 退化诊断（PAC 必须与最大簇占比并报）
ax = axs[0, 1]
ax.scatter([p1[m]["all"] for m in ms], [p1[m]["strong"] for m in ms], s=64,
           color=[C_BAD if "SNF" in m else C_GY for m in ms], zorder=3, linewidths=0)
for m in ms:
    if "SNF" in m or "共联共识" in m:
        ax.annotate(EN(m), (p1[m]["all"], p1[m]["strong"]), textcoords="offset points",
                    xytext=(7, 5), fontsize=8, color=C_BAD if "SNF" in m else C_PY)
ax.set_xlabel("PAC"); ax.set_ylabel("Strong-consensus fraction")
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_xlim(0.14, 0.72); ax.set_ylim(0, 0.72)
ax.set_title("b   Why PAC alone misleads", loc="left", pad=6)
ax.text(0.03, 0.04, "SNF: 2nd-lowest PAC, but 0.573 strong\nconsensus — low PAC from degeneracy.",
        transform=ax.transAxes, fontsize=8, va="bottom", color=C_BAD, linespacing=1.40)
nospines(ax)

# c: PAC → ARI 分箱
ax = axs[1, 0]
cr = pa.get("cross") or pa.get("detail") or []
if cr:
    C = pd.DataFrame(cr)
    pcol = "pac" if "pac" in C.columns else [c for c in C.columns if "pac" in c.lower()][0]
    acol = "ari" if "ari" in C.columns else [c for c in C.columns if "ari" in c.lower()][0]
    p = pd.to_numeric(C[pcol], errors="coerce"); a = pd.to_numeric(C[acol], errors="coerce")
    bins = pd.cut(p, [0, .30, .45, .60, 1.01], right=False)
    g = pd.DataFrame({"a": a, "b": bins}).dropna().groupby("b", observed=True)["a"].agg(["mean", "count"])
    ax.bar(range(len(g)), g["mean"], 0.62, color=[C_PY, C_NEU, C_GY, C_BAD], edgecolor="white")
    for xi, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(xi, m + 0.008, f"{m:.3f}", ha="center", fontsize=8, color="#333333")
        ax.text(xi, 0.012, f"n={int(n)}", ha="center", fontsize=8, color="white")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(["<0.30", "0.30–0.45", "0.45–0.60", ">0.60"], fontsize=8)
    ax.set_xlabel("PAC bin"); ax.set_ylabel("Mean transfer ARI")
    ax.set_ylim(0, 0.36)
    ax.set_title("c   Stability tracks transfer", loc="left", pad=6)
    ax.text(0.03, 0.95, "Spearman −0.158 (p = 0.0015)", transform=ax.transAxes,
            fontsize=8, va="top", color="#333333")
else:
    ax.text(0.5, 0.5, "(no per-pair PAC/ARI table available)", transform=ax.transAxes, ha="center", fontsize=8)
nospines(ax)

# d: 结论
ax = axs[1, 1]
ax.axis("off")
ax.text(0.0, 0.99, "Use of PAC in this work", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax.text(0.0, 0.88,
        "•  PAC is reported as a screening aid for\n"
        "    single-cohort pre-selection only\n\n"
        "•  It must never be read without the\n"
        "    degeneracy diagnostic (panel b)\n\n"
        "•  Its effect size is weak: ρ = −0.158",
        transform=ax.transAxes, fontsize=8, va="top", linespacing=1.30)
ax.text(0.0, 0.02, "Screening, not replacement for cross-cohort\nvalidation.", transform=ax.transAxes,
        fontsize=8, va="bottom", linespacing=1.40, color=C_OK)

fig.suptitle("Figure S4 |  Clustering stability and its degeneracy trap",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS4_pac_stability", "S4")

# ══════════════ S5 | Purity-aware arms ══════════════
pr = J("pai/pai_results.json")
cmp_ = pd.DataFrame(pr["compare"])
ls = J("pai/learn_summary.json")
print("[probe] pai compare cols:", list(cmp_.columns))
print(cmp_.to_string(index=False))

fig, axs = plt.subplots(2, 2, figsize=(7.087, 5.74))
fig.subplots_adjust(left=0.185, right=0.965, top=0.905, bottom=0.140, hspace=0.68, wspace=0.50)

# a: 各臂 vs 基线
ax = axs[0, 0]
C2 = cmp_.sort_values("mean_delta")
ARM = {"G20 纯度无关基因 γ=0.2": "G20 (γ=0.2)", "G30 纯度无关基因 γ=0.3": "G30 (γ=0.3)",
       "S1 剔除纯度PC k=1": "S1 (k=1)", "S2 剔除纯度PC k=2": "S2 (k=2)", "S3 剔除纯度PC k=3": "S3 (k=3)",
       "R1 逐基因纯度残差化": "R1 (residual)"}
lb = [ARM.get(a, a) for a in C2["arm"]]
ax.barh(np.arange(len(C2)), C2["mean_delta"], 0.68,
        color=[C_BAD if "S3" in l else C_GY for l in lb], edgecolor="white")
for yi, (d, p) in enumerate(zip(C2["mean_delta"], C2["p"])):
    ax.text(d - 0.006, yi, f"{d:+.3f}", va="center", ha="right", fontsize=8, color="#333333")
ax.axvline(0, color="#444444", lw=1.0)
ax.set_yticks(np.arange(len(C2))); ax.set_yticklabels(lb, fontsize=8)
ax.invert_yaxis(); ax.set_xlim(-0.22, 0.045)
ax.set_xlabel("ΔARI vs frozen baseline")
ax.set_title("a   Every purity arm is worse", loc="left", pad=6)
ax.text(0.97, 0.62, f"baseline = {pr['g2_base']:.4f}\ntarget = {pr['g2_target']:.4f}",
        transform=ax.transAxes, ha="right", va="center", fontsize=8, color="#333333", linespacing=1.40)
nospines(ax)

# b: 剂量-反应
ax = axs[0, 1]
dose = [("S1\nk=1", -0.0867), ("S2\nk=2", -0.1332), ("S3\nk=3", -0.1800)]
ax.plot([0, 1, 2, 3], [0.0] + [d[1] for d in dose], color=C_BAD, marker="o", lw=1.7, ms=6, zorder=3)
for xi, d in enumerate(dose, 1):
    ax.text(xi, d[1] - 0.012, f"{d[1]:.3f}", ha="center", fontsize=8, color=C_BAD, va="top")
ax.axhline(0, color="#444444", lw=1.0, ls="--")
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["none", "S1\nk=1", "S2\nk=2", "S3\nk=3"], fontsize=8)
ax.set_xlabel("Purity-correlated PCs removed"); ax.set_ylabel("ΔARI vs baseline")
ax.set_ylim(-0.235, 0.045)
ax.set_title("b   More removal, more damage", loc="left", pad=6)
ax.text(0.03, 0.06, "monotone dose–response\nacross two mechanisms", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8, color=C_BAD, linespacing=1.40)
nospines(ax)

# c: 胜率与显著性
ax = axs[1, 0]
ax.scatter(C2["win_rate"] * 100, -np.log10(np.maximum(C2["p"], 1e-300)), s=68,
           color=C_BAD, zorder=3, linewidths=0)
for _, r in C2.iterrows():
    ax.annotate(ARM.get(r["arm"], r["arm"])[:9], (r["win_rate"] * 100, -np.log10(max(r["p"], 1e-300))),
                textcoords="offset points", xytext=(6, 3), fontsize=8, color="#333333")
ax.axhline(-np.log10(0.05), color=C_OK, lw=1.2, ls="--")
ax.text(0.5, -np.log10(0.05) + 0.15, "p = 0.05", fontsize=8, color=C_OK, va="bottom")
ax.set_xlabel("Win rate vs baseline (%)"); ax.set_ylabel("−log10 paired p")
ax.set_xlim(0, 36); ax.set_ylim(0, 8.5)
ax.set_title("c   All arms significantly worse", loc="left", pad=6)
nospines(ax)

# d: 机理——纯度可解释方差
ax = axs[1, 1]
L = [("mRNA", ls["mRNA"]), ("miRNA", ls["miRNA"]), ("CNA", ls["CNA"]),
     ("Methyl.", ls.get("meth", {})), ("Protein", ls.get("protein", {}))]
nm, fr, ng = [], [], []
for n, v in L:
    if isinstance(v, dict) and "frac_r_gt_0.3" in v:
        nm.append(n); fr.append(v["frac_r_gt_0.3"] * 100); ng.append(v["n_gene"])
ax.barh(np.arange(len(nm)), fr, 0.64, color=C_PY, edgecolor="white")
for yi, (f, g) in enumerate(zip(fr, ng)):
    ax.text(f + 0.25, yi, f"{f:.1f}%", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(nm))); ax.set_yticklabels(nm, fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 12)
ax.set_xlabel("Genes with |r(purity)| > 0.3 (%)")
ax.set_title("d   Purity explains little variance", loc="left", pad=6)
ax.text(0.97, 0.42, "mean r² = 3.68%\n(protein 7.2%)", transform=ax.transAxes,
        ha="right", va="center", fontsize=8, color=C_BAD, linespacing=1.40)
nospines(ax)

fig.suptitle("Figure S5 |  Purity-aware correction degrades cross-cohort transfer",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS5_purity_arms", "S5")
print("S4-S5 done")
