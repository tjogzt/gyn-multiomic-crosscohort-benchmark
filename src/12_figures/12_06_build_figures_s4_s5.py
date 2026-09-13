#!/usr/bin/env python3
"""
Build supplementary Figures S4 and S5 of the benchmark manuscript.

Purpose
    Figure S4 shows that consensus stability (PAC) is a screening aid only: the
    co-association arm is the most stable one, SNF reaches a low PAC through a
    degenerate solution rather than through agreement, and the weak negative
    relation between PAC and transfer ARI holds only when the degeneracy
    diagnostic is read alongside it. Figure S5 shows that every purity-aware
    correction arm performs worse than the frozen baseline, monotonically in the
    number of purity-related components removed, and that purity explains only a
    small part of the variance of any layer. Both figures read the artefacts of
    stages 06 and 09 unmodified; nothing is recomputed here.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("pac_aggregate")      dict, tab1 per-method PAC and cross per-pair PAC/ARI
    work("pai_results")        dict, compare table of the purity-aware arms against
                               the frozen baseline, plus the baseline and target ARI
    work("pai_learn_summary")  dict, per-layer number of genes and share of genes
                               whose correlation with purity exceeds |r| = 0.3

Outputs
    results("figures_dir")/FigS4_pac_stability.<fmt>
    results("figures_dir")/FigS5_purity_arms.<fmt>
    where <fmt> are the formats listed under output.figure_formats.

Usage
    python src/12_figures/12_06_build_figures_s4_s5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Put src/ on the import path so that common.* is importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from common.config import param, work  # noqa: E402
from common.plotting_style import (  # noqa: E402
    C_BAD, C_GY, C_NEU, C_OK, C_PY, EN, load_json, nospines, save,
)

# Figure width: the target journal's two-column limit, read from params.yaml and
# converted to inches at the precision the published panel grids were laid out
# with (180 mm -> 7.087 in). The heights below are per-figure design values and
# are kept as written so the rendered geometry does not move.
FIG_W = round(float(param("output", "figure_width_mm")) / 25.4, 3)

# PAC bin edges of the stability-to-transfer relation, read from params.yaml so
# that this panel, the PAC-to-transfer table (13_01) and the number extraction
# (14_02) all cut the axis at the same values.
PAC_EDGES = list(param("verification", "pac_bin_edges"))

pa = load_json(work("pac_aggregate"))
# Run-log probe: the aggregate is a small dictionary, so its keys and the shape
# of each entry are printed to make a stale or partial artefact obvious in the
# log of a re-run.
print("[probe] pac_agg keys:", list(pa))
for k, v in pa.items():
    if k == "tab1":
        continue
    print(f"   {k}: {type(v).__name__} {json.dumps(v, ensure_ascii=False)[:200]}")

# ═══════════════════════════════════════════════════════════════════
# Figure S4 | Clustering stability (PAC) and its degeneracy trap
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 6.02))
fig.subplots_adjust(left=0.135, right=0.965, top=0.905, bottom=0.135, hspace=0.66, wspace=0.52)

# --- a PAC by method ---
ax = axs[0, 0]
p1 = pa["tab1"]
ms = sorted(p1, key=lambda m: p1[m]["all"])
# Red marks SNF, whose low PAC comes from degeneracy rather than agreement; blue
# marks the co-association arm, which is the stable and comparable one; anything
# else is auxiliary. The co-association method key carries the non-ASCII suffix
# the artefact uses, escaped here so the file stays free of CJK characters while
# still matching the key (docs/coding_standard.md §1).
ax.barh(np.arange(len(ms)), [p1[m]["all"] for m in ms], 0.68,
        color=[C_BAD if "SNF" in m else (C_PY if "\u5171\u8054\u5171\u8bc6" in m else C_GY)
               for m in ms], edgecolor="white")
for yi, m in zip(np.arange(len(ms)), ms):
    ax.text(p1[m]["all"] + 0.010, yi, f"{p1[m]['all']:.3f}", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(ms)))
ax.set_yticklabels([EN(m) for m in ms], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 0.78)
ax.set_xlabel("PAC (lower = more stable)")
ax.set_title("a   Method stability", loc="left", pad=6)
nospines(ax)

# --- b degeneracy diagnostic (PAC must be read with the largest-cluster share) ---
ax = axs[0, 1]
ax.scatter([p1[m]["all"] for m in ms], [p1[m]["strong"] for m in ms], s=64,
           color=[C_BAD if "SNF" in m else C_GY for m in ms], zorder=3, linewidths=0)
for m in ms:
    # Only the two arms the panel is about are annotated; the rest would collide.
    if "SNF" in m or "\u5171\u8054\u5171\u8bc6" in m:
        ax.annotate(EN(m), (p1[m]["all"], p1[m]["strong"]), textcoords="offset points",
                    xytext=(7, 5), fontsize=8, color=C_BAD if "SNF" in m else C_PY)
ax.set_xlabel("PAC")
ax.set_ylabel("Strong-consensus fraction")
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
ax.set_xlim(0.14, 0.72)
ax.set_ylim(0, 0.72)
ax.set_title("b   Why PAC alone misleads", loc="left", pad=6)
ax.text(0.03, 0.04, "SNF: 2nd-lowest PAC, but 0.573 strong\nconsensus — low PAC from degeneracy.",
        transform=ax.transAxes, fontsize=8, va="bottom", color=C_BAD, linespacing=1.40)
nospines(ax)

# --- c PAC-to-ARI bins ---
ax = axs[1, 0]
# The per-pair PAC/ARI table was written under two names across revisions of
# stage 06; the first one present is used and an empty table degrades to a note
# in the panel rather than to a crash.
cr = pa.get("cross") or pa.get("detail") or []
if cr:
    C = pd.DataFrame(cr)
    # The PAC and ARI columns are located by name so that a renamed artefact
    # column is still found; the names differ between the two revisions.
    pcol = "pac" if "pac" in C.columns else [c for c in C.columns if "pac" in c.lower()][0]
    acol = "ari" if "ari" in C.columns else [c for c in C.columns if "ari" in c.lower()][0]
    p = pd.to_numeric(C[pcol], errors="coerce")
    a = pd.to_numeric(C[acol], errors="coerce")
    bins = pd.cut(p, PAC_EDGES, right=False)
    g = pd.DataFrame({"a": a, "b": bins}).dropna().groupby("b", observed=True)["a"].agg(["mean", "count"])
    ax.bar(range(len(g)), g["mean"], 0.62, color=[C_PY, C_NEU, C_GY, C_BAD], edgecolor="white")
    for xi, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(xi, m + 0.008, f"{m:.3f}", ha="center", fontsize=8, color="#333333")
        # The count sits white inside the bar, where it cannot collide with the
        # value label above it.
        ax.text(xi, 0.012, f"n={int(n)}", ha="center", fontsize=8, color="white")
    ax.set_xticks(range(len(g)))
    # Bin labels are derived from the configured edges, so they cannot disagree
    # with the cuts actually applied above.
    ax.set_xticklabels([f"<{PAC_EDGES[1]:.2f}",
                        f"{PAC_EDGES[1]:.2f}–{PAC_EDGES[2]:.2f}",
                        f"{PAC_EDGES[2]:.2f}–{PAC_EDGES[3]:.2f}",
                        f">{PAC_EDGES[3]:.2f}"], fontsize=8)
    ax.set_xlabel("PAC bin")
    ax.set_ylabel("Mean transfer ARI")
    ax.set_ylim(0, 0.36)
    ax.set_title("c   Stability tracks transfer", loc="left", pad=6)
    # Spearman rho and p of the per-pair relation, as reported in the manuscript.
    ax.text(0.03, 0.95, "Spearman −0.158 (p = 0.0015)", transform=ax.transAxes,
            fontsize=8, va="top", color="#333333")
else:
    ax.text(0.5, 0.5, "(no per-pair PAC/ARI table available)", transform=ax.transAxes,
            ha="center", fontsize=8)
nospines(ax)

# --- d conclusion ---
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

# ═══════════════════════════════════════════════════════════════════
# Figure S5 | Purity-aware arms degrade cross-cohort transfer
# ═══════════════════════════════════════════════════════════════════
pr = load_json(work("pai_results"))
cmp_ = pd.DataFrame(pr["compare"])
ls = load_json(work("pai_learn_summary"))
# Run-log probe: the comparison table is small and is printed in full, so the
# numbers behind the three panels can be read directly from the log of a re-run.
print("[probe] pai compare cols:", list(cmp_.columns))
print(cmp_.to_string(index=False))

fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 5.74))
fig.subplots_adjust(left=0.185, right=0.965, top=0.905, bottom=0.140, hspace=0.68, wspace=0.50)

# --- a each arm against the baseline ---
ax = axs[0, 0]
C2 = cmp_.sort_values("mean_delta")
# Arm keys as stored in the artefact -> printed labels. The keys carry the
# non-ASCII wording the arms were named with and are escaped so the file stays
# free of CJK characters while still matching the artefact; the values are the
# English labels printed in the figure.
ARM = {"G20 \u7eaf\u5ea6\u65e0\u5173\u57fa\u56e0 γ=0.2": "G20 (γ=0.2)",
       "G30 \u7eaf\u5ea6\u65e0\u5173\u57fa\u56e0 γ=0.3": "G30 (γ=0.3)",
       "S1 \u5254\u9664\u7eaf\u5ea6PC k=1": "S1 (k=1)",
       "S2 \u5254\u9664\u7eaf\u5ea6PC k=2": "S2 (k=2)",
       "S3 \u5254\u9664\u7eaf\u5ea6PC k=3": "S3 (k=3)",
       "R1 \u9010\u57fa\u56e0\u7eaf\u5ea6\u6b8b\u5dee\u5316": "R1 (residual)"}
lb = [ARM.get(a, a) for a in C2["arm"]]
# The worst arm is drawn red, everything else grey: the panel shows that the
# direction of every arm is the same, so only the magnitude marks the worst one.
ax.barh(np.arange(len(C2)), C2["mean_delta"], 0.68,
        color=[C_BAD if "S3" in l else C_GY for l in lb], edgecolor="white")
for yi, (d, p) in enumerate(zip(C2["mean_delta"], C2["p"])):
    # Labels are right-aligned inside the negative bars.
    ax.text(d - 0.006, yi, f"{d:+.3f}", va="center", ha="right", fontsize=8, color="#333333")
ax.axvline(0, color="#444444", lw=1.0)
ax.set_yticks(np.arange(len(C2)))
ax.set_yticklabels(lb, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-0.22, 0.045)
ax.set_xlabel("ΔARI vs frozen baseline")
ax.set_title("a   Every purity arm is worse", loc="left", pad=6)
ax.text(0.97, 0.62, f"baseline = {pr['g2_base']:.4f}\ntarget = {pr['g2_target']:.4f}",
        transform=ax.transAxes, ha="right", va="center", fontsize=8, color="#333333", linespacing=1.40)
nospines(ax)

# --- b dose-response ---
ax = axs[0, 1]
# Mean ΔARI of the three PC-removal arms as published, in the order in which the
# components are removed. The values are transcribed from the compare table of
# work("pai_results") rather than recomputed, so the panel shows the published
# numbers; the first point (0.0) is the baseline itself.
dose = [("S1\nk=1", -0.0867), ("S2\nk=2", -0.1332), ("S3\nk=3", -0.1800)]
ax.plot([0, 1, 2, 3], [0.0] + [d[1] for d in dose], color=C_BAD, marker="o", lw=1.7, ms=6, zorder=3)
for xi, d in enumerate(dose, 1):
    ax.text(xi, d[1] - 0.012, f"{d[1]:.3f}", ha="center", fontsize=8, color=C_BAD, va="top")
ax.axhline(0, color="#444444", lw=1.0, ls="--")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(["none", "S1\nk=1", "S2\nk=2", "S3\nk=3"], fontsize=8)
ax.set_xlabel("Purity-correlated PCs removed")
ax.set_ylabel("ΔARI vs baseline")
ax.set_ylim(-0.235, 0.045)
ax.set_title("b   More removal, more damage", loc="left", pad=6)
ax.text(0.03, 0.06, "monotone dose–response\nacross two mechanisms", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8, color=C_BAD, linespacing=1.40)
nospines(ax)

# --- c win rate and significance ---
ax = axs[1, 0]
# -log10 of the paired p, clipped at 1e-300 so an underflowed p still plots.
ax.scatter(C2["win_rate"] * 100, -np.log10(np.maximum(C2["p"], 1e-300)), s=68,
           color=C_BAD, zorder=3, linewidths=0)
for _, r in C2.iterrows():
    # Label truncated to nine characters: the full arm label would collide with
    # the neighbouring points.
    ax.annotate(ARM.get(r["arm"], r["arm"])[:9], (r["win_rate"] * 100, -np.log10(max(r["p"], 1e-300))),
                textcoords="offset points", xytext=(6, 3), fontsize=8, color="#333333")
# Significance rule of the study.
ax.axhline(-np.log10(0.05), color=C_OK, lw=1.2, ls="--")
ax.text(0.5, -np.log10(0.05) + 0.15, "p = 0.05", fontsize=8, color=C_OK, va="bottom")
ax.set_xlabel("Win rate vs baseline (%)")
ax.set_ylabel("−log10 paired p")
ax.set_xlim(0, 36)
ax.set_ylim(0, 8.5)
ax.set_title("c   All arms significantly worse", loc="left", pad=6)
nospines(ax)

# --- d mechanism: variance explained by purity ---
ax = axs[1, 1]
L = [("mRNA", ls["mRNA"]), ("miRNA", ls["miRNA"]), ("CNA", ls["CNA"]),
     ("Methyl.", ls.get("meth", {})), ("Protein", ls.get("protein", {}))]
nm, fr, ng = [], [], []
for n, v in L:
    # A layer without a measured purity association is left out of the panel
    # rather than drawn as zero.
    if isinstance(v, dict) and "frac_r_gt_0.3" in v:
        nm.append(n)
        fr.append(v["frac_r_gt_0.3"] * 100)
        ng.append(v["n_gene"])
ax.barh(np.arange(len(nm)), fr, 0.64, color=C_PY, edgecolor="white")
for yi, (f, g) in enumerate(zip(fr, ng)):
    ax.text(f + 0.25, yi, f"{f:.1f}%", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(nm)))
ax.set_yticklabels(nm, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 12)
ax.set_xlabel("Genes with |r(purity)| > 0.3 (%)")
ax.set_title("d   Purity explains little variance", loc="left", pad=6)
# Mean explained variance over the layers, as published.
ax.text(0.97, 0.42, "mean r² = 3.68%\n(protein 7.2%)", transform=ax.transAxes,
        ha="right", va="center", fontsize=8, color=C_BAD, linespacing=1.40)
nospines(ax)

fig.suptitle("Figure S5 |  Purity-aware correction degrades cross-cohort transfer",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS5_purity_arms", "S5")
print("Figures S4 and S5 written.")
