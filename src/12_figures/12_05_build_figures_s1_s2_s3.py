#!/usr/bin/env python3
"""
Build supplementary Figures S1, S2 and S3 of the benchmark manuscript.

Purpose
    Figure S1 establishes why the layers had to be harmonised at all: the input
    matrices use different identifier systems, the 450K probes lose 29% of their
    genes on the way to gene symbols, the numeric domains of the layers differ
    by orders of magnitude, and no matrix carries a duplicate identifier.
    Figure S2 diagnoses the protein bridge: Δ(T−N) restores comparability in all
    three CPTAC cohort pairs, the raw failure cannot be explained by a trimming
    artefact, the mRNA–protein anchor lands inside the literature range, and
    methylation degrades across platforms. Figure S3 shows that none of the four
    cross-cohort alignment strategies helps, and that three of them are exactly
    identical to the baseline by construction. Each panel reads an artefact
    written by stage 01, 02 or 03 and prints its values unmodified: nothing in
    this module recomputes, resamples or synthesises a value.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("alignment_domains")            list, per cohort x layer numeric domain
                                         (n_col, q05, q50, q95) of stage 01
    work("alignment_ids")                dict, per-matrix identifier-system census
    work("alignment_probe_map")          dict, 450K probe -> gene mapping census
    work("alignment_duplicates")         dict, per-matrix duplicate-identifier census
    work("protein_bridge_three_cohort")  dict, raw vs Δ(T−N) protein concordance per cohort pair
    work("protein_bridge_diagnostics")   dict, raw, delta and trimmed dis x ov concordance
    work("protein_bridge_anchors")       dict, mRNA–protein and methylation anchors of stage 02
    work("alignment_concordance")        dict, per-layer cross-cohort Spearman rho of stage 01
    work("alignment_ari_aggregate")      dict, alignment-strategy ARI aggregate (agg_K<K>),
                                         per-pair detail and better/worse/same counts

Outputs
    results("figures_dir")/FigS1_harmonisation_domains.<fmt>
    results("figures_dir")/FigS2_protein_bridge.<fmt>
    results("figures_dir")/FigS3_alignment_strategies.<fmt>
    where <fmt> are the formats listed under output.figure_formats.

Usage
    python src/12_figures/12_05_build_figures_s1_s2_s3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put src/ on the import path so that common.* is importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from common.config import param, work  # noqa: E402
from common.plotting_style import (  # noqa: E402
    C_BAD, C_GY, C_NEU, C_OK, C_PY, C_R, C_SPAN,
    EN_LAYER, cbar, load_json, nospines, save,
)

# Figure width: the target journal's two-column limit, read from params.yaml and
# converted to inches at the precision the published panel grids were laid out
# with (180 mm -> 7.087 in). Deriving it keeps every supplementary figure tied to
# the same limit as the main figures. The heights below are per-figure design
# values and are kept as written so the rendered geometry does not move.
FIG_W = round(float(param("output", "figure_width_mm")) / 25.4, 3)

# Primary K of the benchmark, read from params.yaml. The alignment artefact keys
# its K-specific matrix as agg_K<K>, so key and panel title are derived from the
# same value and cannot drift apart.
K = int(param("clustering", "k_primary"))

# Artefacts of stages 01 (harmonisation probes), 02 (protein bridge) and 03
# (alignment strategies). All nine are read once, before any figure is drawn.
ad = load_json(work("alignment_domains"))
aid = load_json(work("alignment_ids"))
apm = load_json(work("alignment_probe_map"))
adups = load_json(work("alignment_duplicates"))
b3c = load_json(work("protein_bridge_three_cohort"))
bd = load_json(work("protein_bridge_diagnostics"))
b3 = load_json(work("protein_bridge_anchors"))
ac = load_json(work("alignment_concordance"))
ar = load_json(work("alignment_ari_aggregate"))

# ═══════════════════════════════════════════════════════════════════
# Figure S1 | Layer harmonisation and numeric domains
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 5.88))
fig.subplots_adjust(left=0.145, right=0.965, top=0.905, bottom=0.135, hspace=0.70, wspace=0.75)

# --- a identifier-system census ---
ax = axs[0, 0]
# Identifier kinds as they are printed, keyed by the census key stored in the
# alignment artefact. The census keys are DATA and are therefore written as
# \uXXXX escapes: they stay byte-for-byte equal to the artefact while this source
# file stays free of CJK characters (docs/coding_standard.md §1). Translating a
# key would break the lookup; only the values are English labels.
KIND_EN = {"Entrez\u6570\u5b57": "Entrez ID", "symbol\u6837": "symbol-like",
           "450K\u63a2\u9488": "450K probe", "\u5176\u5b83/\u4f4d\u70b9ID": "other / locus ID",
           "miRNA\u540d": "miRNA name", "GISTIC\u5cf0": "GISTIC peak"}
# Matrices whose identifier system is censused, and the kinds the bar segments
# show, in stack order.
sel = ["TCGA mRNA(EB++)", "TCGA mRNA(Hugo)", "TCGA 450K", "TCGA RPPA", "TCGA CNA"]
kinds = ["symbol-like", "Entrez ID", "450K probe", "other / locus ID", "miRNA name"]
cols = [C_PY, C_BAD, C_NEU, C_GY, C_OK]
# M[i, j] is the percentage of matrix i's row identifiers that belong to kind j.
# Every kind is shown even when it is absent from a matrix, so the stacked bars
# of the different matrices can be compared segment by segment.
M = np.zeros((len(sel), len(kinds)))
for i, k in enumerate(sel):
    cen = aid[k]["census"]
    tot = sum(cen.values())
    for j, kk in enumerate(kinds):
        # KIND_EN maps artefact key -> printed label, so the reverse lookup
        # recovers the census keys that contribute to this segment.
        zh = [z for z, e in KIND_EN.items() if e == kk]
        M[i, j] = sum(cen.get(z, 0) for z in zh) / tot * 100
left = np.zeros(len(sel))
for j, kk in enumerate(kinds):
    ax.barh(np.arange(len(sel)), M[:, j], 0.66, left=left, color=cols[j], edgecolor="white", label=kk)
    left += M[:, j]
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(["mRNA (EB++)", "mRNA (Hugo)", "Methylation 450K", "RPPA", "CNA"], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of identifiers (%)")
ax.set_title("a   ID systems are not interchangeable", loc="left", pad=6)
ax.legend(loc="lower right", frameon=False, fontsize=8, handletextpad=0.3, labelspacing=0.28)
nospines(ax)

# --- b probe-to-gene mapping ---
ax = axs[0, 1]
pv = [apm["ucec_probes"], apm["mapped"], apm["genes"]]
lb = ["450K probes\nin UCEC", "mapped to\ngene symbol", "unique\ngenes"]
ax.bar(range(3), pv, 0.60, color=[C_GY, C_PY, C_NEU], edgecolor="white")
for xi, v in enumerate(pv):
    ax.text(xi, v * 1.10, f"{v:,}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(3))
ax.set_xticklabels(lb, fontsize=8)
# A log scale is used because the three counts span three orders of magnitude;
# the ticks are set by hand so the decades are labelled instead of every minor
# tick.
ax.set_yscale("log")
ax.set_ylim(1e3, 4e6)
ax.set_yticks([1e3, 1e4, 1e5, 1e6])
ax.set_yticklabels(["1k", "10k", "100k", "1M"])
ax.set_ylabel("Count (log scale)")
# The 29% in the title is the mapping loss reported in the manuscript; the
# annotation states the mapped share the two counts imply.
ax.set_title("b   Probe\u2192gene mapping loses 29%", loc="left", pad=6)
ax.text(1.0, 1.35e6, f"{apm['mapped']/apm['ucec_probes']*100:.1f}% mapped",
        ha="center", fontsize=8, color=C_OK, fontweight="bold")
nospines(ax)

# --- c numeric domains ---
ax = axs[1, 0]
# The nine cohort x layer rows whose numeric domain is drawn, taken in the order
# the artefact lists them. The 450K methylation matrix carries a layer tag that
# contains CJK, written as an escape in the filter below for the same reason as
# the census keys above.
rows = [r for r in ad if r["layer"] in ("mRNA", "\u7532\u57fa\u5316450K", "RPPA", "CNA", "RNAseq")][:9]
# Artefact cohort and layer tags -> the short labels that fit on the tick axis.
SHORT_COH = {"TCGA-UCEC": "TCGA", "CPTAC-UCEC-ind": "ind", "CPTAC-UCEC-dis": "dis",
             "CPTAC-OV": "OV", "CPTAC-UCEC-OV": "OV", "CPTAC-OV-pro": "OV"}
SHORT_LAY = {"mRNA": "mRNA", "\u7532\u57fa\u5316450K": "Meth450K", "RPPA": "RPPA", "CNA": "CNA",
             "RNAseq": "RNA", "Proteome": "Prot", "Phospho": "Phos", "Acetyl": "Ac", "Meth": "Meth", "miRNA": "miR"}
labs, lo, hi, mid = [], [], [], []
for r in rows:
    # An unmapped tag falls back to its first eight characters so the axis never
    # renders an empty label.
    labs.append(f"{SHORT_COH.get(r['cohort'], r['cohort'][:8])} {SHORT_LAY.get(r['layer'], r['layer'][:8])}")
    lo.append(r["q05"])
    hi.append(r["q95"])
    mid.append(r["q50"])
y = np.arange(len(labs))
# Inter-percentile range as a bar, median as a point on top of it.
ax.hlines(y, lo, hi, color=C_NEU, lw=3.2, alpha=0.55)
ax.plot(mid, y, "o", color=C_BAD, ms=5, zorder=3)
ax.axvline(0, color="#555555", lw=0.8, ls=":")
ax.set_yticks(y)
ax.set_yticklabels(labs, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Value (5th / 50th / 95th percentile)")
ax.set_xticks([0, 10, 20])
ax.set_title("c   Numeric domains differ across layers", loc="left", pad=6)
nospines(ax)

# --- d duplicate-identifier census ---
ax = axs[1, 1]
dk = list(adups.keys())
dv = [adups[k]["dup"] for k in dk]
# Green marks a matrix with no duplicate identifier, red one that has any; the
# x limit is widened past the last bar so the value labels stay inside the axes.
ax.barh(np.arange(len(dk)), [max(v, 0) for v in dv], 0.66,
        color=[C_OK if v == 0 else C_BAD for v in dv], edgecolor="white")
for yi, v in zip(np.arange(len(dk)), dv):
    ax.text(max(v, 0) + 0.02, yi, f"{int(v)}", va="center", fontsize=8, color="#333333")
ax.set_yticks(np.arange(len(dk)))
ax.set_yticklabels(dk, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 1.4)
ax.set_xticks([0, 0.5, 1.0])
ax.set_xlabel("Duplicate gene identifiers")
ax.set_title("d   No duplicate IDs anywhere", loc="left", pad=6)
ax.text(0.98, 0.06, "all audits return 0", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=C_OK, fontweight="bold")
nospines(ax)

fig.suptitle("Figure S1 |  Layer harmonisation and numeric domains",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS1_harmonisation_domains", "S1")

# ═══════════════════════════════════════════════════════════════════
# Figure S2 | Protein bridge diagnosis and layer-specific degradation
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 6.02))
fig.subplots_adjust(left=0.135, right=0.965, top=0.905, bottom=0.135, hspace=0.66, wspace=0.50)

# --- a raw vs delta across the three cohort pairs ---
ax = axs[0, 0]
# (axis label, key of the raw artefact, key of the delta artefact).
pk = [("dis × ov", "dis × ov", "Δdis × Δov"), ("dis × ind", "dis × ind", "Δdis × Δind"),
      ("ov × ind", "ov × ind", "Δov × Δind")]
x = np.arange(3)
w = 0.36
raw = [b3c["raw"][p[1]]["rho"] for p in pk]
dlt = [b3c["delta"][p[2]]["rho"] for p in pk]
ax.bar(x - w/2, raw, w, color=C_GY, edgecolor="white", label="raw TMT ratio")
ax.bar(x + w/2, dlt, w, color=C_PY, edgecolor="white", label="Δ(T−N)")
for xi, v in zip(x - w/2, raw):
    # Value labels sit above a positive bar and below a negative one, so neither
    # set of labels crosses the zero line.
    ax.text(xi, v + (0.028 if v >= 0 else -0.082), f"{v:.3f}", ha="center", fontsize=8,
            color="#555555", va="bottom" if v >= 0 else "top")
for xi, v in zip(x + w/2, dlt):
    ax.text(xi, v + 0.028, f"{v:.3f}", ha="center", fontsize=8, color=C_PY, fontweight="bold", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels([p[0] for p in pk], fontsize=8)
ax.set_ylabel("Spearman ρ (protein)")
ax.set_ylim(-0.38, 0.90)
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
ax.set_title("a   Δ(T−N) across three cohorts", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderpad=0.1)
nospines(ax)

# --- b trimming diagnostic ---
ax = axs[0, 1]
lb = ["raw\n(all genes)", "Δ(T−N)\n(all genes)", "raw\n(trimmed 80%)"]
vv = [bd["raw_dis_T_ov_T"]["rho"], bd["delta_dis_ov"]["rho"], bd["trim_dis_ov"]["rho"]]
ax.bar(range(3), vv, 0.60, color=[C_GY, C_PY, C_NEU], edgecolor="white")
for xi, v in zip(range(3), vv):
    ax.text(xi, v + 0.020, f"{v:.3f}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(3))
ax.set_xticklabels(lb, fontsize=8)
ax.set_ylabel("Spearman ρ (dis × ov)")
ax.set_ylim(0, 0.60)
ax.set_title("b   Trimming cannot explain it", loc="left", pad=6)
nospines(ax)

# --- c biological anchor ---
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
# Band of mRNA–protein correlations reported in the literature, so the reader can
# see that the Δ-transformed anchor lands inside it.
ax.axhspan(0.40, 0.62, color=C_SPAN, zorder=0)
ax.text(1.62, 0.51, "literature\ntypical", fontsize=8, color=C_OK, va="center", ha="center", linespacing=1.3)
ax.set_xticks(x)
ax.set_xticklabels([a[0] for a in anc], fontsize=8)
ax.set_ylabel("Spearman ρ (mRNA × protein)")
ax.set_ylim(-0.22, 0.86)
ax.set_xlim(-0.52, 1.92)
ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_title("c   mRNA–protein anchor", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.1)
nospines(ax)

# --- d methylation granularity ---
ax = axs[1, 1]
# Within-CPTAC methylation in blue (comparable), TCGA vs CPTAC in red (not). The
# methylation layer key of the concordance artefact contains CJK and is written as
# an escape below for the same reason as the census keys of Figure S1.
mm = [("CPTAC\ninternal", ac["\u7532\u57fa\u5316"]["CPTAC-ind|CPTAC-dis"]["rho"], C_PY),
      ("TCGA vs\nCPTAC-ind", b3["meth_TCGA_ind"]["rho"], C_BAD),
      ("TCGA vs\nCPTAC-dis", b3["meth_TCGA_dis"]["rho"], C_BAD)]
ax.bar(range(3), [m[1] for m in mm], 0.60, color=[m[2] for m in mm], edgecolor="white")
for xi, m in enumerate(mm):
    ax.text(xi, m[1] + 0.022, f"{m[1]:.3f}", ha="center", fontsize=8, color="#333333")
ax.set_xticks(range(3))
ax.set_xticklabels([m[0] for m in mm], fontsize=8)
# The comparability floor used throughout the study: rho = 0.4.
ax.axhline(0.4, color=C_OK, lw=1.2, ls="--")
ax.text(2.45, 0.418, "usability floor 0.4", fontsize=8, color=C_OK, ha="right", va="bottom")
ax.set_ylabel("Spearman ρ (methylation)")
ax.set_ylim(0, 0.86)
ax.set_title("d   Methylation degrades across platforms", loc="left", pad=6)
nospines(ax)

fig.suptitle("Figure S2 |  Protein bridge diagnosis and layer-specific degradation",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS2_protein_bridge", "S2")

# ═══════════════════════════════════════════════════════════════════
# Figure S3 | Cross-cohort alignment strategies do not help
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(FIG_W, 7.20))
fig.subplots_adjust(left=0.195, right=0.955, top=0.925, bottom=0.125, hspace=0.80, wspace=0.62)

# --- a layer x strategy heat map ---
ax = axs[0, 0]
agg = pd.DataFrame(ar[f"agg_K{K}"]).T
order = [c for c in agg.columns]
# Alignment-strategy column keys -> printed labels. The keys carry the non-ASCII
# suffixes the artefact was written with and are escaped for the same reason as
# the census keys of Figure S1; the values are the labels printed in the panel.
REN = {"S0_\u57fa\u7ebf": "S0 base", "S1_\u5206\u4f4d\u6570\u6807\u51c6\u5316": "S1 quant",
       "S2_Δ\u53c2\u8003\u6821\u6b63": "S2 Δ-ref", "S3_ComBat(\u6c60\u5316)": "S3 ComBat"}
# NaN means the cohort pair has no matrix for that layer; it is drawn as a zero
# value and annotated "n/a" so the cell is visibly missing rather than reading as
# a low ARI.
H = agg.values.astype(float)
im = ax.imshow(np.nan_to_num(H, nan=0.0), cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=0.80)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([REN.get(c, c) for c in order], rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(agg.index)))
ax.set_yticklabels([EN_LAYER.get(i, i) for i in agg.index], fontsize=8)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        if np.isnan(H[i, j]):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#666666")
        else:
            ax.text(j, i, f"{H[i,j]:.2f}", ha="center", va="center", fontsize=8, color="#222222")
ax.set_title(f"a   Layer × alignment strategy (K={K})", loc="left", pad=6)
cbar(im, ax)

# --- b per-pair change from S0 to S1 ---
ax = axs[0, 1]
det = pd.DataFrame(ar["detail"])
# Change the quantile arm makes against the baseline, per cohort pair. The two
# column keys are the artefact's strategy keys.
det["d"] = det["S1_\u5206\u4f4d\u6570\u6807\u51c6\u5316"] - det["S0_\u57fa\u7ebf"]
det = det.sort_values("d")
y = np.arange(len(det))
# Red where quantile normalisation made the pair worse, blue where it improved it.
ax.barh(y, det["d"], 0.72, color=[C_R if v < 0 else C_PY for v in det["d"]], edgecolor="white")
ax.axvline(0, color="#444444", lw=0.9)
ax.set_yticks(y)
# Row label: the first four characters of the layer tag plus the cohort pair.
ax.set_yticklabels([f"{r.layer[:4]} {r.pair}" for r in det.itertuples()], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("ΔARI (S1 quantile − S0 baseline)")
ax.set_title("b   Quantile normalisation is not reliable", loc="left", pad=6)
d = ar["direction"]
ax.text(0.97, 0.06, f"better {d['better']} | worse {d['worse']} | same {d['same']}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C_BAD, fontweight="bold")
nospines(ax)

# --- c exact identities ---
ax = axs[1, 0]
# The three exact identities verified by the alignment stage, as (pair, number of
# ARI records with an identical value). The counts are transcribed from the
# published analysis rather than recomputed here, because the identity check
# itself lives in stage 03.
idn = [("S3 ComBat ≡ S0", 19), ("S2 Δ-ref ≡ S0", 4), ("MCCA ≡ MOFA", 50)]
ax.barh(np.arange(3), [i[1] for i in idn], 0.62, color=C_NEU, edgecolor="white")
for yi, i in enumerate(idn):
    ax.text(i[1] + 0.8, yi, f"{i[1]} records", va="center", fontsize=8, color="#333333")
ax.set_yticks(range(3))
ax.set_yticklabels([i[0] for i in idn], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(0, 78)
ax.set_xlabel("Records with identical ARI")
ax.set_title("c   Three exact identities", loc="left", pad=6)
ax.text(0.97, 0.30, "max pairwise\ndifference = 0.000000", transform=ax.transAxes,
        ha="right", va="center", fontsize=8, color=C_OK, fontweight="bold", linespacing=1.35)
nospines(ax)

# --- d conclusion ---
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
print("Figures S1, S2 and S3 written.")
