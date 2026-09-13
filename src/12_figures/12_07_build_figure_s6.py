#!/usr/bin/env python3
"""
Build Figure S6: the EEEC label-free proteome decodes its own group semantics.

Purpose
    Establish, from the measurement alone, that the sample families of the EEEC
    label-free proteome are real and agree with the tumour/normal semantics the
    batch testbed assumes. The panel set compares the family mean spectra, shows
    that the first principal component recovers the condition axis rather than
    the batch axis, contrasts the within-family sample similarity with the
    across-family similarity, and reports the variance the first component
    explains. No per-sample phenotype key for this cohort was publicly available
    when the analysis was run, so every statement here is decoded from the frozen
    stage-07 batch-testbed artefacts instead of from an annotation file.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("eeec_batch_sample_meta")  csv, one row per sample: col (the matching
                                    matrix column), family, suffix, batch, cond
    work("eeec_batch_pc_assoc")     csv, one row per principal component:
                                    PC, AUROC_vs_batch, AUROC_vs_cond, var_exp
    work("eeec_batch_matrix")       csv, log2 protein-group matrix; rows are
                                    protein groups, columns are samples

Outputs
    results("figures_dir")/FigS6_eeec_semantics.<fmt>
        one figure file per format listed in output.figure_formats

Usage
    python src/12_figures/12_07_build_figure_s6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/12_figures/12_07_build_figure_s6.py sits one level below src/, so this puts
# src/ on the import path and makes common.* importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from common.config import param, work  # noqa: E402
from common.plotting_style import (  # noqa: E402
    AXIS_LINEWIDTH,
    C_BAD,
    C_GY,
    C_PY,
    TICK_SIZE_PT,
    cbar,
    nospines,
    save,
)

# The canvas is as wide as the target journal allows (output.figure_width_mm,
# 180 mm converted to inches). The height only sets the canvas for this four-panel
# grid and carries no analysis meaning, so it stays a layout literal.
FIG_W_IN = param("output", "figure_width_mm") / 25.4
FIG_H_IN = 5.88

# --- Inputs ------------------------------------------------------------------
meta = pd.read_csv(work("eeec_batch_sample_meta"))
pc = pd.read_csv(work("eeec_batch_pc_assoc"))
mat = pd.read_csv(work("eeec_batch_matrix"), index_col=0)

# The matrix holds every sample the batch testbed measured; the metadata is
# restricted to the samples that are actually columns of the matrix, so the
# per-family means below are computed over an existing sample set only.
meta = meta[meta["col"].isin(mat.columns)].reset_index(drop=True)
sub = mat[meta["col"].tolist()]
fams = sorted(meta["family"].unique())
print("[check] families:", fams, "| counts:", meta["family"].value_counts().to_dict())
print("[check] batch:", meta["batch"].value_counts().to_dict(),
      "| cond:", meta["cond"].value_counts().to_dict())

fig, axs = plt.subplots(2, 2, figsize=(FIG_W_IN, FIG_H_IN))
fig.subplots_adjust(left=0.165, right=0.960, top=0.905, bottom=0.155,
                    hspace=0.68, wspace=0.62)

# --- a  Group-mean spectra ---------------------------------------------------
ax = axs[0, 0]
# One mean spectrum per family: the mean over the samples of that family, per
# protein group. Correlating the means rather than the samples removes the
# within-family measurement noise and exposes the family-level structure.
gm = pd.DataFrame({g: sub.loc[:, meta.loc[meta.family == g, "col"].tolist()].mean(axis=1)
                   for g in fams})
R = gm.corr(method="spearman")
# The colour range is stated rather than taken from the data, so that this panel
# keeps one fixed meaning across renders; 0.55 is the floor the published panel
# was drawn with.
im = ax.imshow(R.values, cmap="RdYlBu_r", vmin=0.55, vmax=1.0, aspect="auto")
for i in range(len(fams)):
    for j in range(len(fams)):
        ax.text(j, i, f"{R.values[i, j]:.3f}", ha="center", va="center",
                fontsize=TICK_SIZE_PT, color="#222222")
ax.set_xticks(range(len(fams)))
ax.set_xticklabels(fams, rotation=40, ha="right", fontsize=TICK_SIZE_PT)
ax.set_yticks(range(len(fams)))
ax.set_yticklabels(fams, fontsize=TICK_SIZE_PT)
ax.set_title("a   Group-mean spectra", loc="left", pad=6)
cbar(im, ax)

# --- b  Which axis the leading components encode -----------------------------
ax = axs[0, 1]
# Only the leading components are shown: past these the AUCs sit at chance, which
# is the point of the panel.
n = min(8, len(pc))
x = np.arange(n)
ax.bar(x - 0.19, pc["AUROC_vs_cond"][:n], 0.38, color=C_PY, edgecolor="white",
       label="vs condition")
ax.bar(x + 0.19, pc["AUROC_vs_batch"][:n], 0.38, color=C_GY, edgecolor="white",
       label="vs batch")
# Chance level of the AUROC: a component that carries no axis information sits on
# this line.
ax.axhline(0.5, color="#666666", lw=1.0, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([f"PC{i + 1}" for i in range(n)], fontsize=TICK_SIZE_PT)
# The upper limit leaves room for the legend above the tallest bar. The tick list
# is given as well: with the limits at 0.4 and 1.12 the automatic locator would
# place a tick at each end (0.4 and 1.1), i.e. two residual ticks, and the
# explicit list keeps only the three values the panel is read against.
ax.set_ylim(0.4, 1.12)
ax.set_yticks([0.5, 0.75, 1.0])
ax.set_ylabel("AUROC")
ax.set_title("b   PC1 recovers the condition axis", loc="left", pad=6)
ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.4,
          columnspacing=0.9, fontsize=TICK_SIZE_PT)
# The PC1 pair of AUCs is repeated as text, because it is the one number the
# panel is quoted for.
ax.text(0.03, 0.66,
        f"PC1: cond {pc['AUROC_vs_cond'][0]:.3f}\n      batch {pc['AUROC_vs_batch'][0]:.3f}",
        transform=ax.transAxes, fontsize=TICK_SIZE_PT, va="top", color=C_BAD,
        linespacing=1.45)
nospines(ax)

# --- c  Within-family versus across-family sample similarity -----------------
ax = axs[1, 0]
gg = meta["family"].tolist()
cols = meta["col"].tolist()
# One correlation matrix over all samples; the pairs are then split by whether the
# two samples belong to the same family.
Cz = np.corrcoef(np.asarray(sub.values, dtype=float).T)
within, cross = [], []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        (within if gg[i] == gg[j] else cross).append(Cz[i, j])
# If the families were arbitrary labels the two boxes would coincide; a within
# box clearly above the across box is what makes the group structure real.
ax.boxplot([within, cross], widths=0.5, patch_artist=True,
           boxprops=dict(facecolor=C_PY, alpha=0.72, linewidth=AXIS_LINEWIDTH),
           medianprops=dict(color=C_BAD, lw=1.4),
           whiskerprops=dict(lw=AXIS_LINEWIDTH),
           capprops=dict(lw=AXIS_LINEWIDTH),
           flierprops=dict(marker=".", ms=2.5, markerfacecolor=C_GY,
                           markeredgecolor="none"))
ax.set_xticks([1, 2])
ax.set_xticklabels(["within\ngroup", "across\ngroups"], fontsize=TICK_SIZE_PT)
ax.set_ylabel("Sample\u2013sample correlation")
ax.set_title("c   Group structure is real", loc="left", pad=6)
ax.text(0.97, 0.05, f"within {np.mean(within):.3f}\nacross {np.mean(cross):.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=TICK_SIZE_PT,
        color="#333333", linespacing=1.45)
nospines(ax)

# --- d  Variance explained by the leading components -------------------------
ax = axs[1, 1]
ve = pc["var_exp"][:n] * 100
# The first component is coloured as the deciding one, the rest as auxiliary:
# the panel's claim is that a single axis dominates the data.
ax.bar(x, ve, 0.62, color=[C_BAD if i == 0 else C_GY for i in range(n)],
       edgecolor="white")
for xi, v in zip(x, ve):
    ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=TICK_SIZE_PT,
            color="#333333")
ax.set_xticks(x)
ax.set_xticklabels([f"PC{i + 1}" for i in range(n)], fontsize=TICK_SIZE_PT)
ax.set_ylabel("Variance explained (%)")
# Headroom above the tallest bar so its value label stays inside the axes.
ax.set_ylim(0, max(ve) * 1.30)
ax.set_title("d   One axis dominates", loc="left", pad=6)
ax.text(0.97, 0.95,
        "decoded from the data alone:\nno per-sample phenotype key\nwas publicly available",
        transform=ax.transAxes, ha="right", va="top", fontsize=TICK_SIZE_PT,
        color="#333333", linespacing=1.45)
nospines(ax)

fig.suptitle("Figure S6 |  EEEC group-level semantic decoding",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS6_eeec_semantics", "S6")
print("Figure S6 written.")
