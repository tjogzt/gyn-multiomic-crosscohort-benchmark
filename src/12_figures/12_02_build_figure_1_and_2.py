#!/usr/bin/env python3
"""
Build Figure 1 and Figure 2 of the benchmark manuscript.

Purpose
    Figure 1 shows that sample availability collapses as layers are added and
    that cross-cohort comparability differs by layer, then shows the protein Δ
    transformation repairing it and the mRNA–protein anchor validating that
    transformation. Figure 2 shows that adding omics layers does not improve
    cross-cohort transfer. Both figures read only the artefacts of stages 01,
    02, 04 and 11; nothing is recomputed here.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_availability")         dict, per-domain/cohort/layer-subset sample counts
    work("alignment_concordance")          dict, per-layer cross-cohort Spearman rho
    work("protein_bridge_three_cohort")    dict, raw vs Δ(T−N) protein concordance
    work("protein_bridge_anchors")         dict, mRNA–protein anchor per cohort
    work("benchmark_summary")              dict, tab1/tab3 benchmark aggregates
    work("benchmark_merged")               dict, merged Python+R ranking and layer effects
    work("benchmark_table_b")              dict, Domain B layer-subset x method ARI table

Outputs
    results("figures_dir")/Fig1_availability_comparability.<fmt>
    results("figures_dir")/Fig2_layer_negative_marginal.<fmt>
    where <fmt> are the formats listed in output.figure_formats.

Usage
    python src/12_figures/12_02_build_figure_1_and_2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put src/ on the import path so that common.* is importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.config import ensure_dir, param, results, work  # noqa: E402
from common.plotting_style import (  # noqa: E402
    C_BAD, C_GY, C_NEU, C_PY, C_R,
    EN, cbar, check, load_json,
)

# Domain identifiers as they are keyed inside the benchmark artefacts. They are
# written once here so the rest of the script can refer to Domain A and Domain B
# through the letters A and B without repeating the artefact keys. The two values
# are DATA: they are the exact strings stored in the artefacts, and the non-ASCII
# part of each is written as a \uXXXX escape so that this source file carries no
# CJK (docs/coding_standard.md §1) while still matching them byte for byte.
# Translating them into English would break every lookup in the availability and
# benchmark tables.
DOMAIN_LABEL_CONFIGURED = param("domains")
DOMAIN_LABEL_LEGACY = {"A": "A_CPTAC\u4e09\u961f\u5217", "B": "B_EC\u56db\u5c42"}
# Resolved against the artefact that is actually loaded, further down. The label
# that keys a domain changed with the refactor: the frozen summary uses the
# legacy strings above, whereas a re-run of stage 04 writes the labels declared
# in params:domains. Whichever is present in the artefact wins, so a figure can
# be rebuilt from either artefact without editing this file.
DOMAIN_KEY: dict = {}

# The two figures of this script are drawn on the same panel grid: 6.94 in =
# 176.3 mm wide, inside the 180 mm two-column limit (params:
# output.figure_width_mm). The size is part of each figure's design and is kept
# as written so that the rendered geometry does not move.
FIG_W, FIG_H = 6.94, 6.14

fig_dir = ensure_dir(results("figures_dir"))
# TIFF is produced by stage 15 from the PNG, so the figure scripts write the
# vector formats and the preview only.
FIG_FORMATS = [f for f in param("output", "figure_formats") if f != "tif"]

av = pd.DataFrame(load_json(work("benchmark_availability")))
ac = load_json(work("alignment_concordance"))
b3c = load_json(work("protein_bridge_three_cohort"))
b3 = load_json(work("protein_bridge_anchors"))
bs = load_json(work("benchmark_summary"))
bm = load_json(work("benchmark_merged"))

# ═══════════════════════════════════════════════════════════════════
# Figure 1
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.145, right=0.955, top=0.915, bottom=0.095, hspace=0.60, wspace=0.52)

# --- a ---
ax = axs[0, 0]
# Largest sample count reached at each layer count, per domain and cohort.
av2 = av.copy()
av2["nL"] = av2["subset"].map(lambda s: len(str(s).split("+")))
# Resolve the domain labels against what the artefacts actually contain, so the
# same script reads the frozen pre-refactor summary and a fresh re-run alike.
_observed = set(av2["domain"].astype(str)) | set(bs.get("tab3", {}))
for _letter in ("A", "B"):
    for _candidate in (DOMAIN_LABEL_CONFIGURED.get(_letter),
                       DOMAIN_LABEL_LEGACY[_letter]):
        if _candidate in _observed:
            DOMAIN_KEY[_letter] = _candidate
            break
    else:
        raise KeyError(
            f"Neither the configured domain label {DOMAIN_LABEL_CONFIGURED.get(_letter)!r} "
            f"nor the legacy key for domain {_letter} appears in the benchmark "
            f"artefacts; observed domain values: {sorted(_observed)}"
        )
piv = av2.groupby(["domain", "cohort", "nL"])["n"].max().reset_index()
# Rows are the six evaluation sets, ordered Domain B first and, inside a domain,
# from the reference cohort outwards.
rowlab = [(DOMAIN_KEY["B"], "TCGA"), (DOMAIN_KEY["B"], "ind"), (DOMAIN_KEY["B"], "dis"),
          (DOMAIN_KEY["A"], "ind"), (DOMAIN_KEY["A"], "dis"), (DOMAIN_KEY["A"], "OV")]
disp = {DOMAIN_KEY["B"]: "EC", DOMAIN_KEY["A"]: "CPTAC"}
names = [f"{disp[d]} {c}" for d, c in rowlab]
# M is the sample-count matrix; it is reused by panel c of Figure 2, which is
# why it is kept at module scope rather than local to this panel.
M = np.full((len(rowlab), 4), np.nan)
for i, (d, c) in enumerate(rowlab):
    for _, r in piv[(piv.domain == d) & (piv.cohort == c)].iterrows():
        M[i, int(r["nL"]) - 1] = r["n"]
# Colour limit 560: just above the largest evaluation set shown (539 samples in
# the EC/TCGA four-layer set), so the top of the scale is not reached.
im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=560)
for i in range(M.shape[0]):
    for j in range(4):
        if np.isfinite(M[i, j]):
            # White type on the dark end of the scale, dark type below it.
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 330 else "#222222")
        else:
            ax.text(j, i, "–", ha="center", va="center", fontsize=8.5, color=C_GY)
ax.set_xticks(range(4))
ax.set_xticklabels(["1", "2", "3", "4"])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Integrated layers")
ax.set_ylabel("")
ax.set_title("a   Availability collapses with layers", loc="left", pad=6)
# White rule between the Domain B block (rows 0-2) and the Domain A block.
ax.axhline(2.5, color="white", lw=1.8)
for i, (d, c) in enumerate(rowlab):
    if d == DOMAIN_KEY["B"]:
        v = M[i]
        eff = v[np.isfinite(v)]
        if len(eff) > 1:
            # Relative sample loss from the first to the last layer count, drawn
            # beside the heat map.
            ax.text(3.72, i, f"−{(1 - eff[-1] / eff[0]) * 100:.0f}%", va="center", ha="left",
                    fontsize=8, color=C_BAD, fontweight="bold")
# The x limit is widened past the last tick to 4.02 so that the loss annotation
# above stays inside the axes and is not clipped by the panel edge.
ax.set_xlim(-0.5, 4.02)
cbar(im, ax)

# --- b ---
ax = axs[0, 1]
rows = []
# The concordance artefact keys its layers by their CJK names. Those keys are DATA
# and are written as \uXXXX escapes so this file carries no CJK while matching the
# artefact exactly; the printed layer labels are the English strings beside them.
for pair, v in ac["\u8868\u8fbe"].items():
    rows.append(("mRNA", v["rho"], pair))
for pair, v in ac["\u62f7\u8d1d\u6570"].items():
    rows.append(("CNA", v["rho"], pair))
for pair, v in ac["\u7532\u57fa\u5316"].items():
    rows.append(("Methyl.\n(CPTAC)", v["rho"], pair))
for pair, v in b3c["raw"].items():
    rows.append(("Protein\n(raw)", v["rho"], pair))
for pair, v in b3c["delta"].items():
    rows.append(("Protein\nΔ(T−N)", v["rho"], pair))
D = pd.DataFrame(rows, columns=["layer", "rho", "pair"])
# A pair involving a CPTAC cohort on both sides is a within-CPTAC comparison;
# anything else is TCGA against CPTAC.
D["inner"] = D["pair"].str.contains("CPTAC-")
order = ["mRNA", "CNA", "Methyl.\n(CPTAC)", "Protein\n(raw)", "Protein\nΔ(T−N)"]
for i, lay in enumerate(order):
    sub = D[D.layer == lay]
    # Range bar behind the points, then the points on top.
    ax.plot([sub["rho"].min(), sub["rho"].max()], [i, i], color="#CCCCCC", lw=1.6, zorder=1)
    inn = sub["inner"].values
    ax.scatter(sub.loc[inn, "rho"], np.full(inn.sum(), i), s=22, color=C_NEU, marker="o",
               zorder=3, linewidths=0)
    ax.scatter(sub.loc[~inn, "rho"], np.full((~inn).sum(), i), s=24, color=C_BAD, marker="s",
               zorder=3, linewidths=0)
ax.axvline(0, color="#555555", lw=0.8, ls=":")
# Usable range: the comparability floor used throughout the study is rho >= 0.4.
ax.axvspan(0.4, 1.10, color="#E8F0E8", zorder=0)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-0.40, 1.14)
ax.set_xticks([-0.3, 0, 0.3, 0.6, 0.9])
ax.set_xlabel("Cross-cohort Spearman ρ")
ax.set_title("b   Comparability differs by layer", loc="left", pad=6)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_NEU, ms=5.5, label="within CPTAC"),
                   Line2D([], [], marker="s", ls="", color=C_BAD, ms=5.5, label="TCGA ↔ CPTAC")],
          loc="lower left", frameon=False, handletextpad=0.3, borderpad=0.1)
ax.text(0.75, 3.72, "usable (ρ ≥ 0.4)", fontsize=8, color="#3F6B3F", ha="center", va="center")
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# --- c ---
ax = axs[1, 0]
pr = [("dis × ov", "dis × ov", "Δdis × Δov"), ("dis × ind", "dis × ind", "Δdis × Δind"),
      ("ov × ind", "ov × ind", "Δov × Δind")]
labs = [p[0] for p in pr]
raw = [b3c["raw"][p[1]]["rho"] for p in pr]
dlt = [b3c["delta"][p[2]]["rho"] for p in pr]
x = np.arange(3)
w = 0.36
ax.bar(x - w / 2, raw, w, color=C_GY, edgecolor="white", label="raw TMT ratio")
ax.bar(x + w / 2, dlt, w, color=C_PY, edgecolor="white", label="Δ(T−N)")
for xi, v in zip(x - w / 2, raw):
    # Value labels sit above a positive bar and below a negative one.
    ax.text(xi, v + (0.030 if v >= 0 else -0.088), f"{v:.3f}", ha="center", fontsize=8,
            color="#555555", va="bottom" if v >= 0 else "top")
for xi, v in zip(x + w / 2, dlt):
    ax.text(xi, v + 0.030, f"{v:.3f}", ha="center", fontsize=8, color=C_PY,
            fontweight="bold", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("Spearman ρ (protein)")
ax.set_ylim(-0.52, 0.92)
ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75])
ax.set_title("c   Δ(T−N) restores comparability", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderpad=0.1)
ax.text(2.02, -0.40, "ov × ind:\n−0.066 → 0.412", fontsize=8, color=C_BAD,
        ha="center", va="center", linespacing=1.30)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# --- d ---
ax = axs[1, 1]
anch = [("Discovery", b3["dis_raw_Prot_RNA"]["rho"], b3["dis_dProt_dRNA"]["rho"]),
        ("Independ.", b3["ind_raw_Prot_RNA"]["rho"], b3["ind_dProt_dRNA"]["rho"])]
x = np.arange(2)
ax.bar(x - w / 2, [a[1] for a in anch], w, color=C_GY, edgecolor="white", label="raw")
ax.bar(x + w / 2, [a[2] for a in anch], w, color=C_PY, edgecolor="white", label="Δ(T−N)")
for xi, a in zip(x - w / 2, anch):
    ax.text(xi, a[1] + (0.032 if a[1] >= 0 else -0.090), f"{a[1]:.3f}", ha="center", fontsize=8,
            color="#555555", va="bottom" if a[1] >= 0 else "top")
for xi, a in zip(x + w / 2, anch):
    ax.text(xi, a[2] + 0.032, f"{a[2]:.3f}", ha="center", fontsize=8, color=C_PY,
            fontweight="bold", va="bottom")
ax.axhline(0, color="#444444", lw=0.9)
# Band of mRNA–protein correlations reported in the literature, used to show
# that the transformed protein layer lands inside it.
ax.axhspan(0.40, 0.62, color="#E8F0E8", zorder=0)
ax.text(1.66, 0.51, "literature\ntypical\n0.40–0.62", fontsize=8, color="#3F6B3F",
        va="center", ha="center")
ax.set_xticks(x)
ax.set_xticklabels([a[0] for a in anch], fontsize=8)
ax.set_ylabel("Spearman ρ (mRNA × protein)")
ax.set_ylim(-0.22, 0.86)
ax.set_yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlim(-0.52, 1.96)
ax.set_title("d   Biological anchor validates Δ", loc="left", pad=6)
ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderpad=0.1)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

fig.suptitle("Figure 1 |  Cohort availability and cross-cohort comparability",
             x=0.012, y=0.977, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig1")
for ext in FIG_FORMATS:
    fig.savefig(fig_dir / f"Fig1_availability_comparability.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════
# Figure 2
# ═══════════════════════════════════════════════════════════════════
fig, axs = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
fig.subplots_adjust(left=0.105, right=0.955, top=0.915, bottom=0.115, hspace=0.58, wspace=0.56)

# Transfer ARI of the eight Python arms, per domain, as a function of the number
# of integrated layers.
PY = {d: [bs["tab3"][k][i]["ari"] for i in range(len(bs["tab3"][k]))]
      for d, k in [("A", DOMAIN_KEY["A"]), ("B", DOMAIN_KEY["B"])]}
# The same curve for the R side comes from the merged benchmark table produced by
# stage 11 rather than being recomputed here, so the figure cannot disagree with
# the verified merge.
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
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(["1", "2", "3", "4"])
    ax.set_xlabel("Integrated layers")
    ax.set_ylabel("Cross-cohort transfer ARI")
    ttl = "a" if dom == "A" else "b"
    nm = "CPTAC three-cohort" if dom == "A" else "TCGA + CPTAC-UCEC"
    ax.set_title(f"{ttl}   Domain {dom} ({nm})", loc="left", pad=6)
    ax.set_ylim(0, 0.66)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

# --- c ---
ax = axs[1, 0]
# Per-cohort curves of the Domain B sample counts, reusing the matrix M built in
# panel a of Figure 1.
for (c, col, mk, off) in [("TCGA", C_BAD, "o", 7), ("ind", C_PY, "s", 6), ("dis", C_NEU, "^", -11)]:
    i = rowlab.index((DOMAIN_KEY["B"], c))
    v = [M[i, j] if np.isfinite(M[i, j]) else np.nan for j in range(4)]
    ax.plot(np.arange(1, 5), v, color=col, marker=mk, lw=1.6, ms=6, label=f"EC {c}", zorder=3)
    for xi, vv in zip(np.arange(1, 5), v):
        if np.isfinite(vv):
            # Per-series offset in points keeps the value labels off the markers
            # and off each other.
            ax.annotate(f"{int(vv)}", xy=(xi, vv), xytext=(0, off), textcoords="offset points",
                        ha="center", fontsize=8, color=col)
ax.set_xticks(np.arange(1, 5))
ax.set_xticklabels(["1", "2", "3", "4"])
ax.set_xlabel("Integrated layers")
ax.set_ylabel("Samples with all layers")
ax.set_ylim(0, 660)
ax.set_yticks([0, 200, 400, 600])
ax.set_title("c   Sample intersection collapse", loc="left", pad=6)
ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.1)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)

# --- d ---
ax = axs[1, 1]
tb = pd.DataFrame(load_json(work("benchmark_table_b")))
# Layer subsets are abbreviated to one letter per layer so the tick labels fit.
ABBR = {"mRNA": "R", "miRNA": "m", "CNA": "C", "meth": "M"}


def compact(s):
    """Abbreviate a layer-subset key ('mRNA+meth' -> 'R+M').

    Args:
        s: layer-subset key as stored in the benchmark table.

    Returns:
        str: the same subset written with one letter per layer.
    """
    return "+".join(ABBR.get(x, x) for x in s.split("+"))


# Rows are ordered by number of layers, then alphabetically.
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
# Outline the best method of each row, so the winner of every subset is visible.
best = np.nanargmax(H, axis=1)
for i, j in enumerate(best):
    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.4))

fig.suptitle("Figure 2 |  More omics layers do not improve cross-cohort transfer",
             x=0.012, y=0.977, ha="left", fontsize=10, fontweight="bold")
check(fig, "Fig2")
for ext in FIG_FORMATS:
    fig.savefig(fig_dir / f"Fig2_layer_negative_marginal.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Done.")
