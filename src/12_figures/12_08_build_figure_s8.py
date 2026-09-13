#!/usr/bin/env python3
"""
Build Figure S8: what data are available, and at what scale.

Purpose
    Summarise the acquisition side of the study in one figure: the sample and
    feature scale of every third-party matrix that entered the harmonisation, the
    on-disk volume each cohort costs, the size of the analysis-ready sample sets
    (the four-layer evaluation set and the five-layer core set) next to the
    copy-number-low (NSMP) subset, and the availability facts that constrain the
    whole design. It reads the stage-01 structure probe, the stage-10 sample
    catalogues and the stage-04 availability table, and plots them unchanged.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("alignment_stage1")        list of dicts, one per source matrix:
                                    cohort, layer, id_kind, genes, samples,
                                    first_id, mb
    work("core5_samples")           list, the frozen five-layer core set
    work("core_nsmp_samples")       dict, keys core and nsmp_core, the frozen
                                    core set and its NSMP subset
    work("benchmark_availability")  list of dicts, analysis-ready sample counts:
                                    domain, subset, cohort, n

Outputs
    results("figures_dir")/FigS8_availability_landscape.<fmt>
        one figure file per format listed in output.figure_formats

Usage
    python src/12_figures/12_08_build_figure_s8.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/12_figures/12_08_build_figure_s8.py sits one level below src/, so this puts
# src/ on the import path and makes common.* importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from common.config import param, work  # noqa: E402
from common.plotting_style import (  # noqa: E402
    C_BAD,
    C_NEU,
    C_PY,
    TICK_SIZE_PT,
    load_json,
    nospines,
    save,
)

# --- Data keys that are not English -----------------------------------------
# The stage-01 and stage-04 artefacts key their rows with labels that were fixed
# while the pipeline was built; two of those labels carry a non-ASCII character.
# They are DATA, not display text: renaming one would make the lookup below miss
# the row. They are therefore written as \uXXXX escapes, so this source file stays
# pure ASCII while the key still matches the artefact exactly. The same convention
# is used for the method and layer keys of common/plotting_style.py.
DOMAIN_EC4 = "B_EC\u56db\u5c42"          # the four-layer EC domain of the availability table
ID_KIND_MIXED = "\u6df7\u5408/\u5176\u5b83"   # identifier system: mixed / other
ID_KIND_PROBE = "\u63a2\u9488"                # identifier system: probe
LAYER_METH450K = "\u7532\u57fa\u5316450K"     # the 450K methylation layer tag

# The canvas is as wide as the target journal allows (output.figure_width_mm,
# 180 mm converted to inches); the height only sets the canvas of this four-panel
# grid, so it stays a layout literal.
FIG_W_IN = param("output", "figure_width_mm") / 25.4
FIG_H_IN = 5.74

# --- Inputs ------------------------------------------------------------------
st = pd.DataFrame(load_json(work("alignment_stage1")))
c5 = load_json(work("core5_samples"))
ns = load_json(work("core_nsmp_samples"))
av = pd.DataFrame(load_json(work("benchmark_availability")))
print("[check] stage1 columns:", list(st.columns))
print(st.head(6).to_string(index=False))
print("[check] core5:", (len(c5) if isinstance(c5, list) else c5), "| nsmp:", ns)

# Both shortened cohorts and layers are display text only: the plotted numbers
# carry no suffix, and shortening keeps the tick labels inside the panel.
st["cohort_s"] = st["cohort"].astype(str).str.replace("TCGA-UCEC", "TCGA", regex=False) \
    .str.replace("CPTAC-UCEC-", "", regex=False).str.replace("CPTAC-OV", "OV", regex=False)
st["layer_s"] = st["layer"].astype(str).str.replace(LAYER_METH450K, "Methyl450K", regex=False) \
    .str.replace("(EB++Adjust)", "", regex=False).str.replace("RNAseq", "RNA", regex=False)

fig, axs = plt.subplots(2, 2, figsize=(FIG_W_IN, FIG_H_IN))
fig.subplots_adjust(left=0.155, right=0.960, top=0.905, bottom=0.155,
                    hspace=0.70, wspace=0.62)

# --- a  Sample x feature scale of every source matrix ------------------------
ax = axs[0, 0]
# Marker shape encodes the identifier system of the matrix; colour separates the
# symbol-based matrices (reusable across cohorts) from the rest.
mk = {"symbol": "o", ID_KIND_MIXED: "s", ID_KIND_PROBE: "^"}
for kind, g in st.groupby("id_kind"):
    ax.scatter(g["samples"], g["genes"], s=44, label=kind,
               marker=mk.get(kind, "s"),
               color=C_PY if "symbol" in kind else C_BAD,
               alpha=0.85, zorder=3, linewidths=0)
# Both axes are logarithmic. A log axis labels its decades with mathtext
# (10^2, 10^3, ...); a mathtext label is taller than a plain-text one and its box
# reaches past the axes edge, which the geometry check reports as out-of-canvas.
# The ticks are therefore stated explicitly WITH PLAIN-TEXT LABELS ("1k", "10k").
# Stating them is also what removes the residual tick left visible once the limits
# are narrowed: at xlim 120..20000 and ylim 50..2.2e6 the automatic locator still
# puts a tick at the top of the range (the 10^6 decade, inside the new upper
# limit), and this explicit tick list no longer contains it.
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([200, 500, 1000, 2000, 5000, 10000])
ax.set_xticklabels(["200", "500", "1k", "2k", "5k", "10k"])
ax.set_yticks([100, 1000, 10000, 100000])
ax.set_yticklabels(["100", "1k", "10k", "100k"])
ax.set_xlim(120, 20000)
ax.set_ylim(50, 2.2e6)
ax.set_xlabel("Samples")
ax.set_ylabel("Features")
ax.set_title("a   Data scale spans four orders", loc="left", pad=6)
ax.legend(loc="lower right", frameon=False, fontsize=TICK_SIZE_PT, handletextpad=0.3)
# The two extremes of the feature axis are annotated, because they are the pair
# the scale argument rests on.
_hi = st.nlargest(1, "genes").iloc[0]
_lo = st.nsmallest(1, "genes").iloc[0]
ax.annotate(f"{_hi['cohort_s'][:6]} {_hi['layer_s'][:9]}", (_hi["samples"], _hi["genes"]),
            textcoords="offset points", xytext=(7, -3), fontsize=TICK_SIZE_PT,
            color="#333333")
ax.annotate(f"{_lo['cohort_s'][:6]} {_lo['layer_s'][:6]}", (_lo["samples"], _lo["genes"]),
            textcoords="offset points", xytext=(7, 4), fontsize=TICK_SIZE_PT,
            color="#333333")
nospines(ax)

# --- b  On-disk volume per cohort --------------------------------------------
ax = axs[0, 1]
g = st.groupby("cohort_s")["mb"].sum().sort_values()
ax.barh(np.arange(len(g)), g.values, 0.64, color=C_NEU, edgecolor="white")
for yi, v in zip(np.arange(len(g)), g.values):
    ax.text(v + 12, yi, f"{v:,.0f} MB", va="center", fontsize=TICK_SIZE_PT,
            color="#333333")
ax.set_yticks(np.arange(len(g)))
ax.set_yticklabels(g.index, fontsize=TICK_SIZE_PT)
# Headroom on the right so the volume labels stay inside the axes.
ax.set_xlim(0, max(g.values) * 1.34)
ax.set_xlabel("On-disk volume (MB)")
ax.set_title("b   Storage per cohort", loc="left", pad=6)
nospines(ax)

# --- c  Analysis-ready sample sets ------------------------------------------
ax = axs[1, 0]
# The number of layers a subset uses is encoded in the subset key itself, so the
# key is parsed rather than looked up: "A+B+C+D" is four layers.
av2 = av.copy()
av2["nL"] = av2["subset"].map(lambda s: len(str(s).split("+")))
lab, val = [], []
for (d, c) in [(DOMAIN_EC4, "TCGA"), (DOMAIN_EC4, "ind"), (DOMAIN_EC4, "dis")]:
    v = av2[(av2.domain == d) & (av2.cohort == c)].groupby("nL")["n"].max()
    lab.append(f"EC {c}")
    val.append(int(v.get(4, 0)))
core = len(c5) if isinstance(c5, list) else 0
# The NSMP (copy-number-low) subset is taken from the frozen catalogue
# work("core_nsmp_samples") and NOT derived from this cohort's annotation table:
# the Xena TCGA-UCEC clinical matrix that the sample universe is read from has no
# molecular-subtype column (it carries only the pan-cancer cluster columns and the
# histology fields), so a per-subtype panel could not be derived from it. That is
# why the per-subtype panel was withdrawn from this figure instead of being built,
# and why the NSMP count appears here as a catalogue value only.
nsmp = len((ns or {}).get("nsmp_core", [])) if isinstance(ns, dict) else 0
print(f"[check] core={core} nsmp={nsmp}")
val.append(core)
lab.append("TCGA\n5-layer")
val.append(nsmp)
lab.append("TCGA\nNSMP")
cols = [C_PY, C_PY, C_PY, C_NEU, C_BAD]
ax.bar(range(len(val)), val, 0.62, color=cols, edgecolor="white")
for xi, v in enumerate(val):
    ax.text(xi, v + 8, str(v), ha="center", fontsize=TICK_SIZE_PT, color="#333333")
ax.set_xticks(range(len(val)))
ax.set_xticklabels(lab, fontsize=TICK_SIZE_PT)
ax.set_ylabel("Samples")
# Headroom so the value labels stay inside the axes.
ax.set_ylim(0, max(val) * 1.28)
ax.set_title("c   Analysis-ready sample sets", loc="left", pad=6)
nospines(ax)

# --- d  Availability facts ---------------------------------------------------
ax = axs[1, 1]
# A text panel: the gallery of availability facts, each of which constrains a
# design decision taken elsewhere in the study.
ax.axis("off")
ax.text(0.0, 0.99, "Availability summary", transform=ax.transAxes, fontsize=9,
        fontweight="bold", va="top")
ax.text(0.0, 0.87,
        "\u2022  Five public cohorts, 4 omics layers\n"
        "    in the common cross-cohort space\n\n"
        "\u2022  TCGA\u2194CPTAC sample-ID overlap = 0\n"
        "    (independent validation, not re-split)\n\n"
        "\u2022  Protein usable only as \u0394(T\u2212N)\n\n"
        "\u2022  TCGA RPPA \u2194 CPTAC overlap = 0\n\n"
        "\u2022  EEEC: 684.8 GB archive \u2192 68.6 MB\n"
        "    via remote central-directory parsing\n"
        "    (0.010% of the original)",
        transform=ax.transAxes, fontsize=TICK_SIZE_PT, va="top", linespacing=1.30)

fig.suptitle("Figure S8 |  Data availability and scale",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS8_availability_landscape", "S8")
print("Figure S8 written.")
