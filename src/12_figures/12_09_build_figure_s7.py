#!/usr/bin/env python3
"""
Build Figure S7: sample availability, and the limits it imposes.

Purpose
    Show what the available TCGA-UCEC data actually support, recomputed here from
    the raw Xena matrices rather than quoted from an artefact: the number of
    primary-tumour samples each omics layer contributes, the five-layer
    intersection they leave, the sample cost of omitting one layer, the size of
    the analysis sets of the gate-G3 comparison, and the survival data available
    for the core set. The last panel is the reason the study reports no survival
    or prognostic endpoint at all, which is an availability limit and not an
    analytical choice.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("xena_pancan_rnaseq_eb_adjusted")  gz, pan-cancer RNA-seq matrix; only its
                                            header (sample columns) is read
    data("xena_ucec_mirna")                 gz, TCGA-UCEC miRNA matrix, header only
    data("xena_ucec_methylation_450k")      gz, TCGA-UCEC 450K beta matrix, header only
    data("xena_ucec_rppa")                  gz, TCGA-UCEC RPPA matrix, header only
    data("xena_ucec_cna_gistic")            gz, TCGA-UCEC GISTIC thresholded CNA
                                            matrix, header only
    data("xena_ucec_clinical_matrix")       tsv, TCGA-UCEC clinical matrix; the
                                            participant sample column defines the
                                            cohort's sample universe
    data("xena_ucec_survival")              tsv, TCGA-UCEC survival export with the
                                            OS and OS.time columns
    work("g3_power")                        dict, gate-G3 thresholds; key "n" holds
                                            the analysis-set size of each cohort

Outputs
    results("figures_dir")/FigS7_availability_limits.<fmt>
        one figure file per format listed in output.figure_formats
    work("s7_final")                        json, the layer sample counts, the
                                            leave-one-layer-out table and the
                                            survival summary shown in the figure

Usage
    python src/12_figures/12_09_build_figure_s7.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

# src/12_figures/12_09_build_figure_s7.py sits one level below src/, so this puts
# src/ on the import path and makes common.* importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from common.config import data, ensure_dir, param, work  # noqa: E402
from common.plotting_style import (  # noqa: E402
    C_BAD,
    C_GY,
    C_NEU,
    C_PY,
    TICK_SIZE_PT,
    TITLE_SIZE_PT,
    load_json,
    nospines,
    save,
)

# TCGA sample-type suffix marking a primary solid tumour, and the length of the
# participant + sample + portion barcode (TCGA-XX-XXXX-01A). Both are properties
# of the TCGA barcode scheme, not tunable settings, so they are stated once here.
PRIMARY_TUMOUR_SUFFIX = "-01"
BARCODE_LENGTH = 15

# Days per year and three years in days; the published follow-up statistics are
# reported on a 365.25-day year.
DAYS_PER_YEAR = 365.25
THREE_YEARS_DAYS = 3 * DAYS_PER_YEAR

# The canvas is as wide as the target journal allows (output.figure_width_mm,
# 180 mm converted to inches); the height only sets the canvas of this four-panel
# grid, so it stays a layout literal.
FIG_W_IN = param("output", "figure_width_mm") / 25.4
FIG_H_IN = 5.88


def column_universe(path):
    """Return the primary-tumour participant barcodes a Xena matrix declares.

    Only the header line is read: the sample universe of a matrix is exactly the
    set of its columns, and the pan-cancer and 450K matrices are far too large to
    open in full for that.

    Args:
        path: a Xena matrix, gzip-compressed or plain, with the samples in the
            header row after the first (feature) column.

    Returns:
        set[str]: the first BARCODE_LENGTH characters of every TCGA column that is
        a primary-tumour sample, upper-cased. Columns are quoted in these exports,
        so the quotes are stripped before the test.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    return {c.strip('"')[:BARCODE_LENGTH].upper() for c in header[1:]
            if c.strip('"').upper().startswith("TCGA-")
            and c.strip('"').endswith(PRIMARY_TUMOUR_SUFFIX)}


# --- Layer sample universes --------------------------------------------------
# Each layer is represented by the matrix the study integrates for it. The keys
# are the labels printed on the panel and in the leave-one-out table; they are the
# English layer names and deliberately not the file-name tags of the harmonised
# matrices, because they are read as a sentence in the figure.
LAY = {"mRNA": column_universe(data("xena_pancan_rnaseq_eb_adjusted")),
       "miRNA": column_universe(data("xena_ucec_mirna")),
       "Methylation": column_universe(data("xena_ucec_methylation_450k")),
       "RPPA": column_universe(data("xena_ucec_rppa")),
       "CNA": column_universe(data("xena_ucec_cna_gistic"))}

# The sample universe of the cohort is read from its clinical table, so that a
# layer cannot contribute a sample the cohort does not have a record for. The
# column holding the barcode is declared in params.yaml (nsmp.tcga_sample_id_column)
# because the same column is read by the gate-G3 subset builder.
cm = pd.read_csv(data("xena_ucec_clinical_matrix"), sep="\t", low_memory=False)
id_col = param("nsmp", "tcga_sample_id_column")
UCEC = {s[:BARCODE_LENGTH].upper() for s in cm[id_col].astype(str)
        if s.endswith(PRIMARY_TUMOUR_SUFFIX)}
print(f"[check] UCEC sample universe (clinical table, primary tumours) = {len(UCEC)}")
# NOTE: this clinical table carries no molecular-subtype column (only the
# pan-cancer cluster columns and the histology fields), so no subtype breakdown
# can be derived here; the subtype subsets used downstream come from the frozen
# stage-10 catalogue instead.
LAY = {k: (v & UCEC) for k, v in LAY.items()}
print("[check] layer sample counts after restriction to UCEC:",
      {k: len(v) for k, v in LAY.items()})
CORE = set.intersection(*LAY.values())
print("[check] five-layer core set =", len(CORE), "(published record: 306)")

# --- Leave-one-layer-out cost ------------------------------------------------
# For every layer, the intersection of all the others: the difference to the
# five-layer core set is the number of samples that one omitted layer releases.
lo = []
for drop in LAY:
    use = {k: v for k, v in LAY.items() if k != drop}
    inter = set.intersection(*use.values())
    lo.append(dict(dropped=drop, n_all=len(inter), gain=len(inter) - len(CORE)))
LO = pd.DataFrame(lo).sort_values("gain", ascending=False)
print("\n[check] leave-one-layer-out over all primary tumours:")
print(LO.to_string(index=False))

# --- Survival data of the core set -------------------------------------------
sv = pd.read_csv(data("xena_ucec_survival"), sep="\t", low_memory=False)
# The survival export is barcode-level; it is collapsed onto participants so that
# a participant contributes one row, then restricted to the core set.
sv["pat15"] = sv["sample"].astype(str).str[:BARCODE_LENGTH].str.upper()
sv = sv.drop_duplicates("pat15").set_index("pat15")
inC = sv.loc[sorted(set(sv.index) & CORE)]
# OS is the event flag and OS.time the follow-up in days; rows missing either are
# dropped, because an event without a time carries no information.
os_e = pd.to_numeric(inC["OS"], errors="coerce")
os_t = pd.to_numeric(inC["OS.time"], errors="coerce")
ok = os_e.notna() & os_t.notna()
os_e, os_t = os_e[ok], os_t[ok]
print(f"\n[check] core-set samples with survival data = {len(os_e)}")
print(f"  OS events = {int(os_e.sum())} ({os_e.mean() * 100:.1f}%)")
print(f"  median follow-up = {os_t.median() / DAYS_PER_YEAR:.2f} years")
print(f"  follow-up < 3 years = {(os_t < THREE_YEARS_DAYS).mean() * 100:.1f}%")

# --- Analysis-set sizes of the gate-G3 comparison ----------------------------
# The cohort labels of the comparison are declared in params.yaml (nsmp.cohorts)
# and work("g3_power")["n"] carries the analysis-set size under the same labels.
g3 = load_json(work("g3_power"))
NS = {c: g3["n"][c] for c in param("nsmp", "cohorts")}

fig, axs = plt.subplots(2, 2, figsize=(FIG_W_IN, FIG_H_IN))
fig.subplots_adjust(left=0.185, right=0.965, top=0.905, bottom=0.145,
                    hspace=0.68, wspace=0.56)

# --- a  Per-layer sample counts and their five-way intersection --------------
ax = axs[0, 0]
nm = list(LAY) + ["5-layer\nintersection"]
vl = [len(LAY[k]) for k in LAY] + [len(CORE)]
# The five layers are drawn in the auxiliary colour and their intersection in the
# failed/degeneracy colour: the panel is read as a funnel, and the last bar is the
# set the study can actually work with.
ax.barh(np.arange(6), vl, 0.66, color=[C_GY] * 5 + [C_BAD], edgecolor="white")
for yi, v in zip(np.arange(6), vl):
    ax.text(v + 8, yi, str(v), va="center", fontsize=TICK_SIZE_PT, color="#333333")
ax.set_yticks(np.arange(6))
ax.set_yticklabels(nm, fontsize=TICK_SIZE_PT)
ax.invert_yaxis()
# The x limit is set beyond the longest bar so the value labels stay inside the
# axes; the locator then still draws its 600 tick, which is left visible as the
# axis reference.
ax.set_xlim(0, 640)
ax.set_xlabel("Primary-tumour samples")
ax.set_title("a   Five-way intersection", loc="left", pad=6)
# White text on the bar, naming the two layers that set the ceiling.
ax.text(0.030, 0.085, "set by miRNA\n+ methylation", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=TICK_SIZE_PT, color="white",
        linespacing=1.35)
nospines(ax)

# --- b  Sample cost of each omitted layer ------------------------------------
ax = axs[0, 1]
ax.barh(np.arange(len(LO)), LO["gain"], 0.64,
        color=[C_BAD if g == LO["gain"].max() else C_PY for g in LO["gain"]],
        edgecolor="white")
for yi, (g, n) in enumerate(zip(LO["gain"], LO["n_all"])):
    ax.text(g + 3, yi, f"+{g}  \u2192  {n}", va="center", fontsize=TICK_SIZE_PT,
            color="#333333")
ax.set_yticks(np.arange(len(LO)))
ax.set_yticklabels(["drop " + d for d in LO["dropped"]], fontsize=TICK_SIZE_PT)
ax.invert_yaxis()
# Headroom for the "+gain -> n" labels of the longest bar.
ax.set_xlim(0, max(LO["gain"]) * 1.75)
ax.set_xlabel("Samples gained (all primary tumours)")
ax.set_title("b   Cost of each omitted layer", loc="left", pad=6)
nospines(ax)

# --- c  Size of the analysis sets --------------------------------------------
ax = axs[1, 0]
# Display order is a layout choice: the five-layer core set first, then the three
# cohort subsets in the order the manuscript reports them.
panel_cohorts = ["TCGA", "V2", "V1"]
lab = ["5-layer\ncore set"] + [f"NSMP\n({c})" for c in panel_cohorts]
val = [len(CORE)] + [NS[c] for c in panel_cohorts]
ax.bar(range(4), val, 0.62, color=[C_NEU, C_BAD, C_PY, C_PY], edgecolor="white")
for xi, v in zip(range(4), val):
    ax.text(xi, v + 9, str(v), ha="center", fontsize=TICK_SIZE_PT, color="#333333")
ax.set_xticks(range(4))
ax.set_xticklabels(lab, fontsize=TICK_SIZE_PT)
ax.set_ylabel("Samples")
# The upper limit is a fixed reference for the panel, above the largest set, so
# the three bars stay comparable across renders.
ax.set_ylim(0, 380)
ax.set_title("c   Size of the analysis sets", loc="left", pad=6)
nospines(ax)

# --- d  Why no survival endpoint is reported ---------------------------------
ax = axs[1, 1]
# A text panel: the numbers are the ones computed above, so the panel and the
# printed log always agree.
ax.axis("off")
ax.text(0.0, 0.99, "Why no survival endpoint", transform=ax.transAxes,
        fontsize=TITLE_SIZE_PT, fontweight="bold", va="top")
ax.text(0.0, 0.87,
        f"\u2022  5-layer core set: n = {len(CORE)}\n"
        f"\u2022  With survival data: n = {len(os_e)}\n"
        f"\u2022  OS events: {int(os_e.sum())} ({os_e.mean() * 100:.1f}%)\n"
        f"\u2022  Median follow-up: {os_t.median() / DAYS_PER_YEAR:.2f} years\n"
        f"\u2022  Follow-up < 3 years: {(os_t < THREE_YEARS_DAYS).mean() * 100:.0f}%",
        transform=ax.transAxes, fontsize=TICK_SIZE_PT, va="top", linespacing=1.42)
ax.text(0.0, 0.42, "\u2192  Event count far below what a\n     multivariate model needs.",
        transform=ax.transAxes, fontsize=TICK_SIZE_PT, va="top", color="#333333",
        linespacing=1.34)
ax.text(0.0, 0.02, "No survival or prognostic endpoint\nis reported anywhere in this work.",
        transform=ax.transAxes, fontsize=TICK_SIZE_PT, va="bottom", color=C_BAD,
        linespacing=1.30)

fig.suptitle("Figure S7 |  Sample availability and the limits it imposes",
             x=0.012, y=0.975, ha="left", fontsize=10, fontweight="bold")
save(fig, "FigS7_availability_limits", "S7")

# The numbers the figure is read for are also written out, so that the verification
# stage can compare the manuscript text against this figure without re-measuring.
payload = dict(
    layer_n={k: len(v) for k, v in LAY.items()},
    core=len(CORE),
    leave_one_out=LO.to_dict("records"),
    survival=dict(n_surv=int(len(os_e)), os_events=int(os_e.sum()),
                  median_fu_years=round(float(os_t.median() / DAYS_PER_YEAR), 2),
                  pct_under_3y=round(float((os_t < THREE_YEARS_DAYS).mean() * 100), 1)),
)
out_path = work("s7_final")
ensure_dir(out_path.parent)
with out_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=1)
print(f"\nFigure S7 written; summary written to {out_path}")
