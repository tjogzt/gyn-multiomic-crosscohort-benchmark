#!/usr/bin/env python3
"""
Check that every number displayed in Figures 4 and 5 was produced by an artefact.

Purpose
    Stage 14, ninth script. Figures 4 and 5 quote the batch-testbed metrics, the
    cross-platform ladder and the G2/G3 decision quantities. The script rebuilds
    the expected value set of each figure from those artefacts, extracts the text
    layer of the rendered PDFs and reports the expected values that are absent.
    It then reads the first page box of all five main figures and reports any
    text block flush against a page edge, which is how a label clipped by the
    tight bounding box of the canvas shows up.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("eeec_batch_verify_csv")   CSV, one row per correction arm with the
                                    columns centroid_AUC, LR_AUC_abs, knn_mix.
    work("platform_results_json")   JSON, "ladder" (one record per comparison
                                    rung), "bins" (effect-size strata) and
                                    "network".
    work("g2_refined")              JSON, "required_n" with the sample size per
                                    relative improvement.
    work("g3_power")                JSON, "split_half" and "bootstrap_ceiling"
                                    per cohort and cluster number.
    results("figures_dir")/<stem>.pdf
                                    the rendered figures whose text layer and
                                    page box are read.
    config/params.yaml verification.figure_5_values
                                    printed values Figure 5 must display.
    config/params.yaml verification.figure_5_available_records
                                    evaluation-record count quoted in Figure 5.
    config/params.yaml nsmp.cohorts, gates.g3.k_values
                                    the cohort labels and cluster numbers that
                                    key the G3 artefact.

Outputs
    None. The per-figure report is printed.

Usage
    python 14_09_verify_figure_4_5_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz
import pandas as pd

# Put src/ on the import path so that common.* is importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import param, results, work  # noqa: E402

FIG_DIR = results("figures_dir")

# Cohorts of the G3 comparison and the cluster numbers of its ceiling; both key
# records of work("g3_power"), so they are taken from the configuration the gate
# itself was computed with rather than from the artefact's key order.
COHORTS = list(param("nsmp", "cohorts"))
CLUSTER_NUMBERS = [str(value) for value in param("gates", "g3", "k_values")]

# Distance in points below which a text block counts as touching a page edge. It
# absorbs the rounding of the page box, not a real margin.
EDGE_TOLERANCE_PT = float(param("verification", "page_edge_tolerance_pt"))

# Characters of an offending block echoed next to the warning.
BLOCK_ECHO_LENGTH = 40

# The figures whose numbers are checked, in reporting order; the same five are
# re-read for the page-geometry pass.
CHECKED_FIGURES = [
    "Fig4_covariance_fragility",
    "Fig5_thresholds_undecidable",
]
GEOMETRY_FIGURES = [
    "Fig1_availability_comparability",
    "Fig2_layer_negative_marginal",
    "Fig3_method_not_resolvable",
    "Fig4_covariance_fragility",
    "Fig5_thresholds_undecidable",
]

# The mathtext renderer emits U+2212 for a minus sign while the artefacts carry
# the ASCII hyphen, so the extracted text is folded before it is compared.
FIGURE_MINUS = "\u2212"


def load_artifact(key: str):
    """Read one JSON artefact from the work root.

    Args:
        key: entry name inside the work section of config/paths.yaml.

    Returns:
        The decoded JSON object.
    """
    with work(key).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_figure_text(stem: str) -> str:
    """Return the text layer of a rendered figure, whitespace collapsed.

    Args:
        stem: file-name stem of the figure; the PDF is <stem>.pdf inside the
            results figures directory.

    Returns:
        The extracted text with every run of whitespace collapsed to one space.
    """
    document = fitz.open(FIG_DIR / f"{stem}.pdf")
    text = " ".join(page.get_text() for page in document)
    return re.sub(r"\s+", " ", text)


def build_expectations() -> dict[str, list[str]]:
    """Build the set of values each verified figure is expected to display.

    Returns:
        A dict mapping a figure stem to the list of printed values, in the order
        the panels present them.
    """
    batch = pd.read_csv(work("eeec_batch_verify_csv"))
    platform = load_artifact("platform_results_json")
    g2 = load_artifact("g2_refined")
    g3 = load_artifact("g3_power")

    figures: dict[str, list[str]] = {}

    # Figure 4: the batch-testbed metrics of every correction arm, then the
    # cross-platform ladder, its effect-size strata and the network level. The
    # sign-concordance column is printed as a percentage, which is why it is
    # scaled and given a percent sign.
    figures["Fig4_covariance_fragility"] = (
        [f"{value:.2f}" for value in batch["centroid_AUC"]]
        + [f"{value:.2f}" for value in batch["LR_AUC_abs"]]
        + [f"{value:.3f}" for value in batch["knn_mix"]]
        + [f"{platform['ladder'][i]['spearman']:.3f}" for i in range(len(platform["ladder"]))]
        + [f"{record['spearman']:.3f}" for record in platform["bins"]]
        + [f"{record['sign_conc'] * 100:.1f}%" for record in platform["bins"]]
        + [f"{platform['network']['network_spearman']:.3f}"]
    )

    # Figure 5: the frozen scalar values, the required sample sizes of the G2
    # table, the record count quoted in the annotation, the split-half ceilings
    # and the largest-cluster fraction behind every bootstrap ceiling. The
    # original applied a no-op "-0." rewrite to the split-half value; it is
    # dropped here because it cannot change the formatted string.
    figures["Fig5_thresholds_undecidable"] = (
        [str(value) for value in param("verification", "figure_5_values")]
        + [str(int(record["n_relevant"])) for record in g2["required_n"]]
        + [str(param("verification", "figure_5_available_records"))]
        + [
            f"{g3['split_half'][cohort][cluster]:.3f}"
            for cohort in COHORTS
            for cluster in CLUSTER_NUMBERS
        ]
        + [
            f"{g3['bootstrap_ceiling'][cohort][cluster]['maxfrac_base']:.3f}"
            for cohort in COHORTS
            for cluster in CLUSTER_NUMBERS
        ]
    )

    return figures


def report_missing_values(figures: dict[str, list[str]]) -> None:
    """Report, per figure, how many expected values the rendered text lacks.

    Args:
        figures: the expected values per figure stem.

    Returns:
        None. One line per figure is printed, plus the missing values.
    """
    for stem, expected in figures.items():
        text = read_figure_text(stem).replace(FIGURE_MINUS, "-")
        # fromkeys keeps first-occurrence order and drops duplicates.
        unique = list(dict.fromkeys(expected))
        missing = [value for value in unique if value not in text]
        print(
            f"-- {stem}: {len(unique)} distinct values expected, "
            f"{len(unique) - len(missing)} present, {len(missing)} missing"
        )
        if missing:
            print(f"     missing: {missing}")


def report_page_geometry() -> None:
    """Report the page box and any edge-flush text block of the main figures.

    Returns:
        None. One line per figure is printed.
    """
    print()
    for stem in GEOMETRY_FIGURES:
        document = fitz.open(FIG_DIR / f"{stem}.pdf")
        page = document[0]
        # A block whose box reaches the page edge was clipped by the tight
        # bounding box of the canvas: the label is there but unreadable.
        edge = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if (
                x1 > page.rect.width - EDGE_TOLERANCE_PT
                or y1 > page.rect.height - EDGE_TOLERANCE_PT
                or x0 < EDGE_TOLERANCE_PT
                or y0 < EDGE_TOLERANCE_PT
            ):
                edge.append(block)
        flag = "OK" if not edge else repr(edge[0][4][:BLOCK_ECHO_LENGTH])
        print(
            f"  {stem}: {page.rect.width:.0f}x{page.rect.height:.0f} pt  "
            f"text blocks touching the edge {len(edge)}  {flag}"
        )


def main() -> None:
    """Check the values of Figures 4 and 5 and the page box of all main figures."""
    report_missing_values(build_expectations())
    report_page_geometry()


if __name__ == "__main__":
    main()
