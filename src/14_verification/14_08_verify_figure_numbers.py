#!/usr/bin/env python3
"""
Check that every number displayed in Figures 1 to 3 was produced by an artefact.

Purpose
    Stage 14, eighth script. A published figure is defensible only if each value
    printed inside it can be traced back to the artefact that produced it. The
    script rebuilds the expected value set of Figures 1 to 3 from the Python
    benchmark, the merged toolchain table, the sample-availability table and the
    protein-bridge records, extracts the text layer of each rendered figure PDF
    and reports the expected values that are absent. It then counts how many
    three-decimal values a figure carries, which is the population the expected
    set is drawn from, so a shrinking set is visible next to the totals.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("benchmark_summary")           JSON, tab1 (method means) and tab3 (layer
                                        curve of the fixed best method).
    work("benchmark_merged")            JSON, r_layer_effect and r_ranking of the
                                        merged Python and R toolchains.
    work("benchmark_availability")      JSON, record per (domain, cohort, subset)
                                        with the usable sample count in "n".
    work("protein_bridge_three_cohort") JSON, "raw" and "delta" concordance per
                                        cohort pair, each with a Spearman "rho".
    work("protein_bridge_anchors")      JSON, within-cohort mRNA-protein anchors.
    results("figures_dir")/<stem>.pdf   the rendered figure whose text layer is
                                        read; one PDF per verified stem.
    config/params.yaml verification.figure_3_extra_tokens
                                        extra values Figure 3 must display.
    config/params.yaml domains          the two evaluation-domain labels.

Outputs
    None. The per-figure report is printed.

Usage
    python 14_08_verify_figure_numbers.py
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

# A benchmark record names its evaluation domain by the label of params.domains,
# so the labels are read from the configuration rather than written out; a
# renamed domain must not be able to hide a missing value.
DOMAIN = param("domains")

# The R-side layer curve is keyed "<domain letter>|<number of layers>". The keys
# are the artefact's own vocabulary and cover both domains and every layer count
# the merge reports, which is why they are listed rather than derived.
R_LAYER_KEYS = ["A|1", "A|2", "A|3", "B|1", "B|2", "B|3", "B|4"]

# mRNA-protein anchor records shown in Figure 1, in panel order: the raw and the
# tumour-minus-normal contrast of the discovery and the independent cohort.
ANCHOR_KEYS = [
    "dis_raw_Prot_RNA",
    "dis_dProt_dRNA",
    "ind_raw_Prot_RNA",
    "ind_dProt_dRNA",
]

# The mathtext renderer emits U+2212 for a minus sign while the artefacts carry
# the ASCII hyphen, so the extracted text is folded before it is compared.
# Without the fold every negative value would be reported as missing.
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
    """Return the text layer of a rendered figure.

    Args:
        stem: file-name stem of the figure; the PDF is <stem>.pdf inside the
            results figures directory.

    Returns:
        The extracted text with every run of whitespace collapsed to one space,
        so a value split over two lines still matches its expectation.
    """
    document = fitz.open(FIG_DIR / f"{stem}.pdf")
    text = " ".join(page.get_text() for page in document)
    return re.sub(r"\s+", " ", text)


def resolve_domain_key(table: dict, label: str) -> str:
    """Resolve the key a summary table uses for one evaluation domain.

    Args:
        table: the summary table, keyed by a domain label.
        label: the domain label declared in params.domains.

    Returns:
        The key of ``table`` holding that domain. The configured label is
        preferred. When the artefact keys the table with a longer display label
        - which the released artefacts did - the single key carrying the first
        token of the configured label is used and the difference is reported, so
        a renamed domain surfaces in the output instead of raising a KeyError and
        hiding every other finding of the run.
    """
    if label in table:
        return label
    token = str(label).split("_")[0]
    matches = [key for key in table if token in str(key)]
    if len(matches) == 1:
        print(f"   note: domain {label} is keyed {matches[0]!r} in the artefact")
        return matches[0]
    return label


def availability_counts(availability: pd.DataFrame) -> list[str]:
    """Return the usable sample count of every (domain, cohort, layer count) block.

    Args:
        availability: the sample-availability records, one row per
            (domain, cohort, layer subset) with the count in column "n".

    Returns:
        One string per block: the largest count over the subsets of a given
        layer count, formatted as an integer, in the group order of the table.
    """
    frame = availability.copy()
    # The number of layers of a subset is the number of layers joined by "+".
    frame["nL"] = frame["subset"].map(lambda subset: len(str(subset).split("+")))
    # Several subsets share a layer count; the panel prints the block maximum.
    grouped = frame.groupby(["domain", "cohort", "nL"])["n"].max()
    return [str(int(value)) for value in grouped.values]


def build_expectations() -> dict[str, list[str]]:
    """Build the set of values every verified figure is expected to display.

    Returns:
        A dict mapping a figure stem to the list of printed values, in the order
        the panels present them.
    """
    summary = load_artifact("benchmark_summary")
    merged = load_artifact("benchmark_merged")
    availability = pd.DataFrame(load_artifact("benchmark_availability"))
    bridge = load_artifact("protein_bridge_three_cohort")
    anchors = load_artifact("protein_bridge_anchors")
    counts = availability_counts(availability)

    figures: dict[str, list[str]] = {}

    # Figure 1: sample availability, the raw and the tumour-minus-normal protein
    # concordance of every cohort pair in "raw"/"delta", and the four anchors.
    # The pair sections are iterated rather than listed, so the set follows the
    # artefact if a cohort pair is ever added.
    figures["Fig1_availability_comparability"] = (
        counts
        + [f"{record['rho']:.3f}" for record in bridge["raw"].values()]
        + [f"{record['rho']:.3f}" for record in bridge["delta"].values()]
        + [f"{anchors[key]['rho']:.3f}" for key in ANCHOR_KEYS]
    )

    # Figure 2: the layer curve of both toolchains plus the intersection counts.
    figures["Fig2_layer_negative_marginal"] = (
        [
            f"{record['ari']:.3f}"
            for label in DOMAIN.values()
            for record in summary["tab3"][resolve_domain_key(summary["tab3"], label)]
        ]
        + [f"{merged['r_layer_effect'][key]:.3f}" for key in R_LAYER_KEYS]
        + counts
    )

    # Figure 3: the two method rankings and the frozen extra tokens of the panel,
    # which are the values the panel annotates rather than plots.
    figures["Fig3_method_not_resolvable"] = (
        [f"{summary['tab1'][method]['all']:.3f}" for method in summary["tab1"]]
        + [f"{merged['r_ranking'][method]:.3f}" for method in merged["r_ranking"]]
        + [str(token) for token in param("verification", "figure_3_extra_tokens")]
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
        # fromkeys keeps first-occurrence order and drops duplicates, so a value
        # printed in several panels is reported once and the count is comparable
        # between runs.
        unique = list(dict.fromkeys(expected))
        missing = [value for value in unique if value not in text]
        print(
            f"-- {stem}: {len(unique)} distinct values expected, "
            f"{len(missing)} missing"
        )
        if missing:
            print(f"     missing: {missing}")


def report_value_totals(figures: dict[str, list[str]]) -> None:
    """Report how many three-decimal values each figure displays.

    Args:
        figures: the expected values per figure stem; only its keys are used.

    Returns:
        None. One line per figure is printed.
    """
    for stem in figures:
        text = read_figure_text(stem).replace(FIGURE_MINUS, "-")
        numbers = set(re.findall(r"-?\d+\.\d{3}", text))
        print(f"   {stem}: {len(numbers)} values with three decimals in the figure")


def main() -> None:
    """Rebuild the expected values and check them against the rendered figures."""
    figures = build_expectations()
    report_missing_values(figures)
    report_value_totals(figures)


if __name__ == "__main__":
    main()
