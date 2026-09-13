#!/usr/bin/env python3
"""
Inventory the NSMP material available for the gate-G3 threshold derivation.

Purpose
    Gate G3 fixes the decision threshold of the study from the reproducibility
    that the three cohorts of the comparison can actually deliver, and that
    derivation starts from the copy-number-low (NSMP) subtype. This module is the
    first, read-only step of the stage: it records what already exists before
    anything is derived from it. It locates NSMP artefacts left by earlier
    analysis phases, lists the molecular-subtype annotation columns of the two
    CPTAC metadata tables, and lists the subtype sources available on the TCGA
    side.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_independent_metadata") -- XLSX metadata table of the CPTAC-UCEC
        confirmatory (independent) cohort; must carry the subtype columns listed
        in params:nsmp.independent_report_columns.
    data("cptac_discovery_clinical") -- tab-separated clinical export of the
        CPTAC-UCEC discovery cohort.
    data("xena_ucec_clinical_matrix") -- TCGA-UCEC clinical matrix from Xena.
    data("xena_pancan_subtypes") -- pan-cancer molecular-subtype table from Xena.
    The work root and the repository root, searched for files whose name contains
    "nsmp" or "NSMP".

Outputs
    None. The module only prints an inventory; no artefact is written.

Usage
    python src/10_thresholds_gate_g3/10_01_inventory_nsmp_subsets.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

# Allow the module to be run from any directory: put src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import PROJECT_ROOT, data, param, work_root  # noqa: E402

# Glob patterns used when inventorying NSMP artefacts written by earlier analysis
# phases. The work root is searched recursively, because those phases wrote both
# flat and nested outputs, while the repository root is searched only at its top
# level, which is the only place such an artefact was ever kept there.
WORK_NSMP_GLOBS: tuple[str, ...] = ("*nsmp*", "**/*nsmp*", "*NSMP*", "**/*NSMP*")
REPO_NSMP_GLOBS: tuple[str, ...] = ("*nsmp*", "*NSMP*")

# Width of the section banners printed by this module.
BANNER_WIDTH = 90


def print_banner(title: str) -> None:
    """Print a section banner around ``title``.

    Args:
        title: the section heading.

    Returns:
        None. The banner is written to standard output.
    """
    print("=" * BANNER_WIDTH)
    print(title)
    print("=" * BANNER_WIDTH)


def search_previous_nsmp_artefacts() -> list[Path]:
    """List NSMP artefacts left in the work root and in the repository root.

    The search is deliberately broad and read-only: it answers "what material
    already exists?" before any of it is trusted, and a name hit is never used as
    evidence of a subtype annotation.

    Returns:
        The matching paths, in the order in which the search roots and patterns
        are declared and with duplicates removed. Every path is printed together
        with its size in bytes.
    """
    search_roots = ((work_root(), WORK_NSMP_GLOBS), (PROJECT_ROOT, REPO_NSMP_GLOBS))
    found: list[Path] = []
    for root, patterns in search_roots:
        for pattern in patterns:
            for hit in glob.glob(str(root / pattern), recursive=True):
                path = Path(hit)
                if path in found:
                    # The recursive patterns also match the flat ones, so the same
                    # file can be reached twice.
                    continue
                found.append(path)
                print(f"  {path}  ({path.stat().st_size} B)")
    if not found:
        print("  (no previous NSMP artefact found)")
    return found


def find_annotation_columns(columns, hints) -> list[str]:
    """Return the columns whose name contains any of ``hints``.

    Args:
        columns: iterable of column names of a metadata table.
        hints: lower-case substrings to look for, e.g. "subtype", "cnv".

    Returns:
        The matching column names, in the order in which they appear in
        ``columns``. A match only nominates a column for inspection; no value is
        ever derived from the column name.
    """
    names = [str(column) for column in columns]
    return [name for name in names if any(hint in name.lower() for hint in hints)]


def print_value_counts(frame: pd.DataFrame, column: str, limit: int | None = None) -> None:
    """Print the value counts of one annotation column.

    Args:
        frame: the metadata table.
        column: the column to summarise.
        limit: keep only the most frequent ``limit`` values; every value when
            None.

    Returns:
        None. The counts are written to standard output. ``dropna=False`` keeps
        the missing entries visible: a subtype column that is mostly empty is not
        usable as an annotation.
    """
    counts = frame[column].value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    print(f"  {column}: {counts.to_dict()}")


def inventory_cptac_independent() -> None:
    """Report the molecular-subtype columns of the CPTAC confirmatory cohort.

    The columns reported are listed in the configuration so that the inventory
    stays comparable between runs.

    Returns:
        None. The value counts are written to standard output.
    """
    frame = pd.read_excel(data("cptac_independent_metadata"))
    print(f"CPTAC independent cohort (V1) shape {frame.shape}")
    for column in param("nsmp", "independent_report_columns"):
        if column in frame.columns:
            print_value_counts(frame, column)


def inventory_cptac_discovery() -> None:
    """Report the candidate molecular-subtype columns of the CPTAC discovery cohort.

    The discovery export has no fixed subtype column list, so candidates are
    located by column name and the leading value counts of each are shown.

    Returns:
        None. The candidate columns and their counts are written to standard
        output.
    """
    frame = pd.read_csv(
        data("cptac_discovery_clinical"),
        sep="\t",
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    print(f"\nCPTAC discovery cohort (V2) shape {frame.shape}")
    candidates = find_annotation_columns(frame.columns, param("nsmp", "discovery_subtype_column_hints"))
    limit = param("nsmp", "value_counts_limit")
    print("  candidate subtype columns:", candidates)
    for column in candidates[:limit]:
        print_value_counts(frame, column, limit=limit)


def inventory_tcga() -> None:
    """Report the subtype sources available on the TCGA side.

    Returns:
        None. The columns found in the two Xena assets and their leading value
        counts are written to standard output.
    """
    clinical_path = data("xena_ucec_clinical_matrix")
    if clinical_path.exists():
        header = pd.read_csv(clinical_path, sep="\t", nrows=0)
        candidates = find_annotation_columns(header.columns, param("nsmp", "tcga_subtype_column_hints"))
        print(f"  clinicalMatrix columns {len(header.columns)} | candidates: {candidates}")
        # Only the sample barcode and the nominated columns are read: the matrix
        # is wide and the remaining columns are not part of the inventory.
        frame = pd.read_csv(
            clinical_path,
            sep="\t",
            low_memory=False,
            usecols=[param("nsmp", "tcga_sample_id_column")] + candidates if candidates else None,
        )
        for column in candidates:
            print_value_counts(frame, column, limit=param("nsmp", "value_counts_limit"))

    subtypes_path = data("xena_pancan_subtypes")
    if subtypes_path.exists():
        frame = pd.read_csv(subtypes_path, sep="\t", low_memory=False)
        print(f"\n  {subtypes_path.name}: {frame.shape} | columns {list(frame.columns)[:8]}")
        for column in frame.columns:
            if "subtype" in str(column).lower():
                print_value_counts(frame, column, limit=param("nsmp", "subtype_table_value_limit"))


def main() -> None:
    """Run the three inventory sections of the module.

    Returns:
        None. Everything is written to standard output.
    """
    print_banner("[1] Search for previous NSMP artefacts")
    search_previous_nsmp_artefacts()

    print("\n")
    print_banner("[2] Molecular-subtype fields per cohort")
    inventory_cptac_independent()
    inventory_cptac_discovery()

    print("\n")
    print_banner("[3] Subtype sources on the TCGA side")
    inventory_tcga()


if __name__ == "__main__":
    main()
