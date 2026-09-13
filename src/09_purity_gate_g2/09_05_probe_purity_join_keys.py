"""
Probe which identifier a harmonised layer matrix shares with the purity tables.

Purpose
    Fifth step of the purity gate (G2), and the step that makes the rest of the
    gate well defined. Every purity estimate is attached to a sample through an
    identifier, and the two CPTAC UCEC cohorts expose several of them (case,
    aliquot, sample). This module reads the column names of the harmonised layer
    matrices, prints the candidate identifier columns of the independent metadata
    table and of the discovery clinical table, and counts the intersection of the
    matrix columns with each candidate key.
    Only the key that intersects the matrix on every sample can be used by 09_06,
    and that is what the output documents. Nothing is stored.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("harmonised_matrix_pattern")  harmonised per-cohort layer matrices; only
        their header is read (nrows=0), so no matrix is loaded in full.
    data("cptac_independent_meta")     Excel metadata table of the CPTAC UCEC
        Independent cohort; the candidate identifier columns are read.
    data("cptac_discovery_clinical")   clinical table of the CPTAC UCEC Discovery
        cohort; the candidate identifier columns are read.

Outputs
    None. Matrix column counts, candidate keys and the size of each intersection
    are printed.

Usage
    python 09_05_probe_purity_join_keys.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md §5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.config import data, param, work

# Identifier columns of the two third-party tables, taken verbatim from their
# headers. They are data-schema names, not tunable settings.
INDEPENDENT_KEYS = ("Idx", "Case_id", "Aliquot_ID", "Sample_ID", "Case_excluded")
DISCOVERY_KEYS = (
    "idx",
    "Proteomics_Participant_ID",
    "Proteomics_Aliquot_ID",
    "Proteomics_Parent_Sample_IDs",
)

# The layer whose matrix header is used as the reference; every layer is keyed the
# same way, and mRNA exists for both cohorts.
REFERENCE_LAYER = "mRNA"

# Display limits: how many matrices per cohort and how many identifiers to show.
MAX_MATRICES_SHOWN = 3
MAX_MATRICES_SHOWN_SECOND = 2
MAX_IDENTIFIERS_SHOWN = 5


def read_csv_header(path: Path) -> pd.DataFrame:
    """Read only the header of a harmonised layer matrix.

    Args:
        path: matrix file.

    Returns:
        An empty DataFrame whose columns are the sample identifiers.
    """
    # `nrows=0` reads the header alone, which is all that is needed to compare
    # identifier systems and avoids loading a large matrix.
    matrix = pd.read_csv(path, index_col=0, nrows=0)
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix


def report_matrix_headers(cohort_tag: str, count: int) -> None:
    """Print the column count and the first sample identifiers of a cohort.

    Args:
        cohort_tag: cohort token as it appears in the file names.
        count: number of matrices to report, in file-name order.

    Returns:
        None. One line per matrix is printed.
    """
    paths = sorted(work("hcsv_dir").glob(f"{cohort_tag}__*.csv.gz"))[:count]
    for path in paths:
        matrix = read_csv_header(path)
        print(
            f"{path.name:26s} columns {matrix.shape[1]:5d} | "
            f"first {MAX_IDENTIFIERS_SHOWN}: {list(matrix.columns[:MAX_IDENTIFIERS_SHOWN])}"
        )


def report_candidate_keys(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    """Print example values of every candidate identifier column present.

    Args:
        frame: table whose identifier columns are reported.
        columns: candidate column names, reported only when present.
        label: heading used in the output.

    Returns:
        None. One line per column that exists is printed.
    """
    print(f"\n--- {label} ---")
    for column in columns:
        if column in frame.columns:
            values = list(frame[column].astype(str).head(MAX_IDENTIFIERS_SHOWN))
            print(f"  {column:28s} first {MAX_IDENTIFIERS_SHOWN}: {values}")


def report_intersections(
    independent: pd.DataFrame, discovery: pd.DataFrame
) -> None:
    """Count how many matrix columns each candidate key can join on.

    Args:
        independent: metadata table of the independent cohort.
        discovery: clinical table of the discovery cohort.

    Returns:
        None. The size of each matrix-column/key intersection is printed.
    """
    print("\n--- intersection check ---")
    independent_tag = param("cohort_tags", "independent")
    discovery_tag = param("cohort_tags", "discovery")
    layer_tag = param("layer_tags", REFERENCE_LAYER)
    matrix_independent = read_csv_header(
        Path(
            str(work("harmonised_matrix_pattern")).format(
                cohort=independent_tag, layer=layer_tag
            )
        )
    )
    matrix_discovery = read_csv_header(
        Path(
            str(work("harmonised_matrix_pattern")).format(
                cohort=discovery_tag, layer=layer_tag
            )
        )
    )
    # Each matrix is tested against both candidate identifiers of its own cohort,
    # so that a key that fails to join is visible rather than assumed away.
    comparisons = (
        (
            f"{independent_tag} Case_id",
            list(matrix_independent.columns),
            independent["Case_id"].astype(str),
        ),
        (
            f"{independent_tag} Aliquot_ID",
            list(matrix_independent.columns),
            independent["Aliquot_ID"].astype(str),
        ),
        (
            f"{discovery_tag} Participant",
            list(matrix_discovery.columns),
            discovery["Proteomics_Participant_ID"].astype(str),
        ),
    )
    for name, matrix_columns, key in comparisons:
        intersection = len(set(matrix_columns) & set(key))
        print(
            f"  {name:26s} matrix columns {len(matrix_columns)} vs key {len(key)} "
            f"-> intersection {intersection}"
        )


def main() -> None:
    """Probe the join keys between layer matrices and purity tables. Returns None."""
    independent_tag = param("cohort_tags", "independent")
    discovery_tag = param("cohort_tags", "discovery")
    report_matrix_headers(independent_tag, MAX_MATRICES_SHOWN)
    report_matrix_headers(discovery_tag, MAX_MATRICES_SHOWN_SECOND)

    independent = pd.read_excel(data("cptac_independent_meta"))
    report_candidate_keys(
        independent, INDEPENDENT_KEYS, f"{independent_tag} metadata identifier fields"
    )

    discovery = pd.read_csv(
        data("cptac_discovery_clinical"),
        sep="\t",
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )
    report_candidate_keys(
        discovery, DISCOVERY_KEYS, f"{discovery_tag} clinical identifier fields"
    )

    report_intersections(independent, discovery)


if __name__ == "__main__":
    main()
