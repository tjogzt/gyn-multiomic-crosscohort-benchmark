"""
Inspect the purity column of the CPTAC UCEC Discovery clinical table.

Purpose
    Third step of the purity gate (G2). The Discovery cohort reports more than one
    candidate purity column and one of them, Tumor_purity, is not usable as a
    number, so the choice of column has to be made on the raw values rather than
    on the column name. This module prints the raw values, the dtype and the
    coverage of Tumor_purity, the list of all other column names containing the
    fragment "pur", the identifier columns the purity table shares with the layer
    matrices, and the distribution of the tumour/normal flag.
    It stores nothing: the output documents why 09_06 reads Purity_Cancer.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_discovery_clinical")  tab-separated clinical table of the CPTAC
        UCEC Discovery cohort; must carry Tumor_purity, the identifier columns and
        the tumour/normal flag.

Outputs
    None. The column contents, coverage counts and value distributions are
    printed.

Usage
    python 09_03_check_discovery_purity_column.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md §5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.config import data

# Column names taken verbatim from the Discovery clinical table.
PURITY_COLUMN = "Tumor_purity"
ALIQUOT_ID = "Proteomics_Aliquot_ID"
PARTICIPANT_ID = "Proteomics_Participant_ID"
TUMOR_NORMAL = "Proteomics_Tumor_Normal"
INDEX_COLUMN = "idx"

# Fragment used to search for further purity-like columns by name.
PURITY_NAME_FRAGMENT = "pur"

# Display limits of the report.
MAX_UNIQUE_VALUES = 10
MAX_RAW_VALUES = 8
MAX_IDENTIFIERS = 3
MAX_COLUMNS_COVERAGE = 25


def read_clinical_tsv(path: Path) -> pd.DataFrame:
    """Read the tab-separated Discovery clinical table.

    Args:
        path: table to read.

    Returns:
        The table as read, with every column kept.
    """
    # See 09_01 for why both decoding options are needed.
    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def report_purity_column(clinical: pd.DataFrame) -> None:
    """Print the raw values and coverage of the nominal purity column.

    Args:
        clinical: the Discovery clinical table.

    Returns:
        None. Raw values, dtype, coverage and coercibility are printed.
    """
    print("shape:", clinical.shape)
    print(f"\n{PURITY_COLUMN} column:")
    series = clinical[PURITY_COLUMN]
    print(
        f"  dtype: {series.dtype} | non-null: {series.notna().sum()} | "
        f"first {MAX_UNIQUE_VALUES} unique values: "
        f"{list(series.dropna().unique())[:MAX_UNIQUE_VALUES]}"
    )
    print(f"  first {MAX_RAW_VALUES} raw values: {list(series.head(MAX_RAW_VALUES))}")
    print("\nstatistics after numeric coercion:")
    numeric = pd.to_numeric(series, errors="coerce")
    print(
        f"  non-null: {numeric.notna().sum()} | mean: {numeric.mean()} | "
        f"range: {(numeric.min(), numeric.max())}"
    )
    # The column may be encoded as text (percentages, or a comma decimal mark),
    # which is exactly what would silently produce an all-NaN purity vector if it
    # were used without inspection.
    print(
        "\nfurther column names containing 'pur':",
        [c for c in clinical.columns if PURITY_NAME_FRAGMENT in str(c).lower()],
    )


def report_identifiers(clinical: pd.DataFrame) -> None:
    """Print sample identifier examples and the tumour/normal distribution.

    Args:
        clinical: the Discovery clinical table.

    Returns:
        None. Identifier examples, the flag distribution and per-column coverage
        are printed.
    """
    print(
        f"\n{ALIQUOT_ID} example values:",
        list(clinical[ALIQUOT_ID].astype(str).head(MAX_IDENTIFIERS)),
    )
    print(
        f"{INDEX_COLUMN} example values:",
        list(clinical[INDEX_COLUMN].astype(str).head(MAX_IDENTIFIERS)),
    )
    print(f"\n{TUMOR_NORMAL} distribution:")
    print(clinical[TUMOR_NORMAL].value_counts().to_string())
    print(f"\nnon-null counts of the first {MAX_COLUMNS_COVERAGE} columns:")
    print(clinical.notna().sum().head(MAX_COLUMNS_COVERAGE).to_string())


def main() -> None:
    """Report the candidate purity column of the Discovery cohort. Returns None."""
    clinical = read_clinical_tsv(data("cptac_discovery_clinical"))
    report_purity_column(clinical)
    report_identifiers(clinical)


if __name__ == "__main__":
    main()
