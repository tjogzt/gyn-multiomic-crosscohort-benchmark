"""
Show the raw values of every purity column of the CPTAC UCEC Discovery cohort.

Purpose
    Fourth step of the purity gate (G2), and the follow-up to 09_03. Three of the
    four candidate purity columns of the Discovery cohort, Purity_Cancer,
    Purity_Immune and Purity_Stroma, come out of one decomposition and only one of
    them is usable as the tumour purity. This module prints the raw values of each
    candidate, how many of them survive numeric coercion, and their range, so that
    the column read by 09_06 is chosen from the observed content.
    It stores nothing.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_discovery_clinical")  tab-separated clinical table of the CPTAC
        UCEC Discovery cohort; the four candidate purity columns are read.

Outputs
    None. Raw values, coercibility and ranges are printed.

Usage
    python 09_04_check_raw_column_types.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md §5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.config import data

# The four purity-related columns of the Discovery clinical table, taken verbatim
# from its header: one tumour fraction and the three component fractions.
PURITY_COLUMNS = ("Tumor_purity", "Purity_Immune", "Purity_Cancer", "Purity_Stroma")

# Display limits of the report.
MAX_UNIQUE_VALUES = 12


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


def report_column(clinical: pd.DataFrame, column: str) -> None:
    """Print the raw content and the coercibility of one purity column.

    Args:
        clinical: the Discovery clinical table.
        column: name of the purity column to report.

    Returns:
        None. Raw values, coercibility and the numeric range are printed.
    """
    if column not in clinical.columns:
        print(f"{column}: column absent")
        return
    series = clinical[column]
    print("=" * 70)
    print(f"{column}  | dtype={series.dtype} | non-null {series.notna().sum()}/{len(series)}")
    values = series.dropna().astype(str)
    print(f"  first {MAX_UNIQUE_VALUES} unique values:", list(values.unique())[:MAX_UNIQUE_VALUES])
    # Some exports carry thousands separators, which would defeat a plain numeric
    # coercion; removing them is what makes the count below meaningful.
    numeric = pd.to_numeric(values.str.replace(",", ""), errors="coerce")
    print(f"  coercible after removing thousands separators: {numeric.notna().sum()}/{len(values)}")
    if numeric.notna().sum():
        print(f"    mean {numeric.mean():.4f} range [{numeric.min():.4f},{numeric.max():.4f}]")


def main() -> None:
    """Report every candidate purity column of the Discovery cohort. Returns None."""
    clinical = read_clinical_tsv(data("cptac_discovery_clinical"))
    for column in PURITY_COLUMNS:
        report_column(clinical, column)


if __name__ == "__main__":
    main()
