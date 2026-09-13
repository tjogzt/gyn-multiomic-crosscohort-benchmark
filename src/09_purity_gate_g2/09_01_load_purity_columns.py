"""
Inventory the purity and cell-composition columns that the cohorts actually ship.

Purpose
    Opening step of the purity gate (G2). The gate asks whether purity-aware
    correction improves cross-cohort transfer, and before any correction can be
    attempted the purity annotation has to be located and typed. This module
    inspects the clinical tables of the CPTAC UCEC Discovery cohort, the CPTAC
    ovarian cohort and the CPTAC UCEC Independent cohort metadata, and lists the
    Xena directory tree, reporting every column whose name suggests a purity or
    cell-composition measurement together with its coverage and numeric range.
    It stores nothing: its console output is the evidence on which the column
    choices of the later stages of stage 09 are based.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_discovery_clinical")  tab-separated clinical table of the CPTAC
        UCEC Discovery cohort; must carry the purity and tumour/normal columns.
    data("cptac_ov_clinical")         tab-separated clinical table of the CPTAC
        ovarian cohort.
    data("cptac_independent_meta")    Excel metadata table of the CPTAC UCEC
        Independent cohort; must carry the ABSOLUTE purity column.
    data("xena_dir")                  directory tree of the Xena downloads, listed
        only, to locate purity, sample-map or phenotype files.

Outputs
    None. Candidate columns, their coverage and their numeric summaries are
    printed.

Usage
    python 09_01_load_purity_columns.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md §5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.config import data, param

# Column-name fragments that nominate a purity or cell-composition column. The
# third-party tables are annotated inconsistently, so candidates are located by
# name. The fragments are a configuration parameter because a further cohort may
# use one that is not covered here; a match only nominates a column for
# inspection and no value is ever derived from the name.
PURITY_HINTS = [str(hint).lower() for hint in param("purity", "column_hints")]

# The Excel metadata table of the independent cohort resolves purity through the
# ABSOLUTE algorithm only and records no leukocyte fraction, so it is searched
# with its own fragment list.
METADATA_PURITY_HINTS = [
    str(hint).lower() for hint in param("purity", "column_hints_metadata")
]

# File-name fragments that make a Xena file worth listing: purity estimates, the
# sample map that links aliquots to patients, and phenotype/clinical annotation.
XENA_FILE_HINTS = ("purit", "absolute", "samplemap", "phenotype", "clinical")

# Display limits of the console report. These truncate the report only; no
# analysis input is ever truncated.
MAX_COLUMNS_SHOWN = 30
MAX_COLUMNS_SHOWN_METADATA = 40
MAX_DETAILED_HINT_COLUMNS = 6


def read_clinical_tsv(path: Path) -> pd.DataFrame:
    """Read a tab-separated clinical table of the CPTAC LinkedOmics export.

    Args:
        path: table to read.

    Returns:
        The table as read, with every column kept.

    Raises:
        Exception: whatever pandas raises; the caller reports it and moves on to
            the next table.
    """
    # `encoding_errors="replace"` keeps a row that contains a byte which is not
    # valid UTF-8 rather than failing the whole file, which the LinkedOmics
    # exports do contain. `low_memory=False` stops pandas from inferring dtypes
    # chunk by chunk, which would otherwise split a mixed column into fragments.
    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def report_table(path: Path, label: str, reader=None) -> pd.DataFrame | None:
    """Print shape, candidate purity columns and column list of one table.

    Args:
        path: file to report on.
        label: cohort name used in the section heading.
        reader: optional zero-argument callable returning the DataFrame; used for
            the Excel table, which needs a different reader.

    Returns:
        The table that was read, or None when it could not be read.
    """
    print("=" * 88)
    print(label, "|", path)
    try:
        frame = reader() if reader else read_clinical_tsv(path)
    except Exception as exc:  # a missing or unreadable cohort must not abort the inventory
        print("  read failed:", str(exc)[:120])
        return None
    print(f"  shape {frame.shape}")
    columns = list(frame.columns)
    candidates = [
        column
        for column in columns
        if any(hint in column.lower() for hint in PURITY_HINTS)
    ]
    print(f"  candidate purity/composition columns: {candidates if candidates else 'none'}")
    if candidates:
        for column in candidates[:MAX_DETAILED_HINT_COLUMNS]:
            values = pd.to_numeric(frame[column], errors="coerce")
            print(
                f"    {column}: non-null {values.notna().sum()}/{len(frame)} "
                f"mean {values.mean():.4f} range [{values.min():.3f},{values.max():.3f}]"
            )
    print(f"  all columns (first {MAX_COLUMNS_SHOWN}): {columns[:MAX_COLUMNS_SHOWN]}")
    return frame


def report_metadata(path: Path) -> None:
    """Print the candidate purity columns of the Excel metadata table.

    Args:
        path: Excel metadata table of the CPTAC UCEC Independent cohort.

    Returns:
        None. The shape, candidate columns and column list are printed.
    """
    print("=" * 88)
    print("CPTAC UCEC independent (Excel metadata table)")
    try:
        meta = pd.read_excel(path)
    except Exception as exc:
        print("  failed:", str(exc)[:150])
        return
    print(f"  shape {meta.shape}")
    columns = [str(column) for column in meta.columns]
    candidates = [
        column
        for column in columns
        if any(hint in column.lower() for hint in METADATA_PURITY_HINTS)
    ]
    print(f"  candidate purity columns: {candidates if candidates else 'none'}")
    for column in candidates[:MAX_DETAILED_HINT_COLUMNS]:
        values = pd.to_numeric(meta[column], errors="coerce")
        print(
            f"    {column}: non-null {values.notna().sum()}/{len(meta)} "
            f"mean {values.mean():.4f}"
        )
    print(f"  all columns: {columns[:MAX_COLUMNS_SHOWN_METADATA]}")


def report_xena_tree(xena_dir: Path) -> None:
    """List Xena files that may carry purity or phenotype annotation.

    Args:
        xena_dir: directory tree of the Xena downloads.

    Returns:
        None. Matching files and a per-directory file count are printed.
    """
    print("\n" + "=" * 88)
    print("Xena directory")
    for entry in sorted(xena_dir.glob("*/*")):
        if any(hint in entry.name.lower() for hint in XENA_FILE_HINTS):
            print("  ", entry.name, f"({entry.stat().st_size // 1024} KB)")
    print("\n  Xena directory overview:")
    for sub in sorted(path for path in xena_dir.glob("*") if path.is_dir()):
        print("   ", sub, "->", len(list(sub.glob("*"))), "files")


def main() -> None:
    """Run the purity-column inventory over every cohort. Returns None."""
    report_table(data("cptac_discovery_clinical"), "CPTAC UCEC Discovery")
    report_table(data("cptac_ov_clinical"), "CPTAC OV Prospective")
    report_metadata(data("cptac_independent_meta"))
    report_xena_tree(data("xena_dir"))


if __name__ == "__main__":
    main()
