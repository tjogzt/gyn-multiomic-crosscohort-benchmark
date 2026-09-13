#!/usr/bin/env python3
"""
Build the EEEC batch-testbed matrix from the MaxQuant proteinGroups table.

Purpose
    Stage 07, first step. The early-onset endometrioid carcinoma (EEEC) cohort is
    a label-free proteome in which the sample families E/L (two processing
    batches) x can/norm (tumour / normal-looking tissue) are fully crossed with
    samples_per_cell samples per cell. That crossing is what makes the cohort
    usable as a batch testbed: batch and condition are orthogonal, so a
    correction that removes batch cannot be confused with one that removes
    biology. This script reads the proteinGroups table, keeps the quantified
    protein groups that pass the MaxQuant quality flags, restricts the samples to
    the four balanced cells, filters protein groups that are not measured in
    enough samples of every cell, imputes the remaining gaps with the per-cell
    median, and writes the log2 matrix with the sample annotation that the later
    steps of the stage consume.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data.eeec_protein_groups
        MaxQuant proteinGroups.txt of PRIDE project PXD046507, tab separated.
        Read columns: protein identifiers, gene names, the three MaxQuant quality
        flags and one "LFQ intensity <sample>" column per sample
        (params batch_testbed.protein_group_columns, batch_testbed.lfq_prefix).
    params batch_testbed.min_cell_completeness, batch_testbed.sample_families,
    params batch_testbed.balanced_families.

Outputs
    work.eeec_batch_matrix
        log2 LFQ matrix, protein groups in rows and samples in columns
        (CSV).
    work.eeec_batch_sample_meta
        One row per sample: column name, family, suffix, batch and condition.
    work.eeec_protein_gene_map
        Protein identifier to gene symbol map (CSV).
    work.eeec_batch_info
        JSON summary: protein and sample counts, the cell design and the missing
        rate after filtering.

Usage
    python src/07_batch_testbed/07_01_build_eeec_matrix.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The package root (one level above src/) so that "common" can be imported from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, ensure_dir, param, work

# --- Configuration -----------------------------------------------------------

# Column layout of the MaxQuant table: identifier, gene name, then the three
# quality flags in the order documented under the parameter's comment.
PROTEIN_GROUP_COLUMNS = param("batch_testbed", "protein_group_columns")
PROTEIN_ID_COLUMN = PROTEIN_GROUP_COLUMNS[0]
GENE_NAME_COLUMN = PROTEIN_GROUP_COLUMNS[1]
QUALITY_FLAG_COLUMNS = PROTEIN_GROUP_COLUMNS[2:]

LFQ_PREFIX = param("batch_testbed", "lfq_prefix")
SAMPLE_FAMILIES = [str(family) for family in param("batch_testbed", "sample_families")]
BALANCED_FAMILIES = [str(family) for family in param("batch_testbed", "balanced_families")]
MIN_CELL_COMPLETENESS = float(param("batch_testbed", "min_cell_completeness"))

SOURCE = data("eeec_protein_groups")
OUT_DIR = ensure_dir(work("eeec_batch_dir"))
MATRIX_PATH = work("eeec_batch_matrix")
SAMPLE_META_PATH = work("eeec_batch_sample_meta")
PROTEIN_GENE_PATH = work("eeec_protein_gene_map")
INFO_PATH = work("eeec_batch_info")

# MaxQuant writes a "+" in a quality flag column for a flagged protein group and
# leaves the cell empty otherwise; the flag columns are compared against this.
FLAG_VALUE = "+"


def read_lfq_column_names(path: Path) -> list:
    """Return the sample columns of the proteinGroups table.

    Args:
        path: proteinGroups.txt.

    Returns:
        The header entries that start with the LFQ intensity prefix, in file
        order. Only the header is read here: the table itself is 700 MB, so the
        column list is derived before the full read.
    """
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    return [column for column in header if column.startswith(LFQ_PREFIX)]


def load_protein_groups(path: Path, lfq_columns: list) -> pd.DataFrame:
    """Read the identity, gene and intensity columns of the proteinGroups table.

    Args:
        path: proteinGroups.txt.
        lfq_columns: sample columns to read.

    Returns:
        The raw table restricted to the configured columns. encoding_errors is
        set to "replace" because the MaxQuant export contains non-UTF-8 bytes.
    """
    return pd.read_csv(
        path,
        sep="\t",
        usecols=list(PROTEIN_GROUP_COLUMNS) + list(lfq_columns),
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def filter_quality_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Drop protein groups flagged by MaxQuant.

    Args:
        table: raw proteinGroups table.

    Returns:
        The table without reversed hits, potential contaminants and protein
        groups identified by the site modification only — none of them is a
        quantified protein of the sample.
    """
    keep = pd.Series(True, index=table.index)
    for column in QUALITY_FLAG_COLUMNS:
        keep &= table[column] != FLAG_VALUE
    filtered = table[keep].copy()
    return filtered


def parse_family(column: str) -> tuple:
    """Split a sample column into its family and its within-family suffix.

    Args:
        column: LFQ intensity column name.

    Returns:
        (family, suffix) where the family is the sample-name prefix listed in
        batch_testbed.sample_families and the suffix is the remainder of the
        name. A name that matches no family is reported as ("other", name) so
        that it stays visible in the design table instead of being dropped
        silently.
    """
    name = column.replace(LFQ_PREFIX, "").strip()
    for family in SAMPLE_FAMILIES:
        if name.startswith(family):
            return family, name[len(family):]
    return "other", name


def build_sample_table(lfq_columns: list) -> pd.DataFrame:
    """Build the sample annotation of the design.

    Args:
        lfq_columns: sample columns of the table.

    Returns:
        One row per sample with its column name, family, suffix, batch and
        condition. The family encodes both design factors: its first letter is
        the batch and the remainder is the condition.
    """
    meta = pd.DataFrame(
        [
            {"col": column, **dict(zip(["family", "suffix"], parse_family(column)))}
            for column in lfq_columns
        ]
    )
    meta["batch"] = meta["family"].str[0]
    meta["cond"] = meta["family"].str[1:]
    return meta


def filter_by_cell_completeness(
    values: pd.DataFrame, meta: pd.DataFrame, min_completeness: float
) -> tuple:
    """Keep protein groups quantified in at least min_completeness of every cell.

    Args:
        values: protein groups x samples intensity table.
        meta: sample annotation with the batch and condition of every sample.
        min_completeness: minimum fraction of non-missing samples per cell.

    Returns:
        (filtered table, boolean mask of the retained protein groups). A protein
        group must pass the threshold in *every* batch x condition cell: a
        protein measured only in one batch would otherwise carry a batch effect
        by construction.
    """
    keep = pd.Series(True, index=values.index)
    for (_, _), group in meta.groupby(["batch", "cond"]):
        block = values[group["col"].tolist()]
        keep &= block.notna().mean(axis=1) >= min_completeness
    return values[keep], keep


def impute_cell_medians(values: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining gaps with the per-protein median of the same cell.

    Args:
        values: protein groups x samples intensity table.
        meta: sample annotation with the batch and condition of every sample.

    Returns:
        A copy of the table in which every missing entry is replaced by the
        median of that protein group within its batch x condition cell. Imputing
        within a cell cannot move the cell mean, so it does not create or remove
        batch separation; the missing rate after the completeness filter is small
        enough for this to be an acceptable substitute for a model-based
        imputation.
    """
    filled = values.copy()
    for (_, _), group in meta.groupby(["batch", "cond"]):
        columns = group["col"].tolist()
        medians = filled[columns].median(axis=1)
        filled[columns] = filled[columns].apply(lambda row: row.fillna(medians), axis=0)
    return filled


def main() -> dict:
    """Build and write the batch-testbed matrix and its annotation tables.

    Returns:
        The summary dict that was written to work.eeec_batch_info.
    """
    lfq_columns = read_lfq_column_names(SOURCE)
    print(f"Reading {SOURCE} ({len(lfq_columns)} LFQ intensity columns)")
    table = load_protein_groups(SOURCE, lfq_columns)
    print(f"Raw protein groups: {len(table)}")

    table = filter_quality_flags(table)
    print(f"After reverse / contaminant / only-by-site filtering: {len(table)}")

    values = table[lfq_columns].apply(pd.to_numeric, errors="coerce")
    # MaxQuant codes a non-quantified protein group as LFQ intensity 0; it is a
    # missing value, not a measurement of zero.
    values[values <= 0] = np.nan
    values.index = table[PROTEIN_ID_COLUMN].astype(str)
    genes = table[GENE_NAME_COLUMN].astype(str).str.split(";").str[0].str.strip()
    print(
        f"Protein groups {values.shape[0]} x samples {values.shape[1]} | "
        f"with a gene name {int((genes != '').sum())}"
    )

    meta = build_sample_table(lfq_columns)
    print("\nFamily x batch x condition:")
    print(meta.groupby(["family", "batch", "cond"]).size().to_string())

    # The four balanced cells are the main analysis subset; the remaining
    # families are outside the crossed design and are not analysed here.
    subset_meta = meta[meta["family"].isin(BALANCED_FAMILIES)].copy()
    print(
        f"\nMain analysis samples: {len(subset_meta)} "
        f"({subset_meta.groupby(['batch', 'cond']).size().to_dict()})"
    )

    # Columns are taken in the order of the sample annotation so that the matrix
    # and the annotation table stay aligned row by row.
    subset_values = values.loc[:, subset_meta["col"].tolist()]
    subset_values, _ = filter_by_cell_completeness(
        subset_values, subset_meta, MIN_CELL_COMPLETENESS
    )
    print(
        f"Protein groups after the {MIN_CELL_COMPLETENESS:.0%} per-cell completeness filter: "
        f"{subset_values.shape[0]}"
    )
    print(f"Missing rate: {subset_values.isna().mean().mean() * 100:.2f}%")

    subset_values = impute_cell_medians(subset_values, subset_meta)
    print(f"Missing entries after imputation: {int(subset_values.isna().sum().sum())}")

    log2_matrix = np.log2(subset_values)
    log2_matrix.index.name = "protein"
    log2_matrix.to_csv(MATRIX_PATH)
    subset_meta.to_csv(SAMPLE_META_PATH, index=False)
    # The gene lookup keeps the legacy positional reindex: the gene series is
    # indexed by the row number of the proteinGroups table, not by the protein
    # identifier, so the column is empty here and is rebuilt in stage 08/11
    # (08_02_repair_gene_identifier_mismatch.py) from the protein table. It is
    # reproduced unchanged so that the repair step keeps receiving the same input.
    pd.DataFrame(
        {"protein": subset_values.index, "gene": genes.reindex(subset_values.index).values}
    ).to_csv(PROTEIN_GENE_PATH, index=False)

    info = {
        "n_protein": int(log2_matrix.shape[0]),
        "n_sample": int(log2_matrix.shape[1]),
        "design": {
            f"{batch}_{condition}": int(count)
            for (batch, condition), count in subset_meta.groupby(["batch", "cond"]).size().items()
        },
        "missing_pct_after_filter": float(subset_values.isna().mean().mean() * 100),
    }
    with INFO_PATH.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=1)
    print(f"\nWritten to {OUT_DIR}")
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return info


if __name__ == "__main__":
    main()
