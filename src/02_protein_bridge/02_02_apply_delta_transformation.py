#!/usr/bin/env python3
"""
Apply the tumour-minus-normal delta transformation to the protein layer and
re-test the standardisation strategies on the expression layer.

Purpose
    This module is the second half of the protein bridge. Part one quantifies the
    bridge itself: it splits the Independent cohort into tumour and normal arms
    from its metadata, reduces each cohort's protein matrix to a gene-mean
    spectrum, and compares (a) the raw cross-cohort spectra with (b) their
    within-cohort tumour-minus-normal contrasts, reporting the factor by which the
    contrast improves cross-cohort agreement. Part two asks whether a purely
    computational standardisation reaches the same improvement: the raw, the
    quantile-normalised and the per-sample-ranked expression matrices of four
    cohorts are compared pairwise on their gene-mean spectra. The delta
    transformation is the bridge that the later stages use; the standardisation
    comparison shows why a matrix-level transform cannot replace it.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_discovery_prot_tumor")    .cct, gene-level TMT tumour matrix
    data("cptac_discovery_prot_normal")   .cct, gene-level TMT normal matrix
    data("cptac_ov_prot_tumor")           .cct, gene-level OV tumour matrix
    data("cptac_ov_prot_normal")          .cct, gene-level OV normal matrix
    data("cptac_independent_prot_ratio")  .cct, tumour/normal log2 ratio matrix
    data("cptac_independent_metadata")    .xlsx, its sample metadata; the Group
        and Case_id columns are used to split the Independent cohort
    data("cptac_discovery_rna_tumor")     .cct, Discovery cohort RNA-seq
    data("cptac_ov_rna")                  .cct, Ovarian cohort RNA-seq
    data("cptac_independent_rna")         .cct, Independent cohort RNA-seq
    data("xena_pancan_rnaseq_eb_adjusted") .xena.gz, TCGA pan-cancer RNA-seq

Outputs
    work("protein_bridge_three_cohort")   .json, raw and delta correlation records
        for the three cohort pairs under the keys "raw" and "delta".

Usage
    python 02_02_apply_delta_transformation.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, data, ensure_dir, param, work  # noqa: E402

# Identifier vocabulary of the LinkedOmics gene-level matrices; see 02_01.
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\-\._@]*$")

# Column names of the Independent-cohort metadata workbook.
METADATA_CASE_ID_COLUMN = "Case_id"
METADATA_GROUP_COLUMN = "Group"

# Group values that identify a tumour and a normal specimen respectively.
TUMOUR_GROUP = "Tumor"
NORMAL_GROUPS = ("Adjacent_normal", "Enriched_Normal")

# Matrix sample identifiers carry a one-letter suffix (for example "-A") that the
# metadata case identifiers do not; it is stripped before the join.
SAMPLE_SUFFIX_PATTERN = re.compile(r"-[A-Z]$")

# Protein matrices read in this stage, keyed by internal cohort label.
PROTEIN_FILES = {
    "dis_T": "cptac_discovery_prot_tumor",
    "dis_N": "cptac_discovery_prot_normal",
    "ov_T": "cptac_ov_prot_tumor",
    "ov_N": "cptac_ov_prot_normal",
    "ind_TN": "cptac_independent_prot_ratio",
}

# The three cohort pairs compared in this stage. Their labels are kept verbatim
# from the pre-registered analysis (the cross and delta symbols denote a cohort
# pair and its tumour-minus-normal contrast) because downstream verification
# scripts index bridge_3cohort.json by these literal key names.
COHORT_PAIRS = (("dis", "ov"), ("dis", "ind"), ("ov", "ind"))

# Expression matrices of the four cohorts used for the standardisation re-test,
# keyed by the cohort token used in the output labels.
EXPRESSION_FILES = {
    "TCGA": "xena_pancan_rnaseq_eb_adjusted",
    "ind": "cptac_independent_rna",
    "dis": "cptac_discovery_rna_tumor",
    "ov": "cptac_ov_rna",
}

# Expression-layer cohort pairs shown in the standardisation comparison.
EXPRESSION_PAIRS = (("TCGA", "ind"), ("TCGA", "ov"), ("ind", "ov"), ("ind", "dis"))


def load_matrix(path: Path) -> pd.DataFrame:
    """Read a tab-separated gene-level matrix of a LinkedOmics export.

    Args:
        path: Matrix file; gzip compression is detected from the file suffix.

    Returns:
        DataFrame of float values indexed by gene symbol with one column per
        sample. Duplicate gene symbols are averaged and non-numeric entries
        become NaN.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    # Merge the comment lines through pandas' comment character rather than
    # stripping them here, so the column count is derived from the header alone.
    frame = pd.read_csv(
        path, sep="\t", comment="#", header=0, dtype=str, engine="python"
    )
    frame = frame.rename(columns={frame.columns[0]: "ID"})
    frame["ID"] = frame["ID"].astype(str).str.strip().str.strip('"')
    frame = frame[frame["ID"].str.match(IDENTIFIER_PATTERN, na=False)]
    matrix = frame.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return matrix.groupby(level=0).mean()


def read_group_map(metadata_path: Path) -> dict[str, str]:
    """Read the case-identifier-to-group map of the Independent cohort.

    Args:
        metadata_path: Excel workbook with Case_id and Group columns.

    Returns:
        Mapping of upper-cased case identifier to its group label.
    """
    workbook = openpyxl.load_workbook(metadata_path, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    header = [str(value) for value in rows[0]]
    group_index = header.index(METADATA_GROUP_COLUMN)
    case_index = header.index(METADATA_CASE_ID_COLUMN)
    group_map: dict[str, str] = {}
    for row in rows[1:]:
        case_id = str(row[case_index]).strip() if row[case_index] else None
        if case_id:
            group_map[case_id.upper()] = str(row[group_index])
    return group_map


def split_tumour_normal(matrix: pd.DataFrame, group_map: dict[str, str]) -> tuple:
    """Split a cohort matrix into tumour-mean and normal-mean spectra.

    Args:
        matrix: Gene-by-sample matrix of one cohort.
        group_map: Case identifier to group label, from read_group_map.

    Returns:
        Pair of gene-indexed Series: the mean over the tumour columns and the mean
        over the normal columns. A cohort without a map entry for a column drops
        that column from both spectra.
    """
    column_groups: dict[str, str] = {}
    for column in matrix.columns:
        # Sample identifiers carry a replicate suffix that the metadata does not.
        key = SAMPLE_SUFFIX_PATTERN.sub("", str(column).upper())
        if key in group_map:
            column_groups[column] = group_map[key]
    tumour_columns = [
        column for column, group in column_groups.items() if group == TUMOUR_GROUP
    ]
    normal_columns = [
        column for column, group in column_groups.items() if group in NORMAL_GROUPS
    ]
    print(
        "ind: mapped columns=%d  tumour=%d  normal=%d"
        % (len(column_groups), len(tumour_columns), len(normal_columns))
    )
    return (
        matrix[tumour_columns].mean(axis=1, skipna=True),
        matrix[normal_columns].mean(axis=1, skipna=True),
    )


def correlate_profiles(a: pd.Series, b: pd.Series, label: str) -> dict | None:
    """Spearman-correlate two gene-indexed profiles on their shared genes.

    Args:
        a: First gene-indexed profile.
        b: Second gene-indexed profile.
        label: Text used in the console line.

    Returns:
        Record with the number of genes compared, the Spearman rho and its
        p-value, or None when the shared gene set is smaller than the minimum
        required by the pre-registration.
    """
    shared = a.index.intersection(b.index)
    x = a.reindex(shared).values.astype(float)
    y = b.reindex(shared).values.astype(float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < param("protein_bridge", "min_shared_genes"):
        print("   %-26s too few shared genes (%d)" % (label, finite.sum()))
        return None
    rho, p_value = spearmanr(x[finite], y[finite])
    print("   %-26s n=%-6d rho=%6.3f  p=%.1e" % (label, finite.sum(), rho, p_value))
    return {"n": int(finite.sum()), "rho": float(rho), "p": float(p_value)}


def quantile_normalise(matrix: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalise every sample of a matrix onto a common distribution.

    Args:
        matrix: Gene-by-sample matrix, possibly with missing values.

    Returns:
        DataFrame of the same shape in which each column has been replaced by the
        sorted values of that column, i.e. a rank-preserving quantile map.
    """
    values = matrix.values.astype(float).copy()
    ranks = np.empty_like(values, dtype=float)
    ranks[:] = np.nan
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        measured = np.isfinite(column)
        if measured.sum() == 0:
            continue
        sorted_values = np.sort(column[measured])
        filled = np.empty(measured.sum())
        filled[np.argsort(column[measured])] = sorted_values
        ranks[measured, column_index] = filled
    return pd.DataFrame(ranks, index=matrix.index, columns=matrix.columns)


def rank_transform(matrix: pd.DataFrame) -> pd.DataFrame:
    """Replace every sample by its within-sample ranks.

    Args:
        matrix: Gene-by-sample matrix, possibly with missing values.

    Returns:
        DataFrame of ranks, NaN where the input was missing. Samples with fewer
        than two measured genes are left missing.
    """
    values = matrix.values.astype(float)
    ranks = np.full_like(values, np.nan)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        measured = np.isfinite(column)
        if measured.sum() < 2:
            continue
        ranks[measured, column_index] = pd.Series(column[measured]).rank().values
    return pd.DataFrame(ranks, index=matrix.index, columns=matrix.columns)


def main() -> None:
    """Split the Independent cohort, compare raw and delta spectra, re-test
    standardisation strategies and write the bridge record file."""
    np.random.seed(SEED)

    independent = load_matrix(data("cptac_independent_prot_ratio"))
    group_map = read_group_map(data("cptac_independent_metadata"))
    ind_tumour, ind_normal = split_tumour_normal(independent, group_map)

    dis_tumour = load_matrix(data("cptac_discovery_prot_tumor")).mean(
        axis=1, skipna=True
    )
    dis_normal = load_matrix(data("cptac_discovery_prot_normal")).mean(
        axis=1, skipna=True
    )
    ov_tumour = load_matrix(data("cptac_ov_prot_tumor")).mean(axis=1, skipna=True)
    ov_normal = load_matrix(data("cptac_ov_prot_normal")).mean(axis=1, skipna=True)

    print("\n" + "=" * 92)
    print("[A] Raw tumour gene-mean spectra")
    print("=" * 92)
    spectra = {
        "dis": dis_tumour,
        "ov": ov_tumour,
        "ind": ind_tumour,
    }
    raw_records: dict[str, dict | None] = {}
    for first, second in COHORT_PAIRS:
        raw_records["%s \u00d7 %s" % (first, second)] = correlate_profiles(
            spectra[first], spectra[second], "%s \u00d7 %s" % (first, second)
        )

    print("\n" + "=" * 92)
    print("[B] Within-cohort delta(T-N) contrasts")
    print("=" * 92)
    contrasts = {
        "dis": dis_tumour - dis_normal,
        "ov": ov_tumour - ov_normal,
        "ind": ind_tumour - ind_normal,
    }
    delta_records: dict[str, dict | None] = {}
    for first, second in COHORT_PAIRS:
        label = "\u0394%s \u00d7 \u0394%s" % (first, second)
        delta_records[label] = correlate_profiles(
            contrasts[first], contrasts[second], label
        )

    print("\n" + "=" * 92)
    print("[C] Improvement factor of the delta contrast over the raw spectrum")
    print("=" * 92)
    for first, second in COHORT_PAIRS:
        raw_label = "%s \u00d7 %s" % (first, second)
        delta_label = "\u0394%s \u00d7 \u0394%s" % (first, second)
        raw_record = raw_records.get(raw_label) or {}
        delta_record = delta_records.get(delta_label)
        if raw_record and delta_record:
            print(
                "   %-14s raw %.3f -> delta(T-N) %.3f   (x%.1f)"
                % (
                    raw_label,
                    raw_record["rho"],
                    delta_record["rho"],
                    # The floor keeps the ratio finite for a zero raw correlation.
                    delta_record["rho"] / max(raw_record["rho"], 1e-6),
                )
            )

    bridge_path = work("protein_bridge_three_cohort")
    ensure_dir(bridge_path.parent)
    with bridge_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"raw": raw_records, "delta": delta_records},
            handle,
            indent=1,
            ensure_ascii=False,
        )

    print("\n" + "=" * 92)
    print("[D] Effect of the standardisation strategy on cross-cohort agreement")
    print("     (expression layer, four cohorts)")
    print("=" * 92)
    expression = {token: load_matrix(data(key)) for token, key in EXPRESSION_FILES.items()}
    for token, matrix in expression.items():
        print("   %-6s %d genes x %d samples" % (token, matrix.shape[0], matrix.shape[1]))

    strategies = [
        ("raw", lambda matrix: matrix),
        ("quantile normalisation (qnorm)", quantile_normalise),
        ("per-sample rank", rank_transform),
    ]
    for name, transform in strategies:
        profiles = {
            token: transform(matrix).mean(axis=1, skipna=True)
            for token, matrix in expression.items()
        }
        print("\n   [%s]" % name)
        for first, second in EXPRESSION_PAIRS:
            shared = profiles[first].index.intersection(profiles[second].index)
            x = profiles[first].reindex(shared).values
            y = profiles[second].reindex(shared).values
            finite = np.isfinite(x) & np.isfinite(y)
            rho, _ = spearmanr(x[finite], y[finite])
            print(
                "      %-12s n=%-6d rho=%.3f"
                % ("%s vs %s" % (first, second), finite.sum(), rho)
            )

    print("\nWrote the three-cohort bridge records to %s" % bridge_path)


if __name__ == "__main__":
    main()
