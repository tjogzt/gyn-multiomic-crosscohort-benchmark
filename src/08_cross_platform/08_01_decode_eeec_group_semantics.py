#!/usr/bin/env python3
"""
Decide, from the data alone, what the EEEC sample-group prefixes mean.

Purpose
    The EEEC batch testbed of stage 07 is built from a crossed label-free
    design whose sample names abbreviate the condition ("Ecan", "Enorm",
    "Llymp", ...). Before that testbed may be interpreted, the assumption that
    "can" means tumour and "norm" means normal has to hold in the data itself.
    This module rebuilds the label-free protein matrix from the MaxQuant
    proteinGroups table and tests the assumption four ways: pairwise Pearson
    correlation of the family mean spectra, the family distribution on the
    leading principal components, a per-protein Mann-Whitney comparison of can
    versus norm, and the within- versus between-family sample correlation.
    The cleaned log2 matrix is written out for the cross-platform comparison of
    stage 08.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("eeec_protein_groups")
        str, MaxQuant proteinGroups.txt of PRIDE PXD046507. Streamed twice: once
        for the header, once for the rows. Must contain the columns "Gene
        names", "Reverse", "Potential contaminant", "Unique peptides" and the
        "LFQ intensity <sample>" block.

Outputs
    work("eeec_proteome_matrix")
        pandas DataFrame serialised with to_pickle: log2 protein intensities,
        protein groups x samples, per-sample median centred and filtered.

Usage
    python src/08_cross_platform/08_01_decode_eeec_group_semantics.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, work, param, ensure_dir

# MaxQuant names every label-free quantification column with this prefix; the
# remainder of the column name is the sample name used by the metadata.
LFQ_PREFIX = "LFQ intensity "


def read_protein_groups(protein_groups):
    """Stream the MaxQuant proteinGroups table into an honest protein matrix.

    Args:
        protein_groups: Path to the MaxQuant proteinGroups.txt file.

    Returns:
        pandas.DataFrame of log2 LFQ intensities with one row per gene symbol
        and one column per sample. Protein groups with fewer than
        min_unique_peptides unique peptides, reverse hits, potential
        contaminants and site-only identifications are removed; rows are
        collapsed to the gene level by mean; zero intensities are treated as
        not detected and logged; columns are median centred per sample and rows
        quantified in fewer than min_sample_completeness of the samples are
        dropped.
    """
    # Pass 1: the header fixes the column positions of the annotation fields and
    # of the LFQ block, whose order defines the samples.
    with open(protein_groups, encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    idx = {name: i for i, name in enumerate(header)}
    lfq_columns = [i for i, name in enumerate(header) if name.startswith(LFQ_PREFIX)]
    sample_names = [header[i].replace(LFQ_PREFIX, "") for i in lfq_columns]
    col_gene = idx["Gene names"]
    col_reverse = idx["Reverse"]
    col_contaminant = idx["Potential contaminant"]
    col_peptides = idx["Unique peptides"]
    # Older MaxQuant exports do not carry the "Only identified by site" column.
    col_site_only = idx.get("Only identified by site")

    min_peptides = param("eeec_semantics", "min_unique_peptides")
    rows, genes = [], []
    # Pass 2: the table is read as text and filtered while streaming, so only
    # the identifications that can support a family comparison enter memory.
    with open(protein_groups, encoding="utf-8", errors="replace") as handle:
        handle.readline()
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= max(lfq_columns):
                continue  # truncated row: cannot carry the full LFQ block
            # Decoys, contaminants and site-only identifications are not real
            # protein measurements and would bias the family contrast.
            if cells[col_reverse].strip() == "+" or cells[col_contaminant].strip() == "+":
                continue
            if col_site_only is not None and cells[col_site_only].strip() == "+":
                continue
            # A group supported by fewer unique peptides than the frozen minimum
            # is too weakly identified to be compared across families.
            try:
                if float(cells[col_peptides] or 0) < min_peptides:
                    continue
            except (ValueError, TypeError):
                continue
            # Multi-gene groups are attributed to their first gene name, the
            # convention already used by the batch testbed.
            gene = cells[col_gene].strip().split(";")[0].strip()
            if not gene:
                continue
            rows.append([cells[i] for i in lfq_columns])
            genes.append(gene)

    print(f"Clean protein groups (>= {min_peptides} unique peptides): {len(rows)}")
    matrix = pd.DataFrame(np.array(rows, dtype=float), index=genes, columns=sample_names)
    # Several protein groups can map to the same gene; averaging them keeps the
    # index unique, which every family-level operation below relies on.
    matrix = matrix.groupby(level=0).mean()
    values = np.log2(matrix.replace(0, np.nan))  # a zero LFQ value means not detected
    values = values.dropna(axis=1, how="all")  # drop samples holding no protein at all
    # Per-sample median centring removes the loading difference between runs
    # without touching the tumour/normal contrast under test.
    values = values.sub(values.median(axis=0), axis=1)
    min_samples = int(param("eeec_semantics", "min_sample_completeness") * values.shape[1])
    values = values.dropna(axis=0, thresh=min_samples)
    print(
        f"Matrix: {values.shape[0]} proteins x {values.shape[1]} samples; "
        f"missing rate {100 * (1 - values.notna().mean().mean()):.1f}%"
    )
    return values


def family_of(sample):
    """Return the sample family, i.e. the leading alphabetic run of the name.

    Args:
        sample: Sample name as it appears in the MaxQuant LFQ column.

    Returns:
        str, the leading alphabetic prefix ("Ecan12" -> "Ecan").
    """
    return re.match(r"([A-Za-z]+)", sample).group(1)


def collect_families(matrix):
    """Group the sample columns of the matrix by their family prefix.

    Args:
        matrix: log2 protein matrix, samples as columns.

    Returns:
        dict mapping family name to the list of its sample columns. Families
        listed in the configuration but absent from the matrix are skipped.
    """
    groups = {}
    for family in param("eeec_semantics", "families"):
        columns = [c for c in matrix.columns if family_of(c) == family]
        if columns:
            groups[family] = columns
    return groups


def report_mean_spectra_correlation(matrix, groups):
    """Print the pairwise Pearson correlation of the family mean spectra.

    The mean spectrum of a family is its average protein abundance profile. If
    the can families are tumours and the norm families normal tissue, the two
    can families must correlate more strongly with each other than with the
    norm families.

    Args:
        matrix: log2 protein matrix, proteins as rows, samples as columns.
        groups: Family -> sample columns mapping.

    Returns:
        None. The correlation table is printed.
    """
    group_means = {k: matrix[v].mean(axis=1) for k, v in groups.items()}
    print("\n" + "=" * 96)
    print("[1] Group-mean spectra: pairwise Pearson correlation "
          "(does it support can=tumour / norm=normal?)")
    print("=" * 96)
    keys = list(group_means)
    print(f"{'':<9}" + "".join(f"{k:>10}" for k in keys))
    for a in keys:
        line = f"{a:<9}"
        for b in keys:
            shared = group_means[a].index.intersection(group_means[b].index)
            x = group_means[a].reindex(shared).values
            y = group_means[b].reindex(shared).values
            finite = np.isfinite(x) & np.isfinite(y)
            # A correlation on fewer shared proteins than the frozen minimum is
            # not interpretable and is reported as a dash.
            min_shared = param("eeec_semantics", "min_shared_proteins")
            line += f"{np.corrcoef(x[finite], y[finite])[0, 1]:>10.4f}" if finite.sum() > min_shared else f"{'-':>10}"
        print(line)


def report_principal_components(matrix, groups):
    """Print the family distribution on the first two principal components.

    Args:
        matrix: log2 protein matrix, proteins as rows, samples as columns.
        groups: Family -> sample columns mapping.

    Returns:
        None. Explained variance and the per-family PC means are printed.
    """
    print("\n" + "=" * 96)
    print("[2] Global structure (family distribution on the first two principal components)")
    print("=" * 96)
    # Each protein is centred across samples before the decomposition, matching
    # the per-protein centring used for the correlation analysis below.
    centred = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0)
    u, s, _ = np.linalg.svd(centred.sub(centred.mean(axis=1), axis=0).values, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    print(f"  PC1 explained variance {s[0] ** 2 / np.sum(s ** 2) * 100:.1f}% | "
          f"PC2 {s[1] ** 2 / np.sum(s ** 2) * 100:.1f}%")
    for family, columns in groups.items():
        positions = [centred.columns.get_loc(c) for c in columns]
        print(f"  {family:<9} PC1 mean {pcs[positions, 0].mean():+8.3f}  "
              f"PC2 mean {pcs[positions, 1].mean():+8.3f}   n={len(columns)}")


def report_differential_families(matrix, groups):
    """Test every configured can/norm family pair protein by protein.

    Args:
        matrix: log2 protein matrix, proteins as rows, samples as columns.
        groups: Family -> sample columns mapping.

    Returns:
        None. For each pair the number of testable proteins, the fraction of
        proteins reaching the configured significance level and the median
        p-value are printed.
    """
    print("\n" + "=" * 96)
    print("[3] can vs norm: between-family difference test (expected to be significant)")
    print("=" * 96)
    min_group_samples = param("eeec_semantics", "min_group_samples")
    alpha = param("statistics", "alpha")
    try:
        # scipy is imported here rather than at module level so that the matrix
        # reconstruction can still be inspected if scipy is unavailable.
        from scipy.stats import mannwhitneyu

        for a, b in param("eeec_semantics", "differential_family_pairs"):
            if a in groups and b in groups:
                pvalues = []
                for gene in matrix.index:
                    xa = matrix.loc[gene, groups[a]].dropna().values
                    xb = matrix.loc[gene, groups[b]].dropna().values
                    # Both groups need enough quantified samples for the rank
                    # test to have any power.
                    if len(xa) >= min_group_samples and len(xb) >= min_group_samples:
                        pvalues.append(mannwhitneyu(xa, xb).pvalue)
                pvalues = np.array(pvalues)
                ok = np.isfinite(pvalues)
                print(f"  {a} vs {b}: proteins tested {ok.sum()} | "
                      f"fraction p<{alpha} {100 * np.mean(pvalues[ok] < alpha):.1f}% | "
                      f"median p {np.median(pvalues[ok]):.2e}")
    except Exception as exc:
        print("  Test failed:", str(exc)[:120])


def report_sample_correlation(matrix, groups):
    """Print the median sample-sample Spearman correlation per family pair.

    Within-family pairs give the reproducibility ceiling; between-family pairs
    show how far the tumour and normal profiles are apart.

    Args:
        matrix: log2 protein matrix, proteins as rows, samples as columns.
        groups: Family -> sample columns mapping.

    Returns:
        None. One line per family pair with the median correlation is printed.
    """
    print("\n" + "=" * 96)
    print("[4] Sample-sample correlation: within versus across families")
    print("=" * 96)
    correlation = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0).corr(method="spearman")
    for a, b in param("eeec_semantics", "correlation_family_pairs"):
        if a in groups and b in groups:
            first, second = groups[a], groups[b]
            values = []
            for _, ca in enumerate(first):
                for cb in second:
                    # A within-family pair is unordered, so report it once.
                    if a == b and first.index(ca) >= first.index(cb):
                        continue
                    values.append(correlation.loc[ca, cb])
            if values:
                print(f"  {a:<8}x{b:<8} median Spearman {np.median(values):.4f}   "
                      f"n pairs {len(values)}")


def main():
    """Run the four semantic checks and write the cleaned EEEC protein matrix.

    Args:
        None.

    Returns:
        None. The matrix is written to work("eeec_proteome_matrix").
    """
    print("Reading proteinGroups (key columns only) ...")
    matrix = read_protein_groups(data("eeec_protein_groups"))

    groups = collect_families(matrix)
    print("\nSample counts per family:", {k: len(v) for k, v in groups.items()})

    report_mean_spectra_correlation(matrix, groups)
    report_principal_components(matrix, groups)
    report_differential_families(matrix, groups)
    report_sample_correlation(matrix, groups)

    out_path = work("eeec_proteome_matrix")
    ensure_dir(out_path.parent)
    matrix.to_pickle(out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
