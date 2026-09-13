#!/usr/bin/env python3
"""
Rebuild the EEEC protein -> gene map that stage 07 wrote with the wrong index.

Purpose
    The batch testbed of stage 07 joins its protein matrix to a gene-name map.
    That map was originally produced by reindexing the MaxQuant annotation with
    the protein index of the matrix, which never matches and left every gene
    name empty; without gene names the batch testbed cannot be aggregated to
    the gene level. This module streams the MaxQuant proteinGroups table again,
    takes the gene symbol of every protein group that the testbed matrix
    actually contains, and rewrites the map file in place. It reads the matrix
    index as its definition of the proteins to keep, so it only ever restores
    the entries the testbed needs.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("eeec_batch_matrix")
        pandas DataFrame (csv), the log2 batch-testbed matrix; its index is the
        set of protein group identifiers to look up. No value column is used.
    data("eeec_protein_groups")
        str, MaxQuant proteinGroups.txt of PRIDE PXD046507. Streamed with the
        csv module because the table exceeds the default csv field size limit.
        Must contain the columns "Protein IDs" and "Gene names".
    work("eeec_protein_gene_map")
        pandas DataFrame (csv) with columns "protein" and "gene", the defective
        map. Read only to report how many entries were empty before the repair.

Outputs
    work("eeec_protein_gene_map")
        pandas DataFrame (csv) with columns "protein" and "gene", rewritten;
        the gene column is empty only where the source table has no gene name.

Usage
    python src/08_cross_platform/08_02_repair_gene_identifier_mismatch.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, work

# MaxQuant exports rows far wider than the default csv field size limit of
# 131072 characters; the limit is raised so the reader does not raise.
csv.field_size_limit(10 ** 9)


def read_source_gene_map(protein_groups, wanted_proteins):
    """Collect the gene name of every requested protein group.

    Args:
        protein_groups: Path to the MaxQuant proteinGroups.txt file.
        wanted_proteins: Collection of protein group identifiers (the index of
            the batch-testbed matrix).

    Returns:
        dict mapping protein group identifier to gene symbol. The symbol is the
        first entry of the semicolon-separated "Gene names" field, and is the
        empty string when the source table holds no gene name for the group.
        Groups absent from the source table are simply not present.
    """
    wanted = set(wanted_proteins)
    gene_map = {}
    with open(protein_groups, newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        col_protein = header.index("Protein IDs")
        col_gene = header.index("Gene names")
        for row in reader:
            if len(row) <= max(col_protein, col_gene):
                continue  # truncated row: cannot carry both requested columns
            protein = row[col_protein]
            # Only the protein groups the testbed matrix contains are needed.
            if protein in wanted:
                gene_map[protein] = (row[col_gene] or "").split(";")[0].strip()
    return gene_map


def count_filled_genes(gene_series):
    """Count the gene entries that carry a real symbol.

    Args:
        gene_series: pandas Series of gene values as read from csv.

    Returns:
        int, the number of entries that are neither NaN, the empty string nor
        the string "nan". Empty entries are counted the same way for the
        defective and the repaired file, so the two counts are comparable.
    """
    text = gene_series.astype(str)
    return int((~gene_series.isna() & (text.str.strip() != "") & (text != "nan")).sum())


def main():
    """Rewrite the EEEC protein -> gene map from the MaxQuant annotation.

    Args:
        None.

    Returns:
        None. work("eeec_protein_gene_map") is overwritten in place.
    """
    matrix = pd.read_csv(work("eeec_batch_matrix"), index_col=0)
    wanted = set(matrix.index.astype(str))
    print("Target protein groups:", len(wanted))

    gene_map = read_source_gene_map(data("eeec_protein_groups"), wanted)
    print("Recovered from the source table:", len(gene_map),
          "| with a gene name:", sum(1 for v in gene_map.values() if v))

    old_map = pd.read_csv(work("eeec_protein_gene_map"))
    print("Non-empty genes in the old file (by truth value):", count_filled_genes(old_map["gene"]))

    out = pd.DataFrame({
        "protein": list(wanted),
        "gene": [gene_map.get(p, "") for p in wanted],
    })
    out.to_csv(work("eeec_protein_gene_map"), index=False)
    print("Rebuilt protein_gene.csv | non-empty:", int((out["gene"] != "").sum()), "/", len(out))
    print(out.head(4).to_string(index=False))
    print("\nUnique genes:", out.loc[out["gene"] != "", "gene"].nunique())


if __name__ == "__main__":
    main()
