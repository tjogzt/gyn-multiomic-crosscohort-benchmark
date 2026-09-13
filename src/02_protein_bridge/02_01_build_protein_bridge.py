#!/usr/bin/env python3
"""
Build the protein-layer bridge from the raw CPTAC gene-level protein matrices.

Purpose
    The protein layer cannot be compared across cohorts in its raw form: each
    CPTAC cohort is quantified on its own scale and against its own reference, so
    a raw cross-cohort comparison of gene-mean spectra measures the platform
    rather than biology. This module is the first diagnostic of the protein
    bridge. It loads the gene-level protein matrices of the Discovery, Ovarian
    and Independent cohorts, reports their dimensions, inspects the Independent
    cohort metadata for the sample-grouping column that 02_02 needs, and compares
    (i) the raw tumour gene-mean spectra of two cohorts, (ii) the within-cohort
    tumour-minus-normal contrasts of the same two cohorts, and (iii) the same raw
    comparison after trimming the most extreme 10% of genes. Arguing that the
    delta contrast removes the cohort-specific reference effect, the diagnostic
    records are the evidence quoted for the bridge in the manuscript.

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
    data("cptac_independent_metadata")    .xlsx, metadata workbook of the above.
        Only read to list candidate grouping columns; every file is optional and
        a missing one is reported and skipped.

Outputs
    work("protein_bridge_diagnostics")    .json, Spearman diagnostics per cohort
        pair and comparison (raw, delta, trimmed).

Usage
    python 02_01_build_protein_bridge.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, data, ensure_dir, param, work  # noqa: E402

# Identifier vocabulary of the LinkedOmics gene-level matrices: a gene symbol
# starts with a letter and continues with letters, digits or - . _ @. Rows that
# do not match are annotation or blank rows and are dropped before the matrix is
# built.
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9\-\._@]*$")

# A metadata column is worth listing as a candidate grouping column only when it
# carries at least two and at most this many distinct values; wider columns are
# identifiers, not groups. This bound affects the console report only.
MAX_GROUP_LEVELS = 6

# Cohort label -> configuration key of its gene-level protein matrix. The labels
# are the internal names used in this stage and in its output records.
PROTEIN_FILES = {
    "dis_T": "cptac_discovery_prot_tumor",
    "dis_N": "cptac_discovery_prot_normal",
    "ov_T": "cptac_ov_prot_tumor",
    "ov_N": "cptac_ov_prot_normal",
    "ind_TN": "cptac_independent_prot_ratio",
}


def load_matrix(path: Path, max_cols: int | None = None) -> pd.DataFrame:
    """Read a tab-separated gene-level matrix of a LinkedOmics export.

    Args:
        path: Matrix file; gzip compression is detected from the file suffix.
        max_cols: If given and the file has more sample columns than this, that
            many columns are drawn at random, together with the identifier
            column. Used to keep very wide matrices affordable.

    Returns:
        DataFrame of float values indexed by gene symbol with one column per
        sample. Duplicate gene symbols are averaged and non-numeric entries
        become NaN.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    # The comment block at the top of a .cct file has to be skipped before the
    # column count can be established, because pandas cannot report it up front.
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            columns = line.rstrip("\n").split("\t")
            break
        else:
            # A file with nothing but a comment block cannot be read; fail with
            # the offending path instead of an unrelated NameError.
            raise ValueError(
                "Matrix %s contains no data line after its comment block." % path
            )
    n_columns = len(columns) - 1
    if not max_cols or n_columns <= max_cols:
        usecols = [0] + list(range(1, n_columns + 1))
    else:
        # Columns are drawn from the global stream seeded with SEED so that a
        # partial read is reproducible.
        usecols = [0] + sorted(
            np.random.choice(range(1, n_columns + 1), max_cols, replace=False)
        )
    frame = pd.read_csv(
        path, sep="\t", comment="#", header=0, usecols=usecols, dtype=str, engine="python"
    )
    frame = frame.rename(columns={frame.columns[0]: "ID"})
    frame["ID"] = frame["ID"].astype(str).str.strip().str.strip('"')
    frame = frame[frame["ID"].str.match(IDENTIFIER_PATTERN, na=False)]
    matrix = frame.set_index("ID").apply(pd.to_numeric, errors="coerce")
    return matrix.groupby(level=0).mean()


def gene_mean_spectrum(matrix: pd.DataFrame) -> pd.Series:
    """Return the mean profile of a sample-by-gene matrix, one value per gene.

    Args:
        matrix: Gene-by-sample matrix, as returned by load_matrix.

    Returns:
        Series of gene means, NaN where a gene is unmeasured in every sample.
    """
    return matrix.mean(axis=1, skipna=True)


def compare_spectra(a: pd.Series, b: pd.Series, label: str) -> dict:
    """Spearman-correlate two gene-indexed profiles on their shared genes.

    Args:
        a: First gene-indexed profile.
        b: Second gene-indexed profile.
        label: Text used in the console line.

    Returns:
        Record with the number of genes compared, the Spearman rho and its
        p-value.
    """
    shared = a.index.intersection(b.index)
    x = a.reindex(shared).values
    y = b.reindex(shared).values
    finite = np.isfinite(x) & np.isfinite(y)
    rho, p_value = spearmanr(x[finite], y[finite])
    print("   %s  n=%d  rho=%.3f  p=%.1e" % (label, finite.sum(), rho, p_value))
    return {"n": int(finite.sum()), "rho": float(rho), "p": float(p_value)}


def list_group_candidates(metadata_path: Path) -> list[str]:
    """Report metadata columns that could carry the tumour/normal grouping.

    Args:
        metadata_path: Excel workbook of the Independent cohort.

    Returns:
        Names of the columns whose header mentions a grouping term and whose
        values form a small number of categories. The list is reported for the
        reader only; the split itself is applied in 02_02 using the Group column.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(metadata_path, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = list(sheet.iter_rows(values_only=True))
    header = [str(value) for value in rows[0]]
    candidates = [
        name
        for name in header
        if re.search(r"tumor|normal|sample_type|group", name, re.I)
    ]
    print("\nCandidate grouping columns in the Independent metadata:", candidates[:6])
    for name in candidates:
        values = [
            row[header.index(name)]
            for row in rows[1:]
            if row[header.index(name)] is not None
        ]
        counts = Counter(str(value) for value in values)
        # Only a low-cardinality column can be a group column.
        if 1 < len(counts) <= MAX_GROUP_LEVELS:
            print("   [%s] %s" % (name, counts.most_common(MAX_GROUP_LEVELS)))
    return candidates


def main() -> None:
    """Run the three protein-bridge diagnostics and write the record file."""
    np.random.seed(SEED)

    matrices: dict[str, pd.DataFrame] = {}
    for label, key in PROTEIN_FILES.items():
        path = data(key)
        if path.exists():
            matrices[label] = load_matrix(path)
            print(
                "%-8s genes=%-6d samples=%d"
                % (label, matrices[label].shape[0], matrices[label].shape[1])
            )
        else:
            print("%-8s file missing" % label)

    if data("cptac_independent_metadata").exists():
        list_group_candidates(data("cptac_independent_metadata"))

    result: dict[str, dict] = {}

    print("\n" + "=" * 96)
    print("[Diagnostic 1] Raw gene-mean spectrum (tumour)")
    print("=" * 96)
    for a, b in [("dis_T", "ov_T")]:
        if a in matrices and b in matrices:
            spectrum_a = gene_mean_spectrum(matrices[a])
            spectrum_b = gene_mean_spectrum(matrices[b])
            result["raw_%s_%s" % (a, b)] = compare_spectra(
                spectrum_a, spectrum_b, "%s vs %s" % (a, b)
            )

    print("\n" + "=" * 96)
    print("[Diagnostic 2] Within-cohort tumour-minus-normal contrast (log2 T/N);")
    print("               the platform reference cancels inside a cohort")
    print("=" * 96)
    contrasts: dict[str, pd.Series] = {}
    for cohort, tumour, normal in [("dis", "dis_T", "dis_N"), ("ov", "ov_T", "ov_N")]:
        if tumour in matrices and normal in matrices:
            spectrum_t = gene_mean_spectrum(matrices[tumour])
            spectrum_n = gene_mean_spectrum(matrices[normal])
            shared = spectrum_t.index.intersection(spectrum_n.index)
            contrasts[cohort] = (
                spectrum_t.reindex(shared) - spectrum_n.reindex(shared)
            ).dropna()
            print(
                "   %s: shared genes=%d  delta range[%.2f, %.2f]  median=%.3f"
                % (
                    cohort,
                    len(contrasts[cohort]),
                    contrasts[cohort].min(),
                    contrasts[cohort].max(),
                    contrasts[cohort].median(),
                )
            )
    if len(contrasts) == 2:
        shared = contrasts["dis"].index.intersection(contrasts["ov"].index)
        x = contrasts["dis"].reindex(shared).values
        y = contrasts["ov"].reindex(shared).values
        finite = np.isfinite(x) & np.isfinite(y)
        rho, p_value = spearmanr(x[finite], y[finite])
        result["delta_dis_ov"] = {
            "n": int(finite.sum()),
            "rho": float(rho),
            "p": float(p_value),
        }
        print(
            "   delta(dis) vs delta(ov)  n=%d  rho=%.3f  p=%.1e"
            % (finite.sum(), rho, p_value)
        )

    print("\n" + "=" * 96)
    print("[Diagnostic 3] Trimming robustness: the most extreme 10% of genes dropped")
    print("=" * 96)
    if "dis_T" in matrices and "ov_T" in matrices:
        spectrum_a = gene_mean_spectrum(matrices["dis_T"])
        spectrum_b = gene_mean_spectrum(matrices["ov_T"])
        shared = spectrum_a.index.intersection(spectrum_b.index)
        frame = pd.DataFrame(
            {"a": spectrum_a.reindex(shared), "b": spectrum_b.reindex(shared)}
        ).dropna()
        # A gene is kept only when both cohorts place it inside their own
        # [q, 1-q] interval of the pooled gene-mean spectrum.
        trim = param("protein_bridge", "trim_quantile")
        low, high = frame.stack().quantile([trim, 1.0 - trim])
        kept = frame[
            (frame["a"].between(low, high)) & (frame["b"].between(low, high))
        ]
        rho, p_value = spearmanr(kept["a"], kept["b"])
        result["trim_dis_ov"] = {"n": len(kept), "rho": float(rho), "p": float(p_value)}
        print("   after trimming n=%d  rho=%.3f  p=%.1e" % (len(kept), rho, p_value))

    out_path = work("protein_bridge_diagnostics")
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1, ensure_ascii=False)
    print("\nWrote protein-bridge diagnostics to %s" % out_path)


if __name__ == "__main__":
    main()
