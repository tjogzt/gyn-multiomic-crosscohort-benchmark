#!/usr/bin/env python3
"""
Build the cross-platform comparability ladder between label-free and TMT.

Purpose
    The benchmark transfers cluster labels between cohorts measured on the same
    platform, so it must be shown that adding a platform change costs more than
    adding a cohort change. This module constructs the three-step ladder from
    tumour-minus-normal delta vectors: step 1 compares the two label-free
    sub-cohorts of EEEC (same cohort, same platform), step 2 compares the two
    CPTAC cohorts (same platform, different cohort) and step 3 compares EEEC
    against each CPTAC cohort (different platform and different cohort). Every
    step is summarised by a Spearman and a Pearson correlation on the genes the
    five delta vectors share, with a bootstrap confidence interval for the
    Spearman coefficient. The CPTAC confirmatory cohort enters only through its
    median-polished ratio matrix, whose shape documents why it cannot be used
    for a tumour-versus-normal contrast. The delta vectors are written out for
    stage 08's network analysis.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_discovery_proteomics_normal"),
    data("cptac_discovery_proteomics_tumor")
        str, LinkedOmics gene-level TMT .cct matrices. Rows are genes, columns
        are samples; the delta vector of the cohort is the difference of the two
        row means.
    data("cptac_confirmatory_proteomics_ratio")
        str, LinkedOmics median-polished log2 tumour/normal ratio matrix of the
        CPTAC confirmatory cohort.
    work("eeec_batch_matrix")
        pandas DataFrame (csv), log2 EEEC label-free intensities with protein
        group identifiers as index and sample names as columns.
    work("eeec_batch_sample_meta")
        pandas DataFrame (csv) with the columns "col" (matrix column) and
        "family" (sample family of the crossed design).
    work("eeec_protein_gene_map")
        pandas DataFrame (csv) with columns "protein" and "gene"; produced by
        08_02, so it must be run first.

Outputs
    work("platform_ladder_csv")
        pandas DataFrame (csv), one row per ladder step: pair label, number of
        shared genes, Spearman with its bootstrap interval, Pearson and the
        fraction of same-sign genes.
    work("platform_ladder_json")
        json list, the same records as platform_ladder_csv.
    work("platform_delta_vectors")
        pandas DataFrame (csv), the five delta vectors as columns, indexed by
        gene symbol.

Usage
    python src/08_cross_platform/08_03_compare_platform_ladder.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, work, param, ensure_dir, SEED

# Ladder-step markers and the delta sign of the published comparison table. They
# are written as unicode escapes so that the source file stays pure ASCII while
# the labels written into the artefacts keep the characters used by the report.
STEP_MARKERS = ("\u2460", "\u2461", "\u2462", "\u2463")
DELTA = "\u0394"


def read_cct(path):
    """Read a LinkedOmics .cct matrix into a numeric DataFrame.

    Args:
        path: Path to the .cct file (tab separated, one header line, the first
            column holding the row identifiers).

    Returns:
        pandas.DataFrame of floats, genes as index and samples as columns. Rows
        whose field count differs from the header are skipped because their
        columns cannot be aligned; quotes around identifiers are stripped.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
        rows, row_ids = [], []
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            if len(cells) != len(header):
                continue  # ragged row: cannot be aligned to the header
            row_ids.append(cells[0])
            rows.append(cells[1:])
    matrix = pd.DataFrame(
        rows,
        index=[i.strip('"') for i in row_ids],
        columns=[h.strip('"') for h in header[1:]],
    )
    return matrix.apply(pd.to_numeric, errors="coerce")


def build_eeec_deltas():
    """Build the combined and per-sub-cohort EEEC tumour-minus-normal contrasts.

    Args:
        None. The EEEC matrix, sample metadata and protein-to-gene map are read
        from the configuration.

    Returns:
        tuple(dict, pandas.Index) -- the dict holds the three delta vectors
        ("combined", "E", "L") of type pandas.Series indexed by gene symbol, and
        the returned Index is the gene symbol of every row of the EEEC matrix,
        aligned to the protein order of the matrix.
    """
    matrix = pd.read_csv(work("eeec_batch_matrix"), index_col=0)
    meta = pd.read_csv(work("eeec_batch_sample_meta"))
    # The pilot families of the design are not part of the crossed comparison.
    meta = meta[meta["family"].isin(param("cross_platform", "core_families"))].reset_index(drop=True)
    protein_gene = pd.read_csv(work("eeec_protein_gene_map"))
    gene_map = dict(zip(protein_gene["protein"].astype(str), protein_gene["gene"].astype(str)))

    sample_columns = meta["col"].tolist()
    # Transpose to samples x protein groups so that X.loc[family] selects the
    # samples of one family; the row labels are the sample families.
    samples = matrix[sample_columns].T
    samples.index = meta["family"].values
    genes = np.array([gene_map.get(str(p), "") for p in matrix.index])

    def delta_from(tumour_families, normal_families):
        """Return the mean tumour minus mean normal contrast per gene.

        Args:
            tumour_families: Family labels pooled into the tumour group.
            normal_families: Family labels pooled into the normal group.

        Returns:
            pandas.Series indexed by gene symbol, the mean difference of the
            two group means. Protein groups without a gene symbol are dropped
            and several protein groups of one gene are averaged.
        """
        tumour = samples.loc[tumour_families]
        normal = samples.loc[normal_families]
        delta = tumour.mean(axis=0) - normal.mean(axis=0)
        per_protein = pd.DataFrame({"g": genes, "d": delta.values}).query("g != ''")
        return per_protein.groupby("g")["d"].mean()

    subcohorts = param("cross_platform", "subcohorts")
    deltas = {
        "combined": delta_from(
            param("cross_platform", "pooled_tumour_families"),
            param("cross_platform", "pooled_normal_families"),
        ),
        "E": delta_from(subcohorts["E"]["tumour"], subcohorts["E"]["normal"]),
        "L": delta_from(subcohorts["L"]["tumour"], subcohorts["L"]["normal"]),
    }
    return deltas, genes


def correlate_pair(a, b, label, gene_space, n_bootstrap=None):
    """Correlate two delta vectors on their shared genes.

    Args:
        a: pandas.Series, first delta vector indexed by gene symbol.
        b: pandas.Series, second delta vector indexed by gene symbol.
        label: str, label of the ladder step written to the result table.
        gene_space: Collection of gene symbols the two vectors are restricted to
            (the intersection of all five delta vectors).
        n_bootstrap: int or None, number of bootstrap resamples for the Spearman
            confidence interval. None uses the configured full count.

    Returns:
        dict with the pair label, the number of genes actually correlated, the
        Spearman coefficient with its 2.5% and 97.5% bootstrap bounds, the
        Pearson coefficient and the fraction of genes with the same sign.
    """
    if n_bootstrap is None:
        n_bootstrap = param("cross_platform", "n_bootstrap_ci")
    shared = sorted(gene_space)
    x = a.reindex(shared).values
    y = b.reindex(shared).values
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    rho, _ = spearmanr(x, y)
    r, _ = pearsonr(x, y)
    # Sign concordance is the share of genes whose direction of change agrees,
    # a correlation-free statement about effect direction.
    sign = float(np.mean(np.sign(x) == np.sign(y)))
    rng = np.random.RandomState(SEED)
    bootstrap = []
    for _ in range(n_bootstrap):
        # Genes are resampled with replacement, not permuted: the interval
        # describes the uncertainty of the coefficient on this gene space.
        picks = rng.randint(0, len(x), len(x))
        bootstrap.append(spearmanr(x[picks], y[picks])[0])
    lo, hi = np.percentile(bootstrap, [2.5, 97.5])
    record = {
        "pair": label,
        "n": int(len(x)),
        "spearman": round(float(rho), 4),
        "spearman_lo": round(float(lo), 4),
        "spearman_hi": round(float(hi), 4),
        "pearson": round(float(r), 4),
        "sign_concordance": round(sign, 4),
    }
    print(f"  {label:34s} n={len(x):5d}  Spearman={rho:+.4f} [{lo:+.4f},{hi:+.4f}]  "
          f"Pearson={r:+.4f}  same sign={sign * 100:.1f}%")
    return record


def main():
    """Run the three-step cross-platform ladder and write its artefacts.

    Args:
        None.

    Returns:
        None. work("platform_ladder_csv"), work("platform_ladder_json") and
        work("platform_delta_vectors") are written.
    """
    out_dir = ensure_dir(work("platform_dir"))

    print("Loading CPTAC Discovery ...")
    discovery_normal = read_cct(data("cptac_discovery_proteomics_normal"))
    discovery_tumor = read_cct(data("cptac_discovery_proteomics_tumor"))
    print(f"  Discovery normal {discovery_normal.shape} | tumour {discovery_tumor.shape}")
    print("Loading CPTAC Confirmatory ratio ...")
    confirmatory_ratio = read_cct(data("cptac_confirmatory_proteomics_ratio"))
    print(f"  Confirmatory ratio {confirmatory_ratio.shape}")

    # The delta vector of a cohort is the difference of the group mean spectra
    # of its tumour and normal samples.
    delta_discovery = (discovery_tumor.mean(axis=1) - discovery_normal.mean(axis=1)).dropna()
    delta_confirmatory = confirmatory_ratio.mean(axis=1).dropna()
    print(f"\nDelta vectors: Discovery {len(delta_discovery)} genes | "
          f"Confirmatory {len(delta_confirmatory)} genes")

    eeec, _ = build_eeec_deltas()
    delta_eeec = eeec["combined"]
    delta_eeec_e = eeec["E"]
    delta_eeec_l = eeec["L"]
    print(f"Delta vectors: EEEC(combined) {len(delta_eeec)} | EEEC-E {len(delta_eeec_e)} | "
          f"EEEC-L {len(delta_eeec_l)}")

    # Every coefficient is computed on one common gene space, so the five
    # vectors are restricted to the intersection before any pair is correlated.
    sets = [set(delta_discovery.index), set(delta_confirmatory.index), set(delta_eeec.index),
            set(delta_eeec_e.index), set(delta_eeec_l.index)]
    common = set.intersection(*sets)
    print(f"\nGenes shared by all five vectors: {len(common)}")

    print("\n=== Effect-size ladder (delta = log2 tumour - log2 normal) ===")
    records = []
    # Step 1 is the within-EEEC comparison of the two label-free sub-cohorts:
    # it is the ceiling of the ladder and needs only a reduced bootstrap.
    records.append(correlate_pair(delta_eeec_e, delta_eeec_l,
                                  f"{STEP_MARKERS[0]} EEEC-E {DELTA} vs EEEC-L {DELTA}",
                                  common, param("cross_platform", "n_bootstrap_reduced")))
    records.append(correlate_pair(delta_discovery, delta_confirmatory,
                                  f"{STEP_MARKERS[1]} CPTAC Discovery vs Confirmatory", common))
    records.append(correlate_pair(delta_eeec, delta_discovery,
                                  f"{STEP_MARKERS[2]} EEEC vs CPTAC Discovery", common))
    records.append(correlate_pair(delta_eeec, delta_confirmatory,
                                  f"{STEP_MARKERS[2]} EEEC vs CPTAC Confirmatory", common))
    records.append(correlate_pair(delta_discovery, delta_eeec_e,
                                  f"{STEP_MARKERS[3]} CPTAC-Dis vs EEEC-E", common))
    records.append(correlate_pair(delta_confirmatory, delta_eeec_l,
                                  f"{STEP_MARKERS[3]} CPTAC-Conf vs EEEC-L", common))

    ladder_csv = work("platform_ladder_csv")
    pd.DataFrame(records).to_csv(ladder_csv, index=False)
    ladder_json = work("platform_ladder_json")
    with open(ladder_json, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=1)

    # The delta vectors are kept on disk because the network analysis of stage
    # 08 reuses them instead of rebuilding the cohort contrasts.
    delta_vectors = work("platform_delta_vectors")
    pd.DataFrame({
        "Discovery": delta_discovery,
        "Confirmatory": delta_confirmatory,
        "EEEC": delta_eeec,
        "EEEC_E": delta_eeec_e,
        "EEEC_L": delta_eeec_l,
    }).to_csv(delta_vectors)
    print(f"\nWrote {ladder_csv}, {ladder_json} and {delta_vectors} (directory {out_dir})")


if __name__ == "__main__":
    main()
