#!/usr/bin/env python3
"""
Formalise the cross-platform comparison: ladder, effect strata, network layer.

Purpose
    The first ladder pass of stage 08 compared delta vectors on a single gene
    space. This module makes the comparison formal. It re-establishes the
    ladder on the six matrices that must share a gene space, adds a label
    permutation test to every step, checks that the agreement is not driven by
    the strongest effects by stratifying on the absolute CPTAC discovery
    effect, and then moves from single genes to the correlation structure
    between proteins: the agreement of the protein-protein correlation matrices
    of the two tumour cohorts and the agreement of their first principal
    component loadings. The CPTAC confirmatory cohort is absent by design,
    because its median-polished ratio matrix no longer carries a
    tumour-versus-normal contrast. All results are written as one json file.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_discovery_proteomics_normal"),
    data("cptac_discovery_proteomics_tumor")
        str, LinkedOmics gene-level TMT .cct matrices of the discovery cohort,
        genes as rows and samples as columns.
    work("platform_delta_vectors")
        pandas DataFrame (csv) holding the "Discovery", "EEEC", "EEEC_E" and
        "EEEC_L" delta vectors as columns; produced by 08_03, so it must run
        first.
    work("eeec_batch_matrix")
        pandas DataFrame (csv), log2 EEEC label-free intensities, protein group
        identifiers as index.
    work("eeec_batch_sample_meta")
        pandas DataFrame (csv) with the columns "col" and "family".
    work("eeec_protein_gene_map")
        pandas DataFrame (csv) with columns "protein" and "gene".

Outputs
    work("platform_ladder_v2_csv")
        pandas DataFrame (csv), the permuted ladder: pair label, gene count,
        Spearman, Pearson, sign concordance, slope and permutation p-value.
    work("platform_effect_bins")
        pandas DataFrame (csv), one row per effect-size quartile.
    work("platform_results_json")
        json object with the keys "ladder", "bins" and "network".

Usage
    python src/08_cross_platform/08_04_compare_platform_networks.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import svd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import data, work, param, SEED

# MaxQuant/LinkedOmics tables exceed the default csv field size limit.
csv.field_size_limit(10 ** 9)

# Ladder-step markers of the published table. They are written as unicode escapes
# so that the source file stays pure ASCII while the labels written into the
# artefacts keep the characters used by the report.
STEP_MARKERS = ("\u2460", "\u2461", "\u2462", "\u2463")


def read_cct(path):
    """Read a LinkedOmics .cct matrix into a numeric DataFrame.

    Args:
        path: Path to the .cct file (tab separated, one header line, first
            column holding the row identifiers).

    Returns:
        pandas.DataFrame of floats, genes as index and samples as columns. Rows
        whose field count differs from the header are skipped.
    """
    with open(path, encoding="utf-8", errors="replace") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
        rows, row_ids = [], []
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            if len(cells) != len(header):
                continue  # ragged row: cannot be aligned to the header
            row_ids.append(cells[0].strip('"'))
            rows.append([x.strip('"') for x in cells[1:]])
    matrix = pd.DataFrame(rows, index=row_ids, columns=[h.strip('"') for h in header[1:]])
    return matrix.apply(pd.to_numeric, errors="coerce")


def build_eeec_tumor_matrix():
    """Build the gene-level EEEC tumour matrix for the network comparison.

    Args:
        None. The EEEC matrix, its sample metadata and the protein-to-gene map
        are read from the configuration.

    Returns:
        pandas.DataFrame of log2 intensities, genes as index and the pooled
        label-free tumour samples as columns.
    """
    matrix = pd.read_csv(work("eeec_batch_matrix"), index_col=0)
    meta = pd.read_csv(work("eeec_batch_sample_meta"))
    meta = meta[meta["family"].isin(param("cross_platform", "core_families"))].reset_index(drop=True)
    protein_gene = pd.read_csv(work("eeec_protein_gene_map"))
    gene_map = dict(zip(protein_gene["protein"].astype(str), protein_gene["gene"].astype(str)))

    genes = np.array([gene_map.get(str(p), "") for p in matrix.index])
    tumour_columns = [c for c, f in zip(meta["col"], meta["family"])
                      if f in param("cross_platform", "pooled_tumour_families")]
    tumour = matrix[tumour_columns].copy()
    tumour["__g"] = genes
    # Several protein groups of one gene are averaged; unnamed groups are
    # dropped because they cannot be matched across platforms.
    return tumour[tumour["__g"] != ""].groupby("__g").mean()


def permutation_test(x, y, n=None):
    """Label-permutation test for the Spearman agreement of two vectors.

    Args:
        x: array-like, first vector.
        y: array-like, second vector; its labels are permuted.
        n: int or None, number of permutations. None uses the configured
            default count.

    Returns:
        tuple(float, float, float) -- the observed Spearman coefficient, the
        two-sided permutation p-value with the add-one correction, and the mean
        absolute coefficient of the null distribution.
    """
    if n is None:
        n = param("cross_platform", "n_permutation_default")
    rng = np.random.RandomState(SEED)
    observed = spearmanr(x, y)[0]
    null = [spearmanr(x, rng.permutation(y))[0] for _ in range(n)]
    # Add-one correction: the observed value counts as one draw of the null, so
    # the p-value is never exactly zero.
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n + 1)
    return float(observed), float(p_value), float(np.mean(np.abs(null)))


def first_component(matrix):
    """Return the first principal component loadings of a gene x sample matrix.

    Args:
        matrix: array-like, genes as rows and samples as columns.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray) -- the loadings of the first
        component with a fixed sign convention, and the boolean mask of the
        rows that were kept.
    """
    values = np.asarray(matrix, float)
    keep = np.isfinite(values).all(axis=1)  # the SVD requires complete rows
    if keep.sum() < values.shape[0]:
        values = values[keep]
    values = values - values.mean(axis=1, keepdims=True)
    u, _, _ = svd(values, full_matrices=False)
    # The sign of an eigenvector is arbitrary; flipping by the sign of the sum
    # makes the two cohorts comparable.
    loadings = u[:, 0] * np.sign(u[:, 0].sum())
    return loadings, keep


def main():
    """Run the formal cross-platform comparison and write its artefacts.

    Args:
        None.

    Returns:
        None. work("platform_ladder_v2_csv"), work("platform_effect_bins") and
        work("platform_results_json") are written.
    """
    discovery_normal = read_cct(data("cptac_discovery_proteomics_normal"))
    discovery_tumor = read_cct(data("cptac_discovery_proteomics_tumor"))
    delta = pd.read_csv(work("platform_delta_vectors"), index_col=0)
    delta_discovery = delta["Discovery"]
    delta_eeec = delta["EEEC"]
    delta_eeec_e = delta["EEEC_E"]
    delta_eeec_l = delta["EEEC_L"]

    # ---------- EEEC sample-level tumour matrix ----------
    eeec_tumor = build_eeec_tumor_matrix()
    print(f"EEEC tumour matrix: {eeec_tumor.shape[0]} genes x {eeec_tumor.shape[1]} samples")

    # The gene space must be shared by the delta vectors, the EEEC matrix and
    # both arms of the discovery cohort, otherwise the comparisons would mix
    # different gene spaces.
    common = sorted(set(delta_discovery.index) & set(delta_eeec.index)
                    & set(eeec_tumor.index) & set(discovery_tumor.index)
                    & set(discovery_normal.index))
    print(f"Genes shared by all inputs: {len(common)}")
    d_discovery = delta_discovery.reindex(common)
    d_eeec = delta_eeec.reindex(common)
    d_e = delta_eeec_e.reindex(common)
    d_l = delta_eeec_l.reindex(common)

    rows = []

    def add(x, y, label, permute=True):
        """Append one ladder record and print its summary line.

        Args:
            x: pandas.Series, first delta vector on the common gene space.
            y: pandas.Series, second delta vector on the common gene space.
            label: str, label of the ladder step.
            permute: bool, whether to run the label permutation test.

        Returns:
            dict, the record appended to the ladder table.
        """
        mask = np.isfinite(x.values) & np.isfinite(y.values)
        a, b = x.values[mask], y.values[mask]
        rho = float(spearmanr(a, b)[0])
        r = float(pearsonr(a, b)[0])
        sign = float(np.mean(np.sign(a) == np.sign(b)))
        # The regression slope calibrates the scale of the EEEC effects against
        # the CPTAC ones; a slope far from 1 means the two platforms disagree
        # in magnitude even when they agree in rank.
        slope = float(np.polyfit(a, b, 1)[0])
        record = {"pair": label, "n": int(mask.sum()), "spearman": round(rho, 4),
                  "pearson": round(r, 4), "sign_conc": round(sign, 4),
                  "slope_EEEC_on_CPTAC": round(slope, 4)}
        if permute:
            _, p_value, null_abs = permutation_test(a, b, param("statistics", "n_permutations"))
            record["perm_p"] = p_value
            record["null_abs_mean"] = round(null_abs, 4)
        rows.append(record)
        print(f"  {label:38s} n={mask.sum():5d}  rho={rho:+.4f}  r={r:+.4f}  "
              f"same sign={sign * 100:5.1f}%  slope={slope:+.3f}"
              + (f"  perm p={record['perm_p']:.4f}" if permute else ""))
        return record

    print("\n=== Delta concordance ladder ===")
    add(d_e, d_l, f"{STEP_MARKERS[0]} EEEC-E vs EEEC-L (same cohort, cross-batch)")
    add(d_eeec, d_discovery, f"{STEP_MARKERS[2]} EEEC vs CPTAC-Discovery")
    add(d_e, d_discovery, f"{STEP_MARKERS[3]} CPTAC-Dis vs EEEC-E")
    add(d_l, d_discovery, f"{STEP_MARKERS[3]} CPTAC-Dis vs EEEC-L")
    pd.DataFrame(rows).to_csv(work("platform_ladder_v2_csv"), index=False)

    # ---------- Effect-size stratification ----------
    print("\n=== Effect-size strata (quantiles of |delta_CPTAC-Dis|) ===")
    # If the agreement were driven by a handful of very strong effects, the
    # weakest quartile would lose it entirely.
    quantiles = pd.qcut(d_discovery.abs(), param("statistics", "n_effect_bins"),
                        labels=param("cross_platform", "effect_bin_labels"))
    bins = []
    for label, index in d_discovery.groupby(quantiles, observed=True).groups.items():
        index = pd.Index(index)
        a = d_discovery.reindex(index).values
        b = d_eeec.reindex(index).values
        mask = np.isfinite(a) & np.isfinite(b)
        rho = float(spearmanr(a[mask], b[mask])[0])
        bins.append({"bin": str(label), "n": int(mask.sum()), "spearman": round(rho, 4),
                     "sign_conc": round(float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))), 4)})
        print(f"  {str(label):10s} n={mask.sum():5d}  rho={rho:+.4f}  "
              f"same sign={np.mean(np.sign(a[mask]) == np.sign(b[mask])) * 100:.1f}%")
    pd.DataFrame(bins).to_csv(work("platform_effect_bins"), index=False)

    # ---------- Network layer (protein-protein correlation inside tumour samples) ----------
    print("\n=== Network layer agreement ===")
    network_genes = [g for g in common if g in eeec_tumor.index and g in discovery_tumor.index]
    eeec_values = eeec_tumor.loc[network_genes].values.astype(float)  # genes x EEEC tumour samples
    cptac_values = discovery_tumor.loc[network_genes].values.astype(float)  # genes x CPTAC tumour samples
    eeec_corr = np.corrcoef(eeec_values)
    cptac_corr = np.corrcoef(cptac_values)
    upper = np.triu_indices(len(network_genes), 1)
    eeec_pairs, cptac_pairs = eeec_corr[upper], cptac_corr[upper]
    ok = np.isfinite(eeec_pairs) & np.isfinite(cptac_pairs)
    network_rho = float(spearmanr(eeec_pairs[ok], cptac_pairs[ok])[0])
    network_r = float(pearsonr(eeec_pairs[ok], cptac_pairs[ok])[0])
    print(f"  protein pairs {ok.sum():,} | network structure Spearman = {network_rho:+.4f} | "
          f"Pearson = {network_r:+.4f}")

    # PC1 loading comparison
    pc1_eeec, eeec_kept = first_component(eeec_values)
    pc1_cptac, cptac_kept = first_component(cptac_values)
    eeec_positions = np.where(eeec_kept)[0]
    cptac_positions = np.where(cptac_kept)[0]
    common_positions = np.intersect1d(eeec_positions, cptac_positions)  # positions in the full gene space
    min_pc1_genes = param("cross_platform", "min_pc1_genes")
    if len(common_positions) >= min_pc1_genes:
        # Map the shared positions onto the retained subsets of each matrix.
        keep_eeec = np.searchsorted(eeec_positions, common_positions)
        keep_cptac = np.searchsorted(cptac_positions, common_positions)
        pc1_rho = float(spearmanr(pc1_eeec[keep_eeec], pc1_cptac[keep_cptac])[0])
        print(f"  PC1 loading agreement Spearman = {pc1_rho:+.4f} "
              f"({len(common_positions)} complete genes)")
    else:
        pc1_rho = float("nan")
        print("  too few genes with complete profiles for the PC1 comparison")

    # Network layer permutation
    _, network_perm_p, _ = permutation_test(eeec_pairs[ok], cptac_pairs[ok],
                                            param("cross_platform", "n_permutation_network"))
    print(f"  network structure permutation p = {network_perm_p:.4f}")

    network = {
        "n_pairs": int(ok.sum()),
        "network_spearman": round(network_rho, 4),
        "network_pearson": round(network_r, 4),
        "network_perm_p": network_perm_p,
        "pc1_loading_spearman": round(pc1_rho, 4),
        "pc1_n_genes": int(len(common_positions)),
        "n_genes": len(network_genes),
        "n_tumor_EEEC": int(eeec_values.shape[1]),
        "n_tumor_CPTAC": int(cptac_values.shape[1]),
    }
    results_path = work("platform_results_json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump({"ladder": rows, "bins": bins, "network": network},
                  handle, ensure_ascii=False, indent=1)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
