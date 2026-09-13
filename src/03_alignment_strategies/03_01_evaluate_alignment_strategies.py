#!/usr/bin/env python3
"""
Evaluate the cross-cohort alignment strategies as a transfer-ARI matrix over
layers, cohorts pairs and standardisation arms.

Purpose
    Cohorts are measured on their own scales, so before any integration method can
    be benchmarked the layers must be brought onto a comparable footing. This
    module quantifies how much of the cross-cohort agreement each alignment
    strategy recovers, under gate G4. Every strategy emits a sample-by-gene matrix
    that can be clustered directly; the main loop performs no further
    standardisation. For each layer and each unordered cohort pair it centres the
    samples, restricts the genes to the intersection of the two cohorts' top-MAD
    sets, builds four arms (S0 baseline, S1 quantile normalisation, S2
    normal-tissue delta reference, S3 pooled ComBat location/scale) and, for each
    arm and each K of the pre-registered grid, clusters cohort A and transfers the
    samples of cohort B to A's centroids, scoring the transfer against B's own
    clustering by adjusted Rand index. Degenerate arms, in which one side collapses
    to a single cluster, are flagged because their ARI is not interpretable.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("harmonised_matrix_pattern")  .csv.gz per cohort x layer, the harmonised
        matrices produced by stage 01. The cohort and layer tokens in the file
        name come from params:alignment:layer_cohorts. Cohorts without a file are
        skipped for that layer, and a missing normal-tissue matrix only disables
        the delta-reference arm.

Outputs
    work("alignment_ari")              .json, one record per (layer, pair,
        strategy) with the ARI at every K of the grid and the degeneracy flags,
        plus one record per skipped pair stating the reason.

Usage
    python 03_01_evaluate_alignment_strategies.py
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, ensure_dir, param, work  # noqa: E402

# Libraries emit convergence and dtype warnings that do not affect the result and
# would otherwise flood the log of a long sweep.
warnings.filterwarnings("ignore")

# Separator of the cohort-pair labels. It is kept as the multiplication sign used
# by the pre-registered analysis (U+00D7) because the cross-implementation merge
# in stage 11 normalises exactly this separator when it matches the Python pair
# labels against the R ones.
PAIR_SEPARATOR = "\u00d7"

# A quantile map needs at least this many measured genes in a sample to describe a
# distribution; samples below the bound are left untouched.
MIN_GENES_PER_SAMPLE = 3


def load_layer(cohort: str, layer_token: str) -> pd.DataFrame | None:
    """Load the harmonised matrix of one cohort and one layer.

    Args:
        cohort: Cohort token as used in the harmonised file names.
        layer_token: Layer token as used in the harmonised file names.

    Returns:
        Gene-by-sample DataFrame, or None when no file exists for that
        cohort-layer combination.
    """
    path = Path(
        str(work("harmonised_matrix_pattern")).format(
            cohort=cohort, layer=layer_token
        )
    )
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


def centre_samples(matrix: np.ndarray) -> np.ndarray:
    """Subtract every sample's median, the first step of the S0 baseline.

    Args:
        matrix: Sample-by-gene array.

    Returns:
        Array of the same shape, centred rowwise.
    """
    return matrix - np.nanmedian(matrix, axis=1, keepdims=True)


def quantile_normalise(matrix: np.ndarray) -> np.ndarray:
    """Map every sample onto the joint reference quantiles of the arm.

    Args:
        matrix: Sample-by-gene array.

    Returns:
        Array of the same shape in which each row holds the reference values
        assigned by the within-row rank order, so all rows share one
        distribution.
    """
    values = np.array(matrix, dtype=float)
    # Reference distribution: the mean of the sorted rows, i.e. the quantiles of
    # the pooled samples of this arm.
    reference = np.nanmean(np.sort(values, axis=1), axis=0)
    n_reference = len(reference)
    for row_index in range(values.shape[0]):
        row = values[row_index]
        measured = np.isfinite(row)
        n_measured = int(measured.sum())
        if n_measured < MIN_GENES_PER_SAMPLE:
            continue
        column_index = np.where(measured)[0]
        order = np.argsort(row[column_index])
        mapped = np.interp(
            np.linspace(0, 1, n_measured), np.linspace(0, 1, n_reference), reference
        )
        updated = row.copy()
        updated[column_index[order]] = mapped
        values[row_index] = updated
    return values


def zscore_genes(matrix: np.ndarray) -> np.ndarray:
    """Standardise every gene across the samples of one arm.

    Args:
        matrix: Sample-by-gene array.

    Returns:
        Array of z-scores with non-finite entries replaced by zero; genes without
        variance are left unscaled so they contribute no noise.
    """
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    std = np.where((std < 1e-9) | ~np.isfinite(std), 1.0, std)
    zscored = (matrix - mean) / std
    return np.nan_to_num(zscored, nan=0.0, posinf=0.0, neginf=0.0)


def combat_location_scale(first: np.ndarray, second: np.ndarray) -> tuple:
    """Pooled ComBat location/scale correction with the cohort as batch.

    Implements the location/scale part of ComBat (Johnson, Li and Rabinovic,
    2007, Biostatistics 8:118-127) without the empirical-Bayes shrinkage, because
    each batch holds a single cohort. After removing the batch location and scale,
    the pooled variance is restored on every feature; the pooled mean is
    deliberately not added back, since injecting a common mean would create
    between-cohort similarity that the data does not support.

    Args:
        first: Sample-by-gene array of cohort A.
        second: Sample-by-gene array of cohort B.

    Returns:
        Pair of corrected arrays, in the order of the arguments.
    """
    combined = np.vstack([first, second])
    batch = np.array([0] * len(first) + [1] * len(second))
    pooled_variance = combined.var(axis=0)
    pooled_variance[pooled_variance < 1e-12] = 1.0
    corrected = np.zeros_like(combined)
    for batch_index in (0, 1):
        in_batch = batch == batch_index
        batch_mean = combined[in_batch].mean(axis=0)
        batch_variance = combined[in_batch].var(axis=0)
        batch_variance[batch_variance < 1e-12] = 1.0
        corrected[in_batch] = (combined[in_batch] - batch_mean) / np.sqrt(
            batch_variance
        )
    corrected = corrected * np.sqrt(pooled_variance)
    return corrected[: len(first)], corrected[len(first) :]


def select_top_mad_genes(matrix: np.ndarray, n_genes: int) -> set:
    """Select the genes with the largest median absolute deviation.

    Args:
        matrix: Sample-by-gene array.
        n_genes: Number of genes to keep.

    Returns:
        Set of selected gene positions. Genes with a non-finite MAD sort last.
    """
    median = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - median), axis=0)
    mad = np.where(np.isfinite(mad), mad, -1)
    return set(np.argsort(-mad)[:n_genes].tolist())


def compute_transfer_ari(
    first: np.ndarray, second: np.ndarray, k: int, seed: int = SEED
) -> tuple:
    """Score a cross-cohort label transfer against the target's own clustering.

    Clusters cohort A and cohort B independently, assigns every sample of B to the
    nearest centroid of A, then compares that assignment with B's own labels. Both
    labellings describe the same B samples, which is why the ARI is meaningful
    even though the two clusterings are not nested.

    Args:
        first: Sample-by-gene array of cohort A.
        second: Sample-by-gene array of cohort B.
        k: Number of clusters.
        seed: Random seed for the two k-means fits.

    Returns:
        Pair (adjusted Rand index, degeneracy flag). The flag is True when any of
        the three labellings holds fewer than two clusters.
    """
    n_init = param("alignment", "kmeans_n_init")
    kmeans_a = KMeans(k, n_init=n_init, random_state=seed).fit(first)
    kmeans_b = KMeans(k, n_init=n_init, random_state=seed).fit(second)
    centroids = kmeans_a.cluster_centers_
    transferred = (
        ((second[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    )
    # A single-cluster arm yields an ARI that cannot be interpreted.
    degenerate = (
        (len(set(kmeans_b.labels_)) < 2)
        or (len(set(transferred)) < 2)
        or (len(set(kmeans_a.labels_)) < 2)
    )
    return float(adjusted_rand_score(kmeans_b.labels_, transferred)), bool(degenerate)


def main() -> None:
    """Build the transfer-ARI matrix and write it to the work root."""
    np.random.seed(SEED)

    layer_cohorts = param("alignment", "layer_cohorts")
    normal_layers = param("alignment", "normal_layers")
    baseline, quantile, delta_reference, combat_pooled = param(
        "alignment", "strategies"
    )
    k_values = param("clustering", "k_values")
    n_top_mad = param("alignment", "top_mad_genes")
    min_shared = param("alignment", "min_shared_genes")
    min_hvg = param("alignment", "min_hvar_genes")

    out_path = work("alignment_ari")
    ensure_dir(out_path.parent)

    records: list[dict] = []
    print("=" * 104)
    print(
        "%-9s%-18s%-14s%11s%7s    ARI(K=%s)      degenerate"
        % (
            "layer",
            "strategy",
            "pair",
            "n(A/B)",
            "genes",
            "/".join(str(k) for k in k_values),
        )
    )
    print("=" * 104)
    for layer, cohorts in layer_cohorts.items():
        matrices = {cohort: load_layer(cohort, layer) for cohort in cohorts}
        matrices = {cohort: matrix for cohort, matrix in matrices.items() if matrix is not None}
        normals = {}
        for cohort in matrices:
            normal_token = normal_layers.get(cohort, {}).get(layer)
            normals[cohort] = load_layer(cohort, normal_token) if normal_token else None
        for first_cohort, second_cohort in itertools.combinations(sorted(matrices), 2):
            matrix_a, matrix_b = matrices[first_cohort], matrices[second_cohort]
            normal_a, normal_b = normals[first_cohort], normals[second_cohort]
            pair_label = "%s%s%s" % (first_cohort, PAIR_SEPARATOR, second_cohort)
            shared_genes = matrix_a.index.intersection(matrix_b.index)
            if len(shared_genes) < min_shared:
                records.append(
                    {
                        "layer": layer,
                        "pair": pair_label,
                        "ari": None,
                        "reason": "too few shared genes (%d)" % len(shared_genes),
                    }
                )
                continue
            # S0 baseline: samples are centred, genes are z-scored below once the
            # arm is built. Both cohorts are centred independently so that the
            # arm's input differs between them only in its distribution.
            centred_a = centre_samples(matrix_a.loc[shared_genes].values.T.astype(float))
            centred_b = centre_samples(matrix_b.loc[shared_genes].values.T.astype(float))
            centred_a = np.nan_to_num(centred_a, nan=0.0)
            centred_b = np.nan_to_num(centred_b, nan=0.0)
            # High-variance genes are selected per cohort on the centred data and
            # intersected, so a gene enters only when both cohorts vary in it.
            kept = np.array(
                sorted(
                    select_top_mad_genes(centred_a, n_top_mad)
                    & select_top_mad_genes(centred_b, n_top_mad)
                ),
                dtype=int,
            )
            if len(kept) < min_hvg:
                records.append(
                    {
                        "layer": layer,
                        "pair": pair_label,
                        "ari": None,
                        "reason": "too few high-variance genes",
                    }
                )
                continue
            kept_a, kept_b = centred_a[:, kept], centred_b[:, kept]
            arms = {baseline: (kept_a, kept_b)}
            arms[quantile] = (quantile_normalise(kept_a), quantile_normalise(kept_b))
            if normal_a is not None and normal_b is not None:
                # S2: subtract each cohort's own normal-tissue gene mean, which
                # removes the cohort-specific reference before any comparison.
                reference_a = np.nanmean(
                    normal_a.reindex(shared_genes).values[kept, :].astype(float), axis=1
                )
                reference_b = np.nanmean(
                    normal_b.reindex(shared_genes).values[kept, :].astype(float), axis=1
                )
                arms[delta_reference] = (
                    np.nan_to_num(kept_a - reference_a, nan=0.0),
                    np.nan_to_num(kept_b - reference_b, nan=0.0),
                )
            combat_a, combat_b = combat_location_scale(kept_a, kept_b)
            arms[combat_pooled] = (combat_a, combat_b)
            for strategy, (arm_a, arm_b) in arms.items():
                arm_a, arm_b = zscore_genes(arm_a), zscore_genes(arm_b)
                aris: dict[int, float | None] = {}
                degeneracies: dict[int, bool | None] = {}
                for k in k_values:
                    try:
                        value, degenerate = compute_transfer_ari(arm_a, arm_b, k)
                        aris[k] = round(value, 4)
                        degeneracies[k] = degenerate
                    except Exception:
                        # A cluster can fail to fit, for example when an arm
                        # collapses; the grid entry is then recorded as missing
                        # instead of aborting the sweep.
                        aris[k] = None
                        degeneracies[k] = None
                records.append(
                    {
                        "layer": layer,
                        "strategy": strategy,
                        "pair": pair_label,
                        "nA": int(kept_a.shape[0]),
                        "nB": int(kept_b.shape[0]),
                        "ngene": int(len(kept)),
                        "ari": aris,
                        "degenerate": degeneracies,
                    }
                )
                print(
                    "%-9s%-18s%-14s%11s%7d    %s      %d/%d"
                    % (
                        layer,
                        strategy,
                        pair_label,
                        "%d/%d" % (kept_a.shape[0], kept_b.shape[0]),
                        len(kept),
                        "/".join(
                            str(aris.get(k))
                            for k in k_values
                        ),
                        sum(1 for value in degeneracies.values() if value),
                        len(k_values),
                    )
                )
            # The file is rewritten after every pair so that an interrupted sweep
            # keeps the pairs it already completed.
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=1, ensure_ascii=False)
    print("\nWrote the transfer-ARI matrix to %s" % out_path)


if __name__ == "__main__":
    main()
