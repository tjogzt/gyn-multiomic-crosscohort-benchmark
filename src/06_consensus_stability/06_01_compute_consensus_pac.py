#!/usr/bin/env python3
"""
Bootstrap consensus stability (PAC) of every within-cohort integration arm.

Purpose
    Stage 06 of the pipeline, the consensus-stability analysis that follows the
    Python and R benchmarks (stages 04-05). For every combination of cohort,
    layer subset and integration method it rebuilds the feature space of that
    arm, resamples it with a bootstrap, and derives from the bootstrap consensus
    matrix the proportion of ambiguous clustering (PAC) together with the strong
    and weak consensus fractions, the median consensus value and the number of
    usable bootstrap replicates. PAC is the fraction of sample pairs whose
    consensus value falls strictly between pac_lower and pac_upper; a low PAC
    means the sample partition is stable under resampling, which is what a
    transferable clustering needs (Senbabaoglu et al., Scientific Reports 2014,
    "Critical limitations of consensus clustering in class discovery").
    Records are appended to work.pac_matrix after every arm, so an interrupted
    run keeps the work it already completed; 06_02 aggregates the result.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work.harmonised_matrix_pattern
        One harmonised matrix per cohort and layer (gzipped CSV, genes in rows,
        samples in columns), addressed as
        hcsv/<cohort>__<layer>.csv.gz for the cohorts and layers of the two
        evaluation domains, taken from params cohort_tags and layer_tags. A
        missing file simply removes that layer from the enumeration.
    params clustering.k_primary, clustering.n_bootstrap, features.top_mad_genes,
    params stability.pac_lower, stability.pac_upper, stability.strong_consensus,
    params stability.min_samples, stability.min_block_genes,
    params stability.bootstrap_unique_factor, stability.min_bootstrap_replicates,
    params stability.min_valid_pairs, domains, cohort_tags, layer_tags.

Outputs
    work.pac_matrix
        JSON list of per-arm records; each record carries the domain, cohort,
        layer subset, method label, sample and feature counts, pac, strong,
        weak, median_cons and boot_ok, or pac=None plus an err field when the arm
        could not be scored.

Usage
    python src/06_consensus_stability/06_01_compute_consensus_pac.py
"""

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances

# The package root (one level above src/) so that "common" can be imported from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, ensure_dir, param, work

# A library warning in this stage would be repeated once per bootstrap replicate
# over thousands of arms and would drown the progress output.
warnings.filterwarnings("ignore")

np.random.seed(SEED)

# --- Configuration -----------------------------------------------------------

# Clustering protocol.
K = int(param("clustering", "k_primary"))
N_BOOTSTRAP = int(param("clustering", "n_bootstrap"))
N_GENES = int(param("features", "top_mad_genes"))

# Stability definition (see the module docstring for the references).
PAC_LOWER = float(param("stability", "pac_lower"))
PAC_UPPER = float(param("stability", "pac_upper"))
STRONG_CONSENSUS = float(param("stability", "strong_consensus"))

# Guards on what may be scored at all.
MIN_SAMPLES = int(param("stability", "min_samples"))
MIN_BLOCK_GENES = int(param("stability", "min_block_genes"))
UNIQUE_FACTOR = int(param("stability", "bootstrap_unique_factor"))
MIN_REPLICATES = int(param("stability", "min_bootstrap_replicates"))
MIN_VALID_PAIRS = int(param("stability", "min_valid_pairs"))

# Evaluation domains, the cohort and layer tags of their harmonised matrices and
# the method vocabulary. Domain labels, method labels, layer tags and cohort tags
# are the strings 04_01 writes into work.benchmark_matrix, because 06_02 joins
# this stage's records to those on (domain, cohort, subset, method); they are
# therefore taken from the shared configuration and from the same label strings,
# never translated at the join.
DOMAIN = param("domains")
COHORT = param("cohort_tags")
LAYER = param("layer_tags")

# Domain A: the three CPTAC cohorts on three layers, the protein layer included.
# Domain B: TCGA-UCEC plus the two CPTAC-UCEC cohorts on four layers, with
# methylation and miRNA instead of protein.
DOMAINS = {
    DOMAIN["A"]: {
        "cohorts": [COHORT["independent"], COHORT["discovery"], COHORT["ov"]],
        "layers": [LAYER["mRNA"], LAYER["CNA"], LAYER["protein"]],
    },
    DOMAIN["B"]: {
        "cohorts": [COHORT["tcga"], COHORT["independent"], COHORT["discovery"]],
        "layers": [LAYER["mRNA"], LAYER["miRNA"], LAYER["CNA"], LAYER["methylation"]],
    },
}

MATRIX_PATTERN = work("harmonised_matrix_pattern")
OUT_PATH = work("pac_matrix")


# --- Matrix access -----------------------------------------------------------


def load_layer(cohort: str, layer: str) -> pd.DataFrame | None:
    """Read one harmonised cohort x layer matrix.

    Args:
        cohort: cohort stem used in the harmonised file name (e.g. "ind").
        layer: layer stem used in the harmonised file name (e.g. "mRNA").

    Returns:
        The matrix with genes in the index and samples in the columns, with the
        column labels forced to str so that they compare against the sample
        identifiers written by the harmonisation stage; None when the file does
        not exist, which is how a cohort without that layer drops out.
    """
    path = Path(str(MATRIX_PATTERN).format(cohort=cohort, layer=layer))
    if not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix


def zscore_genes(matrix: np.ndarray) -> np.ndarray:
    """Standardise every column (gene) of a samples x genes matrix.

    Args:
        matrix: samples x genes array.

    Returns:
        The column-wise z-scored array. Columns with a standard deviation below
        the numerical floor or with a non-finite deviation are left unscaled,
        because such a column carries no variance to standardise; remaining
        non-finite entries are replaced by zero so that downstream estimators
        (NMF, PCA, spectral embedding) receive a finite matrix.
    """
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    std = np.where((std < 1e-9) | ~np.isfinite(std), 1.0, std)
    return np.nan_to_num((matrix - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def select_top_mad(matrix: np.ndarray, n_genes: int) -> np.ndarray:
    """Select the n_genes most variable columns by median absolute deviation.

    Args:
        matrix: samples x genes array.
        n_genes: number of genes to retain at most.

    Returns:
        Indices of the retained columns, ordered by decreasing MAD. Genes with a
        MAD of exactly zero are dropped: they are constant and add no signal.
    """
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    order = np.argsort(-mad)[:n_genes]
    return order[mad[order] > 0]


# --- Method feature spaces ---------------------------------------------------
# The same feature spaces as the Python benchmark (stage 04), so that a stability
# verdict and a transfer verdict refer to the same construction.


def knn_affinity(matrix: np.ndarray, k: int) -> np.ndarray:
    """Build a symmetrised k-nearest-neighbour affinity matrix.

    Args:
        matrix: samples x features array.
        k: number of neighbours kept per sample.

    Returns:
        Row-normalised affinity matrix, the input the SNF fusion operates on.
        The Gaussian bandwidth is the median positive pairwise distance, a
        scale-free choice that does not depend on the feature count.
    """
    distances = euclidean_distances(matrix)
    positive = distances[distances > 0]
    mu = np.median(positive) if len(positive) else 1.0
    affinity = np.exp(-(distances**2) / (2 * mu**2))
    n = affinity.shape[0]
    sparse = np.zeros_like(affinity)
    for i in range(n):
        idx = np.argsort(-affinity[i])[:k]
        sparse[i, idx] = affinity[i, idx]
    sparse = (sparse + sparse.T) / 2
    row_sums = sparse.sum(1, keepdims=True)
    row_sums[row_sums < 1e-12] = 1
    return sparse / row_sums


def m_kmeans_concat(blocks: list, k: int) -> np.ndarray:
    """KMeans arm: concatenate the layer blocks and return the sample space.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters (unused here; the space is what is compared).

    Returns:
        The concatenated samples x features array.
    """
    return np.hstack(blocks)


def m_nmf_concat(blocks: list, k: int) -> np.ndarray:
    """NMF arm: non-negative factorisation of the concatenated blocks.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters (unused; the factorisation rank is fixed below).

    Returns:
        The samples x components matrix of the NMF factorisation. The shift to a
        strictly positive range is required because NMF accepts non-negative
        input only, and z-scored blocks contain negatives.
    """
    matrix = np.hstack(blocks)
    matrix = matrix - matrix.min(0, keepdims=True) + 1e-3
    return NMF(
        n_components=min(10, matrix.shape[1] - 1),
        init="nndsvd",
        random_state=SEED,
        max_iter=400,
    ).fit_transform(matrix)


def m_pca_concat(blocks: list, k: int) -> np.ndarray:
    """PCA arm: principal components of the concatenated blocks.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters (unused; the number of components is fixed below).

    Returns:
        The samples x components score matrix.
    """
    matrix = np.hstack(blocks)
    return PCA(
        n_components=min(10, matrix.shape[0] - 1, matrix.shape[1]), random_state=SEED
    ).fit_transform(matrix)


def m_spec_concat(blocks: list, k: int) -> np.ndarray:
    """Spectral-embedding arm on the concatenated blocks.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters (unused; the embedding dimension is fixed below).

    Returns:
        The spectral embedding of the concatenated blocks. The neighbour count is
        bounded by the sample size so that the density graph stays connected on
        small cohorts.
    """
    matrix = np.hstack(blocks)
    n = matrix.shape[0]
    return SpectralEmbedding(
        n_components=min(8, n - 1),
        affinity="nearest_neighbors",
        n_neighbors=max(5, min(15, n // 5)),
        random_state=SEED,
    ).fit_transform(matrix)


def m_mcca_lite(blocks: list, k: int) -> np.ndarray:
    """MCCA-lite arm: per-layer principal components concatenated side by side.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters (unused).

    Returns:
        The concatenated per-layer PCA scores, an unweighted approximation of
        multiple-set canonical correlation analysis.
    """
    return np.hstack(
        [
            PCA(
                n_components=min(5, block.shape[0] - 1, block.shape[1]), random_state=SEED
            ).fit_transform(block)
            for block in blocks
        ]
    )


def m_mofa_lite(blocks: list, k: int) -> np.ndarray:
    """MOFA-lite arm: per-layer principal components weighted by variance share.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters (unused).

    Returns:
        The concatenated per-layer PCA scores, each layer scaled by the square
        root of the share of variance it explains. The square root is used
        because the scores themselves are linear in the loadings, so the variance
        share scales the squared norm of the block.
    """
    outputs, weights = [], []
    for block in blocks:
        fitted = PCA(
            n_components=min(5, block.shape[0] - 1, block.shape[1]), random_state=SEED
        ).fit(block)
        outputs.append(fitted.transform(block))
        # Floor the share so that a layer with no explained variance keeps a
        # small non-zero weight instead of dropping out of the concatenation.
        weights.append(max(float(fitted.explained_variance_ratio_.sum()), 1e-6))
    weights = np.array(weights)
    weights /= weights.sum()
    return np.hstack([out * np.sqrt(weight) for out, weight in zip(outputs, weights)])


def m_consensus(blocks: list, k: int) -> np.ndarray:
    """Co-association arm: one-hot cluster memberships of every layer side by side.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters used per layer.

    Returns:
        The concatenated one-hot membership matrix (samples x layers*k). A layer
        whose KMeans fails contributes an all-zero block so that the remaining
        layers are still comparable across arms.
    """
    outputs = []
    for block in blocks:
        try:
            labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(block)
            one_hot = np.zeros((len(labels), k))
            one_hot[np.arange(len(labels)), labels] = 1.0
            outputs.append(one_hot)
        except Exception:
            outputs.append(np.zeros((block.shape[0], k)))
    return np.hstack(outputs)


def m_snf(blocks: list, k: int, iterations: int = 20) -> np.ndarray:
    """SNF arm: similarity-network fusion followed by a spectral embedding.

    Args:
        blocks: list of per-layer arrays, all with the same samples in rows.
        k: number of clusters (unused; the embedding dimension is fixed below).
        iterations: number of fusion iterations of the message-passing update
            (Wang et al., Nature Methods 2014, similarity network fusion).

    Returns:
        The spectral embedding of the fused network. With fewer than 15 samples
        the fusion is undefined (the density graph degenerates), so the single
        available block is returned unchanged.
    """
    n = blocks[0].shape[0]
    if n < 15:
        return blocks[0]
    neighbours = max(3, min(20, n // 10))
    affinities = [knn_affinity(block, neighbours) for block in blocks]
    current = [affinity.copy() for affinity in affinities]
    for _ in range(iterations):
        updated = []
        for i in range(len(current)):
            others = np.zeros_like(current[i])
            for j in range(len(current)):
                if j != i:
                    others = others + current[j]
            others /= max(len(current) - 1, 1)
            updated.append(affinities[i] @ others @ affinities[i].T)
        current = [(matrix + matrix.T) / 2 for matrix in updated]
    fused = np.mean(current, axis=0)
    fused = np.clip((fused + fused.T) / 2, 0, None)
    dimension = min(8, n - 2)
    try:
        return SpectralEmbedding(
            n_components=dimension, affinity="precomputed", random_state=SEED
        ).fit_transform(fused)
    except Exception:
        return np.hstack(blocks)


# Method label -> feature-space constructor. The label is written into every
# record and is what 06_02 and the result tables group by.
METHODS = {
    "1_SNF": lambda blocks, k: m_snf(blocks, k),
    "2_NMF(concat)": lambda blocks, k: m_nmf_concat(blocks, k),
    "3_KMeans(concat)": lambda blocks, k: m_kmeans_concat(blocks, k),
    "4_PCA(concat)": lambda blocks, k: m_pca_concat(blocks, k),
    "5_SpectralEmbedding(concat)": lambda blocks, k: m_spec_concat(blocks, k),
    "6_CoassociationConsensus": lambda blocks, k: m_consensus(blocks, k),
    "7_MCCA-lite": lambda blocks, k: m_mcca_lite(blocks, k),
    "8_MOFA-lite": lambda blocks, k: m_mofa_lite(blocks, k),
}


# --- PAC ---------------------------------------------------------------------


def compute_pac(
    feature_space: np.ndarray,
    k: int = K,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict | None:
    """Bootstrap the feature space and measure consensus stability (PAC).

    Every bootstrap replicate resamples the samples with replacement, clusters
    the distinct samples with KMeans and adds one co-clustering vote to every
    pair of distinct samples that falls into the same cluster. The consensus
    value of a pair is its vote share across the replicates that contained both
    of its samples.

    Args:
        feature_space: samples x features array of the arm.
        k: number of clusters of the consensus experiment.
        n_bootstrap: number of bootstrap replicates.
        seed: seed of the resampling and of KMeans.

    Returns:
        A dict with pac (fraction of pairs whose consensus value lies strictly
        between PAC_LOWER and PAC_UPPER), strong (fraction above
        STRONG_CONSENSUS), weak (fraction below PAC_LOWER), median_cons,
        boot_ok (number of usable replicates), or None when the cohort is below
        MIN_SAMPLES, when fewer than MIN_REPLICATES replicates succeeded, or when
        fewer than MIN_VALID_PAIRS sample pairs are available.
    """
    rng = np.random.RandomState(seed)
    n = feature_space.shape[0]
    if n < MIN_SAMPLES:
        return None
    accumulated = np.zeros((n, n))
    counted = np.zeros((n, n))
    replicates_ok = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        unique = np.unique(idx)
        # A replicate that draws too few distinct samples cannot fill every
        # cluster, which would bias PAC downwards; those replicates are dropped.
        if len(unique) < k * UNIQUE_FACTOR:
            continue
        try:
            labels = KMeans(k, n_init=5, random_state=seed).fit_predict(feature_space[unique])
        except Exception:
            continue
        # A replicate that returns a single cluster votes for every pair and
        # would push PAC to zero; such replicates carry no information.
        if len(set(labels)) < 2:
            continue
        present = np.zeros(n, bool)
        present[unique] = True
        labels_full = np.full(n, -1)
        labels_full[unique] = labels
        same_cluster = (labels_full[:, None] == labels_full[None, :]) & present[:, None] & present[None, :]
        accumulated += same_cluster
        counted += present[:, None] & present[None, :]
        replicates_ok += 1
    if replicates_ok < MIN_REPLICATES:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = np.where(counted > 0, accumulated / np.maximum(counted, 1), np.nan)
    upper = np.triu_indices(n, 1)
    values = consensus[upper]
    finite = np.isfinite(values)
    if finite.sum() < MIN_VALID_PAIRS:
        return None
    values = values[finite]
    return {
        "pac": float(np.mean((values > PAC_LOWER) & (values < PAC_UPPER))),
        "strong": float(np.mean(values > STRONG_CONSENSUS)),
        "weak": float(np.mean(values < PAC_LOWER)),
        "median_cons": float(np.median(values)),
        "boot_ok": replicates_ok,
    }


# --- Driver ------------------------------------------------------------------


def main() -> list:
    """Score every (domain, cohort, layer subset, method) arm and write the records.

    Returns:
        The list of per-arm records that was written to work.pac_matrix.
    """
    ensure_dir(OUT_PATH.parent)
    records = []
    start = time.time()
    # The harmonised matrices are read once per (cohort, layer) and reused across
    # every layer subset that contains that layer.
    cache: dict = {}
    for domain_name, domain in DOMAINS.items():
        cohorts = domain["cohorts"]
        layers = domain["layers"]
        print(
            "\n"
            + "=" * 100
            + f"\nDomain {domain_name}  {cohorts} x {layers}\n"
            + "=" * 100,
            flush=True,
        )
        for cohort in cohorts:
            for size in range(1, len(layers) + 1):
                for subset in itertools.combinations(layers, size):
                    matrices = {}
                    for layer in subset:
                        key = (cohort, layer)
                        if key not in cache:
                            cache[key] = load_layer(cohort, layer)
                        if cache[key] is None:
                            matrices = None
                            break
                        matrices[layer] = cache[key]
                    if not matrices:
                        continue
                    # Sample intersection across the layers of the subset.
                    shared = None
                    for matrix in matrices.values():
                        shared = (
                            set(matrix.columns)
                            if shared is None
                            else (shared & set(matrix.columns))
                        )
                    shared = sorted(shared)
                    if len(shared) < MIN_SAMPLES:
                        continue
                    blocks = []
                    for layer in subset:
                        matrix = matrices[layer][shared]
                        values = zscore_genes(matrix.values.T.astype(float))
                        genes = select_top_mad(values, N_GENES)
                        # Fall back to the leading genes of the ranking when the
                        # MAD filter leaves too few, so that a heavily imputed
                        # layer still contributes instead of dropping the subset.
                        if len(genes) < MIN_BLOCK_GENES:
                            genes = np.arange(min(N_GENES, values.shape[1]))
                        blocks.append(values[:, genes])
                    for method_name, constructor in METHODS.items():
                        try:
                            space = zscore_genes(
                                np.nan_to_num(
                                    np.asarray(constructor(blocks, K), float), nan=0.0
                                )
                            )
                        except Exception as exc:
                            records.append(
                                {
                                    "domain": domain_name,
                                    "cohort": cohort,
                                    "subset": "+".join(subset),
                                    "method": method_name,
                                    "pac": None,
                                    "err": str(exc)[:60],
                                }
                            )
                            continue
                        if space.shape[0] < MIN_SAMPLES:
                            records.append(
                                {
                                    "domain": domain_name,
                                    "cohort": cohort,
                                    "subset": "+".join(subset),
                                    "method": method_name,
                                    "pac": None,
                                    "err": f"n<{MIN_SAMPLES}",
                                }
                            )
                            continue
                        stability = compute_pac(space)
                        record = {
                            "domain": domain_name,
                            "cohort": cohort,
                            "subset": "+".join(subset),
                            "method": method_name,
                            "n": int(space.shape[0]),
                            "nfeat": int(space.shape[1]),
                        }
                        record.update(stability if stability else {"pac": None})
                        records.append(record)
                    # Persist after every subset: the grid is large and an
                    # interrupted run must keep the arms already scored.
                    with OUT_PATH.open("w", encoding="utf-8") as handle:
                        json.dump(records, handle, indent=1, ensure_ascii=False)
                    print(
                        f"  {cohort:6s} {'+'.join(subset):<26} n={len(shared):4d} "
                        f"({time.time() - start:.0f}s, {len(records)} records)",
                        flush=True,
                    )
    print(f"\nCompleted {len(records)} records in {time.time() - start:.0f}s")
    print(f"Written to {OUT_PATH}")
    return records


if __name__ == "__main__":
    main()
