"""
Benchmark eight unsupervised multi-omic integration methods by cross-cohort
transfer ARI (first, per-layer-intersection protocol).

Purpose
    Evaluate every integration method in ``METHODS`` by clustering one cohort of
    a pair and transferring the labels to the second cohort through nearest
    centroid, scoring the transfer with the adjusted Rand index. The unit of
    evaluation is a (cohort pair x non-empty layer subset) combination. Domain A
    uses the three CPTAC cohorts on mRNA + CNA + protein; domain B adds TCGA-UCEC
    and methylation, on mRNA + miRNA + CNA + methylation. The module consumes the
    harmonised per-cohort layer matrices and writes the raw benchmark record
    file used by the aggregation stage. It is the first of two Python benchmark
    drivers; the second re-runs the protocol with a corrected sample-intersection
    rule and a method-independent reference partition.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("harmonised_dir")  Directory holding one <cohort>__<layer>.pkl matrix
                            per cohort and layer (samples x genes, pandas
                            DataFrame). Missing files are skipped.
    param("cohort_tags")    Frozen cohort tags, used to build the file names and
                            the record fields.
    param("layer_tags")     Frozen layer tags, used to build the file names.
    param("domains")        The two evaluation domains and their labels.

Outputs
    work("benchmark_matrix")  JSON list of benchmark records, one per
                              (domain, layer subset, cohort pair, method), with
                              the per-K transfer ARI, degeneracy flag, sample
                              counts and feature count.

Usage
    python 04_01_benchmark_python_methods.py
"""

import sys
from pathlib import Path

# Make the shared configuration loader importable regardless of the caller's
# working directory: this file sits in src/<stage>/, so ``parents[1]`` is src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import work, param, ensure_dir, SEED

# Separator inside a cohort-pair label; declared in params.yaml because
# downstream stages split records on it. See the note there.
PAIR_SEPARATOR = param("benchmark", "pair_separator")

import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# --- Configuration -----------------------------------------------------------
HARMONISED_DIR = work("harmonised_dir")
OUT_MATRIX = work("benchmark_matrix")

NGENE = param("features", "top_mad_genes")
KGRID = param("clustering", "k_values")
NBOOT = param("clustering", "n_bootstrap")
K_PRIMARY = param("clustering", "k_primary")

# Gates deciding whether a (cohort pair x layer) combination is evaluated at all.
MIN_COMMON_GENES = param("benchmark", "min_common_genes")
MIN_COMMON_MAD_GENES = param("benchmark", "min_common_mad_genes")
MIN_SAMPLES = param("benchmark", "min_samples")
# Extra floor for the resample-based PAC statistic, which needs more samples
# than a deterministic evaluation to produce a stable estimate.
MIN_SAMPLES_PAC = param("benchmark", "min_samples_pac")

# Consensus interval that defines an ambiguous pair for the PAC statistic.
PAC_LOWER = param("stability", "pac_lower")
PAC_UPPER = param("stability", "pac_upper")

COHORT = param("cohort_tags")
LAYER = param("layer_tags")
DOMAIN = param("domains")


# --- Data access -------------------------------------------------------------


def load_layer(cohort, layer):
    """Load one harmonised layer matrix.

    Args:
        cohort: Frozen cohort tag, e.g. ``ind``.
        layer: Frozen layer tag, e.g. ``mRNA``.

    Returns:
        The pandas DataFrame if the pickled matrix exists, otherwise ``None`` so
        that absent layers are skipped rather than treated as an error.
    """
    path = HARMONISED_DIR / f"{cohort}__{layer}.pkl"
    return pd.read_pickle(path) if os.path.exists(path) else None


# --- Shared numerical helpers ------------------------------------------------


def zgene(X):
    """Standardise every column (gene) to mean 0 and unit standard deviation.

    Args:
        X: Sample x gene matrix.

    Returns:
        The standardised matrix. Non-finite entries become 0, and columns with a
        degenerate (near-zero or non-finite) standard deviation are left as-is by
        substituting a unit denominator instead of dividing by it. This is the
        per-layer standardisation the protocol applies before concatenation.
    """
    mean = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    # Guard the denominator: a zero or non-finite SD would produce inf/NaN.
    sd = np.where((sd < 1e-9) | ~np.isfinite(sd), 1.0, sd)
    return np.nan_to_num((X - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def topmad_k(X, ng):
    """Select the ``ng`` genes with the largest median absolute deviation.

    Args:
        X: Sample x gene matrix.
        ng: Number of genes to retain.

    Returns:
        Integer column indices of the selected genes, restricted to those whose
        MAD is strictly positive (a zero-MAD gene carries no signal).
    """
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    k = np.argsort(-mad)[:ng]
    return k[mad[k] > 0]


def knn_affinity(X, k):
    """Build a k-nearest-neighbour Gaussian affinity matrix, row-normalised.

    Args:
        X: Sample x feature matrix.
        k: Number of neighbours kept per sample.

    Returns:
        An n x n doubly symmetrised, row-normalised affinity matrix. The Gaussian
        bandwidth is the median non-zero pairwise distance, the standard SNF
        choice.
    """
    D = euclidean_distances(X)
    mu = np.median(D[D > 0]) if (D > 0).any() else 1.0
    S = np.exp(-(D**2) / (2 * mu**2))
    n = S.shape[0]
    K = np.zeros_like(S)
    for i in range(n):
        idx = np.argsort(-S[i])[:k]
        K[i, idx] = S[i, idx]
    # Symmetrise, then normalise rows; a row of zeros is left untouched.
    K = (K + K.T) / 2
    rs = K.sum(axis=1, keepdims=True)
    rs[rs < 1e-12] = 1
    return K / rs


# --- Integration methods -----------------------------------------------------
# Each method maps a list of per-layer blocks to a sample x feature matrix. The
# label prefixes number the arms in the order reported in the results; the labels
# are the frozen arm identifiers carried into every downstream table.


def m_kmeans_concat(ls, K):
    """Concatenate the layers without any reduction (the baseline arm).

    Args:
        ls: List of per-layer sample x feature blocks.
        K: Number of clusters (unused; the arm returns features, not labels).

    Returns:
        Horizontally concatenated blocks.
    """
    return np.hstack(ls)


def m_nmf_concat(ls, K):
    """Concatenate the layers and reduce them with non-negative matrix factorisation.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The NMF sample loadings. A small positive offset makes the concatenated
        matrix non-negative, which NMF requires.
    """
    X = np.hstack(ls)
    X = X - X.min(0, keepdims=True) + 1e-3
    return NMF(
        n_components=min(10, X.shape[1] - 1),
        init="nndsvd",
        random_state=SEED,
        max_iter=400,
    ).fit_transform(X)


def m_pca_concat(ls, K):
    """Concatenate the layers and reduce them with PCA.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The PCA scores of the concatenated matrix.
    """
    X = np.hstack(ls)
    return PCA(n_components=min(10, X.shape[0] - 1), random_state=SEED).fit_transform(X)


def m_spec_concat(ls, K):
    """Concatenate the layers and embed them with a spectral embedding.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The spectral embedding coordinates. The neighbour count is scaled to the
        sample size so that small cohort pairs remain connected.
    """
    X = np.hstack(ls)
    return SpectralEmbedding(
        n_components=min(8, X.shape[0] - 1),
        affinity="nearest_neighbors",
        n_neighbors=max(5, min(15, X.shape[0] // 5)),
        random_state=SEED,
    ).fit_transform(X)


def m_mcca_lite(ls, K):
    """Per-layer PCA followed by concatenation (a lightweight MCCA surrogate).

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The concatenated per-layer PCA scores.
    """
    return np.hstack(
        [
            PCA(n_components=min(5, X.shape[0] - 1), random_state=SEED).fit_transform(X)
            for X in ls
        ]
    )


def m_mofa_lite(ls, K):
    """Per-layer PCA reweighted by explained variance and concatenated (MOFA surrogate).

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The concatenated, variance-weighted per-layer scores. The weight of a
        layer is the square root of its explained-variance fraction, so layers
        that carry more structure contribute proportionally more.
    """
    outs, weights = [], []
    for X in ls:
        p = PCA(n_components=min(5, X.shape[0] - 1), random_state=SEED).fit(X)
        outs.append(p.transform(X))
        weights.append(max(float(p.explained_variance_ratio_.sum()), 1e-6))
    weights = np.array(weights)
    weights /= weights.sum()
    return np.hstack([o * np.sqrt(w) for o, w in zip(outs, weights)])


def m_consensus(ls, K):
    """Per-layer KMeans, co-association matrix, average-linkage partition.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters.

    Returns:
        The average-linkage partition of the co-association distance, reshaped to
        a single-column feature block. If no layer can be clustered the first
        layer is returned unchanged.
    """
    n = ls[0].shape[0]
    C = np.zeros((n, n))
    m = 0
    for X in ls:
        try:
            lb = KMeans(K, n_init=10, random_state=SEED).fit_predict(X)
            C += (lb[:, None] == lb[None, :]).astype(float)
            m += 1
        except Exception:
            # A layer that cannot be clustered simply does not vote.
            pass
    if m == 0:
        return ls[0]
    C /= m
    D = 1 - C
    np.fill_diagonal(D, 0)
    try:
        return (
            AgglomerativeClustering(
                n_clusters=K, metric="precomputed", linkage="average"
            )
            .fit_predict(D)
            .reshape(-1, 1)
            .astype(float)
        )
    except Exception:
        return ls[0]


def m_snf(ls, K, t=20):
    """Standard similarity network fusion followed by average-linkage clustering.

    Implements the SNF message passing ``P_i <- W_i x mean_{j != i}(P_j) x W_i^T``
    (Wang et al., Nature Methods 2014), iterated ``t`` times.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters.
        t: Number of fusion iterations.

    Returns:
        The average-linkage partition of the fused network, reshaped to a
        single-column feature block. Very small cohorts are returned unchanged.
    """
    n = ls[0].shape[0]
    if n < MIN_SAMPLES:
        return ls[0]
    k = max(3, min(20, n // 10))
    W = [knn_affinity(X, k) for X in ls]
    P = [w.copy() for w in W]
    for _ in range(t):
        newP = []
        for i in range(len(P)):
            others = np.zeros_like(P[i])
            for j in range(len(P)):
                if j != i:
                    others = others + P[j]
            others /= max(len(P) - 1, 1)
            newP.append(W[i] @ others @ W[i].T)
        P = [(p + p.T) / 2 for p in newP]
    Sf = np.mean(P, axis=0)
    D = 1 - Sf / (Sf.max() + 1e-12)
    np.fill_diagonal(D, 0)
    try:
        return (
            AgglomerativeClustering(
                n_clusters=K, metric="precomputed", linkage="average"
            )
            .fit_predict(D)
            .reshape(-1, 1)
            .astype(float)
        )
    except Exception:
        return ls[0]


METHODS = {
    "1_SNF": lambda ls, K: m_snf(ls, K),
    "2_NMF(concat)": lambda ls, K: m_nmf_concat(ls, K),
    "3_KMeans(concat)": lambda ls, K: m_kmeans_concat(ls, K),
    "4_PCA(concat)": lambda ls, K: m_pca_concat(ls, K),
    "5_SpectralEmbedding(concat)": lambda ls, K: m_spec_concat(ls, K),
    "6_CoassociationConsensus": lambda ls, K: m_consensus(ls, K),
    "7_MCCA-lite": lambda ls, K: m_mcca_lite(ls, K),
    "8_MOFA-lite": lambda ls, K: m_mofa_lite(ls, K),
}


# --- Transfer protocol -------------------------------------------------------


def transfer_ari(Xa, Xb, K, seed=SEED):
    """Transfer a clustering fitted on A to the samples of B and score it.

    Args:
        Xa: Feature matrix of the first cohort.
        Xb: Feature matrix of the second cohort, in the same feature space.
        K: Number of clusters.
        seed: Random seed for the two KMeans fits.

    Returns:
        ``(ari, degeneracy)`` where ``ari`` is the adjusted Rand index between the
        clustering of B and the centroid assignment of B, and ``degeneracy`` is 1
        when either partition has fewer than two distinct labels (such a record is
        flagged and later excluded from verdicts).
    """
    kmA = KMeans(K, n_init=20, random_state=seed).fit(Xa)
    kmB = KMeans(K, n_init=20, random_state=seed).fit(Xb)
    # Assign each B sample to its nearest centroid of the A clustering.
    pred = ((Xb[:, None, :] - kmA.cluster_centers_[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    deg = int(len(set(kmB.labels_)) < 2 or len(set(pred)) < 2)
    return float(adjusted_rand_score(kmB.labels_, pred)), deg


def pac(X, K, nboot=NBOOT, seed=SEED):
    """Proportion of ambiguous clustering, the consensus stability statistic.

    Args:
        X: Sample x feature matrix.
        K: Number of clusters fitted on each resample.
        nboot: Number of bootstrap resamples.
        seed: Random seed for the shared RandomState.

    Returns:
        The fraction of sample pairs whose co-association value falls strictly
        inside the ambiguity interval, or ``None`` when the cohort is too small or
        too few resamples succeeded (the record is then left empty rather than
        guessed).
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if n < MIN_SAMPLES_PAC:  # too few samples for a resample-based statistic
        return None
    acc = np.zeros((n, n))
    cnt = 0
    for _ in range(nboot):
        idx = rng.choice(n, n, replace=True)
        uniq = np.unique(idx)
        # A resample must retain enough distinct samples to fit K clusters.
        if len(uniq) < K * 3:
            continue
        try:
            lb = KMeans(K, n_init=10, random_state=seed).fit_predict(X[uniq])
        except Exception:
            continue
        pos = {u: i for i, u in enumerate(uniq)}
        for a in range(n):
            for b in range(a + 1, n):
                if a in pos and b in pos:
                    acc[a, b] += lb[pos[a]] == lb[pos[b]]
                    acc[b, a] = acc[a, b]
        cnt += 1
    if cnt < 4:  # too few usable resamples to form a stable estimate
        return None
    acc /= cnt
    iu = np.triu_indices(n, 1)
    v = acc[iu]
    return float(np.mean((v > PAC_LOWER) & (v < PAC_UPPER)))


# --- Evaluation domains ------------------------------------------------------
# Domain A: the three CPTAC cohorts on three layers.
# Domain B: TCGA-UCEC plus the two CPTAC-UCEC cohorts on four layers.

DOMAINS = {
    DOMAIN["A"]: {
        "cohorts": [COHORT["discovery"], COHORT["independent"], COHORT["ov"]],
        "layers": [LAYER["mRNA"], LAYER["CNA"], LAYER["protein"]],
    },
    DOMAIN["B"]: {
        "cohorts": [COHORT["tcga"], COHORT["discovery"], COHORT["independent"]],
        "layers": [LAYER["mRNA"], LAYER["miRNA"], LAYER["CNA"], LAYER["methylation"]],
    },
}


def main():
    """Run the transfer benchmark over every domain, subset, pair and method."""
    ensure_dir(OUT_MATRIX.parent)
    res = []
    t0 = time.time()
    for dname, D in DOMAINS.items():
        cohs, lays = D["cohorts"], D["layers"]
        # Cache the layer matrices once per domain; each pair reuses them.
        RAW = {}
        for c in cohs:
            for l in lays:
                M = load_layer(c, l)
                if M is not None:
                    RAW[(c, l)] = M
        print(
            f"\n{'=' * 106}\nDomain {dname}  cohorts={cohs}  layers={lays}\n{'=' * 106}",
            flush=True,
        )
        for c in cohs:
            print(f"  {c:6s} available layers: {[l for l in lays if (c, l) in RAW]}", flush=True)
        # Enumerate every non-empty layer subset in increasing size.
        subsets = [s for r in range(1, len(lays) + 1) for s in itertools.combinations(lays, r)]
        for sub in subsets:
            for a, b in itertools.combinations(cohs, 2):
                # Keep only the layers that both cohorts of the pair actually have.
                common = [l for l in sub if (a, l) in RAW and (b, l) in RAW]
                if not common:
                    continue
                A_sel, B_sel, ok = [], [], True
                for l in common:
                    ga = set(RAW[(a, l)].index)
                    gb = set(RAW[(b, l)].index)
                    gk = sorted(ga & gb)
                    if len(gk) < MIN_COMMON_GENES:
                        ok = False
                        break
                    Xa = zgene(RAW[(a, l)].loc[gk].values.T.astype(float))
                    Xb = zgene(RAW[(b, l)].loc[gk].values.T.astype(float))
                    ka, kb = topmad_k(Xa, NGENE), topmad_k(Xb, NGENE)
                    k = np.array(sorted(set(ka.tolist()) & set(kb.tolist())))
                    # Fall back to the first NGENE genes when too few MAD genes agree.
                    if len(k) < MIN_COMMON_MAD_GENES:
                        k = np.arange(min(NGENE, Xa.shape[1]))
                    A_sel.append(Xa[:, k])
                    B_sel.append(Xb[:, k])
                if not ok or not A_sel:
                    continue
                for mname, fn in METHODS.items():
                    try:
                        Fa = fn(A_sel, K_PRIMARY)
                        Fb = fn(B_sel, K_PRIMARY)
                    except Exception as ex:
                        res.append(
                            {
                                "domain": dname,
                                "subset": "+".join(common),
                                "pair": f"{a}{PAIR_SEPARATOR}{b}",
                                "method": mname,
                                "ari": None,
                                "err": str(ex)[:70],
                            }
                        )
                        continue
                    # The two blocks must share a feature space to be comparable.
                    Fa = zgene(np.nan_to_num(np.asarray(Fa, dtype=float), nan=0.0))
                    Fb = zgene(np.nan_to_num(np.asarray(Fb, dtype=float), nan=0.0))
                    if (
                        Fa.shape[1] != Fb.shape[1]
                        or Fa.shape[0] < MIN_SAMPLES
                        or Fb.shape[0] < MIN_SAMPLES
                    ):
                        res.append(
                            {
                                "domain": dname,
                                "subset": "+".join(common),
                                "pair": f"{a}{PAIR_SEPARATOR}{b}",
                                "method": mname,
                                "ari": None,
                                "err": f"dim {Fa.shape}/{Fb.shape}",
                            }
                        )
                        continue
                    aris, degs = {}, {}
                    for K in KGRID:
                        try:
                            v, d = transfer_ari(Fa, Fb, K)
                            aris[K] = round(v, 4)
                            degs[K] = d
                        except Exception:
                            aris[K] = None
                            degs[K] = None
                    res.append(
                        {
                            "domain": dname,
                            "subset": "+".join(common),
                            "pair": f"{a}{PAIR_SEPARATOR}{b}",
                            "method": mname,
                            "ari": aris,
                            "degen": degs,
                            "nA": int(Fa.shape[0]),
                            "nB": int(Fb.shape[0]),
                            "nfeat": int(Fa.shape[1]),
                        }
                    )
                # Persist after every pair so a long run survives interruption.
                with open(OUT_MATRIX, "w") as handle:
                    json.dump(res, handle, indent=1, ensure_ascii=False)
            print(
                f"  subset {'+'.join(sub)} done ({time.time() - t0:.0f}s, "
                f"{len(res)} records so far)",
                flush=True,
            )
    print(f"\nDone: {len(res)} records in {time.time() - t0:.0f}s")
    print("written:", OUT_MATRIX)


if __name__ == "__main__":
    main()
