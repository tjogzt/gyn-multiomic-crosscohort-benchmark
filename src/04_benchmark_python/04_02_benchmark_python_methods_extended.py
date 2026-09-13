"""
Benchmark eight unsupervised multi-omic integration methods by cross-cohort
transfer ARI (second, per-(cohort x subset)-intersection protocol).

Purpose
    Re-run the Python integration benchmark with two corrections relative to the
    first driver. First, the sample set of a (cohort, layer subset) cell is the
    intersection of the samples present in *all* selected layers of that cohort,
    so the number of usable samples becomes the core quantity of the per-layer
    marginal-value analysis. Second, the transfer score of a method is compared
    against a method-independent reference partition derived from the raw
    concatenated representation of cohort B, which removes the structural
    self-confirmation of consensus and graph methods. The module additionally
    writes the sample-availability table. It reads the harmonised per-cohort
    layer matrices, materialises a compressed-CSV cache for the R side, and
    writes the benchmark record file consumed by the aggregation stage.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("harmonised_dir")  Directory holding one <cohort>__<layer>.pkl matrix
                            per cohort and layer (samples x genes).
    param("cohort_tags")    Frozen cohort tags used in file names and records.
    param("layer_tags")     Frozen layer tags used in file names.
    param("domains")        The two evaluation domains and their labels.

Outputs
    work("hcsv_dir")                     Compressed-CSV mirror of the layer
                                         matrices, read by the R benchmark.
    work("benchmark_matrix")             JSON benchmark records as in the first
                                         driver, extended with the cluster-balance
                                         diagnostics.
    work("benchmark_availability")       JSON records of the number of usable
                                         samples per (domain, subset, cohort).

Usage
    python 04_02_benchmark_python_methods_extended.py
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
HCSV_DIR = work("hcsv_dir")
# Template for the per-cohort per-layer compressed CSV, e.g. <hcsv>/ind__mRNA.csv.gz.
HCSV_PATTERN = str(work("harmonised_matrix_pattern"))
OUT_MATRIX = work("benchmark_matrix")
OUT_AVAILABILITY = work("benchmark_availability")

NGENE = param("features", "top_mad_genes")
KGRID = param("clustering", "k_values")
K_PRIMARY = param("clustering", "k_primary")

MIN_COMMON_GENES = param("benchmark", "min_common_genes")
MIN_COMMON_MAD_GENES = param("benchmark", "min_common_mad_genes")
MIN_SAMPLES = param("benchmark", "min_samples")

COHORT = param("cohort_tags")
LAYER = param("layer_tags")
DOMAIN = param("domains")


# --- Data access -------------------------------------------------------------


def hcsv_path(cohort, layer):
    """Return the compressed-CSV path for one cohort and layer.

    Args:
        cohort: Frozen cohort tag.
        layer: Frozen layer tag.

    Returns:
        The path built from the configured matrix-name pattern.
    """
    return Path(HCSV_PATTERN.format(cohort=cohort, layer=layer))


def build_csv_cache():
    """Mirror every pickled layer matrix to a compressed CSV, once.

    The R benchmark cannot read Python pickles, so the harmonised matrices are
    exported as gzip-compressed CSV. Existing exports are left untouched, which
    keeps the function cheap on re-runs.

    Returns:
        None. Writes files into ``work("hcsv_dir")``.
    """
    ensure_dir(HCSV_DIR)
    for name in os.listdir(HARMONISED_DIR):
        if not name.endswith(".pkl"):
            continue
        cohort, layer = name[:-4].split("__")
        out_path = hcsv_path(cohort, layer)
        if os.path.exists(out_path):
            continue
        M = pd.read_pickle(HARMONISED_DIR / name)
        M.to_csv(out_path, compression="gzip")
    print("CSV cache ready:", len(os.listdir(HCSV_DIR)))


def load_layer(cohort, layer):
    """Load one layer matrix from the compressed-CSV cache.

    Args:
        cohort: Frozen cohort tag.
        layer: Frozen layer tag.

    Returns:
        The DataFrame with string sample names, or ``None`` when the cached file
        does not exist.
    """
    path = hcsv_path(cohort, layer)
    if not os.path.exists(path):
        return None
    M = pd.read_csv(path, index_col=0)
    # Force string sample identifiers: CSV round-tripping can coerce them.
    M.columns = [str(c) for c in M.columns]
    return M


# --- Shared numerical helpers ------------------------------------------------


def zgene(X):
    """Standardise every column (gene) to mean 0 and unit standard deviation.

    Args:
        X: Sample x gene matrix.

    Returns:
        The standardised matrix with non-finite entries set to 0; degenerate
        columns are protected by substituting a unit denominator.
    """
    mean = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd = np.where((sd < 1e-9) | ~np.isfinite(sd), 1.0, sd)
    return np.nan_to_num((X - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def topmad_k(X, ng):
    """Select the ``ng`` genes with the largest median absolute deviation.

    Args:
        X: Sample x gene matrix.
        ng: Number of genes to retain.

    Returns:
        Integer column indices of the selected genes with strictly positive MAD.
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
        An n x n symmetrised, row-normalised affinity matrix using the median
        non-zero distance as the Gaussian bandwidth.
    """
    D = euclidean_distances(X)
    nz = D[D > 0]
    mu = np.median(nz) if len(nz) else 1.0
    S = np.exp(-(D**2) / (2 * mu**2))
    n = S.shape[0]
    K = np.zeros_like(S)
    for i in range(n):
        idx = np.argsort(-S[i])[:k]
        K[i, idx] = S[i, idx]
    K = (K + K.T) / 2
    rs = K.sum(1, keepdims=True)
    rs[rs < 1e-12] = 1
    return K / rs


# --- Integration methods -----------------------------------------------------
# Each method maps a list of per-layer blocks to a sample x feature matrix. The
# label prefixes number the arms in the order reported in the results.


def m_kmeans_concat(ls, K):
    """Concatenate the layers without reduction (the baseline arm).

    Args:
        ls: List of per-layer sample x feature blocks.
        K: Number of clusters (unused).

    Returns:
        Horizontally concatenated blocks.
    """
    return np.hstack(ls)


def m_nmf_concat(ls, K):
    """Concatenate and reduce with non-negative matrix factorisation.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The NMF sample loadings; a small positive offset enforces non-negativity.
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
    """Concatenate and reduce with PCA.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The PCA scores of the concatenated matrix.
    """
    X = np.hstack(ls)
    return PCA(
        n_components=min(10, X.shape[0] - 1, X.shape[1]), random_state=SEED
    ).fit_transform(X)


def m_spec_concat(ls, K):
    """Concatenate and embed with a nearest-neighbour spectral embedding.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The spectral embedding coordinates of the concatenated matrix.
    """
    X = np.hstack(ls)
    n = X.shape[0]
    return SpectralEmbedding(
        n_components=min(8, n - 1),
        affinity="nearest_neighbors",
        n_neighbors=max(5, min(15, n // 5)),
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
            PCA(
                n_components=min(5, X.shape[0] - 1, X.shape[1]), random_state=SEED
            ).fit_transform(X)
            for X in ls
        ]
    )


def m_mofa_lite(ls, K):
    """Per-layer PCA reweighted by explained variance and concatenated.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).

    Returns:
        The concatenated, variance-weighted per-layer scores.
    """
    outs, weights = [], []
    for X in ls:
        p = PCA(n_components=min(5, X.shape[0] - 1, X.shape[1]), random_state=SEED).fit(X)
        outs.append(p.transform(X))
        weights.append(max(float(p.explained_variance_ratio_.sum()), 1e-6))
    weights = np.array(weights)
    weights /= weights.sum()
    return np.hstack([o * np.sqrt(w) for o, w in zip(outs, weights)])


def m_consensus(ls, K):
    """Per-layer KMeans, each turning into a K-column cluster-membership block.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters.

    Returns:
        The horizontally concatenated one-hot membership blocks. Membership is
        returned as features rather than as labels so that the downstream
        clustering step does not simply read the answer back out; a layer whose
        KMeans fails contributes an all-zero block to preserve the width.
    """
    outs = []
    for X in ls:
        try:
            lb = KMeans(K, n_init=10, random_state=SEED).fit_predict(X)
            ind = np.zeros((len(lb), K))
            ind[np.arange(len(lb)), lb] = 1.0
            outs.append(ind)
        except Exception:
            outs.append(np.zeros((X.shape[0], K)))
    return np.hstack(outs)


def m_snf(ls, K, t=20):
    """Standard similarity network fusion, returned as a spectral embedding.

    Implements the SNF message passing ``P_i <- W_i x mean_{j != i}(P_j) x W_i^T``
    (Wang et al., Nature Methods 2014) and returns the leading eigenvectors of the
    fused network rather than a hard partition.

    Args:
        ls: List of per-layer blocks.
        K: Number of clusters (unused).
        t: Number of fusion iterations.

    Returns:
        The spectral embedding of the fused network, or the concatenated blocks
        when fusion or eigendecomposition fails.
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
            o = np.zeros_like(P[i])
            for j in range(len(P)):
                if j != i:
                    o = o + P[j]
            o /= max(len(P) - 1, 1)
            newP.append(W[i] @ o @ W[i].T)
        P = [(p + p.T) / 2 for p in newP]
    Sf = np.mean(P, axis=0)
    Sf = (Sf + Sf.T) / 2
    Sf = np.clip(Sf, 0, None)
    d = min(8, n - 2)
    try:
        return SpectralEmbedding(
            n_components=d, affinity="precomputed", random_state=SEED
        ).fit_transform(Sf)
    except Exception:
        return np.hstack(ls)


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


def transfer_ari(Xa, Xb, K, seed=SEED, ref=None):
    """Transfer a clustering fitted on A to B and score it against a fixed reference.

    Args:
        Xa: Feature matrix of the first cohort.
        Xb: Feature matrix of the second cohort, in the same feature space.
        K: Number of clusters.
        seed: Random seed for the KMeans fits.
        ref: Optional method-independent reference partition of B, derived from
            the raw concatenated representation. When omitted, the reference is
            the KMeans clustering of B in the method's own feature space.

    Returns:
        ``(ari, degeneracy, balance_A, balance_B, balance_pred)``. The balance
        values are the size of the largest cluster as a fraction of the samples,
        which diagnoses degenerate solutions.
    """
    kmA = KMeans(K, n_init=20, random_state=seed).fit(Xa)
    pred = ((Xb[:, None, :] - kmA.cluster_centers_[None, :, :]) ** 2).sum(2).argmin(1)
    balA = float(np.bincount(kmA.labels_, minlength=K).max() / len(kmA.labels_))
    balP = float(np.bincount(pred, minlength=K).max() / len(pred))
    if ref is None:
        kmB = KMeans(K, n_init=20, random_state=seed).fit(Xb)
        ref = kmB.labels_
        balB = float(np.bincount(kmB.labels_, minlength=K).max() / len(kmB.labels_))
    else:
        balB = float(np.bincount(ref, minlength=K).max() / len(ref))
    # A record is degenerate when any of the three partitions has one label.
    deg = int(min(len(set(kmA.labels_)), len(set(ref)), len(set(pred))) < 2)
    return float(adjusted_rand_score(ref, pred)), deg, balA, balB, balP


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
    """Run the corrected transfer benchmark and write the benchmark and availability files."""
    ensure_dir(OUT_MATRIX.parent)
    build_csv_cache()
    res, avail = [], []
    t0 = time.time()
    for dname, D in DOMAINS.items():
        cohs, lays = D["cohorts"], D["layers"]
        CACHE = {}

        def g(coh, lay):
            """Memoised layer lookup for the current domain (avoids re-reading CSVs)."""
            if (coh, lay) not in CACHE:
                CACHE[(coh, lay)] = load_layer(coh, lay)
            return CACHE[(coh, lay)]

        print(
            f"\n{'=' * 106}\nDomain {dname}  cohorts={cohs}  layers={lays}\n{'=' * 106}",
            flush=True,
        )
        subsets = [s for r in range(1, len(lays) + 1) for s in itertools.combinations(lays, r)]
        for sub in subsets:
            # Within-cohort sample intersection: a sample must be present in every
            # selected layer to enter the cell, so dropping a layer can only add
            # samples back. This is the marginal-value quantity of the analysis.
            blocks = {}
            for c in cohs:
                Ms = {l: g(c, l) for l in sub}
                if any(v is None for v in Ms.values()):
                    continue
                s = None
                for M in Ms.values():
                    s = set(M.columns) if s is None else (s & set(M.columns))
                s = sorted(s)
                if len(s) < MIN_SAMPLES:
                    continue
                blocks[c] = ({l: M[s] for l, M in Ms.items()}, len(s))
                avail.append({"domain": dname, "subset": "+".join(sub), "cohort": c, "n": len(s)})
            if len(blocks) < 2:
                continue
            for a, b in itertools.combinations([c for c in cohs if c in blocks], 2):
                A_sel, B_sel, ok = [], [], True
                for l in sub:
                    Ma, Mb = blocks[a][0][l], blocks[b][0][l]
                    gk = sorted(set(Ma.index) & set(Mb.index))
                    if len(gk) < MIN_COMMON_GENES:
                        ok = False
                        break
                    Xa = zgene(Ma.loc[gk].values.T.astype(float))
                    Xb = zgene(Mb.loc[gk].values.T.astype(float))
                    ka, kb = topmad_k(Xa, NGENE), topmad_k(Xb, NGENE)
                    k = np.array(sorted(set(ka.tolist()) & set(kb.tolist())))
                    if len(k) < MIN_COMMON_MAD_GENES:
                        k = np.arange(min(NGENE, Xa.shape[1]))
                    A_sel.append(Xa[:, k])
                    B_sel.append(Xb[:, k])
                if not ok or not A_sel:
                    continue
                nA, nB = blocks[a][1], blocks[b][1]
                # Method-independent reference partition, derived once per K from
                # the raw concatenated representation of cohort B.
                REFS = {}
                Bcat = np.hstack(B_sel)
                for K_ in KGRID:
                    try:
                        REFS[K_] = KMeans(K_, n_init=20, random_state=SEED).fit_predict(Bcat)
                    except Exception:
                        REFS[K_] = None
                for mname, fn in METHODS.items():
                    try:
                        Fa = zgene(np.nan_to_num(np.asarray(fn(A_sel, K_PRIMARY), float), nan=0.0))
                        Fb = zgene(np.nan_to_num(np.asarray(fn(B_sel, K_PRIMARY), float), nan=0.0))
                    except Exception as ex:
                        res.append(
                            {
                                "domain": dname,
                                "subset": "+".join(sub),
                                "pair": f"{a}{PAIR_SEPARATOR}{b}",
                                "method": mname,
                                "ari": None,
                                "err": str(ex)[:70],
                                "nA": nA,
                                "nB": nB,
                            }
                        )
                        continue
                    if Fa.shape[1] != Fb.shape[1] or min(Fa.shape[0], Fb.shape[0]) < MIN_SAMPLES:
                        res.append(
                            {
                                "domain": dname,
                                "subset": "+".join(sub),
                                "pair": f"{a}{PAIR_SEPARATOR}{b}",
                                "method": mname,
                                "ari": None,
                                "err": f"dim {Fa.shape}/{Fb.shape}",
                                "nA": nA,
                                "nB": nB,
                            }
                        )
                        continue
                    aris, degs, bals = {}, {}, {}
                    for K in KGRID:
                        try:
                            v, d, ba, bb, bp = transfer_ari(Fa, Fb, K, ref=REFS[K])
                            aris[K] = round(v, 4)
                            degs[K] = d
                            bals[K] = [round(ba, 3), round(bb, 3), round(bp, 3)]
                        except Exception:
                            aris[K] = None
                            degs[K] = None
                            bals[K] = None
                    res.append(
                        {
                            "domain": dname,
                            "subset": "+".join(sub),
                            "pair": f"{a}{PAIR_SEPARATOR}{b}",
                            "method": mname,
                            "ari": aris,
                            "degen": degs,
                            "balance": bals,
                            "nA": nA,
                            "nB": nB,
                            "nfeat": int(Fa.shape[1]),
                        }
                    )
                # Persist after every pair so a long run survives interruption.
                with open(OUT_MATRIX, "w") as handle:
                    json.dump(res, handle, indent=1, ensure_ascii=False)
            with open(OUT_AVAILABILITY, "w") as handle:
                json.dump(avail, handle, indent=1, ensure_ascii=False)
            print(
                f"  subset {'+'.join(sub)} done ({time.time() - t0:.0f}s, "
                f"{len(res)} records so far)",
                flush=True,
            )
    print(f"\nDone: {len(res)} records in {time.time() - t0:.0f}s")
    print("written:", OUT_MATRIX, "|", OUT_AVAILABILITY)


if __name__ == "__main__":
    main()
