"""
Run the purity-aware integration (PAI) arms against the frozen G2 baseline.

Purpose
    Core experiment of the purity gate (G2). The gate asks whether removing the
    purity signal from a representation improves cross-cohort transfer. Every
    purity direction and every per-gene purity correlation is learned on the
    CPTAC UCEC Independent cohort alone; the Discovery, TCGA and ovarian cohorts
    are held-out evaluation cohorts and never contribute to the model. The arms
    are the B0 baseline without any purity handling, the S1-S3 arms that project
    out the first k purity-related principal components, the G20/G30 arms that
    zero every gene whose |r| with purity exceeds gamma, and the R1 arm that
    residualises each gene on purity, which is only defined for the two cohorts
    that publish a per-sample purity.
    The protocol is the frozen benchmark protocol reused unchanged (same gene
    count, same k grid, same nearest-centroid transfer), so that a difference
    between an arm and B0 is attributable to the purity handling and nothing else.
    This module is the slowest of the stage and rewrites its result file after
    every arm, so a run that is interrupted keeps its completed arms.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    data("cptac_independent_meta")     Excel metadata table of the CPTAC UCEC
        Independent cohort; Case_id and ABSOLUTE_tumor_purity are the learning
        cohort's purity source.
    data("cptac_discovery_clinical")   clinical table of the CPTAC UCEC Discovery
        cohort; Proteomics_Participant_ID, Purity_Cancer and
        Proteomics_Tumor_Normal are the held-out cohort's purity source.
    work("harmonised_matrix_pattern")  harmonised per-cohort layer matrices
        (hcsv/{cohort}__{layer}.csv.gz); genes are rows, samples are columns.
    param("layers"), param("cohort_tags"), param("layer_tags")
        the layer vocabulary and the tokens used in those file names.

Outputs
    work("pai_matrix")        one record per (domain, layer subset, cohort pair,
        arm); the ari and degen fields are keyed by the k of param("clustering",
        "k_values"), and the record keys match the benchmark records so that
        09_08 can join the two.
    work("pai_learn_summary") per-layer summary of the learned purity model.

Usage
    python 09_06_run_purity_aware_arms.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

# Insert src/ on sys.path so that `common` is importable whatever the working
# directory is; see docs/coding_standard.md §5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from common.config import SEED, data, ensure_dir, param, work

# --- Data-schema names of the third-party purity sources ---------------------

CASE_ID = "Case_id"
ABSOLUTE_PURITY = "ABSOLUTE_tumor_purity"
PARTICIPANT_ID = "Proteomics_Participant_ID"
PURITY_CANCER = "Purity_Cancer"
TUMOR_NORMAL = "Proteomics_Tumor_Normal"
TUMOR_LABEL = "Tumor"

# --- Frozen settings, all read from config/params.yaml -----------------------

# Layer vocabulary in display order, and the tokens the harmonised file names and
# the benchmark subset labels use. The record keys have to match the benchmark
# records, so the token and not the display name is written to the output.
LAYER_NAMES = list(param("layers"))
LAYER_TAGS = [param("layer_tags", name) for name in LAYER_NAMES]

# Gates of the frozen benchmark protocol, reused unchanged by every arm.
TOP_MAD_GENES = param("features", "top_mad_genes")
K_GRID = list(param("clustering", "k_values"))
KMEANS_N_INIT = param("alignment", "kmeans_n_init")
MIN_COMMON_GENES = param("benchmark", "min_common_genes")
MIN_COMMON_MAD_GENES = param("benchmark", "min_common_mad_genes")
MIN_SAMPLES = param("benchmark", "min_samples")

# Purity-model settings.
GENE_COMPLETENESS = param("purity", "gene_completeness")
ABS_R_GENE = param("purity", "abs_r_gene")
MIN_SAMPLES_LEARNED = param("purity", "min_samples_learned")
MIN_ARM_GENES = param("purity", "min_arm_genes")
N_PC_LEARNING = param("purity", "n_pc_learning")

# Purity estimates are available for exactly these two cohorts; every other
# cohort is evaluated without the R1 arm, which is what the gate is testing.
PURITY_COHORT_KEYS = ("independent", "discovery")


def as_list(value) -> list:
    """Normalise a configuration value that may be a list or a comma string.

    Args:
        value: value from param().

    Returns:
        A list; a string is split on commas. The environment override mechanism of
        common.config replaces a list with a bare string, so an override of
        param("purity", "arms") arrives here as a string.
    """
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def read_clinical_tsv(path: Path) -> pd.DataFrame:
    """Read the tab-separated Discovery clinical table.

    Args:
        path: table to read.

    Returns:
        The table as read, with every column kept.
    """
    # See 09_01 for why both decoding options are needed.
    return pd.read_csv(
        path,
        sep="\t",
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def load_layer(cohort_tag: str, layer_tag: str) -> pd.DataFrame | None:
    """Load one harmonised cohort x layer matrix.

    Args:
        cohort_tag: cohort token as it appears in the file names.
        layer_tag: layer token as it appears in the file names.

    Returns:
        The matrix with genes as rows and sample identifiers as columns, or None
        when the cohort has no matrix for that layer.
    """
    path = Path(
        str(work("harmonised_matrix_pattern")).format(
            cohort=cohort_tag, layer=layer_tag
        )
    )
    if not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix


def zscore_genes(matrix: np.ndarray) -> np.ndarray:
    """Standardise every gene (column) across the samples.

    Args:
        matrix: array of shape (samples, genes).

    Returns:
        The standardised array, every non-finite entry replaced by zero.
    """
    mean = np.nanmean(matrix, 0, keepdims=True)
    std = np.nanstd(matrix, 0, keepdims=True)
    # A constant gene has no spread; leaving its scale at 1 keeps it at zero after
    # centring instead of producing an infinity.
    std = np.where((std < 1e-9) | ~np.isfinite(std), 1.0, std)
    return np.nan_to_num((matrix - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def top_mad_indices(matrix: np.ndarray, count: int) -> np.ndarray:
    """Select the genes with the largest median absolute deviation.

    Args:
        matrix: array of shape (samples, genes), standardised.
        count: number of genes to select.

    Returns:
        Index array of the selected genes; genes whose deviation is exactly zero
        are dropped, because they carry no information and would only add noise
        to the distance computation.
    """
    median = np.median(matrix, 0)
    mad = np.median(np.abs(matrix - median), 0)
    selected = np.argsort(-mad)[:count]
    return selected[mad[selected] > 0]


def transfer_ari(
    features_a: np.ndarray,
    features_b: np.ndarray,
    k: int,
    reference: np.ndarray,
    seed: int = SEED,
) -> tuple[float, int]:
    """Cluster one cohort and transfer the labels to the other, then score them.

    Args:
        features_a: (samples_a, genes) matrix of the cohort the clustering is fit
            on.
        features_b: (samples_b, genes) matrix of the cohort the labels are
            transferred to.
        k: number of clusters.
        reference: cluster labels of cohort B obtained from the unmodified
            representation; the transferred labels are compared with these, which
            is what makes the score a transfer score rather than a
            self-consistency score.
        seed: k-means seed.

    Returns:
        Tuple of the adjusted Rand index and a degeneracy flag that is 1 when
        either the fitted clustering, the reference or the transferred labels
        collapses to a single cluster.
    """
    kmeans_a = KMeans(k, n_init=KMEANS_N_INIT, random_state=seed).fit(features_a)
    # Nearest-centroid transfer: squared Euclidean distance to every centroid of
    # cohort A, then the index of the minimum. Written out explicitly so that the
    # assignment rule is identical to the benchmark protocol.
    prediction = (
        (features_b[:, None, :] - kmeans_a.cluster_centers_[None, :, :]) ** 2
    ).sum(2).argmin(1)
    degenerate = int(
        min(len(set(kmeans_a.labels_)), len(set(reference)), len(set(prediction))) < 2
    )
    return float(adjusted_rand_score(reference, prediction)), degenerate


def load_purity_independent(path: Path) -> pd.Series:
    """Build the sample -> purity series of the independent (learning) cohort.

    Args:
        path: Excel metadata table of the CPTAC UCEC Independent cohort.

    Returns:
        Purity indexed by Case_id.
    """
    meta = pd.read_excel(path)
    frame = pd.DataFrame(
        {
            "id": meta[CASE_ID].astype(str),
            "purity": pd.to_numeric(meta[ABSOLUTE_PURITY], errors="coerce"),
        }
    ).dropna()
    # A purity of zero is a missing estimate in this cohort, not an observation.
    frame = frame[frame["purity"] > 0]
    # One row per case: the metadata table repeats a case across its aliquots.
    return frame.drop_duplicates("id").set_index("id")["purity"]


def load_purity_discovery(path: Path) -> pd.Series:
    """Build the sample -> purity series of the Discovery cohort.

    Args:
        path: tab-separated clinical table of the CPTAC UCEC Discovery cohort.

    Returns:
        Purity indexed by Proteomics_Participant_ID.
    """
    clinical = read_clinical_tsv(path)
    frame = pd.DataFrame(
        {
            "id": clinical[PARTICIPANT_ID].astype(str),
            "purity": pd.to_numeric(clinical[PURITY_CANCER], errors="coerce"),
            "tissue": clinical[TUMOR_NORMAL].astype(str),
        }
    )
    # Only the tumour aliquots carry a purity estimate.
    frame = frame[(frame["tissue"] == TUMOR_LABEL) & frame["purity"].notna()]
    return frame.drop_duplicates("id").set_index("id")["purity"]


def learn_purity_model(purity: pd.Series) -> dict:
    """Learn the purity directions and per-gene purity correlations of every layer.

    Args:
        purity: per-sample purity of the learning cohort.

    Returns:
        Mapping layer token -> model with keys:
            genes        gene identifiers in the order of the learned vectors,
            r            per-gene Pearson correlation with purity,
            dirs         gene-space unit vectors, ordered by decreasing |r| of the
                         principal component they come from,
            slope        per-gene ordinary-least-squares slope on purity, used by
                         the R1 residualisation arm,
            pc_r_sorted  |r| of the purity-related principal components.

    Only the learning cohort is passed in; nothing about the evaluation cohorts
    enters this function, which is what keeps the gate non-circular.
    """
    learned: dict = {}
    learning_tag = param("cohort_tags", PURITY_COHORT_KEYS[0])
    for layer in LAYER_TAGS:
        matrix = load_layer(learning_tag, layer)
        if matrix is None:
            continue
        samples = [column for column in matrix.columns if column in purity.index]
        if len(samples) < MIN_SAMPLES_LEARNED:
            continue
        # Samples as rows, genes as columns.
        values = matrix[samples].T.astype(float)
        values = values.loc[:, values.notna().mean() > GENE_COMPLETENESS]
        values = values.fillna(values.median())
        standardised = zscore_genes(values.values)
        target = purity.reindex([str(index) for index in values.index]).values.astype(float)

        # Vectorised Pearson correlation, gene by gene. The explicit form is used
        # instead of scipy.stats.pearsonr because it computes all genes in one
        # pass; the result is the same.
        centred = standardised - standardised.mean(0, keepdims=True)
        target_centred = target - target.mean()
        denominator = np.sqrt((centred**2).sum(0)) * np.sqrt((target_centred**2).sum())
        denominator = np.where(denominator < 1e-12, 1.0, denominator)
        correlations = np.nan_to_num(
            (centred * target_centred[:, None]).sum(0) / denominator
        )

        # Per-gene OLS slope on purity, used by the residualisation arm.
        target_centred = target - target.mean()
        slope = (standardised * target_centred[:, None]).sum(0) / max(
            (target_centred**2).sum(), 1e-9
        )

        # Purity directions come from the principal components of the learning
        # matrix that correlate with purity, not from the genes themselves: a
        # single purity variable spans one direction in gene space, so deflating
        # it directly would leave a zero covariance and give no removable
        # subspace of dimension k. Taking the purity-correlated principal
        # components instead yields a genuine k-dimensional subspace.
        n_components = min(
            N_PC_LEARNING, standardised.shape[0] - 1, standardised.shape[1]
        )
        pca = PCA(n_components=n_components, random_state=SEED).fit(standardised)
        scores = pca.transform(standardised)
        pc_correlations = np.array(
            [
                abs(np.corrcoef(scores[:, i], target)[0, 1])
                if np.std(scores[:, i]) > 0
                else 0.0
                for i in range(n_components)
            ]
        )
        pc_correlations = np.nan_to_num(pc_correlations)
        # Principal components ordered by decreasing |r| with purity; each
        # component is a unit vector in gene space.
        order = np.argsort(-pc_correlations)
        directions = [pca.components_[i] for i in order]

        learned[layer] = {
            "genes": list(values.columns),
            "r": correlations,
            "dirs": directions,
            "slope": slope,
            "pc_r_sorted": pc_correlations[order].round(4).tolist(),
        }
        print(
            f"  {layer:8s} genes {len(values.columns):6d} | "
            f"share with |r|>{ABS_R_GENE:g}: {np.mean(np.abs(correlations) > ABS_R_GENE) * 100:5.1f}% | "
            f"top 3 |r| of the purity-related PCs = "
            f"{pc_correlations[order][:3].round(3).tolist()}"
        )
    return learned


def align_to_learned_genes(model: dict, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Align a learned model to the genes a cohort pair actually shares.

    Args:
        model: one entry of the learned purity model.
        genes: gene identifiers of the pair's selected feature space.

    Returns:
        Tuple (positions, keep) where `positions` index the learned vectors and
        `keep` the columns of the feature matrix that are found in the model.
        Only the genes present on both sides can be corrected; every other gene
        has no learned purity statistic and must be left untouched.
    """
    index = {gene: i for i, gene in enumerate(model["genes"])}
    positions = np.array([index[gene] for gene in genes if gene in index])
    keep = np.array([i for i, gene in enumerate(genes) if gene in index])
    return positions, keep


def apply_arm(
    matrix: np.ndarray,
    genes: list[str],
    layer: str,
    arm: str,
    purity: np.ndarray | None = None,
) -> tuple[np.ndarray | None, int]:
    """Apply one purity-aware arm to a cohort's selected feature matrix.

    Args:
        matrix: (samples, genes) feature matrix, already standardised.
        genes: gene identifiers of the columns of `matrix`.
        layer: layer token the matrix comes from.
        arm: arm label; S<k> removes k purity principal components, G<gamma>
            keeps genes with |r| below gamma percent, R1 residualises on purity
            and B0 leaves the matrix untouched.
        purity: per-sample purity of this cohort, or None when the cohort has no
            purity estimate, in which case the R1 arm is undefined.

    Returns:
        Tuple (transformed matrix, number of genes the arm reports as
        purity-independent). Returns (None, 0) when the arm cannot be applied to
        this cohort, which marks the record as inapplicable rather than dropping
        it silently.
    """
    model = LEARNED.get(layer)
    if model is None:
        return matrix, 0
    positions, keep = align_to_learned_genes(model, genes)
    if len(keep) < MIN_ARM_GENES:
        return matrix, 0

    if arm.startswith("S"):
        # Project the feature matrix onto the complement of the first k
        # purity-related principal components.
        components = int(arm[1:])
        transformed = matrix.copy()
        for direction in model["dirs"][:components]:
            restricted = direction[positions]
            norm = np.linalg.norm(restricted)
            if norm < 1e-9:
                continue
            unit = restricted / norm
            transformed[:, keep] = transformed[:, keep] - (
                transformed[:, keep] @ unit
            )[:, None] * unit[None, :]
        return zscore_genes(transformed), len(keep)

    if arm.startswith("G"):
        # Keep every gene whose |r| with purity is below gamma and zero the
        # others, which removes them from the Euclidean distance without changing
        # the width of the matrix.
        gamma = float(arm[1:]) / 100.0
        independent = np.abs(model["r"][positions]) < gamma
        transformed = matrix.copy()
        transformed[:, keep[~independent]] = 0.0
        return zscore_genes(transformed), int(independent.sum())

    if arm == "R1":
        if purity is None:
            # The cohort publishes no per-sample purity, so the arm is undefined
            # rather than ineffective.
            return None, 0
        slope = model["slope"][positions]
        centred_purity = purity - np.nanmean(purity)
        transformed = matrix.copy()
        transformed[:, keep] = transformed[:, keep] - np.outer(centred_purity, slope)
        return zscore_genes(transformed), len(keep)

    # B0 and any unrecognised label leave the matrix untouched, which is the
    # baseline of the comparison.
    return matrix, 0


def build_domains() -> dict:
    """Build the evaluation domains from the configured cohort and layer sets.

    Returns:
        Mapping domain label -> {"cohorts": cohort tokens, "layers": layer
        tokens}, in the configured order. The label is param("domains", key), so
        it is the same domain vocabulary the benchmark records use.
    """
    domains = {}
    for key in param("purity", "domain_cohorts"):
        cohort_keys = list(param("purity", "domain_cohorts", key))
        layer_keys = list(param("purity", "domain_layers", key))
        domains[param("domains", key)] = {
            "cohorts": [param("cohort_tags", name) for name in cohort_keys],
            "layers": [param("layer_tags", name) for name in layer_keys],
        }
    return domains


def collect_purity_by_cohort() -> dict:
    """Read the per-sample purity of every cohort that publishes one.

    Returns:
        Mapping cohort token -> purity series, holding only the two cohorts for
        which an estimate exists.
    """
    sources = {
        "independent": load_purity_independent(data("cptac_independent_meta")),
        "discovery": load_purity_discovery(data("cptac_discovery_clinical")),
    }
    return {
        param("cohort_tags", key): series
        for key, series in sources.items()
        if key in PURITY_COHORT_KEYS
    }


def run() -> None:
    """Evaluate every purity-aware arm over both domains. Returns None."""
    ensure_dir(work("pai_dir"))
    arms = as_list(param("purity", "arms"))
    output_path = work("pai_matrix")

    print("Learning the purity directions and per-gene correlations on the independent cohort ...")
    learning_purity = load_purity_independent(data("cptac_independent_meta"))
    print(f"  learning samples {len(learning_purity)} | mean purity {learning_purity.mean():.4f}")
    learn_purity_model(learning_purity)

    with open(work("pai_learn_summary"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                layer: {
                    "n_gene": len(model["genes"]),
                    "frac_r_gt_0.3": float(np.mean(np.abs(model["r"]) > ABS_R_GENE)),
                    "n_dir": len(model["dirs"]),
                }
                for layer, model in LEARNED.items()
            },
            handle,
            ensure_ascii=False,
            indent=1,
        )
    print(f"wrote {work('pai_learn_summary')}")

    purity_by_cohort = collect_purity_by_cohort()
    print(f"per-sample purity available for: {[(k, len(v)) for k, v in purity_by_cohort.items()]}")

    # Matrices are cached because the same cohort x layer pair is revisited for
    # every layer subset that contains it.
    cache: dict = {}

    def cached_layer(cohort_tag: str, layer_tag: str) -> pd.DataFrame | None:
        """Return a harmonised matrix, reading it at most once per run.

        Args:
            cohort_tag: cohort token.
            layer_tag: layer token.

        Returns:
            The matrix, or None when the cohort has no matrix for that layer.
        """
        key = (cohort_tag, layer_tag)
        if key not in cache:
            cache[key] = load_layer(cohort_tag, layer_tag)
        return cache[key]

    records: list[dict] = []
    start = time.time()
    for domain_label, domain in build_domains().items():
        cohorts = domain["cohorts"]
        layers = domain["layers"]
        print(f"\n{'=' * 100}\ndomain {domain_label}\n{'=' * 100}", flush=True)
        # Every non-empty layer subset, shortest first, exactly as the benchmark
        # enumerates them.
        subsets = [
            subset
            for size in range(1, len(layers) + 1)
            for subset in itertools.combinations(layers, size)
        ]
        for subset in subsets:
            blocks = {}
            for cohort in cohorts:
                matrices = {layer: cached_layer(cohort, layer) for layer in subset}
                if any(matrix is None for matrix in matrices.values()):
                    continue
                shared = None
                for matrix in matrices.values():
                    shared = (
                        set(matrix.columns)
                        if shared is None
                        else (shared & set(matrix.columns))
                    )
                shared = sorted(shared)
                # A cohort pair needs enough shared samples to cluster at all; the
                # threshold is the benchmark's.
                if len(shared) < MIN_SAMPLES:
                    continue
                blocks[cohort] = (
                    {layer: matrix[shared] for layer, matrix in matrices.items()},
                    len(shared),
                )
            if len(blocks) < 2:
                continue

            for cohort_a, cohort_b in itertools.combinations(
                [cohort for cohort in cohorts if cohort in blocks], 2
            ):
                raw_a, raw_b, usable = [], [], True
                for layer in subset:
                    matrix_a = blocks[cohort_a][0][layer]
                    matrix_b = blocks[cohort_b][0][layer]
                    shared_genes = sorted(set(matrix_a.index) & set(matrix_b.index))
                    if len(shared_genes) < MIN_COMMON_GENES:
                        usable = False
                        break
                    features_a = zscore_genes(
                        matrix_a.loc[shared_genes].values.T.astype(float)
                    )
                    features_b = zscore_genes(
                        matrix_b.loc[shared_genes].values.T.astype(float)
                    )
                    selected_a = top_mad_indices(features_a, TOP_MAD_GENES)
                    selected_b = top_mad_indices(features_b, TOP_MAD_GENES)
                    selected = np.array(
                        sorted(set(selected_a.tolist()) & set(selected_b.tolist()))
                    )
                    # Fall back to the leading genes of the layer when the two
                    # cohorts' high-variance sets barely overlap; this is the
                    # benchmark's rule and must not be changed here.
                    if len(selected) < MIN_COMMON_MAD_GENES:
                        selected = np.arange(min(TOP_MAD_GENES, features_a.shape[1]))
                    # Only the shared top-MAD subspace is used, so the two cohorts
                    # are compared on identical genes.
                    raw_a.append(
                        (features_a[:, selected], [shared_genes[i] for i in selected], layer)
                    )
                    raw_b.append(
                        (features_b[:, selected], [shared_genes[i] for i in selected], layer)
                    )
                if not usable or not raw_a:
                    continue

                # The reference labels come from cohort B's unmodified
                # concatenation, so every arm is scored against the same partition
                # and a change in ARI is a change of the transferred labels only.
                reference_space = np.hstack([entry[0] for entry in raw_b])
                references = {
                    k: KMeans(k, n_init=KMEANS_N_INIT, random_state=SEED).fit_predict(
                        reference_space
                    )
                    for k in K_GRID
                }

                for arm in arms:
                    features_arm_a, features_arm_b, kept, skip = [], [], [], False
                    for (matrix_a, genes, layer), (matrix_b, _, _) in zip(raw_a, raw_b):
                        purity_a = (
                            None
                            if cohort_a not in purity_by_cohort
                            else purity_by_cohort[cohort_a]
                            .reindex([str(i) for i in blocks[cohort_a][0][layer].columns])
                            .values
                        )
                        purity_b = (
                            None
                            if cohort_b not in purity_by_cohort
                            else purity_by_cohort[cohort_b]
                            .reindex([str(i) for i in blocks[cohort_b][0][layer].columns])
                            .values
                        )
                        transformed_a, kept_a = apply_arm(matrix_a, genes, layer, arm, purity_a)
                        transformed_b, _ = apply_arm(matrix_b, genes, layer, arm, purity_b)
                        if transformed_a is None or transformed_b is None:
                            skip = True
                            break
                        features_arm_a.append(transformed_a)
                        features_arm_b.append(transformed_b)
                        kept.append(kept_a)
                    if skip:
                        records.append(
                            {
                                "domain": domain_label,
                                "subset": "+".join(subset),
                                "pair": f"{cohort_a}\u00d7{cohort_b}",
                                "arm": arm,
                                "ari": None,
                                "err": "cohort has no purity data (R1 not applicable)",
                                "nA": blocks[cohort_a][1],
                                "nB": blocks[cohort_b][1],
                                "nfeat": None,
                                "n_purity_indep": None,
                            }
                        )
                        with open(output_path, "w", encoding="utf-8") as handle:
                            json.dump(records, handle, indent=1, ensure_ascii=False)
                        continue

                    final_a = zscore_genes(np.hstack(features_arm_a))
                    final_b = zscore_genes(np.hstack(features_arm_b))
                    aris, degeneracies = {}, {}
                    for k in K_GRID:
                        value, degeneracy = transfer_ari(final_a, final_b, k, references[k])
                        aris[k] = round(value, 4)
                        degeneracies[k] = degeneracy
                    records.append(
                        {
                            "domain": domain_label,
                            "subset": "+".join(subset),
                            "pair": f"{cohort_a}\u00d7{cohort_b}",
                            "arm": arm,
                            "ari": aris,
                            "degen": degeneracies,
                            "nA": blocks[cohort_a][1],
                            "nB": blocks[cohort_b][1],
                            "nfeat": int(final_a.shape[1]),
                            "n_purity_indep": int(np.sum(kept)),
                        }
                    )
                    # Rewritten after every arm so that an interrupted run keeps
                    # its completed arms.
                    with open(output_path, "w", encoding="utf-8") as handle:
                        json.dump(records, handle, indent=1, ensure_ascii=False)
            print(
                f"  subset {'+'.join(subset)} done "
                f"({time.time() - start:.0f}s, {len(records)} records so far)",
                flush=True,
            )
    print(f"\nfinished {len(records)} records in {time.time() - start:.0f}s")
    print(f"wrote {output_path}")


# The learned model is filled by run() and read by apply_arm(); it is module level
# because every arm of every cohort pair is expressed in terms of it.
LEARNED: dict = {}


def _learn_purity_model(purity: pd.Series) -> dict:
    """Learn the model and publish it in the module-level LEARNED mapping.

    Args:
        purity: per-sample purity of the learning cohort.

    Returns:
        The learned model, identical to LEARNED.
    """
    global LEARNED
    LEARNED = learn_purity_model(purity)
    return LEARNED


def main() -> None:
    """Run the purity-aware arms and write the result matrix. Returns None."""
    ensure_dir(work("pai_dir"))
    print("Learning the purity directions and per-gene correlations on the independent cohort ...")
    learning_purity = load_purity_independent(data("cptac_independent_meta"))
    print(f"  learning samples {len(learning_purity)} | mean purity {learning_purity.mean():.4f}")
    _learn_purity_model(learning_purity)

    with open(work("pai_learn_summary"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                layer: {
                    "n_gene": len(model["genes"]),
                    "frac_r_gt_0.3": float(np.mean(np.abs(model["r"]) > ABS_R_GENE)),
                    "n_dir": len(model["dirs"]),
                }
                for layer, model in LEARNED.items()
            },
            handle,
            ensure_ascii=False,
            indent=1,
        )
    print(f"wrote {work('pai_learn_summary')}")

    purity_by_cohort = collect_purity_by_cohort()
    print(f"per-sample purity available for: {[(k, len(v)) for k, v in purity_by_cohort.items()]}")

    cache: dict = {}

    def cached_layer(cohort_tag: str, layer_tag: str) -> pd.DataFrame | None:
        """Return a harmonised matrix, reading it at most once per run.

        Args:
            cohort_tag: cohort token.
            layer_tag: layer token.

        Returns:
            The matrix, or None when the cohort has no matrix for that layer.
        """
        key = (cohort_tag, layer_tag)
        if key not in cache:
            cache[key] = load_layer(cohort_tag, layer_tag)
        return cache[key]

    records: list[dict] = []
    start = time.time()
    for domain_label, domain in build_domains().items():
        cohorts = domain["cohorts"]
        layers = domain["layers"]
        print(f"\n{'=' * 100}\ndomain {domain_label}\n{'=' * 100}", flush=True)
        # Every non-empty layer subset, shortest first, exactly as the benchmark
        # enumerates them.
        subsets = [
            subset
            for size in range(1, len(layers) + 1)
            for subset in itertools.combinations(layers, size)
        ]
        for subset in subsets:
            blocks = {}
            for cohort in cohorts:
                matrices = {layer: cached_layer(cohort, layer) for layer in subset}
                if any(matrix is None for matrix in matrices.values()):
                    continue
                shared_samples = None
                for matrix in matrices.values():
                    shared_samples = (
                        set(matrix.columns)
                        if shared_samples is None
                        else (shared_samples & set(matrix.columns))
                    )
                shared_samples = sorted(shared_samples)
                # A cohort pair needs enough shared samples to cluster at all; the
                # threshold is the benchmark's.
                if len(shared_samples) < MIN_SAMPLES:
                    continue
                blocks[cohort] = (
                    {
                        layer: matrix[shared_samples]
                        for layer, matrix in matrices.items()
                    },
                    len(shared_samples),
                )
            if len(blocks) < 2:
                continue

            for cohort_a, cohort_b in itertools.combinations(
                [cohort for cohort in cohorts if cohort in blocks], 2
            ):
                raw_a, raw_b, usable = [], [], True
                for layer in subset:
                    matrix_a = blocks[cohort_a][0][layer]
                    matrix_b = blocks[cohort_b][0][layer]
                    shared_genes = sorted(set(matrix_a.index) & set(matrix_b.index))
                    if len(shared_genes) < MIN_COMMON_GENES:
                        usable = False
                        break
                    features_a = zscore_genes(
                        matrix_a.loc[shared_genes].values.T.astype(float)
                    )
                    features_b = zscore_genes(
                        matrix_b.loc[shared_genes].values.T.astype(float)
                    )
                    selected_a = top_mad_indices(features_a, TOP_MAD_GENES)
                    selected_b = top_mad_indices(features_b, TOP_MAD_GENES)
                    selected = np.array(
                        sorted(set(selected_a.tolist()) & set(selected_b.tolist()))
                    )
                    # Fall back to the leading genes of the layer when the two
                    # cohorts' high-variance sets barely overlap; this is the
                    # benchmark's rule and must not be changed here.
                    if len(selected) < MIN_COMMON_MAD_GENES:
                        selected = np.arange(min(TOP_MAD_GENES, features_a.shape[1]))
                    # Only the shared top-MAD subspace is used, so both cohorts are
                    # compared on exactly the same genes.
                    raw_a.append(
                        (
                            features_a[:, selected],
                            [shared_genes[i] for i in selected],
                            layer,
                        )
                    )
                    raw_b.append(
                        (
                            features_b[:, selected],
                            [shared_genes[i] for i in selected],
                            layer,
                        )
                    )
                if not usable or not raw_a:
                    continue

                # The reference labels come from cohort B's unmodified
                # concatenation, so every arm is scored against the same partition
                # and a change in ARI is a change of the transferred labels only.
                reference_space = np.hstack([entry[0] for entry in raw_b])
                references = {
                    k: KMeans(
                        k, n_init=KMEANS_N_INIT, random_state=SEED
                    ).fit_predict(reference_space)
                    for k in K_GRID
                }

                for arm in arms:
                    features_arm_a, features_arm_b, kept, skip = [], [], [], False
                    for (matrix_a, genes, layer), (matrix_b, _, _) in zip(raw_a, raw_b):
                        purity_a = (
                            None
                            if cohort_a not in purity_by_cohort
                            else purity_by_cohort[cohort_a]
                            .reindex(
                                [str(i) for i in blocks[cohort_a][0][layer].columns]
                            )
                            .values
                        )
                        purity_b = (
                            None
                            if cohort_b not in purity_by_cohort
                            else purity_by_cohort[cohort_b]
                            .reindex(
                                [str(i) for i in blocks[cohort_b][0][layer].columns]
                            )
                            .values
                        )
                        transformed_a, kept_a = apply_arm(
                            matrix_a, genes, layer, arm, purity_a
                        )
                        transformed_b, _ = apply_arm(
                            matrix_b, genes, layer, arm, purity_b
                        )
                        if transformed_a is None or transformed_b is None:
                            skip = True
                            break
                        features_arm_a.append(transformed_a)
                        features_arm_b.append(transformed_b)
                        kept.append(kept_a)
                    if skip:
                        records.append(
                            {
                                "domain": domain_label,
                                "subset": "+".join(subset),
                                "pair": f"{cohort_a}\u00d7{cohort_b}",
                                "arm": arm,
                                "ari": None,
                                "err": "cohort has no purity data (R1 not applicable)",
                                "nA": blocks[cohort_a][1],
                                "nB": blocks[cohort_b][1],
                                "nfeat": None,
                                "n_purity_indep": None,
                            }
                        )
                        with open(work("pai_matrix"), "w", encoding="utf-8") as handle:
                            json.dump(records, handle, indent=1, ensure_ascii=False)
                        continue

                    final_a = zscore_genes(np.hstack(features_arm_a))
                    final_b = zscore_genes(np.hstack(features_arm_b))
                    aris, degeneracies = {}, {}
                    for k in K_GRID:
                        value, degeneracy = transfer_ari(
                            final_a, final_b, k, references[k]
                        )
                        aris[k] = round(value, 4)
                        degeneracies[k] = degeneracy
                    records.append(
                        {
                            "domain": domain_label,
                            "subset": "+".join(subset),
                            "pair": f"{cohort_a}\u00d7{cohort_b}",
                            "arm": arm,
                            "ari": aris,
                            "degen": degeneracies,
                            "nA": blocks[cohort_a][1],
                            "nB": blocks[cohort_b][1],
                            "nfeat": int(final_a.shape[1]),
                            "n_purity_indep": int(np.sum(kept)),
                        }
                    )
                    # Rewritten after every arm so that an interrupted run keeps
                    # its completed arms.
                    with open(work("pai_matrix"), "w", encoding="utf-8") as handle:
                        json.dump(records, handle, indent=1, ensure_ascii=False)
            print(
                f"  subset {'+'.join(subset)} done "
                f"({time.time() - start:.0f}s, {len(records)} records so far)",
                flush=True,
            )
    print(f"\nfinished {len(records)} records in {time.time() - start:.0f}s")
    print(f"wrote {work('pai_matrix')}")


if __name__ == "__main__":
    main()
