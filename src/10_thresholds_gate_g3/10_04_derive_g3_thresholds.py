#!/usr/bin/env python3
"""
Derive the gate-G3 decision thresholds from the measured reproducibility ceiling.

Purpose
    Gate G3 asks whether the copy-number-low (NSMP) subtype reproduces across
    cohorts, and the threshold that decides the gate has to be read off the
    reproducibility the data actually supports rather than assumed. This module
    measures the three quantities that threshold is derived from: the
    within-cohort split-half ARI, which is the ceiling a cohort imposes on its
    own reproducibility; the ARI of two random partitions of the same sample
    size, which is the metric's noise scale under the null that no structure
    exists; and the measured cross-cohort transfer ARI, which is the statistic
    the gate is decided on. It is the fourth step of the stage: the JSON it
    writes is extended by 10_05 and read by the figure and table stages.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("core_nsmp_samples") -- JSON catalogue of the frozen document package;
        its NSMP participant list, under params:nsmp.catalogue_nsmp_key, is the
        TCGA arm of the comparison.
    data("cptac_independent_metadata") and data("cptac_discovery_clinical") --
        the metadata tables from which params:nsmp.cohorts reads the NSMP
        participants of the two CPTAC cohorts.
    work("harmonised_matrix_pattern") -- the harmonised per-cohort per-layer
        matrices; one file per cohort tag and per layer suffix. Samples must be
        in the columns and genes in the rows.

Outputs
    work("g3_power") -- JSON with "split_half" (mean split-half ARI per cohort
        and K), "null" (mean, SD, q95 and q99 of the random-partition ARI per
        sample size and K), "cross" (measured transfer ARI per cohort pair and
        K) and "n" (four-layer sample count of each cohort).

Usage
    python src/10_thresholds_gate_g3/10_04_derive_g3_thresholds.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Allow the module to be run from any directory: put src/ on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, data, param, work  # noqa: E402


# --- Layer and cohort inputs -------------------------------------------------


def layer_file_suffixes() -> dict[str, str]:
    """Return layer name -> harmonised file-name suffix for the layers used here.

    The comparison runs on the four layers that the study integrates across
    cohorts; their file-name suffixes are the values of params:layer_tags, so
    the suffix vocabulary is not repeated in code.

    Returns:
        A dict such as {"mRNA": "mRNA", "methylation": "meth"}, in the order of
        params:integrable_layers.
    """
    layer_tags = param("layer_tags")
    return {name: layer_tags[name] for name in param("integrable_layers")}


def layer_matrix_path(cohort_tag: str, layer_suffix: str) -> Path:
    """Return the harmonised matrix path of one cohort and one layer.

    Args:
        cohort_tag: tag of the cohort, a value of params:cohort_tags.
        layer_suffix: file-name suffix of the layer, a value of params:layer_tags.

    Returns:
        The resolved path of the matrix file.
    """
    pattern = str(work("harmonised_matrix_pattern"))
    return Path(pattern.format(cohort=cohort_tag, layer=layer_suffix))


def load_layer_matrix(cohort_tag: str, layer_suffix: str) -> pd.DataFrame | None:
    """Load one harmonised layer matrix, or None when the file does not exist.

    Args:
        cohort_tag: tag of the cohort.
        layer_suffix: file-name suffix of the layer.

    Returns:
        The matrix with samples in the columns, or None when the cohort has no
        matrix for that layer. Column names are cast to ``str`` because the
        harmonised identifiers are compared as strings.
    """
    path = layer_matrix_path(cohort_tag, layer_suffix)
    if not path.exists():
        return None
    matrix = pd.read_csv(path, index_col=0)
    matrix.columns = [str(column) for column in matrix.columns]
    return matrix


def read_metadata_table(path: Path) -> pd.DataFrame:
    """Read a cohort metadata table.

    Args:
        path: path of the table, as declared in paths.yaml.

    Returns:
        The table. The reader follows the file suffix: the confirmatory cohort
        publishes an Excel workbook, the discovery cohort a tab-separated text
        export whose encoding is declared as UTF-8 with undecodable bytes
        replaced, because the export is not consistently encoded.
    """
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(
        path, sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace"
    )


def nsmp_identifiers_from_metadata(spec: dict) -> set[str]:
    """Return the NSMP participants of a cohort read from its metadata table.

    Args:
        spec: the cohort entry of params:nsmp.cohorts, giving the metadata table,
            the participant column, the subtype column and the NSMP class.

    Returns:
        The set of participant identifiers whose subtype column carries the NSMP
        class of that cohort.
    """
    table = read_metadata_table(data(spec["metadata_file"]))
    selected = table.loc[
        table[spec["subtype_column"]] == spec["nsmp_value"], spec["participant_column"]
    ]
    return set(selected.astype(str))


def cohort_inputs() -> dict[str, tuple[str, set[str], bool]]:
    """Return the three cohorts of the comparison with their NSMP participants.

    The cohorts, their harmonised file-name tag and the source of their NSMP
    annotation are declared once in params:nsmp.cohorts, so the configuration --
    and not this module -- decides which cohort a label refers to.

    Returns:
        label -> (cohort tag, NSMP participant identifiers, match on barcode
        prefix), in the order the labels appear in params:nsmp.cohorts. The flag
        is true for the cohort whose harmonised matrices carry full barcodes
        while its NSMP list carries shorter identifiers.
    """
    catalogue_key = param("nsmp", "catalogue_nsmp_key")
    barcode_prefix = int(param("nsmp", "barcode_prefix_length"))
    with work("core_nsmp_samples").open(encoding="utf-8") as handle:
        catalogue = json.load(handle)

    cohort_tags = param("cohort_tags")
    inputs: dict[str, tuple[str, set[str], bool]] = {}
    for label, spec in param("nsmp", "cohorts").items():
        tag = cohort_tags[spec["cohort_tag"]]
        match_barcode_prefix = bool(spec["match_barcode_prefix"])
        if spec["nsmp_source"] == "catalogue":
            # The frozen catalogue stores full barcodes for some participants and
            # shorter identifiers for others; truncating every entry to the
            # declared prefix length makes the list comparable with the matrix
            # columns, which is also how the downstream matching works.
            identifiers = {str(entry)[:barcode_prefix] for entry in catalogue[catalogue_key]}
        else:
            identifiers = nsmp_identifiers_from_metadata(spec)
        inputs[label] = (tag, identifiers, match_barcode_prefix)
    return inputs


# --- Feature-space construction ---------------------------------------------


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


def build_cohort_blocks(
    cohort_tag: str,
    identifiers: set[str],
    match_barcode_prefix: bool,
    layer_suffixes: dict[str, str],
    min_samples: int,
) -> tuple[list[np.ndarray] | None, list[str] | None]:
    """Build the per-layer standardised blocks of one cohort.

    Only the samples that carry every layer of the comparison are kept, because
    the transfer ARI concatenates the layers and a sample missing one of them
    would have to be imputed.

    Args:
        cohort_tag: tag of the cohort, used to address its harmonised matrices.
        identifiers: NSMP participant identifiers of the cohort.
        match_barcode_prefix: true when identifiers must be matched on the
            declared barcode prefix rather than exactly.
        layer_suffixes: layer name -> file-name suffix.
        min_samples: number of samples below which the cohort is not analysed.

    Returns:
        (blocks, sample identifiers) where blocks holds one standardised
        (samples x genes) array per layer and the identifiers are the four-layer
        intersection, sorted; (None, None) when a layer matrix is missing or the
        intersection is too small.
    """
    barcode_prefix = int(param("nsmp", "barcode_prefix_length"))
    if match_barcode_prefix:
        wanted = {str(identifier)[:barcode_prefix] for identifier in identifiers}
    else:
        wanted = set(identifiers)

    per_layer: list[tuple[str, pd.DataFrame, dict[str, str]]] = []
    # Start from the wanted identifiers and narrow by one layer at a time; every
    # per-layer availability set is already a subset of the wanted identifiers, so
    # this equals intersecting the per-layer sets with one another.
    intersection: set[str] = set(wanted)
    for suffix in layer_suffixes.values():
        matrix = load_layer_matrix(cohort_tag, suffix)
        if matrix is None:
            print(f"    [{cohort_tag}/{suffix}] layer missing")
            return None, None
        # Shortening the keys to the barcode prefix is what allows a cohort whose
        # matrices carry full barcodes to be matched against an NSMP list that
        # carries shorter identifiers. The map is keyed by the shortened name and
        # valued by the exact column label, so a truncated key still selects the
        # right column.
        column_map = {
            (str(column)[:barcode_prefix] if match_barcode_prefix else str(column)): str(column)
            for column in matrix.columns
        }
        available = set(column_map) & wanted
        print(f"    [{cohort_tag}/{suffix}] columns {len(column_map)} | intersection {len(available)}")
        intersection &= available
        per_layer.append((suffix, matrix, column_map))

    keep = sorted(intersection)
    print(f"    [{cohort_tag}] four-layer intersection {len(keep)}")
    if len(keep) < min_samples:
        print(f"    [{cohort_tag}] too few samples -> returning empty")
        return None, None

    blocks = []
    for suffix, matrix, column_map in per_layer:
        # Selecting by label, not by position, keeps the layer blocks aligned on
        # the sample. Two identifiers shortened to the same prefix collapse into
        # one key in column_map, and the label selection then expands the frame
        # to every column carrying that label; the harmonisation stage resolves
        # duplicates upstream, so each label selects exactly one column here and
        # the block keeps len(keep) samples.
        columns = [column_map[key] for key in keep]
        blocks.append(zscore_genes(matrix[columns].T.values.astype(float)))
    return blocks, keep


def sample_count(samples: list[str] | None) -> int:
    """Return the number of four-layer samples, or zero when there is no block set.

    Args:
        samples: the sample identifiers returned with a cohort's blocks.

    Returns:
        The count, zero for a cohort that was not analysable.
    """
    return len(samples) if samples else 0


def select_shared_features(
    blocks_a: list[np.ndarray], blocks_b: list[np.ndarray], top_genes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate two cohorts' blocks on the top-MAD genes they share.

    Args:
        blocks_a: standardised layer blocks of the first cohort.
        blocks_b: standardised layer blocks of the second cohort, same order.
        top_genes: number of genes kept per layer by median absolute deviation.

    Returns:
        (features A, features B), each standardised after concatenation.
    """
    selected_a, selected_b = [], []
    for matrix_a, matrix_b in zip(blocks_a, blocks_b):
        # Genes are matched positionally, which requires both blocks of a layer to
        # list the same genes in the same order -- the common genome space the
        # harmonisation stage establishes. A layer can still carry a different
        # number of genes in the two cohorts, so both blocks are truncated to the
        # shorter width before the index sets are intersected; intersecting
        # indices beyond that width would compare different genes.
        shared_width = min(matrix_a.shape[1], matrix_b.shape[1])
        matrix_a, matrix_b = matrix_a[:, :shared_width], matrix_b[:, :shared_width]
        # Descending MAD order: the negation turns argsort's ascending order into
        # a descending one. The median absolute deviation is the robustness
        # filter of the feature space, taken per layer on the standardised block.
        median_a = np.median(matrix_a, 0)
        median_b = np.median(matrix_b, 0)
        index_a = np.argsort(-np.median(np.abs(matrix_a - median_a), 0))[:top_genes]
        index_b = np.argsort(-np.median(np.abs(matrix_b - median_b), 0))[:top_genes]
        common = np.array(sorted(set(index_a.tolist()) & set(index_b.tolist())))
        selected_a.append(matrix_a[:, common])
        selected_b.append(matrix_b[:, common])
    return zscore_genes(np.hstack(selected_a)), zscore_genes(np.hstack(selected_b))


# --- The three measured quantities ------------------------------------------


def split_half_ceiling(
    blocks: list[np.ndarray],
    sample_count: int,
    rng: np.random.RandomState,
    k_values: list[int],
    replicates: int,
    n_init: int,
) -> dict[str, list[float]]:
    """Measure the within-cohort split-half ARI, the cohort's own ceiling.

    The samples are split at random, each half is clustered on its own and the
    agreement of the two partitions is the ARI a second, independent sample of
    the same cohort would be expected to reach. It is an upper bound for the
    cross-cohort transfer ARI: two different cohorts cannot agree better than
    two halves of one cohort do.

    Args:
        blocks: standardised layer blocks of the cohort.
        sample_count: number of samples available.
        rng: random state, shared with the caller so that the draw order of the
            whole module is fixed.
        k_values: cluster numbers compared.
        replicates: number of random splits.
        n_init: k-means restarts of each half fit.

    Returns:
        K (as a string) -> list of the per-replicate ARIs.
    """
    values: dict[int, list[float]] = {k: [] for k in k_values}
    for replicate in range(replicates):
        order = rng.permutation(sample_count)
        first = order[: len(order) // 2]
        # The two slices are deliberately the same length: the halves are taken
        # from the first 2*floor(n/2) samples so that an odd sample is dropped
        # rather than unbalanced between the halves.
        second = order[len(order) // 2: 2 * (len(order) // 2)]
        features_first = zscore_genes(np.hstack([block[first] for block in blocks]))
        features_second = zscore_genes(np.hstack([block[second] for block in blocks]))
        for k in k_values:
            # The two halves are fitted with different seeds so that they cannot
            # agree merely because they started from the same initialisation;
            # the offset keeps the replicate reproducible from its index.
            partition_first = KMeans(k, n_init=n_init, random_state=replicate).fit_predict(features_first)
            partition_second = KMeans(
                k, n_init=n_init, random_state=replicate + int(param("gates", "g3", "second_fit_seed_offset"))
            ).fit_predict(features_second)
            values[k].append(adjusted_rand_score(partition_first, partition_second))
    return {str(k): value for k, value in values.items()}


def random_partition_null(
    rng: np.random.RandomState,
    sample_sizes: list[int],
    k_values: list[int],
    draws: int,
) -> dict[str, dict[str, float]]:
    """Measure the ARI distribution of two independent random partitions.

    This is the noise scale of the metric: it states how much agreement two
    partitions of n samples into k groups reach when there is no structure at
    all, so that a measured ARI can be read against it.

    Args:
        rng: random state, shared with the caller.
        sample_sizes: sample sizes to measure, typically those of the cohorts.
        k_values: cluster numbers compared.
        draws: number of random partition pairs per (sample size, K).

    Returns:
        "n<size>_K<k>" -> dict with mean, sd, q95 and q99 of the draws. The two
        quantiles are reporting positions of the null, not decision thresholds.
    """
    null: dict[str, dict[str, float]] = {}
    for sample_size in sample_sizes:
        for k in k_values:
            values = []
            for _ in range(draws):
                # The permutation randomises which cluster index labels which
                # group, so indexing it by a vector of random labels gives a
                # uniformly random partition; the two partitions are drawn
                # independently, which is what makes the ARI a null draw.
                labels_first = rng.permutation(k)[rng.randint(0, k, sample_size)]
                labels_second = rng.permutation(k)[rng.randint(0, k, sample_size)]
                values.append(adjusted_rand_score(labels_first, labels_second))
            array = np.array(values)
            null[f"n{sample_size}_K{k}"] = {
                "mean": float(array.mean()),
                "sd": float(array.std()),
                "q95": float(np.percentile(array, 95)),
                "q99": float(np.percentile(array, 99)),
            }
    return null


def cross_cohort_transfer(
    blocks_a: list[np.ndarray],
    blocks_b: list[np.ndarray],
    k_values: list[int],
    top_genes: int,
    n_init: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Measure the cross-cohort transfer ARI, the statistic gate G3 decides on.

    Cohort B is clustered on its own, and separately labelled by the nearest
    cluster centre of a model fitted on cohort A; the ARI between the two
    labellings is the transfer. The nearest-centre labelling is used rather than
    fitting a classifier, so that the transfer carries no extra fitting step.

    Args:
        blocks_a: standardised layer blocks of the first cohort.
        blocks_b: standardised layer blocks of the second cohort.
        k_values: cluster numbers compared.
        top_genes: genes kept per layer before the layers are concatenated.
        n_init: k-means restarts of both fits.

    Returns:
        (features A, features B, K (as a string) -> transfer ARI), rounded to
        four decimals as recorded in the artefact.
    """
    features_a, features_b = select_shared_features(blocks_a, blocks_b, top_genes)
    results: dict[str, float] = {}
    for k in k_values:
        model_a = KMeans(k, n_init=n_init, random_state=SEED).fit(features_a)
        # Squared Euclidean distance of every sample of B to every centre of A,
        # then the nearest centre; computed by broadcasting rather than through
        # the model's own predict, which is the same arithmetic.
        predicted = ((features_b[:, None, :] - model_a.cluster_centers_[None]) ** 2).sum(2).argmin(1)
        own_b = KMeans(k, n_init=n_init, random_state=SEED).fit_predict(features_b)
        results[str(k)] = round(float(adjusted_rand_score(own_b, predicted)), 4)
    return features_a, features_b, results


def main() -> None:
    """Measure the three quantities and write the G3 threshold artefact.

    Returns:
        None. The measurements are written to work("g3_power").
    """
    rng = np.random.RandomState(SEED)
    layer_suffixes = layer_file_suffixes()
    k_values = [int(k) for k in param("gates", "g3", "k_values")]
    top_genes = int(param("gates", "g3", "top_mad_genes"))
    cohorts = cohort_inputs()

    print("NSMP lists: " + " | ".join(
        f"{label} {len(identifiers)}" for label, (_, identifiers, _) in cohorts.items()
    ))

    cohort_blocks: dict[str, tuple[list[np.ndarray] | None, list[str] | None]] = {}
    for label, (tag, identifiers, match_barcode_prefix) in cohorts.items():
        blocks, keep = build_cohort_blocks(
            tag,
            identifiers,
            match_barcode_prefix,
            layer_suffixes,
            int(param("gates", "g3", "min_samples_build")),
        )
        cohort_blocks[label] = (blocks, keep)
        print(f"  {label} (layer files {tag}): four-layer samples {sample_count(keep)}")

    # (1) Within-cohort split-half ARI: the reproducibility ceiling.
    print("\n=== (1) Within-cohort split-half ARI (reproducibility ceiling) ===")
    split_half: dict[str, dict[str, float]] = {}
    for label, (blocks, keep) in cohort_blocks.items():
        if blocks is None or sample_count(keep) < int(param("gates", "g3", "min_samples_split_half")):
            print(f"  {label}: too few samples")
            continue
        values = split_half_ceiling(
            blocks,
            len(keep),
            rng,
            k_values,
            int(param("gates", "g3", "n_split_half")),
            int(param("gates", "g3", "kmeans_n_init_resample")),
        )
        split_half[label] = {k: float(np.mean(value)) for k, value in values.items()}
        print(f"  {label} (n={len(keep)}): " + " | ".join(
            f"K={k} ARI {np.mean(value):.4f}+/-{np.std(value):.4f}" for k, value in values.items()
        ))

    # (2) Random-partition null: the noise scale of the ARI.
    print("\n=== (2) Random-partition null (noise scale of the ARI) ===")
    sample_sizes = sorted(int(size) for size in param("gates", "g3", "cohort_sizes"))
    null = random_partition_null(
        rng, sample_sizes, k_values, int(param("gates", "g3", "n_null_draws"))
    )
    # Only the largest and the smallest cohort size are printed; the artefact
    # carries every size.
    print(f"  (only the key values of n={max(sample_sizes)} and n={min(sample_sizes)} are listed)")
    reported_keys = [f"n{size}_K{k}" for size in (max(sample_sizes), min(sample_sizes)) for k in k_values]
    for key in reported_keys:
        record = null[key]
        print(
            f"  {key}: mean {record['mean']:+.4f} SD {record['sd']:.4f} "
            f"Q95 {record['q95']:+.4f} Q99 {record['q99']:+.4f}"
        )

    # (3) Cross-cohort transfer ARI: the measured value of the G3 statistic.
    # Every cohort pair is measured; the two-cohort pairs are skipped when a
    # cohort has no usable four-layer block at all.
    print("\n=== (3) Cross-cohort transfer ARI (measured G3 statistic) ===")
    pair_separator = str(param("benchmark", "pair_separator"))
    cross: dict[str, dict[str, float]] = {}
    for label_a, label_b in itertools.combinations(list(cohort_blocks), 2):
        blocks_a = cohort_blocks[label_a][0]
        blocks_b = cohort_blocks[label_b][0]
        if blocks_a is None or blocks_b is None:
            continue
        features_a, features_b, results = cross_cohort_transfer(
            blocks_a,
            blocks_b,
            k_values,
            top_genes,
            int(param("gates", "g3", "kmeans_n_init")),
        )
        cross[f"{label_a}{pair_separator}{label_b}"] = results
        print(
            f"  {label_a}{pair_separator}{label_b} "
            f"(n={features_a.shape[0]}->{features_b.shape[0]}, d={features_a.shape[1]}): "
            + " | ".join(f"K={k} {value:+.4f}" for k, value in results.items())
        )

    out = {
        "split_half": split_half,
        "null": null,
        "cross": cross,
        "n": {
            label: sample_count(keep)
            for label, (_, keep) in cohort_blocks.items()
        },
    }
    with work("g3_power").open("w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print(f"\nWrote {work('g3_power')}")


if __name__ == "__main__":
    main()
