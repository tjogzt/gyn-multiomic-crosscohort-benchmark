#!/usr/bin/env python3
"""
Measure the whole-sample bootstrap ceiling and the cross-cohort degeneracy of gate G3.

Purpose
    Two additional criteria are needed before gate G3 can be frozen, and both of
    them guard against a reading of the transfer ARI that the metric alone cannot
    exclude. The first is the bootstrap stability of a clustering of the whole
    cohort: the split-half ceiling of 10_04 is measured on half the samples, so a
    whole-sample fit resampled with replacement is the correct ceiling for the
    full-sample transfer the gate reports. The second is a degeneracy diagnostic
    of the cross-cohort runs: the ARI compares two partitions, and a partition
    that puts almost every sample in one cluster agrees with any other partition
    of the same shape. A transfer ARI of 1.0 therefore is not by itself evidence
    of shared structure -- if both partitions are degenerate, the agreement is a
    property of the two clusterings and not of the cohorts. This module measures
    the largest-cluster fraction of the reference partition, of the transferred
    labelling and of the model fitted on the first cohort, and flags the run as
    degenerate when any of the three exceeds the configured fraction. It extends
    the artefact written by 10_04 and is the fifth step of the stage.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work("g3_power") -- JSON written by 10_04; read in full and written back with
        the two new sections added, so the earlier measurements are preserved.
    work("core_nsmp_samples") -- JSON catalogue of the frozen document package;
        its NSMP participant list, under params:nsmp.catalogue_nsmp_key, is the
        TCGA arm of the comparison.
    data("cptac_independent_metadata") and data("cptac_discovery_clinical") --
        the metadata tables from which params:nsmp.cohorts reads the NSMP
        participants of the two CPTAC cohorts.
    work("harmonised_matrix_pattern") -- the harmonised per-cohort per-layer
        matrices, one file per cohort tag and layer suffix. The four layers must
        be present: this module rebuilds the blocks without the missing-layer
        guard of 10_04, because a cohort cannot be resampled without them.

Outputs
    work("g3_power") -- the same JSON, extended with "bootstrap_ceiling"
        (bootstrap ARI, its SD and the full-sample largest-cluster fraction per
        cohort and K) and "cross_degeneracy" (transfer ARI and the largest-cluster
        fractions of the three labellings, plus the degeneracy flag, per cohort
        pair and K).

Usage
    python src/10_thresholds_gate_g3/10_05_measure_reproducibility_ceiling.py
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


def load_layer_matrix(cohort_tag: str, layer_suffix: str) -> pd.DataFrame:
    """Load one harmonised layer matrix.

    Args:
        cohort_tag: tag of the cohort.
        layer_suffix: file-name suffix of the layer.

    Returns:
        The matrix with samples in the columns and genes in the rows. A file that
        does not exist is not caught here: 10_04 has already established which
        layers each cohort has, and a cohort missing one of the four layers
        cannot be resampled at all, so the read is allowed to fail.
    """
    matrix = pd.read_csv(layer_matrix_path(cohort_tag, layer_suffix), index_col=0)
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


def cohort_inputs() -> dict[str, tuple[str, set[str], bool]]:
    """Return the three cohorts of the comparison with their NSMP participants.

    The cohorts, their harmonised file-name tag and the source of their NSMP
    annotation are declared once in params:nsmp.cohorts, so the configuration --
    and not this module -- decides which cohort a label refers to. The blocks are
    rebuilt here rather than read back from 10_04, so that this module is
    self-contained and a rerun does not depend on an earlier save.

    Returns:
        label -> (cohort tag, NSMP participant identifiers, match on barcode
        prefix), in the order the labels appear in params:nsmp.cohorts.
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
            table = read_metadata_table(data(spec["metadata_file"]))
            selected = table.loc[
                table[spec["subtype_column"]] == spec["nsmp_value"], spec["participant_column"]
            ]
            identifiers = set(selected.astype(str))
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
    cohort_tag: str, identifiers: set[str], match_barcode_prefix: bool, layer_suffixes: dict[str, str]
) -> tuple[list[np.ndarray], list[str]]:
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

    Returns:
        (blocks, sample identifiers), where blocks holds one standardised
        (samples x genes) array per layer and the identifiers are the four-layer
        intersection, sorted.
    """
    barcode_prefix = int(param("nsmp", "barcode_prefix_length"))
    if match_barcode_prefix:
        wanted = {str(identifier)[:barcode_prefix] for identifier in identifiers}
    else:
        wanted = set(identifiers)

    per_layer: list[tuple[pd.DataFrame, dict[str, str]]] = []
    # Start from the wanted identifiers and narrow by one layer at a time; every
    # per-layer availability set is already a subset of the wanted identifiers, so
    # this equals intersecting the per-layer sets with one another.
    intersection: set[str] = set(wanted)
    for suffix in layer_suffixes.values():
        matrix = load_layer_matrix(cohort_tag, suffix)
        # Shortening the keys to the barcode prefix is what allows a cohort whose
        # matrices carry full barcodes to be matched against an NSMP list that
        # carries shorter identifiers. The map is keyed by the shortened name and
        # valued by the exact column label.
        column_map = {
            (str(column)[:barcode_prefix] if match_barcode_prefix else str(column)): str(column)
            for column in matrix.columns
        }
        intersection &= set(column_map) & wanted
        per_layer.append((matrix, column_map))

    keep = sorted(intersection)
    blocks = []
    for matrix, column_map in per_layer:
        # Selecting by label, not by position, keeps the layer blocks aligned on
        # the sample. Two identifiers shortened to the same prefix collapse into
        # one key in column_map, and the label selection then expands the frame
        # to every column carrying that label; the harmonisation stage resolves
        # duplicates upstream, so each label selects exactly one column here and
        # the block keeps len(keep) samples.
        columns = [column_map[key] for key in keep]
        blocks.append(zscore_genes(matrix[columns].T.values.astype(float)))
    return blocks, keep


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


# --- The two supplementary criteria -----------------------------------------


def bootstrap_ceiling(
    label: str,
    blocks: list[np.ndarray],
    rng: np.random.RandomState,
    k_values: list[int],
    n_bootstrap: int,
    n_init: int,
    n_init_resample: int,
    unique_margin: int,
) -> dict[str, dict[str, float]]:
    """Measure the bootstrap stability of a whole-cohort clustering.

    The cohort is clustered as a whole and the same clustering is repeated on
    samples drawn with replacement; the ARI between the labels of the drawn
    samples in the full-sample partition and the labels of the resample is the
    stability. This is the ceiling that applies to the transfer ARI the gate
    reports, which is measured on all samples rather than on a split half.

    Args:
        label: cohort label, used in the console output.
        blocks: standardised layer blocks of the cohort.
        rng: random state, shared with the caller so that the draw order of the
            whole module is fixed.
        k_values: cluster numbers compared.
        n_bootstrap: number of resamples.
        n_init: k-means restarts of the full-sample fit.
        n_init_resample: k-means restarts of a resample fit.
        unique_margin: a resample is used only when it retains at least
            k + this many distinct samples.

    Returns:
        K (as a string) -> dict with the mean bootstrap ARI, its SD and the
        largest-cluster fraction of the full-sample partition, the last rounded
        to three decimals as recorded in the artefact.
    """
    features = zscore_genes(np.hstack(blocks))
    sample_size = features.shape[0]
    row: dict[str, dict[str, float]] = {}
    for k in k_values:
        base = KMeans(k, n_init=n_init, random_state=SEED).fit_predict(features)
        # Largest-cluster fraction of the full-sample partition: the same
        # degeneracy diagnostic that is applied to the cross-cohort runs below,
        # reported for the ceiling so that a stable but degenerate partition is
        # not read as a high ceiling.
        max_fraction = float(np.bincount(base, minlength=k).max() / len(base))
        values = []
        for replicate in range(n_bootstrap):
            index = rng.randint(0, sample_size, sample_size)
            # A resample with too few distinct samples cannot fill k clusters, and
            # the replicates it would contribute are biased; they are dropped.
            if len(set(index.tolist())) < k + unique_margin:
                continue
            partition = KMeans(k, n_init=n_init_resample, random_state=replicate).fit_predict(
                features[index]
            )
            values.append(adjusted_rand_score(base[index], partition))
        row[str(k)] = {
            "boot_ari": float(np.mean(values)),
            "boot_sd": float(np.std(values)),
            "maxfrac_base": round(max_fraction, 3),
        }
        print(
            f"  {label} (n={sample_size}) K={k}: bootstrap ARI {np.mean(values):+.4f}"
            f"+/-{np.std(values):.4f} | full-sample largest-cluster fraction {max_fraction:.3f}"
        )
    return row


def cross_cohort_degeneracy(
    label: str,
    blocks_a: list[np.ndarray],
    blocks_b: list[np.ndarray],
    k_values: list[int],
    top_genes: int,
    n_init: int,
    degeneracy_fraction: float,
) -> dict[str, dict[str, float | int]]:
    """Measure the transfer ARI of a cohort pair and diagnose its degeneracy.

    The ARI is the same quantity 10_04 reports; the diagnostic is the check that
    the number can be believed. A partition that puts almost every sample into
    one cluster is reproduced by any other partition of the same shape, so two
    such partitions reach an ARI close to one while sharing no structure at all.
    The largest-cluster fraction is therefore measured on all three labellings of
    the pair -- the model fitted on cohort A, the reference partition of cohort B
    and the labelling transferred from A to B -- and the pair is flagged when any
    of them exceeds the configured fraction.

    Args:
        label: cohort-pair label, used in the console output.
        blocks_a: standardised layer blocks of the first cohort.
        blocks_b: standardised layer blocks of the second cohort.
        k_values: cluster numbers compared.
        top_genes: genes kept per layer before the layers are concatenated.
        n_init: k-means restarts of both fits.
        degeneracy_fraction: largest-cluster fraction above which the pair is
            flagged as degenerate.

    Returns:
        K (as a string) -> dict with the transfer ARI (rounded to four decimals),
        the three largest-cluster fractions (rounded to three) and the degeneracy
        flag.
    """
    features_a, features_b = select_shared_features(blocks_a, blocks_b, top_genes)
    row: dict[str, dict[str, float | int]] = {}
    for k in k_values:
        model_a = KMeans(k, n_init=n_init, random_state=SEED).fit(features_a)
        # Squared Euclidean distance of every sample of B to every centre of A,
        # then the nearest centre; computed by broadcasting rather than through
        # the model's own predict, which is the same arithmetic.
        predicted = ((features_b[:, None, :] - model_a.cluster_centers_[None]) ** 2).sum(2).argmin(1)
        own_b = KMeans(k, n_init=n_init, random_state=SEED).fit_predict(features_b)
        ari = float(adjusted_rand_score(own_b, predicted))
        max_a = float(np.bincount(model_a.labels_, minlength=k).max() / len(model_a.labels_))
        max_reference = float(np.bincount(own_b, minlength=k).max() / len(own_b))
        max_predicted = float(np.bincount(predicted, minlength=k).max() / len(predicted))
        row[str(k)] = {
            "ari": round(ari, 4),
            "maxfrac_A": round(max_a, 3),
            "maxfrac_refB": round(max_reference, 3),
            "maxfrac_pred": round(max_predicted, 3),
            "degen": int(max(max_a, max_reference, max_predicted) > degeneracy_fraction),
        }
        marker = "  DEGENERATE" if row[str(k)]["degen"] else ""
        print(
            f"  {label} K={k}: ARI {ari:+.4f} | largest cluster A {max_a:.3f} "
            f"refB {max_reference:.3f} pred {max_predicted:.3f}{marker}"
        )
    return row


def main() -> None:
    """Measure the two criteria and write them back into the G3 artefact.

    Returns:
        None. work("g3_power") is extended in place.
    """
    rng = np.random.RandomState(SEED)
    layer_suffixes = layer_file_suffixes()
    k_values = [int(k) for k in param("gates", "g3", "k_values")]
    top_genes = int(param("gates", "g3", "top_mad_genes"))
    n_init = int(param("gates", "g3", "kmeans_n_init"))

    with work("g3_power").open(encoding="utf-8") as handle:
        g3 = json.load(handle)

    cohort_blocks = {
        label: build_cohort_blocks(tag, identifiers, match_barcode_prefix, layer_suffixes)
        for label, (tag, identifiers, match_barcode_prefix) in cohort_inputs().items()
    }

    # (1) Bootstrap stability of the whole-cohort clustering: the ceiling that
    # applies to a transfer measured on all samples.
    print("=== (1) Bootstrap stability of the whole-cohort clustering (the correct ceiling) ===")
    bootstrap: dict[str, dict[str, dict[str, float]]] = {}
    for label, (blocks, _samples) in cohort_blocks.items():
        bootstrap[label] = bootstrap_ceiling(
            label,
            blocks,
            rng,
            k_values,
            int(param("gates", "g3", "n_bootstrap")),
            n_init,
            int(param("gates", "g3", "kmeans_n_init_resample")),
            int(param("gates", "g3", "bootstrap_unique_margin")),
        )

    # (2) Degeneracy diagnostic of the cross-cohort runs: a high ARI produced by
    # two near-single-cluster partitions is not evidence of shared structure.
    print("\n=== (2) Cross-cohort degeneracy diagnostic (largest-cluster fraction) ===")
    pair_separator = str(param("benchmark", "pair_separator"))
    degeneracy: dict[str, dict[str, dict[str, float | int]]] = {}
    for label_a, label_b in itertools.combinations(list(cohort_blocks), 2):
        pair_label = f"{label_a}{pair_separator}{label_b}"
        degeneracy[pair_label] = cross_cohort_degeneracy(
            pair_label,
            cohort_blocks[label_a][0],
            cohort_blocks[label_b][0],
            k_values,
            top_genes,
            n_init,
            float(param("gates", "g3", "cross_degeneracy_fraction")),
        )

    # The file is rewritten from the artefact read at the start, so the
    # split-half, null and cross measurements of 10_04 survive unchanged.
    out = dict(g3)
    out["bootstrap_ceiling"] = bootstrap
    out["cross_degeneracy"] = degeneracy
    with work("g3_power").open("w", encoding="utf-8") as handle:
        json.dump(out, handle, ensure_ascii=False, indent=1)
    print(f"\nUpdated {work('g3_power')}")


if __name__ == "__main__":
    main()
