#!/usr/bin/env python3
"""
Apply the correction arms of the EEEC batch testbed.

Purpose
    Stage 07, second step. It loads the log2 EEEC matrix built by 07_01 and
    applies the six correction arms that can be computed in Python (raw,
    per-gene z globally, per-gene z within batch, per-sample quantile and
    rank-to-inverse-normal normalisation, and Harmony on the embedding layer).
    The three remaining arms (ComBat, ComBat mean-only and removeBatchEffect)
    are computed by the R side from the matrix exported here. The central arm of
    the testbed is the within-batch per-gene z transform: it removes every
    per-gene mean and variance difference between the batches exactly, and is
    therefore the theoretical ceiling of any per-gene correction. Whatever batch
    separability survives it cannot be a per-gene location or scale effect.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work.eeec_batch_matrix
        log2 LFQ matrix (protein groups x samples) from 07_01.
    work.eeec_batch_sample_meta
        Sample annotation (col/family/suffix/batch/cond) from 07_01.
    params batch_testbed.balanced_families, batch_testbed.reference_batch,
    params batch_testbed.positive_condition, batch_testbed.harmony_n_pcs,
    params batch_testbed.harmony_max_iter.

Outputs
    work.eeec_batch_arms_npz
        Every Python arm as a numeric matrix, keyed by the arm label.
    work.eeec_batch_arms_json
        Shape summary of every Python arm (JSON).
    work.eeec_batch_for_r_matrix
        The uncorrected sample x protein matrix handed to the R corrections.
    work.eeec_batch_for_r_meta
        Batch and condition of every row of that matrix.
    work.eeec_batch_y_batch, work.eeec_batch_y_cond
        Integer targets used by the metric scripts.

Usage
    python src/07_batch_testbed/07_02_apply_batch_corrections.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# The package root (one level above src/) so that "common" can be imported from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, param, work

# --- Configuration -----------------------------------------------------------

BALANCED_FAMILIES = [str(family) for family in param("batch_testbed", "balanced_families")]
REFERENCE_BATCH = str(param("batch_testbed", "reference_batch"))
POSITIVE_CONDITION = str(param("batch_testbed", "positive_condition"))
HARMONY_N_PCS = int(param("batch_testbed", "harmony_n_pcs"))
HARMONY_MAX_ITER = int(param("batch_testbed", "harmony_max_iter"))

# Arm labels are a configuration vocabulary because 07_03 and 07_04 read the arms
# back by label; the tag is the key under which the label is declared.
ARM_LABELS = {str(tag): str(label) for tag, label in param("batch_testbed", "arms").items()}

MATRIX_PATH = work("eeec_batch_matrix")
SAMPLE_META_PATH = work("eeec_batch_sample_meta")
FOR_R_MATRIX_PATH = work("eeec_batch_for_r_matrix")
FOR_R_META_PATH = work("eeec_batch_for_r_meta")
Y_BATCH_PATH = work("eeec_batch_y_batch")
Y_COND_PATH = work("eeec_batch_y_cond")
ARMS_JSON_PATH = work("eeec_batch_arms_json")
ARMS_NPZ_PATH = work("eeec_batch_arms_npz")

# Name of the batch column handed to Harmony as the covariate to correct.
BATCH_COVARIATE = "batch"


def load_design() -> tuple:
    """Load the matrix, the sample annotation and the analysis targets.

    Returns:
        (matrix, meta, targets) where matrix is samples x protein groups with the
        rows in the order of the balanced design, meta is the matching sample
        annotation, and targets is a dict with the integer batch and condition
        vectors. The row filter keeps only the four balanced cells, so the
        remaining families never reach an arm.
    """
    log2_matrix = pd.read_csv(MATRIX_PATH, index_col=0)
    meta = pd.read_csv(SAMPLE_META_PATH)
    meta = meta[meta["family"].isin(BALANCED_FAMILIES)].reset_index(drop=True)
    matrix = log2_matrix[meta["col"].tolist()].T.copy()
    targets = {
        "batch": (meta["batch"] == REFERENCE_BATCH).astype(int).values,
        "cond": (meta["cond"] == POSITIVE_CONDITION).astype(int).values,
    }
    return matrix, meta, targets


# --- Correction arms ---------------------------------------------------------


def s0_raw(matrix: pd.DataFrame) -> np.ndarray:
    """Arm S0: no correction.

    Args:
        matrix: samples x protein groups log2 matrix.

    Returns:
        The matrix unchanged, the reference every other arm is compared against.
    """
    return matrix.values


def s1_zscore(matrix: pd.DataFrame) -> np.ndarray:
    """Arm S1: global per-gene z-scoring.

    Args:
        matrix: samples x protein groups log2 matrix.

    Returns:
        Every protein standardised across all samples. Genes whose standard
        deviation is zero are left as they are (division guard): a constant gene
        carries no scale to standardise.
    """
    zscored = (matrix - matrix.mean()) / matrix.std(ddof=1).replace(0, 1)
    return zscored.values


def s2_zscore_within_batch(matrix: pd.DataFrame, batch: np.ndarray) -> np.ndarray:
    """Arm S2: per-gene z-scoring inside each batch.

    Args:
        matrix: samples x protein groups log2 matrix.
        batch: integer batch vector aligned with the rows of the matrix.

    Returns:
        Every protein standardised within its batch. This removes the per-gene
        mean and variance difference between the batches exactly, which is why it
        is the upper bound of any per-gene correction: no per-gene location or
        scale transform can do more.
    """
    zscored = matrix.copy()
    for value in [0, 1]:
        block = matrix.iloc[batch == value]
        zscored.iloc[batch == value] = (block - block.mean()) / block.std(ddof=1).replace(0, 1)
    return zscored.values


def s3_quantile(matrix: pd.DataFrame) -> np.ndarray:
    """Arm S3: per-sample quantile normalisation to a common reference.

    Args:
        matrix: samples x protein groups log2 matrix.

    Returns:
        The matrix in which every sample is replaced by its own rank order mapped
        onto the mean sorted profile of all samples, so that every sample ends up
        with the same intensity distribution.
    """
    result = matrix.values.copy()
    # Reference profile: sort within each sample, then average across samples by
    # rank, which yields one value per protein rank.
    reference = np.sort(matrix.values, axis=1).mean(axis=0)
    for row in range(result.shape[0]):
        order = np.argsort(result[row])
        result[row, order] = reference
    return result


def s4_rankint(matrix: pd.DataFrame) -> np.ndarray:
    """Arm S4: per-sample ranks mapped to the inverse normal distribution.

    Args:
        matrix: samples x protein groups log2 matrix.

    Returns:
        The matrix in which every sample is replaced by the inverse standard
        normal quantile of its ranks: a distributional correction that does not
        assume a shared mean profile across samples.
    """
    from scipy.stats import norm

    result = matrix.values.copy()
    n = result.shape[1]
    for row in range(result.shape[0]):
        order = np.argsort(result[row])
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        result[row] = norm.ppf((ranks - 0.5) / n)
    return result


def s5_harmony(matrix: pd.DataFrame, meta_batch: pd.DataFrame) -> np.ndarray:
    """Arm S5: Harmony correction on the embedding layer.

    Args:
        matrix: samples x protein groups log2 matrix.
        meta_batch: covariate frame holding the batch of every sample.

    Returns:
        The batch-corrected embedding (harmonypy run_harmony; Korsunsky et al.,
        Nature Methods 2019). Unlike the per-gene arms this corrects the
        covariance structure: Harmony operates on principal components and
        rotates and rescales the embedding until the batches overlap, so no
        per-gene location or scale transformation is applied at all.
    """
    import harmonypy

    scaled = StandardScaler().fit_transform(matrix.values)
    fitted = PCA(n_components=HARMONY_N_PCS, random_state=SEED).fit(scaled)
    harmony = harmonypy.run_harmony(
        fitted.transform(scaled),
        meta_batch,
        [BATCH_COVARIATE],
        max_iter_harmony=HARMONY_MAX_ITER,
    )
    corrected = np.asarray(harmony.Z_corr)
    # harmonypy returns the corrected embedding transposed in some versions, so
    # the orientation is aligned with the sample count instead of assumed.
    if corrected.shape[0] != matrix.shape[0]:
        corrected = corrected.T
    return corrected


def build_arm_functions(batch: np.ndarray, meta_batch: pd.DataFrame) -> dict:
    """Bind the correction arms to the design they are applied to.

    Args:
        batch: integer batch vector.
        meta_batch: covariate frame for Harmony.

    Returns:
        Mapping arm tag -> callable taking the sample x protein matrix. The arms
        that need the design (within-batch z, Harmony covariates) close over it
        here, so the arm loop below stays a plain apply.
    """
    return {
        "raw": lambda matrix: s0_raw(matrix),
        "pygen": lambda matrix: s1_zscore(matrix),
        "pygen_max": lambda matrix: s2_zscore_within_batch(matrix, batch),
        "quantile": lambda matrix: s3_quantile(matrix),
        "rankint": lambda matrix: s4_rankint(matrix),
        "embed": lambda matrix: s5_harmony(matrix, meta_batch),
    }


def main() -> dict:
    """Apply every Python arm and export the artefacts of the stage.

    Returns:
        Mapping arm label -> corrected matrix for the arms that were applied
        successfully. An arm that raises is reported and left out, so that a
        missing optional dependency (Harmony) does not abort the other arms.
    """
    matrix, meta, targets = load_design()
    y_batch = targets["batch"]
    y_cond = targets["cond"]
    print(
        f"Matrix: {matrix.shape[0]} samples x {matrix.shape[1]} protein groups | "
        f"batch {REFERENCE_BATCH}={int(y_batch.sum())} | "
        f"condition {POSITIVE_CONDITION}={int(y_cond.sum())}"
    )

    meta_batch = meta[["batch"]].copy()
    # Harmony addresses covariates by name, so the batch column must be a string.
    meta_batch["batch"] = meta_batch["batch"].astype(str)

    corrected_arms = {}
    for tag, function in build_arm_functions(y_batch, meta_batch).items():
        label = ARM_LABELS[tag]
        try:
            values = np.asarray(function(matrix), dtype=float)
            # Proteomic embeddings occasionally emit non-finite entries; the
            # metrics need a finite matrix and zero is the neutral value of a
            # standardised feature.
            values[~np.isfinite(values)] = 0.0
            corrected_arms[label] = values
            print(f"  [ok]   {label:34s} {values.shape}")
        except Exception as exc:
            print(f"  [fail] {label:34s} {type(exc).__name__}: {str(exc)[:90]}")

    # Export the uncorrected matrix and the design for the R arms
    # (ComBat / limma), which run outside this script.
    matrix.to_csv(FOR_R_MATRIX_PATH)
    pd.DataFrame({"batch": meta["batch"].values, "cond": meta["cond"].values}).to_csv(
        FOR_R_META_PATH, index=False
    )
    np.save(Y_BATCH_PATH, y_batch)
    np.save(Y_COND_PATH, y_cond)
    # Only the shapes are written as JSON; the matrices themselves go to the
    # compressed archive because they are two orders of magnitude larger.
    with ARMS_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {label: [list(values.shape)] for label, values in corrected_arms.items()},
            handle,
            ensure_ascii=False,
            indent=1,
        )
    np.savez_compressed(ARMS_NPZ_PATH, **corrected_arms)
    print(
        f"\nWrote {len(corrected_arms)} Python arms to {ARMS_NPZ_PATH.name} and exported "
        f"the matrix for the R corrections (ComBat / limma)"
    )
    return corrected_arms


if __name__ == "__main__":
    main()
