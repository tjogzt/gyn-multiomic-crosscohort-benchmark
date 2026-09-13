#!/usr/bin/env python3
"""
Compute the five batch-testbed metrics per correction arm.

Purpose
    Stage 07, third step. For every arm of the testbed it quantifies what the arm
    did to the batch effect and to the biology: batch separability (cross-
    validated AUC of a classifier that predicts the processing batch), biological
    separability (the same classifier predicting tumour vs normal), cross-batch
    label transfer accuracy (train the tumour/normal classifier in one batch,
    test it in the other), the within-minus-between-batch correlation gap, and
    the kNN batch-mixing rate. It then repeats the two separability metrics on
    the per-gene ceiling arm after removing its leading principal components, and
    reports how strongly each of the first principal components tracks the batch,
    which localises the surviving batch signal in the covariance layer.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work.eeec_batch_y_batch, work.eeec_batch_y_cond
        Integer targets written by 07_02.
    work.eeec_batch_sample_meta
        Sample annotation from 07_01 (used to order the arms).
    work.eeec_batch_arms_npz
        Python correction arms from 07_02, keyed by arm label.
    work.eeec_batch_arm_combat, work.eeec_batch_arm_combat_mean_only,
    work.eeec_batch_arm_remove_batch_effect
        The three R correction arms; each is read only when it exists.
    params batch_testbed.knn_neighbours, batch_testbed.cv_folds,
    params batch_testbed.logistic_c, batch_testbed.logistic_max_iter,
    params batch_testbed.pca_components, batch_testbed.pc_removal_sweep,
    params batch_testbed.pc_association_top, batch_testbed.arm_order,
    params batch_testbed.arms, batch_testbed.r_arms.

Outputs
    work.eeec_batch_metrics
        One row per arm with every metric (CSV).
    work.eeec_batch_pc_sweep
        The S2 arm after removing 0, 1, 2, ... leading components (CSV).
    work.eeec_batch_pc_assoc
        Association of the first components with batch and condition (CSV).
    work.eeec_batch_results
        All three tables as one JSON object.

Usage
    python src/07_batch_testbed/07_03_compute_batch_metrics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The package root (one level above src/) so that "common" can be imported from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, ensure_dir, param, work

# --- Configuration -----------------------------------------------------------

BALANCED_FAMILIES = [str(family) for family in param("batch_testbed", "balanced_families")]
KNN_NEIGHBOURS = int(param("batch_testbed", "knn_neighbours"))
CV_FOLDS = int(param("batch_testbed", "cv_folds"))
LOGISTIC_C = float(param("batch_testbed", "logistic_c"))
LOGISTIC_MAX_ITER = int(param("batch_testbed", "logistic_max_iter"))
PCA_COMPONENTS = int(param("batch_testbed", "pca_components"))
PC_SWEEP = [int(k) for k in param("batch_testbed", "pc_removal_sweep")]
PC_ASSOC_TOP = int(param("batch_testbed", "pc_association_top"))

# Arm vocabulary, shared with 07_02 and 07_04.
ARM_LABELS = {str(tag): str(label) for tag, label in param("batch_testbed", "arms").items()}
R_ARM_LABELS = {str(tag): str(label) for tag, label in param("batch_testbed", "r_arms").items()}
ARM_ORDER = [str(tag) for tag in param("batch_testbed", "arm_order")]
LABEL_BY_TAG = {**ARM_LABELS, **R_ARM_LABELS}
ORDER = [LABEL_BY_TAG[tag] for tag in ARM_ORDER]

# Where each R arm is read from. A missing file simply removes that arm from the
# table, which is how a run without the R side still produces the Python metrics.
R_ARM_PATHS = {
    R_ARM_LABELS["combat"]: work("eeec_batch_arm_combat"),
    R_ARM_LABELS["combat_mean_only"]: work("eeec_batch_arm_combat_mean_only"),
    R_ARM_LABELS["remove_batch_effect"]: work("eeec_batch_arm_remove_batch_effect"),
}

Y_BATCH_PATH = work("eeec_batch_y_batch")
Y_COND_PATH = work("eeec_batch_y_cond")
SAMPLE_META_PATH = work("eeec_batch_sample_meta")
ARMS_NPZ_PATH = work("eeec_batch_arms_npz")
METRICS_PATH = work("eeec_batch_metrics")
PC_SWEEP_PATH = work("eeec_batch_pc_sweep")
PC_ASSOC_PATH = work("eeec_batch_pc_assoc")
RESULTS_PATH = work("eeec_batch_results")


def load_arms() -> dict:
    """Load every correction arm that is available.

    Returns:
        Mapping arm label -> sample x feature matrix, holding the Python arms
        from the archive plus each R arm whose CSV exists.
    """
    arms = {}
    with np.load(ARMS_NPZ_PATH) as archive:
        for label in archive.files:
            arms[label] = archive[label]
    for label, path in R_ARM_PATHS.items():
        if Path(path).exists():
            arms[label] = pd.read_csv(path, index_col=0).values
    return arms


# --- Metrics -----------------------------------------------------------------


def clean_matrix(values: np.ndarray) -> np.ndarray:
    """Return the arm matrix with every non-finite entry replaced by zero.

    Args:
        values: sample x feature matrix of an arm.

    Returns:
        The finiteness-repaired matrix. The metrics are computed on this matrix
        rather than on the raw arm, so that a missing entry cannot turn a
        correlation (which has no built-in guard) into NaN while the AUCs, which
        do guard, stay finite.
    """
    cleaned = np.asarray(values, float)
    cleaned[~np.isfinite(cleaned)] = 0.0
    return cleaned


def auc_cv(values: np.ndarray, target: np.ndarray, seed: int = SEED) -> float:
    """Cross-validated AUC of a logistic classifier on a feature matrix.

    Args:
        values: samples x features matrix of the arm.
        target: binary integer target vector.
        seed: seed of the fold split and of the classifier.

    Returns:
        The cross-validated AUC. This is a *signed* quantity: an AUC near 0 means
        the classifier separates the two groups perfectly but in the opposite
        direction (the fitted axis points the other way), not that the groups are
        mixed. The metrics that must be read as separability use abs_auc below.
    """
    values = np.asarray(values, float).copy()
    values[~np.isfinite(values)] = 0
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=LOGISTIC_MAX_ITER, C=LOGISTIC_C)
    )
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    probability = cross_val_predict(classifier, values, target, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(target, probability))


def abs_auc(value: float) -> float:
    """Fold a signed AUC onto the separability scale.

    Args:
        value: AUC as returned by auc_cv.

    Returns:
        max(value, 1 - value), i.e. the separation strength irrespective of the
        orientation of the fitted axis. The sign of an AUC is an artefact of
        which class the classifier happened to call positive and of the sign of
        the feature axis; separation strength, which is what the testbed asks
        about, is sign-free. A raw AUC of 0.0000 therefore means complete,
        reversed separation - the batch effect was *not* removed.
    """
    return float(max(value, 1 - value))


def transfer_accuracy(values: np.ndarray, target: np.ndarray, batch: np.ndarray) -> tuple:
    """Measure cross-batch label transfer of the tumour/normal classifier.

    Args:
        values: samples x features matrix of the arm.
        target: binary integer condition vector.
        batch: binary integer batch vector.

    Returns:
        (mean accuracy, per-direction accuracies) where each direction trains on
        one batch and predicts the other. The mean over the two directions is the
        reported value; both directions are kept so that an asymmetric failure
        stays visible.
    """
    values = np.asarray(values, float).copy()
    values[~np.isfinite(values)] = 0
    values = StandardScaler().fit_transform(values)
    accuracies = []
    for train_batch, test_batch in [(0, 1), (1, 0)]:
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=LOGISTIC_MAX_ITER, C=LOGISTIC_C)
        )
        model.fit(values[batch == train_batch], target[batch == train_batch])
        accuracies.append(
            float(
                (
                    model.predict(values[batch == test_batch])
                    == target[batch == test_batch]
                ).mean()
            )
        )
    return float(np.mean(accuracies)), accuracies


def correlation_gap(values: np.ndarray, batch: np.ndarray) -> float:
    """Difference between mean within-batch and between-batch sample correlation.

    Args:
        values: samples x features matrix of the arm.
        batch: binary integer batch vector.

    Returns:
        Mean correlation of sample pairs from the same batch minus the mean
        correlation of pairs from different batches. Zero means the batch leaves
        no trace in the correlation structure; a positive value means samples of
        the same batch still resemble each other.
    """
    correlations = np.corrcoef(np.asarray(values, float))
    upper = np.triu_indices(len(correlations), 1)
    same_batch = batch[upper[0]] == batch[upper[1]]
    return float(correlations[upper][same_batch].mean() - correlations[upper][~same_batch].mean())


def knn_mixing(values: np.ndarray, batch: np.ndarray, k: int = KNN_NEIGHBOURS) -> float:
    """Fraction of a sample's k nearest neighbours that come from the other batch.

    Args:
        values: samples x features matrix of the arm.
        batch: binary integer batch vector.
        k: number of neighbours.

    Returns:
        The mixing rate. 0.5 is perfect mixing, 0 means every neighbourhood is
        single-batch, so higher is better.
    """
    values = np.asarray(values, float).copy()
    values[~np.isfinite(values)] = 0
    values = StandardScaler().fit_transform(values)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(values)
    _, indices = neighbours.kneighbors(values)
    return float(np.mean(batch[indices[:, 1:]] != batch[:, None]))


def build_metric_table(arms: dict, y_batch: np.ndarray, y_cond: np.ndarray) -> pd.DataFrame:
    """Compute the five metrics for every arm in the reporting order.

    Args:
        arms: arm label -> sample x feature matrix.
        y_batch: binary integer batch vector.
        y_cond: binary integer condition vector.

    Returns:
        One row per available arm, in the order of batch_testbed.arm_order.
    """
    rows = []
    for label in ORDER:
        if label not in arms:
            continue
        values = clean_matrix(arms[label])
        record = {
            "arm": label,
            "n_feat": int(values.shape[1]),
            "batch_AUC": round(auc_cv(values, y_batch), 4),
            "bio_AUC": round(auc_cv(values, y_cond), 4),
        }
        transfer_mean, transfer_directions = transfer_accuracy(values, y_cond, y_batch)
        record["transfer_acc"] = round(transfer_mean, 4)
        record["transfer_E2L"] = round(transfer_directions[0], 4)
        record["transfer_L2E"] = round(transfer_directions[1], 4)
        record["corr_gap"] = round(correlation_gap(values, y_batch), 4)
        record["knn_mix"] = round(knn_mixing(values, y_batch), 4)
        rows.append(record)
    return pd.DataFrame(rows)


def covariance_sweep(arms: dict, y_batch: np.ndarray, y_cond: np.ndarray) -> tuple:
    """Remove the leading components of the ceiling arm one at a time.

    Args:
        arms: arm label -> sample x feature matrix.
        y_batch: binary integer batch vector.
        y_cond: binary integer condition vector.

    Returns:
        (sweep table, fitted PCA, component scores). The sweep shows which metric
        survives the removal of a given number of covariance directions: the
        per-gene ceiling arm has no per-gene mean or variance difference left, so
        anything that disappears here lived in the covariance layer.
    """
    ceiling = clean_matrix(arms[ARM_LABELS["pygen_max"]])
    scaled = StandardScaler().fit_transform(ceiling)
    fitted = PCA(n_components=min(PCA_COMPONENTS, scaled.shape[0] - 1), random_state=SEED).fit(scaled)
    scores = fitted.transform(scaled)
    sweep = []
    for k in PC_SWEEP:
        # Removing every component would leave nothing to score; the last
        # iteration therefore keeps a single column instead.
        kept = scores[:, k:] if k < scores.shape[1] else scores[:, :1]
        sweep.append(
            {
                "k_removed": k,
                "batch_AUC": round(auc_cv(kept, y_batch), 4),
                "bio_AUC": round(auc_cv(kept, y_cond), 4),
                "transfer_acc": round(transfer_accuracy(kept, y_cond, y_batch)[0], 4),
                "corr_gap": round(correlation_gap(kept, y_batch), 4),
            }
        )
    return pd.DataFrame(sweep), fitted, scores


def component_association(scores: np.ndarray, explained: np.ndarray, y_batch: np.ndarray, y_cond: np.ndarray) -> pd.DataFrame:
    """Associate each leading component with batch and with condition.

    Args:
        scores: component scores of the ceiling arm.
        explained: explained variance ratio per component.
        y_batch: binary integer batch vector.
        y_cond: binary integer condition vector.

    Returns:
        One row per component with the sign-fixed AUROC against batch (that is
        max(AUC, 1-AUC): a component whose raw AUC is 0.0000 separates the
        batches perfectly along its reversed axis), the same against condition,
        and the variance fraction it explains.
    """
    rows = []
    for component in range(PC_ASSOC_TOP):
        against_batch = float(roc_auc_score(y_batch, scores[:, component]))
        against_condition = float(roc_auc_score(y_cond, scores[:, component]))
        rows.append(
            {
                "PC": component + 1,
                "AUROC_vs_batch": round(abs_auc(against_batch), 4),
                "AUROC_vs_cond": round(abs_auc(against_condition), 4),
                "var_exp": round(float(explained[component]), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> dict:
    """Compute every metric table and write them to the stage artefacts.

    Returns:
        The payload written to work.eeec_batch_results.
    """
    ensure_dir(RESULTS_PATH.parent)
    y_batch = np.load(Y_BATCH_PATH)
    y_cond = np.load(Y_COND_PATH)
    meta = pd.read_csv(SAMPLE_META_PATH)
    meta = meta[meta["family"].isin(BALANCED_FAMILIES)].reset_index(drop=True)
    arms = load_arms()
    print(f"Arms loaded: {len(arms)} ({', '.join(sorted(arms))})")

    metrics = build_metric_table(arms, y_batch, y_cond)
    print("\n=== Batch-correction arms, all metrics ===")
    print(metrics.to_string(index=False))
    metrics.to_csv(METRICS_PATH, index=False)

    print(f"\n=== Covariance-layer check: {ARM_LABELS['pygen_max']} after removing leading PCs ===")
    sweep, fitted, scores = covariance_sweep(arms, y_batch, y_cond)
    print(sweep.to_string(index=False))
    sweep.to_csv(PC_SWEEP_PATH, index=False)

    print("\n=== Association of the first components with batch (|AUROC|) and explained variance ===")
    association = component_association(
        scores,
        fitted.explained_variance_ratio_,
        y_batch,
        y_cond,
    )
    print(association.to_string(index=False))
    association.to_csv(PC_ASSOC_PATH, index=False)

    payload = {
        "metrics": metrics.to_dict(orient="records"),
        "pc_sweep": sweep.to_dict(orient="records"),
        "pc_assoc": association.to_dict(orient="records"),
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    print(f"\nWritten to {RESULTS_PATH}")
    return payload


if __name__ == "__main__":
    main()
