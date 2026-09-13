#!/usr/bin/env python3
"""
Verify batch separability with sign-fixed, classifier-independent criteria.

Purpose
    Stage 07, fourth step, and the verification counterpart of the metric script.
    The cross-validated AUC of a fitted classifier is orientation-dependent: a
    raw AUC near 0 means the classifier separated the two batches perfectly along
    an axis whose sign is reversed, not that the batches were mixed. Reading such
    a value as "no batch effect" would invert the conclusion of the testbed, so
    this script reports separability through criteria that cannot change sign:
    the centroid AUC, which projects the samples onto the batch mean-difference
    direction and therefore measures per-gene mean separation with a fixed
    orientation; the sign-fixed cross-validated AUC max(AUC, 1-AUC); the median
    per-protein Kolmogorov-Smirnov statistic, which compares whole distributions
    rather than means; and the kNN mixing rate. The four criteria agree on every
    arm, which is what makes the testbed's verdict robust to the orientation
    artefact.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    work.eeec_batch_y_batch
        Integer batch target written by 07_02.
    work.eeec_batch_arms_npz
        Python correction arms from 07_02, keyed by arm label.
    work.eeec_batch_arm_combat, work.eeec_batch_arm_combat_mean_only,
    work.eeec_batch_arm_remove_batch_effect
        The three R correction arms; each is read only when it exists.
    params batch_testbed.knn_neighbours, batch_testbed.cv_folds,
    params batch_testbed.logistic_c, batch_testbed.logistic_max_iter,
    params batch_testbed.svm_c, batch_testbed.svm_max_iter,
    params batch_testbed.arm_order, batch_testbed.arms, batch_testbed.r_arms.

Outputs
    work.eeec_batch_verify_csv
        One row per arm with centroid_AUC, LR_AUC_raw, LR_AUC_abs, SVM_AUC_abs,
        KS_median and knn_mix (CSV).
    work.eeec_batch_verify_json
        The same table as JSON.

Usage
    python src/07_batch_testbed/07_04_verify_batch_testbed.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# The package root (one level above src/) so that "common" can be imported from
# any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import SEED, ensure_dir, param, work

# --- Configuration -----------------------------------------------------------

KNN_NEIGHBOURS = int(param("batch_testbed", "knn_neighbours"))
CV_FOLDS = int(param("batch_testbed", "cv_folds"))
LOGISTIC_C = float(param("batch_testbed", "logistic_c"))
LOGISTIC_MAX_ITER = int(param("batch_testbed", "logistic_max_iter"))
SVM_C = float(param("batch_testbed", "svm_c"))
SVM_MAX_ITER = int(param("batch_testbed", "svm_max_iter"))

# Arm vocabulary, shared with 07_02 and 07_03.
ARM_LABELS = {str(tag): str(label) for tag, label in param("batch_testbed", "arms").items()}
R_ARM_LABELS = {str(tag): str(label) for tag, label in param("batch_testbed", "r_arms").items()}
ARM_ORDER = [str(tag) for tag in param("batch_testbed", "arm_order")]
LABEL_BY_TAG = {**ARM_LABELS, **R_ARM_LABELS}
ORDER = [LABEL_BY_TAG[tag] for tag in ARM_ORDER]

R_ARM_PATHS = {
    R_ARM_LABELS["combat"]: work("eeec_batch_arm_combat"),
    R_ARM_LABELS["combat_mean_only"]: work("eeec_batch_arm_combat_mean_only"),
    R_ARM_LABELS["remove_batch_effect"]: work("eeec_batch_arm_remove_batch_effect"),
}

Y_BATCH_PATH = work("eeec_batch_y_batch")
Y_COND_PATH = work("eeec_batch_y_cond")
ARMS_NPZ_PATH = work("eeec_batch_arms_npz")
VERIFY_CSV_PATH = work("eeec_batch_verify_csv")
VERIFY_JSON_PATH = work("eeec_batch_verify_json")

# Classifier identifiers used by cv_auc_both.
CLASSIFIER_LOGISTIC = "lr"
CLASSIFIER_SVM = "svm"


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


def clean(values: np.ndarray) -> np.ndarray:
    """Return a finite copy of an arm matrix.

    Args:
        values: sample x feature matrix.

    Returns:
        The matrix with non-finite entries replaced by zero. The comparison
        metrics below (KS, correlations) do not guard against NaN, so the repair
        happens once here.
    """
    cleaned = np.asarray(values, float).copy()
    cleaned[~np.isfinite(cleaned)] = 0.0
    return cleaned


# --- Sign-fixed criteria ------------------------------------------------------


def centroid_auc(values: np.ndarray, target: np.ndarray) -> float:
    """AUC of the projection onto the batch mean-difference direction.

    Args:
        values: samples x features matrix of the arm.
        target: binary integer batch vector.

    Returns:
        The AUC of the score (mean of group 1 minus mean of group 0) projected on
        the samples. The projection direction is built from the group means, so
        the score increases with separation by construction and the AUC is never
        below chance because of a flipped axis: 0.5 means the group means are
        identical (for the per-gene arms exactly 0.5) and 1.0 means they are
        perfectly separated. If the two group means coincide, the direction is
        undefined and chance (0.5) is returned.
    """
    values = clean(values)
    mean_one = values[target == 1].mean(axis=0)
    mean_zero = values[target == 0].mean(axis=0)
    direction = mean_one - mean_zero
    if np.allclose(direction, 0):
        return 0.5
    return float(roc_auc_score(target, values @ direction))


def cv_auc_both(values: np.ndarray, target: np.ndarray, seed: int = SEED, classifier: str = CLASSIFIER_LOGISTIC) -> tuple:
    """Cross-validated AUC, raw and sign-fixed.

    Args:
        values: samples x features matrix of the arm.
        target: binary integer target vector.
        seed: seed of the fold split and of the classifier.
        classifier: CLASSIFIER_LOGISTIC or CLASSIFIER_SVM.

    Returns:
        (raw AUC, max(raw AUC, 1 - raw AUC)). The raw value is kept for the audit
        trail; the sign-fixed value is what measures separability, because the
        orientation of a fitted decision function is arbitrary and a raw AUC near
        0 describes a perfect but reversed separation, not the absence of one.
    """
    values = clean(values)
    if classifier == CLASSIFIER_LOGISTIC:
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=LOGISTIC_MAX_ITER, C=LOGISTIC_C)
        )
    else:
        estimator = make_pipeline(
            StandardScaler(), LinearSVC(C=SVM_C, max_iter=SVM_MAX_ITER)
        )
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    if classifier == CLASSIFIER_LOGISTIC:
        score = cross_val_predict(estimator, values, target, cv=folds, method="predict_proba")[:, 1]
    else:
        score = cross_val_predict(estimator, values, target, cv=folds, method="decision_function")
    raw = float(roc_auc_score(target, score))
    return raw, float(max(raw, 1 - raw))


def ks_median(values: np.ndarray, target: np.ndarray) -> float:
    """Median per-protein Kolmogorov-Smirnov statistic between the two batches.

    Args:
        values: samples x features matrix of the arm.
        target: binary integer batch vector.

    Returns:
        The median over proteins of the two-sample KS statistic between the batch
        distributions. A distribution-level criterion: the per-gene arms can make
        the means coincide while the shapes still differ, and the KS statistic
        detects exactly that.
    """
    values = clean(values)
    return float(
        np.median(
            [
                ks_2samp(values[target == 1, column], values[target == 0, column]).statistic
                for column in range(values.shape[1])
            ]
        )
    )


def knn_mixing(values: np.ndarray, target: np.ndarray, k: int = KNN_NEIGHBOURS) -> float:
    """Fraction of a sample's k nearest neighbours that come from the other batch.

    Args:
        values: samples x features matrix of the arm.
        target: binary integer batch vector.
        k: number of neighbours.

    Returns:
        The mixing rate: 0.5 is complete mixing and 0 means every neighbourhood
        is single-batch. A local, model-free criterion that does not fit
        anything, so it cannot inherit an orientation from a fit.
    """
    values = clean(values)
    values = StandardScaler().fit_transform(values)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(values)
    _, indices = neighbours.kneighbors(values)
    return float(np.mean(target[indices[:, 1:]] != target[:, None]))


def main() -> pd.DataFrame:
    """Score every arm with the four sign-fixed criteria and write the table.

    Returns:
        The verification table that was written to work.eeec_batch_verify_csv.
    """
    ensure_dir(VERIFY_CSV_PATH.parent)
    y_batch = np.load(Y_BATCH_PATH)
    arms = load_arms()
    print(
        f"{'arm':34s} {'centroid_AUC':>12s} {'LR_AUC_raw':>10s} {'LR_AUC_abs':>10s} "
        f"{'SVM_AUC_abs':>11s} {'KS_median':>9s} {'knn_mix':>8s}"
    )
    rows = []
    for label in ORDER:
        if label not in arms:
            continue
        values = arms[label]
        centroid = centroid_auc(values, y_batch)
        logistic_raw, logistic_abs = cv_auc_both(values, y_batch, classifier=CLASSIFIER_LOGISTIC)
        _, svm_abs = cv_auc_both(values, y_batch, classifier=CLASSIFIER_SVM)
        ks = ks_median(values, y_batch)
        mixing = knn_mixing(values, y_batch)
        rows.append(
            {
                "arm": label,
                "centroid_AUC": round(centroid, 4),
                "LR_AUC_raw": round(logistic_raw, 4),
                "LR_AUC_abs": round(logistic_abs, 4),
                "SVM_AUC_abs": round(svm_abs, 4),
                "KS_median": round(ks, 4),
                "knn_mix": round(mixing, 4),
            }
        )
        print(
            f"{label:34s} {centroid:12.4f} {logistic_raw:10.4f} {logistic_abs:10.4f} "
            f"{svm_abs:11.4f} {ks:9.4f} {mixing:8.4f}"
        )
    table = pd.DataFrame(rows)
    table.to_csv(VERIFY_CSV_PATH, index=False)
    with VERIFY_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(table.to_dict(orient="records"), handle, ensure_ascii=False, indent=1)
    print(f"\nWritten to {VERIFY_JSON_PATH}")
    return table


if __name__ == "__main__":
    main()
