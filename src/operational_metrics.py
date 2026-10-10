"""Operational Evaluation & Leakage-Safe Thresholding Module.

Provides:
  1. derive_operational_threshold: Finite-sample order-statistic threshold
     selection calibrated strictly on normal flows (y_cal == 0) from X_cal / y_cal.
  2. evaluate_operational_metrics: Reusable operational evaluation reporting
     Recall, empirical FPR, Precision, F1, ROC-AUC, standardized pAUC (max_fpr=0.02),
     and confusion matrix at a previously frozen threshold.
  3. extract_attack_scores: Model adapter ensuring a 1D continuous score where
     higher scores consistently indicate higher attack likelihood (class 1).

Scientific & Statistical Principles:
  - Thresholds are chosen ONLY from calibration negatives (y_cal == 0) on X_cal.
  - Hyperparameter tuning data (X_tune) and test data (X_test) are NEVER used
    for threshold calibration.
  - Test labels are strictly read-only and never alter a frozen threshold.
  - Finite-sample conservative order-statistic rule guarantees empirical
    calibration FPR <= target_fpr. Under distribution shift (e.g. UNSW-NB15 test set),
    achieved test FPR may deviate from target FPR due to covariate/label shifts and
    sampling variability; achieved test FPR is therefore reported explicitly alongside
    target calibration FPR.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def extract_attack_scores(estimator: Any, X: Any) -> np.ndarray:
    """Extract continuous 1D attack scores from an estimator.

    Orientation Convention:
      Higher score -> higher likelihood of attack (class 1).

    Compatibility:
      - Random Forest / Classifiers with predict_proba:
        Extracts probability of class 1: predict_proba(X)[:, class_1_idx].
      - RBF-SVM / Classifiers with decision_function:
        Extracts signed margin: decision_function(X). For binary classification
        with classes (0, 1), positive values correspond to class 1.
      - QSVC (precomputed kernel SVC):
        Extracts signed margin: decision_function(K).

    Hard predictions (predict) are rejected because operational thresholding
    requires continuous ranking scores.

    Parameters
    ----------
    estimator : Any
        Fitted classifier implementing predict_proba or decision_function.
    X : Any
        Feature matrix or precomputed kernel matrix.

    Returns
    -------
    np.ndarray
        1D float array of continuous scores.

    Raises
    ------
    ValueError
        If estimator provides only hard predictions, outputs multiclass scores (>2),
        produces non-finite values, or output cannot be converted to 1D float array.
    """
    if estimator is None:
        raise ValueError("Estimator cannot be None.")

    # 1. Prefer predict_proba if available (e.g. RandomForestClassifier)
    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(X)
        probs_arr = np.asarray(probs, dtype=float)

        if probs_arr.ndim == 2:
            if probs_arr.shape[1] == 2:
                # Determine column corresponding to class 1
                col_idx = 1
                if hasattr(estimator, "classes_"):
                    classes = list(estimator.classes_)
                    if 1 in classes:
                        col_idx = classes.index(1)
                scores = probs_arr[:, col_idx]
            elif probs_arr.shape[1] == 1:
                scores = probs_arr[:, 0]
            else:
                raise ValueError(
                    f"Multiclass predict_proba with {probs_arr.shape[1]} classes is not supported; "
                    "binary attack classification requires 2 classes."
                )
        elif probs_arr.ndim == 1:
            scores = probs_arr
        else:
            raise ValueError(f"Unexpected predict_proba shape: {probs_arr.shape}")

    # 2. Use decision_function if available (e.g. SVC, QSVC)
    elif hasattr(estimator, "decision_function"):
        df = estimator.decision_function(X)
        df_arr = np.asarray(df, dtype=float)

        if df_arr.ndim == 2 and df_arr.shape[1] == 1:
            scores = df_arr[:, 0]
        elif df_arr.ndim == 1:
            scores = df_arr
        else:
            raise ValueError(f"Unexpected decision_function shape: {df_arr.shape}")

    else:
        raise ValueError(
            "Estimator does not implement 'predict_proba' or 'decision_function'. "
            "Hard predictions must not be used as continuous scores."
        )

    # Validate output
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("Estimator produced an empty score array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Estimator produced non-finite (NaN or Inf) scores.")

    return scores


def derive_operational_threshold(
    y_cal: np.ndarray,
    scores_cal: np.ndarray,
    target_fpr: float = 0.01,
) -> Dict[str, Any]:
    """Derive a frozen attack-score threshold from calibration negatives only.

    Order-Statistic Selection Rule:
      Let S_0 = {s_i : y_cal[i] == 0} be the set of N_0 calibration negative
      (normal flow) scores, sorted in ascending order: s_(0) <= s_(1) <= ... <= s_(N_0 - 1).
      A test sample is flagged as an attack alert if score >= threshold.
      The maximum allowed false positives on the calibration set is:
          k_max = floor(target_fpr * N_0)
      The nominal order-statistic threshold is:
          tau_nominal = s_(N_0 - k_max)   (rank N_0 - k_max + 1, 1-indexed)
      Under this threshold, exactly k_max negative samples have score >= tau_nominal
      if all negative scores are distinct, guaranteeing calibration FPR <= target_fpr.

    Conservative Tie Handling:
      If ties occur at tau_nominal such that count(s in S_0 : s >= tau_nominal) > k_max,
      the threshold is advanced to the smallest unique score u in S_0 satisfying:
          count(s in S_0 : s >= u) <= k_max
      This strictly guarantees that empirical calibration FPR <= target_fpr even
      in the presence of duplicate/tied score values.

    Assumptions and Limitations:
      1. Finite-Sample Calibration Property: On the calibration negative sample S_0 itself,
         achieved empirical calibration FPR is mathematically bounded: FPR_cal <= target_fpr.
      2. Non-Guarantee on Future / Test Data: This rule guarantees ONLY that the
         empirical calibration FPR on S_0 is <= target_fpr. It does NOT provide a
         distribution-free guarantee for future or held-out test data. Even under
         i.i.d. exchangeability, test FPR on finite samples fluctuates around its expectation.
         Under distribution shift (as present between UNSW-NB15 train/calibration and test splits),
         exchangeability is broken, and empirical test FPR may exceed target_fpr.
         Therefore, the frozen threshold must be evaluated on held-out test data and
         achieved test FPR reported explicitly alongside target calibration FPR.
      3. Tie Limitations: When duplicate scores occur at or above the candidate threshold,
         the conservative rule steps up to the smallest unique score satisfying the budget.
         If ties at the maximum score exceed k_max, no threshold in S_0 can satisfy the
         budget under the (score >= threshold) decision rule, and a ValueError is raised.

    Parameters
    ----------
    y_cal : np.ndarray
        1D binary ground-truth labels for calibration (0 = normal, 1 = attack).
    scores_cal : np.ndarray
        1D continuous scores for calibration (higher = more likely attack).
    target_fpr : float, default=0.01
        Target maximum false positive rate on normal flows (e.g. 0.01 or 0.02).
        Must be strictly in (0, 1).

    Returns
    -------
    Dict[str, Any]
        Dictionary with complete reproducibility metadata:
          - "target_fpr": float
          - "threshold": float
          - "n_cal_negatives": int
          - "allowed_false_positives": int
          - "calibration_false_positives": int
          - "calibration_fpr": float
          - "rule": str
          - "order_statistic_rank": int (1-indexed rank of selected threshold in sorted S_0)
          - "nominal_order_statistic_rank": int (1-indexed nominal rank N_0 - k_max + 1)
          - "tie_adjusted": bool

    Raises
    ------
    ValueError
        If inputs are mismatched, empty, non-finite, contain non-binary labels,
        have target_fpr not in (0, 1), have 0 normal flows, have insufficient
        negatives (N_0 < ceil(1/target_fpr)), or have excessive ties exceeding budget.
    """
    # 1. Input validations
    y_arr = np.asarray(y_cal)
    s_arr = np.asarray(scores_cal)

    if y_arr.ndim != 1 or s_arr.ndim != 1:
        if y_arr.squeeze().ndim == 1 and s_arr.squeeze().ndim == 1:
            y_arr = y_arr.squeeze()
            s_arr = s_arr.squeeze()
        else:
            raise ValueError(f"y_cal and scores_cal must be 1-dimensional, got {y_arr.shape} and {s_arr.shape}.")

    if len(y_arr) != len(s_arr):
        raise ValueError(
            f"Length mismatch: len(y_cal)={len(y_arr)} != len(scores_cal)={len(s_arr)}."
        )

    if len(y_arr) == 0:
        raise ValueError("Calibration arrays must not be empty.")

    if not np.all(np.isfinite(s_arr)):
        raise ValueError("scores_cal contains non-finite (NaN or Inf) values.")

    unique_labels = np.unique(y_arr)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"Invalid labels in y_cal: expected binary {{0, 1}}, found {unique_labels.tolist()}.")

    if not isinstance(target_fpr, (int, float)) or np.isnan(target_fpr) or not (0.0 < target_fpr < 1.0):
        raise ValueError(f"Invalid target_fpr: {target_fpr}. Must be a float strictly in (0, 1).")

    # 2. Extract normal flows (y == 0)
    normal_mask = (y_arr == 0)
    neg_scores = s_arr[normal_mask]
    n_neg = len(neg_scores)

    if n_neg == 0:
        raise ValueError("No normal-flow samples (y_cal == 0) found in calibration set.")

    k_max = int(np.floor(target_fpr * n_neg))
    min_required_negatives = int(np.ceil(1.0 / target_fpr))
    if k_max < 1:
        raise ValueError(
            f"Insufficient calibration negatives (N_0={n_neg}) for target FPR {target_fpr:.4f}. "
            f"At least {min_required_negatives} normal calibration samples are required so that "
            f"floor(target_fpr * N_0) >= 1."
        )

    # 3. Sort negative scores
    sorted_neg = np.sort(neg_scores)

    # Nominal order-statistic candidate (0-indexed)
    nominal_idx = n_neg - k_max
    threshold_candidate = float(sorted_neg[nominal_idx])
    nominal_rank = nominal_idx + 1
    rank_1_indexed = nominal_rank

    # Check ties
    fp_cal_count = int(np.sum(sorted_neg >= threshold_candidate))
    tie_adjusted = False

    if fp_cal_count > k_max:
        # Ties at threshold_candidate caused false positives to exceed budget.
        # Find smallest unique value in sorted_neg with count <= k_max.
        unique_vals = np.unique(sorted_neg)
        viable_vals = [u for u in unique_vals if np.sum(sorted_neg >= u) <= k_max]
        if viable_vals:
            threshold_candidate = float(viable_vals[0])
            fp_cal_count = int(np.sum(sorted_neg >= threshold_candidate))
            tie_adjusted = True
            # Update rank_1_indexed to reflect the actual selected threshold in sorted_neg
            rank_1_indexed = int(np.searchsorted(sorted_neg, threshold_candidate, side="left")) + 1
        else:
            raise ValueError(
                "Excessive duplicate/tied scores among calibration negatives prevent "
                f"achieving target FPR {target_fpr:.4f} without exceeding the false positive budget under decision rule (score >= threshold)."
            )

    achieved_cal_fpr = float(fp_cal_count / n_neg)

    return {
        "target_fpr": float(target_fpr),
        "threshold": float(threshold_candidate),
        "n_cal_negatives": int(n_neg),
        "allowed_false_positives": int(k_max),
        "calibration_false_positives": int(fp_cal_count),
        "calibration_fpr": round(achieved_cal_fpr, 6),
        "rule": "conservative_finite_sample_order_statistic",
        "order_statistic_rank": int(rank_1_indexed),
        "nominal_order_statistic_rank": int(nominal_rank),
        "tie_adjusted": bool(tie_adjusted),
    }


def evaluate_operational_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    frozen_threshold: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate operational cyber-threat detection metrics at a previously frozen threshold.

    Guarantees:
      - frozen_threshold is NEVER modified, recalculated, or adapted using test data.
      - Reports achieved empirical test FPR explicitly, never conflating it with target FPR.
      - Standardized partial ROC-AUC is evaluated over FPR in [0, 0.02] (McClish, 1989).

    Parameters
    ----------
    y_true : np.ndarray
        1D binary ground-truth test labels (0 = normal, 1 = attack).
    y_score : np.ndarray
        1D continuous test prediction scores (higher = more likely attack).
    frozen_threshold : float
        Previously frozen decision threshold derived on calibration data.
    metadata : Optional[Dict[str, Any]], default=None
        Optional calibration threshold metadata for audit provenance.

    Returns
    -------
    Dict[str, Any]
        Dictionary of operational metrics:
          - "frozen_threshold": float
          - "recall_at_threshold": float
          - "empirical_test_fpr": float
          - "precision_at_threshold": float
          - "f1_at_threshold": float
          - "roc_auc": float or None (None if degenerate single-class test set)
          - "standardized_pauc_02": float or None (standardized pAUC with max_fpr=0.02)
          - "confusion_matrix": {"tn": int, "fp": int, "fn": int, "tp": int}
          - "metadata": dict (if metadata provided)

    Raises
    ------
    ValueError
        If inputs are mismatched, empty, non-finite, contain non-binary labels,
        or frozen_threshold is non-finite.
    """
    y_arr = np.asarray(y_true)
    s_arr = np.asarray(y_score)

    if y_arr.ndim != 1 or s_arr.ndim != 1:
        if y_arr.squeeze().ndim == 1 and s_arr.squeeze().ndim == 1:
            y_arr = y_arr.squeeze()
            s_arr = s_arr.squeeze()
        else:
            raise ValueError(f"y_true and y_score must be 1-dimensional, got {y_arr.shape} and {s_arr.shape}.")

    if len(y_arr) != len(s_arr):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_arr)} != len(y_score)={len(s_arr)}.")

    if len(y_arr) == 0:
        raise ValueError("Test arrays must not be empty.")

    if not np.all(np.isfinite(s_arr)):
        raise ValueError("y_score contains non-finite (NaN or Inf) values.")

    unique_labels = np.unique(y_arr)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"Invalid labels in y_true: expected binary {{0, 1}}, found {unique_labels.tolist()}.")

    if not isinstance(frozen_threshold, (int, float)) or not np.isfinite(frozen_threshold):
        raise ValueError(f"frozen_threshold must be a finite float, got {frozen_threshold}.")

    # Binary prediction at frozen threshold: higher score -> attack
    y_pred = (s_arr >= frozen_threshold).astype(int)

    # Confusion matrix with explicit labels=[0, 1] for fixed 2x2 shape
    cm = confusion_matrix(y_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # Metric calculations
    n_pos = tp + fn
    n_neg = tn + fp

    recall = float(tp / n_pos) if n_pos > 0 else 0.0
    empirical_fpr = float(fp / n_neg) if n_neg > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # ROC-AUC and Standardized pAUC (max_fpr=0.02)
    # Undefined if test set is degenerate (contains only 1 class)
    roc_auc_val: Optional[float] = None
    pauc_val: Optional[float] = None

    if len(unique_labels) == 2:
        try:
            roc_auc_val = round(float(roc_auc_score(y_arr, s_arr)), 4)
        except Exception as exc:
            logger.warning("ROC-AUC computation failed: %s", exc)
            roc_auc_val = None

        try:
            pauc_val = round(float(roc_auc_score(y_arr, s_arr, max_fpr=0.02)), 4)
        except Exception as exc:
            logger.warning("Standardized pAUC computation failed: %s", exc)
            pauc_val = None
    else:
        logger.warning(
            "Degenerate label set with unique labels %s; ROC-AUC and standardized pAUC are undefined.",
            unique_labels.tolist(),
        )

    results: Dict[str, Any] = {
        "frozen_threshold": float(frozen_threshold),
        "recall_at_threshold": round(recall, 4),
        "empirical_test_fpr": round(empirical_fpr, 4),
        "precision_at_threshold": round(precision, 4),
        "f1_at_threshold": round(f1, 4),
        "roc_auc": roc_auc_val,
        "standardized_pauc_02": pauc_val,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }

    if metadata is not None:
        results["calibration_metadata"] = metadata

    return results
