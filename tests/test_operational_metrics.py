"""Unit tests for Stage 8B Operational Metrics and Leakage-Safe Thresholding.

Covers:
  - Known threshold example with expected selected order statistic
  - 1% and 2% target FPR handling
  - Tie handling at candidate threshold and guaranteed FPR behavior
  - Empty calibration negatives validation
  - Invalid labels, NaN/Inf scores, mismatched lengths, and invalid target FPR
  - Correct confusion matrix, recall, FPR, precision, F1, ROC-AUC, and standardized pAUC
  - Score orientation for RF, SVM, and QSVC test doubles
  - Proof of frozen threshold immutability across evaluation passes
"""

import numpy as np
import pytest

from src.operational_metrics import (
    derive_operational_threshold,
    evaluate_operational_metrics,
    extract_attack_scores,
)


# =====================================================================
# 1. Known Threshold & Order-Statistic Verification
# =====================================================================

def test_known_threshold_order_statistic():
    """Verify exact order statistic formula on a deterministic sequence.

    With N_0 = 100 negatives with scores 1, 2, ..., 100:
      target_fpr = 0.02 -> allowed_fp = floor(0.02 * 100) = 2.
      Nominal index = 100 - 2 = 98 (0-indexed). Score is 99.0.
      1-indexed order statistic rank = 99.
      Values >= 99.0 are {99.0, 100.0} (exactly 2 negatives).
      Empirical calibration FPR = 2 / 100 = 0.02 (2.0%).
    """
    scores_neg = np.arange(1.0, 101.0, 1.0)  # 1.0 to 100.0
    labels = np.zeros(100, dtype=int)

    meta = derive_operational_threshold(labels, scores_neg, target_fpr=0.02)

    assert meta["target_fpr"] == 0.02
    assert meta["n_cal_negatives"] == 100
    assert meta["allowed_false_positives"] == 2
    assert meta["calibration_false_positives"] == 2
    assert meta["calibration_fpr"] == 0.02
    assert meta["threshold"] == 99.0
    assert meta["order_statistic_rank"] == 99
    assert meta["tie_adjusted"] is False
    assert meta["rule"] == "conservative_finite_sample_order_statistic"


# =====================================================================
# 2. Target FPR 1% and 2% Handling
# =====================================================================

def test_target_fpr_1_and_2_percent():
    """Verify behavior under standard operational budgets of 1% and 2%.

    With N_0 = 200 distinct scores (0.005 to 1.0):
      target 1%: allowed FP = floor(0.01 * 200) = 2 -> threshold rank 199.
      target 2%: allowed FP = floor(0.02 * 200) = 4 -> threshold rank 197.
      Stricter budget (1%) must result in a higher alert threshold.
    """
    scores = np.linspace(0.005, 1.0, 200)
    labels = np.zeros(200, dtype=int)

    meta_1 = derive_operational_threshold(labels, scores, target_fpr=0.01)
    meta_2 = derive_operational_threshold(labels, scores, target_fpr=0.02)

    assert meta_1["allowed_false_positives"] == 2
    assert meta_1["calibration_fpr"] == 0.01
    assert meta_1["threshold"] == pytest.approx(scores[198])

    assert meta_2["allowed_false_positives"] == 4
    assert meta_2["calibration_fpr"] == 0.02
    assert meta_2["threshold"] == pytest.approx(scores[196])

    # Stricter false positive budget requires higher attack score threshold
    assert meta_1["threshold"] > meta_2["threshold"]


# =====================================================================
# 3. Tie Handling at Candidate Threshold
# =====================================================================

def test_ties_at_threshold_handling():
    """Verify conservative stepping when duplicate scores occur near threshold.

    Scores: [1, 2, 3, 4, 5, 6, 7, 8, 8, 9] (N_0 = 10).
    target_fpr = 0.20 -> allowed FP = floor(0.20 * 10) = 2.
    Nominal index = 10 - 2 = 8 -> candidate is 8.0.
    However, elements >= 8.0 are indices 7, 8, 9 (values 8, 8, 9; count = 3).
    Count 3 > allowed 2 -> FPR = 30% > 20%.
    Conservative rule must step up to next unique value 9.0 (count = 1 <= 2).
    """
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0])
    labels = np.zeros(10, dtype=int)

    meta = derive_operational_threshold(labels, scores, target_fpr=0.20)

    assert meta["threshold"] == 9.0
    assert meta["tie_adjusted"] is True
    assert meta["calibration_false_positives"] == 1
    assert meta["calibration_fpr"] == 0.10  # 10% <= 20%
    assert meta["order_statistic_rank"] == 10  # Rank of selected threshold 9.0
    assert meta["nominal_order_statistic_rank"] == 9  # Rank before tie stepping


def test_ties_at_maximum_score_exceeding_budget():
    """Verify ValueError when top scores are duplicate and exceed the FP budget.

    Scores: [1, 2, 3, 4, 5, 6, 7, 9, 9, 9] (N_0 = 10).
    target_fpr = 0.20 -> allowed FP = floor(0.20 * 10) = 2.
    Maximum score 9.0 has multiplicity 3.
    Under score >= threshold, any threshold in S_0 produces at least 3 false positives (30% > 20%).
    Therefore no threshold in S_0 can satisfy the budget, and ValueError must be raised.
    """
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 9.0, 9.0])
    labels = np.zeros(10, dtype=int)

    with pytest.raises(ValueError, match="Excessive duplicate/tied scores"):
        derive_operational_threshold(labels, scores, target_fpr=0.20)


def test_excessive_ties_raises_error():
    """Verify that if all values are identical exceeding budget, error is raised."""
    # 10 negatives all with score 1.0. target_fpr=0.10 allows 1 FP.
    # But any threshold <= 1.0 yields 10 FPs.
    scores = np.ones(10)
    labels = np.zeros(10, dtype=int)

    with pytest.raises(ValueError, match="Excessive duplicate/tied scores"):
        derive_operational_threshold(labels, scores, target_fpr=0.10)


def test_exact_boundary_conditions_k_max():
    """Verify exact sample size boundary conditions for k_max = floor(alpha * N_0).

    For alpha = 0.02 (min N_0 = 50):
      - N_0 = 49 -> floor(0.02 * 49) = 0 -> ValueError (k_max < 1).
      - N_0 = 50 -> floor(0.02 * 50) = 1 -> Allowed FP = 1, nominal index = 49, rank = 50.
    For alpha = 0.01 (min N_0 = 100):
      - N_0 = 99 -> floor(0.01 * 99) = 0 -> ValueError (k_max < 1).
      - N_0 = 100 -> floor(0.01 * 100) = 1 -> Allowed FP = 1, nominal index = 99, rank = 100.
    """
    # Boundary at alpha = 0.02
    scores_49 = np.linspace(0.1, 1.0, 49)
    labels_49 = np.zeros(49, dtype=int)
    with pytest.raises(ValueError, match="Insufficient calibration negatives"):
        derive_operational_threshold(labels_49, scores_49, target_fpr=0.02)

    scores_50 = np.linspace(0.1, 1.0, 50)
    labels_50 = np.zeros(50, dtype=int)
    meta_50 = derive_operational_threshold(labels_50, scores_50, target_fpr=0.02)
    assert meta_50["allowed_false_positives"] == 1
    assert meta_50["order_statistic_rank"] == 50
    assert meta_50["threshold"] == pytest.approx(scores_50[49])

    # Boundary at alpha = 0.01
    scores_99 = np.linspace(0.1, 1.0, 99)
    labels_99 = np.zeros(99, dtype=int)
    with pytest.raises(ValueError, match="Insufficient calibration negatives"):
        derive_operational_threshold(labels_99, scores_99, target_fpr=0.01)

    scores_100 = np.linspace(0.1, 1.0, 100)
    labels_100 = np.zeros(100, dtype=int)
    meta_100 = derive_operational_threshold(labels_100, scores_100, target_fpr=0.01)
    assert meta_100["allowed_false_positives"] == 1
    assert meta_100["order_statistic_rank"] == 100
    assert meta_100["threshold"] == pytest.approx(scores_100[99])


# =====================================================================
# 4. Empty Calibration Negatives & Validation Errors
# =====================================================================

def test_empty_calibration_negatives():
    """Verify error when calibration set has no normal flows (y == 0)."""
    # Only attacks (y == 1)
    labels = np.ones(20, dtype=int)
    scores = np.random.RandomState(42).rand(20)

    with pytest.raises(ValueError, match="No normal-flow samples"):
        derive_operational_threshold(labels, scores, target_fpr=0.01)

    # Empty array
    with pytest.raises(ValueError, match="empty"):
        derive_operational_threshold(np.array([]), np.array([]), target_fpr=0.01)


def test_validation_errors_derive_threshold():
    """Verify comprehensive input validation in derive_operational_threshold."""
    valid_labels = np.zeros(100, dtype=int)
    valid_scores = np.linspace(0, 1, 100)

    # Invalid labels (non-binary)
    bad_labels = valid_labels.copy()
    bad_labels[0] = 2
    with pytest.raises(ValueError, match="Invalid labels"):
        derive_operational_threshold(bad_labels, valid_scores, target_fpr=0.01)

    # NaN in scores
    bad_scores_nan = valid_scores.copy()
    bad_scores_nan[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        derive_operational_threshold(valid_labels, bad_scores_nan, target_fpr=0.01)

    # Inf in scores
    bad_scores_inf = valid_scores.copy()
    bad_scores_inf[5] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        derive_operational_threshold(valid_labels, bad_scores_inf, target_fpr=0.01)

    # Mismatched lengths
    with pytest.raises(ValueError, match="Length mismatch"):
        derive_operational_threshold(valid_labels, valid_scores[:50], target_fpr=0.01)

    # Invalid target_fpr
    for bad_fpr in [0.0, -0.05, 1.0, 1.5, np.nan, "0.01"]:
        with pytest.raises(ValueError, match="Invalid target_fpr"):
            derive_operational_threshold(valid_labels, valid_scores, target_fpr=bad_fpr)  # type: ignore

    # Insufficient negatives (N_0 = 50 for target_fpr = 0.01 -> floor(0.5) = 0)
    small_labels = np.zeros(50, dtype=int)
    small_scores = np.linspace(0, 1, 50)
    with pytest.raises(ValueError, match="Insufficient calibration negatives"):
        derive_operational_threshold(small_labels, small_scores, target_fpr=0.01)


# =====================================================================
# 5. Operational Evaluation Metrics Verification
# =====================================================================

def test_evaluation_metrics_correctness():
    """Verify accuracy of operational evaluation outputs on known synthetic counts."""
    # 50 normal flows (y=0): 45 with score 0.2, 5 with score 0.8
    # 50 attack flows (y=1): 10 with score 0.2, 40 with score 0.8
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.array([0.2] * 45 + [0.8] * 5 + [0.2] * 10 + [0.8] * 40)

    # Frozen threshold = 0.5
    # Normal flows: 45 score < 0.5 (TN), 5 score >= 0.5 (FP)
    # Attack flows: 10 score < 0.5 (FN), 40 score >= 0.5 (TP)
    res = evaluate_operational_metrics(y_true, y_score, frozen_threshold=0.5)

    assert res["frozen_threshold"] == 0.5
    assert res["confusion_matrix"] == {"tn": 45, "fp": 5, "fn": 10, "tp": 40}

    # Recall = TP / (TP + FN) = 40 / 50 = 0.8000
    assert res["recall_at_threshold"] == 0.8000

    # Empirical test FPR = FP / (FP + TN) = 5 / 50 = 0.1000
    assert res["empirical_test_fpr"] == 0.1000

    # Precision = TP / (TP + FP) = 40 / 45 = 0.8889
    assert res["precision_at_threshold"] == 0.8889

    # F1 = 2 * (0.88888 * 0.8) / (0.88888 + 0.8) = 0.8421
    assert res["f1_at_threshold"] == 0.8421

    # ROC-AUC and Standardized pAUC should be computed and non-None
    assert res["roc_auc"] is not None
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert res["standardized_pauc_02"] is not None
    assert 0.0 <= res["standardized_pauc_02"] <= 1.0


def test_standardized_pauc_scikit_learn_mcclish_value():
    """Verify exact numerical agreement with scikit-learn's McClish (1989) standardized pAUC API."""
    from sklearn.metrics import roc_auc_score
    y_true = np.array([0] * 100 + [1] * 100)
    y_score = np.random.RandomState(42).randn(200)

    res = evaluate_operational_metrics(y_true, y_score, frozen_threshold=0.0)
    expected_pauc = round(float(roc_auc_score(y_true, y_score, max_fpr=0.02)), 4)
    expected_auc = round(float(roc_auc_score(y_true, y_score)), 4)

    assert res["standardized_pauc_02"] == expected_pauc
    assert res["roc_auc"] == expected_auc
    assert 0.0 <= res["standardized_pauc_02"] <= 1.0


def test_evaluation_degenerate_labels():
    """Verify graceful handling when test labels contain only one class."""
    # Only negatives
    y_all_zero = np.zeros(20, dtype=int)
    scores = np.linspace(0.1, 0.9, 20)
    res_zero = evaluate_operational_metrics(y_all_zero, scores, frozen_threshold=0.5)

    assert res_zero["recall_at_threshold"] == 0.0
    assert res_zero["roc_auc"] is None
    assert res_zero["standardized_pauc_02"] is None
    assert res_zero["confusion_matrix"]["tp"] == 0

    # Only positives
    y_all_one = np.ones(20, dtype=int)
    res_one = evaluate_operational_metrics(y_all_one, scores, frozen_threshold=0.5)

    assert res_one["empirical_test_fpr"] == 0.0
    assert res_one["roc_auc"] is None
    assert res_one["standardized_pauc_02"] is None
    assert res_one["confusion_matrix"]["tn"] == 0


def test_evaluation_validation_errors():
    """Verify input validation in evaluate_operational_metrics."""
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])

    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_operational_metrics(y, s[:2], frozen_threshold=0.5)

    with pytest.raises(ValueError, match="empty"):
        evaluate_operational_metrics(np.array([]), np.array([]), frozen_threshold=0.5)

    with pytest.raises(ValueError, match="non-finite"):
        evaluate_operational_metrics(y, np.array([0.1, np.nan, 0.2, 0.8]), frozen_threshold=0.5)

    with pytest.raises(ValueError, match="Invalid labels"):
        evaluate_operational_metrics(np.array([0, 2, 0, 1]), s, frozen_threshold=0.5)

    with pytest.raises(ValueError, match="finite float"):
        evaluate_operational_metrics(y, s, frozen_threshold=np.nan)


# =====================================================================
# 6. Score Orientation & Model Adapters (RF, SVM, QSVC)
# =====================================================================

class DummyRandomForest:
    """Lightweight test double for RandomForestClassifier."""
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        # Return probability matrix: [[P(y=0), P(y=1)], ...]
        # For test input X in [0, 1], P(y=1) = X
        X = np.asarray(X).ravel()
        return np.column_stack([1.0 - X, X])


class DummySVM:
    """Lightweight test double for SVC with RBF kernel."""
    def decision_function(self, X):
        # Signed margin: positive indicates class 1 (attack)
        return np.asarray(X).ravel() - 0.5


class DummyQSVC:
    """Lightweight test double for Quantum SVC with precomputed kernel."""
    def decision_function(self, K):
        # Signed margin on kernel evaluations
        return np.asarray(K).ravel() * 2.0 - 1.0


class DummyHardPredictOnly:
    """Invalid test double providing only hard predictions."""
    def predict(self, X):
        return np.zeros(len(X))


def test_score_orientation_rf():
    """Verify RF scores extract class 1 probability."""
    rf = DummyRandomForest()
    X = np.array([0.1, 0.7, 0.9])
    scores = extract_attack_scores(rf, X)

    # Class 1 probabilities should match X
    np.testing.assert_allclose(scores, [0.1, 0.7, 0.9])
    assert scores[2] > scores[0]  # Higher score = more likely attack


def test_score_orientation_svm():
    """Verify SVM scores extract signed margin."""
    svm = DummySVM()
    X = np.array([0.1, 0.5, 0.9])
    scores = extract_attack_scores(svm, X)

    # [0.1-0.5, 0.5-0.5, 0.9-0.5] = [-0.4, 0.0, 0.4]
    np.testing.assert_allclose(scores, [-0.4, 0.0, 0.4])
    assert scores[2] > scores[0]  # Higher score = more likely attack


def test_score_orientation_qsvc():
    """Verify QSVC scores extract signed decision margin."""
    qsvc = DummyQSVC()
    K = np.array([0.1, 0.5, 0.9])
    scores = extract_attack_scores(qsvc, K)

    np.testing.assert_allclose(scores, [-0.8, 0.0, 0.8])
    assert scores[2] > scores[0]  # Higher score = more likely attack


def test_score_adapter_rejects_hard_predictions():
    """Verify that models with only hard predictions are rejected."""
    bad_model = DummyHardPredictOnly()
    with pytest.raises(ValueError, match="Hard predictions must not be used"):
        extract_attack_scores(bad_model, np.array([1, 2, 3]))


def test_score_adapter_rejects_none_and_nan():
    """Verify adapter rejects None and non-finite outputs."""
    with pytest.raises(ValueError, match="Estimator cannot be None"):
        extract_attack_scores(None, np.array([1]))

    class DummyNaNModel:
        def decision_function(self, X):
            return np.array([0.1, np.nan])

    with pytest.raises(ValueError, match="non-finite"):
        extract_attack_scores(DummyNaNModel(), np.array([1, 2]))


# =====================================================================
# 7. Proof of Frozen Threshold Immutability
# =====================================================================

def test_frozen_threshold_immutability():
    """Verify that evaluate_operational_metrics NEVER alters or adapts the threshold."""
    frozen_tau = 0.8250
    input_tau_copy = float(frozen_tau)

    # Test Set 1: Baseline distribution (negatives have lower scores)
    y_test_1 = np.array([0] * 50 + [1] * 50)
    # Negatives in [0.1, 0.6], Positives in [0.7, 1.0]
    s_test_1 = np.concatenate([np.linspace(0.1, 0.6, 50), np.linspace(0.7, 1.0, 50)])
    eval_1 = evaluate_operational_metrics(y_test_1, s_test_1, frozen_threshold=frozen_tau)

    # Test Set 2: Extreme distribution shift (negatives have inflated scores crossing threshold)
    y_test_2 = np.array([0] * 50 + [1] * 50)
    # Negatives in [0.4, 0.95] (some exceed 0.8250), Positives in [0.85, 1.2]
    s_test_2 = np.concatenate([np.linspace(0.4, 0.95, 50), np.linspace(0.85, 1.2, 50)])
    eval_2 = evaluate_operational_metrics(y_test_2, s_test_2, frozen_threshold=frozen_tau)

    # Threshold passed in must be strictly identical
    assert frozen_tau == input_tau_copy
    assert eval_1["frozen_threshold"] == 0.8250
    assert eval_2["frozen_threshold"] == 0.8250

    # Operational metrics must reflect test distribution shift without adapting threshold
    assert eval_1["empirical_test_fpr"] != eval_2["empirical_test_fpr"]
    assert eval_1["recall_at_threshold"] != eval_2["recall_at_threshold"]
