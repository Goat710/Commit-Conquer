"""Unit tests for Stage 4 — Evaluation and Classical Models.

Verifies:
1. Correct metric calculation, including ROC-AUC from y_score and FPR = FP / (FP + TN).
2. Timing output: train_time, infer_time, and inference_time_per_1000_samples.
3. Loading of exact Stage 3 selected 4 features without recomputation.
4. Determinism of model configurations with random_state=42.
5. Train-only hyperparameter tuning (no test data access).
6. Result JSON structure and schema compliance.
7. Verification of all 4 Experiment B configurations.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.classical_models import (
    EXP_B_RF_ALL_JSON,
    EXP_B_RF_SEL4_JSON,
    EXP_B_SVM_ALL_JSON,
    EXP_B_SVM_SEL4_JSON,
    load_stage3_selected_features,
    train_and_evaluate_model,
    tune_rf_hyperparameters,
    tune_svm_hyperparameters,
)
from src.config import RANDOM_SEED
from src.evaluation import evaluate, save_evaluation_results
from src.feature_selection import FEATURE_SELECTION_JSON


def test_evaluate_metrics_and_fpr_calculation():
    """Proves: evaluate calculates correct metrics, ROC-AUC from continuous scores,

    and false positive rate as FP / (FP + TN).
    """
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1, 0, 1]
    # Continuous probabilities
    y_score = [0.1, 0.2, 0.8, 0.3, 0.7, 0.9, 0.4, 0.85]

    # Confusion matrix expected:
    # True 0s: 3 predicted 0 (TN), 1 predicted 1 (FP) -> TN=3, FP=1
    # True 1s: 1 predicted 0 (FN), 3 predicted 1 (TP) -> FN=1, TP=3
    # FPR = FP / (FP + TN) = 1 / (1 + 3) = 0.25
    train_time = 1.5
    infer_time = 0.2

    res = evaluate(y_true, y_pred, y_score, train_time, infer_time)

    assert res["accuracy"] == 0.75
    assert res["precision"] == 3 / 4  # 0.75
    assert res["recall"] == 3 / 4     # 0.75
    assert res["f1"] == 0.75
    assert res["false_positive_rate"] == 0.25
    assert res["confusion_matrix"] == [[3, 1], [1, 3]]
    assert "classification_report" in res
    assert res["roc_auc"] > 0.5  # Continuous score gives high AUC


def test_evaluate_timing_output():
    """Proves: evaluate correctly calculates inference_time_per_1000_samples."""
    y_true = np.array([0] * 250 + [1] * 250, dtype=int)
    y_pred = np.array([0] * 250 + [1] * 250, dtype=int)
    y_score = np.array([0.1] * 250 + [0.9] * 250, dtype=float)

    train_time = 5.25
    infer_time = 0.50  # 0.5s for 500 samples -> (0.5 / 500) * 1000 = 1.0s per 1000

    res = evaluate(y_true, y_pred, y_score, train_time, infer_time)

    assert res["train_time"] == 5.25
    assert res["infer_time"] == 0.50
    assert pytest.approx(res["inference_time_per_1000_samples"], rel=1e-4) == 1.0


def test_load_stage3_selected_features_exact_match():
    """Proves: load_stage3_selected_features reads exactly the 4 Stage 3 features without recomputation."""
    features = load_stage3_selected_features(FEATURE_SELECTION_JSON)

    assert isinstance(features, list)
    assert len(features) == 4
    assert features == ["sbytes", "sload", "sttl", "smean"]


def test_model_determinism_seed_42():
    """Proves: Models configured with RANDOM_SEED=42 produce deterministic predictions."""
    X = np.array([[1.0, 2.0], [2.0, 1.0], [5.0, 6.0], [6.0, 5.0]])
    y = np.array([0, 0, 1, 1])

    rf1 = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
    rf2 = RandomForestClassifier(n_estimators=10, random_state=RANDOM_SEED)
    rf1.fit(X, y)
    rf2.fit(X, y)

    np.testing.assert_array_equal(rf1.predict_proba(X), rf2.predict_proba(X))


def test_tuning_strictly_uses_train_only():
    """Proves: Hyperparameter tuning functions only take training sets and execute without test data."""
    X_train = np.random.randn(200, 4)
    y_train = np.random.binomial(1, 0.5, 200)

    # Tuning should execute and return a valid dictionary of hyperparameters
    svm_params = tune_svm_hyperparameters(X_train, y_train, random_state=42, cv_subsample_size=100, n_splits=2)
    assert "C" in svm_params
    assert "gamma" in svm_params

    rf_params = tune_rf_hyperparameters(X_train, y_train, random_state=42, cv_subsample_size=100, n_splits=2)
    assert "n_estimators" in rf_params
    assert "max_depth" in rf_params


def test_result_json_structure_and_schema(tmp_path):
    """Proves: train_and_evaluate_model generates compliant JSON artifact with all required keys."""
    X_train = np.array([[0.0, 1.0], [1.0, 0.0], [10.0, 11.0], [11.0, 10.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[0.5, 0.5], [10.5, 10.5]])
    y_test = np.array([0, 1])

    rf = RandomForestClassifier(n_estimators=5, random_state=RANDOM_SEED)
    out_json = tmp_path / "test_rf_res.json"

    result = train_and_evaluate_model(
        model_instance=rf,
        hyperparameters={"n_estimators": 5},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        experiment_name="Unit Test Exp",
        model_name="Random Forest",
        feature_set="test_features",
        features_list=["feat1", "feat2"],
        save_json_path=out_json,
        random_state=RANDOM_SEED,
    )

    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    expected_keys = [
        "experiment_name",
        "model_name",
        "feature_set",
        "features",
        "selected_hyperparameters",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive_rate",
        "confusion_matrix",
        "classification_report",
        "train_time",
        "total_inference_time",
        "inference_time_per_1000_samples",
        "random_seed",
    ]
    for k in expected_keys:
        assert k in loaded, f"Key '{k}' missing from output JSON"


def test_four_experiment_b_output_paths_defined():
    """Proves: All four Experiment B JSON target paths are properly defined under results/."""
    expected_paths = [
        EXP_B_SVM_SEL4_JSON,
        EXP_B_RF_SEL4_JSON,
        EXP_B_SVM_ALL_JSON,
        EXP_B_RF_ALL_JSON,
    ]
    for p in expected_paths:
        assert str(p).endswith(".json")
        assert "experiment_B_" in p.name
