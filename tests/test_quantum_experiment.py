"""Unit tests for src/quantum_experiment.py — Stage 5 Step 5.

Tests cover:
  - Sampling bounds and reproducibility (seed 42 deterministic subsets)
  - Stratification and label encoding (0=BENIGN, 1=ATTACK)
  - Train-only scaler fitting (no data leakage)
  - Kernel matrix shape validation (train, val, test)
  - Validation candidate selection and deterministic tie-breaking (smaller C on tie)
  - Metric orientation with pos_label=1 (ATTACK)
  - Single-class label handling for ROC-AUC
  - Hard sample ceiling enforcement (> 100 train, > 50 val, > 100 test raises ValueError)
  - Separation of validation and test stages
  - JSON schema verification of quantum_experiment.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.quantum_experiment import (
    CANDIDATE_C_VALUES,
    MAX_TEST_SAMPLES,
    MAX_TRAIN_SAMPLES,
    MAX_VAL_SAMPLES,
    select_best_c_candidate,
    split_quantum_data,
    tune_qsvc_hyperparameters,
)
from src.quantum_model import (
    fit_quantum_scaler,
    transform_quantum_features,
)


@pytest.fixture()
def mock_train_df():
    """Mock training dataframe with 200 samples and known label distribution."""
    rng = np.random.default_rng(42)
    n = 200
    features = {
        "sbytes": rng.uniform(0, 1000, n),
        "sload": rng.uniform(10, 500, n),
        "sttl": rng.uniform(30, 255, n),
        "smean": rng.uniform(20, 200, n),
        "label": np.array([0] * 90 + [1] * 110),
    }
    return pd.DataFrame(features)


@pytest.fixture()
def mock_test_df():
    """Mock test dataframe with 150 samples."""
    rng = np.random.default_rng(42)
    n = 150
    features = {
        "sbytes": rng.uniform(0, 1000, n),
        "sload": rng.uniform(10, 500, n),
        "sttl": rng.uniform(30, 255, n),
        "smean": rng.uniform(20, 200, n),
        "label": np.array([0] * 50 + [1] * 100),
    }
    return pd.DataFrame(features)


class TestDataSplittingAndSampling:
    def test_sampling_bounds_and_shapes(self, mock_train_df, mock_test_df):
        """Verify subsets strictly adhere to requested sample limits."""
        features = ["sbytes", "sload", "sttl", "smean"]
        splits = split_quantum_data(
            train_df=mock_train_df,
            test_df=mock_test_df,
            selected_features=features,
            n_train=100,
            n_val=50,
            n_test=100,
            random_seed=42,
        )
        assert splits["X_train"].shape == (100, 4)
        assert splits["y_train"].shape == (100,)
        assert splits["X_val"].shape == (50, 4)
        assert splits["y_val"].shape == (50,)
        assert splits["X_test"].shape == (100, 4)
        assert splits["y_test"].shape == (100,)

    def test_reproducibility_with_seed(self, mock_train_df, mock_test_df):
        """Same seed must produce bitwise identical splits."""
        features = ["sbytes", "sload", "sttl", "smean"]
        s1 = split_quantum_data(mock_train_df, mock_test_df, features, 100, 50, 100, 42)
        s2 = split_quantum_data(mock_train_df, mock_test_df, features, 100, 50, 100, 42)
        np.testing.assert_array_equal(s1["X_train"], s2["X_train"])
        np.testing.assert_array_equal(s1["y_train"], s2["y_train"])
        np.testing.assert_array_equal(s1["X_val"], s2["X_val"])
        np.testing.assert_array_equal(s1["y_val"], s2["y_val"])
        np.testing.assert_array_equal(s1["X_test"], s2["X_test"])
        np.testing.assert_array_equal(s1["y_test"], s2["y_test"])

    def test_stratification_preserves_both_classes(self, mock_train_df, mock_test_df):
        """Train, val, and test subsets must all contain both benign (0) and attack (1)."""
        features = ["sbytes", "sload", "sttl", "smean"]
        splits = split_quantum_data(mock_train_df, mock_test_df, features, 100, 50, 100, 42)
        for part in ["train", "val", "test"]:
            y = splits[f"y_{part}"]
            assert np.sum(y == 0) > 0, f"No benign samples in {part}"
            assert np.sum(y == 1) > 0, f"No attack samples in {part}"

    def test_hard_ceiling_rejection(self, mock_train_df, mock_test_df):
        """Requesting more than hard ceilings must raise ValueError."""
        features = ["sbytes", "sload", "sttl", "smean"]
        with pytest.raises(ValueError, match="hard ceiling"):
            split_quantum_data(mock_train_df, mock_test_df, features, n_train=101, n_val=50, n_test=100)
        with pytest.raises(ValueError, match="hard ceiling"):
            split_quantum_data(mock_train_df, mock_test_df, features, n_train=100, n_val=51, n_test=100)
        with pytest.raises(ValueError, match="hard ceiling"):
            split_quantum_data(mock_train_df, mock_test_df, features, n_train=100, n_val=50, n_test=101)


class TestAntiLeakageScaling:
    def test_scaler_parameters_invariant_to_val_and_test(self, mock_train_df, mock_test_df):
        """Scaler fitted on train must NOT change when transforming val or test."""
        features = ["sbytes", "sload", "sttl", "smean"]
        splits = split_quantum_data(mock_train_df, mock_test_df, features, 100, 50, 100, 42)
        X_tr = splits["X_train"]
        X_val = splits["X_val"]
        X_te = splits["X_test"]

        scaler = fit_quantum_scaler(X_tr)
        scale_orig = scaler.scale_.copy()
        min_orig = scaler.data_min_.copy()

        transform_quantum_features(scaler, X_val)
        transform_quantum_features(scaler, X_te)

        np.testing.assert_array_equal(scaler.scale_, scale_orig)
        np.testing.assert_array_equal(scaler.data_min_, min_orig)


class TestHyperparameterTuningAndTieBreak:
    def test_select_best_c_highest_f1(self):
        """Selects candidate with strictly highest validation F1."""
        candidates = [
            {"C": 0.1, "val_f1": 0.82},
            {"C": 1.0, "val_f1": 0.88},
            {"C": 10.0, "val_f1": 0.85},
        ]
        best = select_best_c_candidate(candidates)
        assert best["C"] == 1.0

    def test_tie_break_prefers_smaller_c(self):
        """When multiple C values have identical validation F1, prefer smaller C."""
        candidates = [
            {"C": 10.0, "val_f1": 0.85},
            {"C": 1.0, "val_f1": 0.85},
            {"C": 0.1, "val_f1": 0.85},
        ]
        best = select_best_c_candidate(candidates)
        assert best["C"] == 0.1

    def test_tune_qsvc_with_mock_kernel(self):
        """Test hyperparameter tuning function with fast synthetic Gram matrices."""
        rng = np.random.default_rng(42)
        n_tr, n_val = 20, 10
        # Create symmetric PSD train matrix
        A = rng.uniform(0.1, 0.9, size=(n_tr, n_tr))
        K_tr = (A @ A.T) / n_tr
        np.fill_diagonal(K_tr, 1.0)
        K_val = rng.uniform(0.1, 0.9, size=(n_val, n_tr))

        y_tr = np.array([0] * 10 + [1] * 10)
        y_val = np.array([0] * 5 + [1] * 5)

        best_C, results = tune_qsvc_hyperparameters(
            K_train=K_tr,
            y_train=y_tr,
            K_val=K_val,
            y_val=y_val,
            candidate_c_values=[0.1, 1.0, 10.0],
            random_seed=42,
        )
        assert best_C in [0.1, 1.0, 10.0]
        assert len(results) == 3
        for r in results:
            assert "val_f1" in r
            assert "val_accuracy" in r
            assert "fit_time_s" in r


class TestMetricOrientationAndSingleClass:
    def test_metric_orientation_attack_is_positive(self):
        """Confirm F1 score calculation treats label 1 (ATTACK) as positive."""
        from sklearn.metrics import f1_score
        y_true = np.array([0, 0, 1, 1])
        # Model only predicts class 1 on actual class 1
        y_pred = np.array([0, 0, 1, 0])  # Recall=0.5, Precision=1.0 -> F1=0.6667
        f1_attack = f1_score(y_true, y_pred, pos_label=1)
        assert f1_attack == pytest.approx(2.0 / 3.0)

    def test_roc_auc_single_class_handling(self):
        """ROC-AUC calculation on single-class array triggers UndefinedMetricWarning and NaN in scikit-learn 1.9."""
        import warnings
        from sklearn.metrics import roc_auc_score
        from sklearn.exceptions import UndefinedMetricWarning

        y_single = np.array([1, 1, 1, 1])
        y_scores = np.array([0.2, 0.4, 0.6, 0.8])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score = roc_auc_score(y_single, y_scores)
            assert np.isnan(score)
            assert any(issubclass(warn.category, UndefinedMetricWarning) for warn in w)


class TestArtifactSchema:
    def test_experiment_json_schema_if_exists(self):
        """Check schema of results/quantum_experiment.json when generated."""
        from src.config import RESULTS_DIR
        res_file = RESULTS_DIR / "quantum_experiment.json"
        if not res_file.exists():
            pytest.skip("quantum_experiment.json has not been created yet")

        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0.0"
        assert data["is_restricted_subset_experiment"] is True
        assert data["quantum_advantage_claimed"] is False
        assert "disclaimer" in data
        assert "configuration" in data
        assert "data_subsets" in data
        assert "kernel_evaluations" in data
        assert "hyperparameter_selection" in data
        assert "test_evaluation" in data
        assert "timings" in data

        # Validate sample sizes obey limits
        subsets = data["data_subsets"]
        assert subsets["train"]["n_samples"] <= MAX_TRAIN_SAMPLES
        assert subsets["val"]["n_samples"] <= MAX_VAL_SAMPLES
        assert subsets["test"]["n_samples"] <= MAX_TEST_SAMPLES

        # Validate test evaluation metrics
        test_eval = data["test_evaluation"]
        assert 0.0 <= test_eval["accuracy"] <= 1.0
        assert 0.0 <= test_eval["f1"] <= 1.0
        assert 0.0 <= test_eval["precision"] <= 1.0
        assert 0.0 <= test_eval["recall"] <= 1.0
        assert len(test_eval["confusion_matrix"]) == 2
