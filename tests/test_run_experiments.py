"""Unit tests for src/run_experiments.py — Stage 6 Unified Experiments & Stability.

Tests cover:
  - Stratified data subsetting, reproducibility, and sample counts
  - Anti-leakage isolation between train and test partitions
  - Classical SVM evaluation and score output
  - Classical Random Forest evaluation and score output
  - QSVC evaluation wrapper (mocked for fast unit testing)
  - Metrics aggregation (mean and sample standard deviation)
  - Stage 4 reference loading and schema
  - Predictions CSV schema and alignment (when generated)
  - Unified results JSON schema (when generated)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.run_experiments import (
    DEFAULT_N_TEST,
    DEFAULT_N_TRAIN,
    DEFAULT_SEEDS,
    compute_metrics_aggregation,
    evaluate_classical_rf,
    evaluate_classical_svm,
    get_stratified_subsets,
    load_feature_sets,
    load_stage4_baselines,
)


@pytest.fixture()
def mock_dfs():
    """Mock train and test DataFrames with 4 features."""
    rng = np.random.default_rng(42)
    features = ["sbytes", "sload", "sttl", "smean"]

    n_tr = 200
    df_tr = pd.DataFrame({
        "id": range(1, n_tr + 1),
        "sbytes": rng.uniform(0, 1000, n_tr),
        "sload": rng.uniform(10, 500, n_tr),
        "sttl": rng.uniform(30, 255, n_tr),
        "smean": rng.uniform(20, 200, n_tr),
        "attack_cat": ["Normal"] * 90 + ["Generic"] * 110,
        "label": [0] * 90 + [1] * 110,
    })

    n_te = 150
    df_te = pd.DataFrame({
        "id": range(1001, 1001 + n_te),
        "sbytes": rng.uniform(0, 1000, n_te),
        "sload": rng.uniform(10, 500, n_te),
        "sttl": rng.uniform(30, 255, n_te),
        "smean": rng.uniform(20, 200, n_te),
        "attack_cat": ["Normal"] * 50 + ["Exploits"] * 100,
        "label": [0] * 50 + [1] * 100,
    })

    return df_tr, df_te, features


class TestSubsettingAndReproducibility:
    def test_sample_counts_and_shapes(self, mock_dfs):
        df_tr, df_te, features = mock_dfs
        subsets = get_stratified_subsets(df_tr, df_te, features, n_train=50, n_test=40, seed=42)
        assert subsets["X_train"].shape == (50, 4)
        assert subsets["y_train"].shape == (50,)
        assert subsets["X_test"].shape == (40, 4)
        assert subsets["y_test"].shape == (40,)

    def test_reproducibility_across_identical_seed(self, mock_dfs):
        df_tr, df_te, features = mock_dfs
        s1 = get_stratified_subsets(df_tr, df_te, features, 50, 40, seed=42)
        s2 = get_stratified_subsets(df_tr, df_te, features, 50, 40, seed=42)
        np.testing.assert_array_equal(s1["X_train"], s2["X_train"])
        np.testing.assert_array_equal(s1["X_test"], s2["X_test"])
        np.testing.assert_array_equal(s1["test_indices"], s2["test_indices"])

    def test_different_seeds_produce_different_subsets(self, mock_dfs):
        df_tr, df_te, features = mock_dfs
        s1 = get_stratified_subsets(df_tr, df_te, features, 50, 40, seed=42)
        s2 = get_stratified_subsets(df_tr, df_te, features, 50, 40, seed=100)
        assert not np.array_equal(s1["train_indices"], s2["train_indices"])
        assert not np.array_equal(s1["test_indices"], s2["test_indices"])


class TestClassicalModelsEvaluation:
    def test_svm_evaluation_output_structure(self, mock_dfs):
        df_tr, df_te, features = mock_dfs
        sub = get_stratified_subsets(df_tr, df_te, features, 30, 20, seed=42)
        res = evaluate_classical_svm(sub["X_train"], sub["y_train"], sub["X_test"], sub["y_test"], seed=42)
        assert "accuracy" in res
        assert "f1" in res
        assert "roc_auc" in res
        assert len(res["y_pred_arr"]) == 20
        assert len(res["y_score_arr"]) == 20

    def test_rf_evaluation_output_structure(self, mock_dfs):
        df_tr, df_te, features = mock_dfs
        sub = get_stratified_subsets(df_tr, df_te, features, 30, 20, seed=42)
        res = evaluate_classical_rf(sub["X_train"], sub["y_train"], sub["X_test"], sub["y_test"], seed=42)
        assert "accuracy" in res
        assert "f1" in res
        assert len(res["y_pred_arr"]) == 20
        assert len(res["y_score_arr"]) == 20
        # Probabilities should be bounded in [0, 1]
        assert all(0.0 <= s <= 1.0 for s in res["y_score_arr"])


class TestMetricsAggregation:
    def test_mean_and_std_calculation(self):
        runs = [
            {"accuracy": 0.80, "f1": 0.70, "roc_auc": 0.85, "precision": 0.75, "recall": 0.65, "false_positive_rate": 0.10, "train_time": 1.0, "infer_time": 0.5},
            {"accuracy": 0.90, "f1": 0.90, "roc_auc": 0.95, "precision": 0.85, "recall": 0.95, "false_positive_rate": 0.05, "train_time": 1.2, "infer_time": 0.6},
        ]
        agg = compute_metrics_aggregation(runs)
        assert agg["accuracy"]["mean"] == pytest.approx(0.85)
        # sample std of [0.8, 0.9] is approx 0.07071
        assert agg["accuracy"]["std"] == pytest.approx(0.070711, abs=1e-4)
        assert agg["f1"]["mean"] == pytest.approx(0.80)


class TestFeatureSetsAndBaselines:
    def test_load_feature_sets_valid(self):
        primary, ablation = load_feature_sets()
        assert len(primary) == 4
        assert len(ablation) == 4
        assert "sttl" in primary
        assert "sttl" not in ablation  # TTL excluded

    def test_load_stage4_baselines(self):
        baselines = load_stage4_baselines()
        assert "svm_selected4_full_test" in baselines
        assert "rf_selected4_full_test" in baselines
        assert baselines["svm_selected4_full_test"]["n_test_samples"] == 175341
        assert baselines["rf_selected4_full_test"]["n_test_samples"] == 175341


class TestArtifactsSchema:
    def test_predictions_csv_schema_if_exists(self):
        from src.config import RESULTS_DIR
        csv_path = RESULTS_DIR / "predictions_A.csv"
        if not csv_path.exists():
            pytest.skip("predictions_A.csv does not exist yet")

        df = pd.read_csv(csv_path)
        assert len(df) == DEFAULT_N_TEST
        expected_cols = [
            "sample_id",
            "original_test_index",
            "y_true",
            "attack_cat",
            "svm_pred",
            "svm_score",
            "rf_pred",
            "rf_score",
            "qsvc_pred",
            "qsvc_score",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"
        assert not df.isnull().values.any(), "predictions_A.csv contains nulls"

    def test_unified_results_json_schema_if_exists(self):
        from src.config import RESULTS_DIR
        json_path = RESULTS_DIR / "stage6_unified_experiments.json"
        if not json_path.exists():
            pytest.skip("stage6_unified_experiments.json does not exist yet")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0.0"
        assert "experiment_A_primary_comparison" in data
        assert "experiment_C_stability_analysis" in data
        assert "experiment_D_feature_ablation" in data
        assert "stage4_ceiling_references" in data

        exp_a = data["experiment_A_primary_comparison"]
        for m in ["SVM", "Random_Forest", "QSVC"]:
            assert m in exp_a
            assert "accuracy" in exp_a[m]
            assert "f1" in exp_a[m]


class TestLeakagePrevention:
    def test_train_scaler_isolation(self, mock_dfs):
        """Verify scaler parameters are fitted strictly on train data."""
        from sklearn.preprocessing import MinMaxScaler
        df_tr, df_te, features = mock_dfs
        sub = get_stratified_subsets(df_tr, df_te, features, n_train=50, n_test=30, seed=42)

        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_tr_scaled = scaler.fit_transform(sub["X_train"])

        # Inject extreme outlier in test data
        X_te_copy = sub["X_test"].copy()
        X_te_copy[0, 0] = 999999.0

        # Transform test data using fitted scaler
        data_min_before = scaler.data_min_.copy()
        scaler.transform(X_te_copy)
        data_min_after = scaler.data_min_

        # Scaler attributes must remain identical to training set parameters
        np.testing.assert_array_equal(data_min_before, data_min_after)
        assert np.all(scaler.data_min_ == sub["X_train"].min(axis=0))


class TestMockedQSVC:
    def test_evaluate_qsvc_with_mock_kernel(self, monkeypatch):
        """Verify QSVC evaluation wrapper with mocked quantum kernel computation."""
        from unittest.mock import MagicMock
        from src.run_experiments import evaluate_qsvc_model

        n_tr = 15
        n_te = 10
        rng = np.random.default_rng(42)
        X_tr = rng.uniform(0, np.pi, (n_tr, 4))
        y_tr = np.array([0] * 7 + [1] * 8)
        X_te = rng.uniform(0, np.pi, (n_te, 4))
        y_te = np.array([0] * 5 + [1] * 5)

        # Mock kernel object whose evaluate method returns valid Gram matrices
        mock_kernel = MagicMock()
        def mock_evaluate(X1, X2=None):
            if X2 is None or X2 is X1:
                # Symmetric PSD matrix
                mat = np.eye(len(X1)) + 0.05 * np.ones((len(X1), len(X1)))
                return mat
            return 0.5 * np.ones((len(X1), len(X2)))

        mock_kernel.evaluate.side_effect = mock_evaluate

        # Mock build_quantum_kernel to return mock_kernel
        monkeypatch.setattr("src.run_experiments.build_quantum_kernel", lambda *args, **kwargs: mock_kernel)

        result = evaluate_qsvc_model(X_tr, y_tr, X_te, y_te, C=10.0, seed=42)
        assert result["model_name"] == "QSVC"
        assert "accuracy" in result
        assert "f1" in result
        assert "roc_auc" in result
        assert len(result["y_pred_arr"]) == n_te
        assert len(result["y_score_arr"]) == n_te


class TestAlignmentAndIntegrity:
    def test_csv_alignment_with_raw_test_data(self):
        """Verify predictions_A.csv sample_id and labels match raw test set exactly."""
        from src.config import RESULTS_DIR, RAW_TEST_CSV
        csv_path = RESULTS_DIR / "predictions_A.csv"
        if not csv_path.exists() or not RAW_TEST_CSV.exists():
            pytest.skip("predictions_A.csv or RAW_TEST_CSV does not exist")

        df_preds = pd.read_csv(csv_path)
        df_raw_test = pd.read_csv(RAW_TEST_CSV)

        assert len(df_preds) == 100
        for _, row in df_preds.iterrows():
            idx = int(row["original_test_index"])
            sample_id = int(row["sample_id"])
            expected_id = int(df_raw_test.loc[idx, "id"])
            expected_label = int(df_raw_test.loc[idx, "label"])
            assert sample_id == expected_id, f"Mismatch at idx {idx}: {sample_id} vs {expected_id}"
            assert int(row["y_true"]) == expected_label, f"Label mismatch at idx {idx}"


