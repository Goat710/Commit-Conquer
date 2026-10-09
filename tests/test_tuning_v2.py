"""Unit tests for Stage 7b Step 4 hyperparameter tuning."""

import numpy as np
import pytest

from src.tuning_v2 import (
    QSVC_C_GRID,
    RF_PARAM_GRID,
    SVM_PARAM_GRID,
    tune_classical_rf,
    tune_classical_svm,
    tune_qsvc_on_partition,
)


@pytest.fixture
def dummy_tuning_data():
    np.random.seed(42)
    X_train = np.random.uniform(0, np.pi, size=(20, 4))
    y_train = np.array([0] * 10 + [1] * 10)
    X_tune = np.random.uniform(0, np.pi, size=(15, 4))
    y_tune = np.array([0] * 7 + [1] * 8)
    return X_train, y_train, X_tune, y_tune


def test_qsvc_tuning_structure_and_timings(dummy_tuning_data):
    X_train, y_train, X_tune, y_tune = dummy_tuning_data
    res = tune_qsvc_on_partition(X_train, y_train, X_tune, y_tune, c_grid=[0.1, 1.0, 10.0])

    for c in [0.1, 1.0, 10.0]:
        key = f"C_{c}"
        assert key in res
        entry = res[key]
        assert 0.0 <= entry["accuracy"] <= 1.0
        assert 0.0 <= entry["f1"] <= 1.0
        assert 0.0 <= entry["roc_auc"] <= 1.0
        # Check separated timings
        assert entry["fit_time_s"] >= 0.0
        assert entry["infer_time_s"] >= 0.0
        assert entry["kernel_train_time_s"] >= 0.0
        assert entry["kernel_tune_time_s"] >= 0.0


def test_classical_models_tuning_structure(dummy_tuning_data):
    X_train, y_train, X_tune, y_tune = dummy_tuning_data

    # Classical SVM
    svm_res = tune_classical_svm(
        X_train, y_train, X_tune, y_tune,
        param_grid=[{"C": 1.0, "gamma": "scale"}]
    )
    assert "C_1.0_gamma_scale" in svm_res
    assert svm_res["C_1.0_gamma_scale"]["fit_time_s"] >= 0.0
    assert svm_res["C_1.0_gamma_scale"]["infer_time_s"] >= 0.0

    # Classical RF
    rf_res = tune_classical_rf(
        X_train, y_train, X_tune, y_tune, seed=42,
        param_grid=[{"n_estimators": 10, "max_depth": 5}]
    )
    assert "trees_10_depth_5" in rf_res
    assert rf_res["trees_10_depth_5"]["fit_time_s"] >= 0.0
    assert rf_res["trees_10_depth_5"]["infer_time_s"] >= 0.0
