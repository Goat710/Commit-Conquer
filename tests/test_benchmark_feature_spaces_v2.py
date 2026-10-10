"""Unit and integration tests for Stage 8E benchmark module."""

import numpy as np
import pandas as pd
import pytest

from src.benchmark_feature_spaces_v2 import (
    FEATURE_SET_F4_BASE,
    FEATURE_SET_F4_NOTTL,
    HGB_PARAM_GRID,
    compute_summary_stats,
    get_feature_sets,
    tune_and_fit_hgb,
    tune_and_fit_rf,
    tune_and_fit_svm,
)
from src.dataset_v2 import load_raw_datasets
from src.tuning_v2 import RF_PARAM_GRID, SVM_PARAM_GRID


def test_feature_sets_definitions():
    """Verify feature sets have exact requested schemas."""
    train_df, _ = load_raw_datasets()
    f_sets = get_feature_sets(train_df)

    assert "F4-Base" in f_sets
    assert "F4-NoTTL" in f_sets
    assert "F39-Full" in f_sets

    assert f_sets["F4-Base"] == ["sbytes", "sload", "sttl", "smean"]
    assert f_sets["F4-NoTTL"] == ["sbytes", "sload", "is_ftp_login", "smean"]
    assert len(f_sets["F39-Full"]) == 39
    assert "sttl" in f_sets["F4-Base"]
    assert "sttl" not in f_sets["F4-NoTTL"]
    assert "is_ftp_login" in f_sets["F4-NoTTL"]


def test_hyperparameter_grids_budget():
    """Verify each model has an explicit search budget of exactly 5 configurations."""
    assert len(RF_PARAM_GRID) == 5
    assert len(HGB_PARAM_GRID) == 5
    assert len(SVM_PARAM_GRID) == 5


def test_summary_statistics_dual_std():
    """Verify compute_summary_stats properly distinguishes ddof=0 and ddof=1."""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = compute_summary_stats(vals)

    expected_mean = 3.0
    expected_std0 = float(np.std(vals, ddof=0))
    expected_std1 = float(np.std(vals, ddof=1))

    assert stats["mean"] == expected_mean
    assert stats["std_ddof0"] == round(expected_std0, 4)
    assert stats["std_ddof1"] == round(expected_std1, 4)
    assert stats["std_ddof1"] > stats["std_ddof0"]


def test_tune_and_fit_functions_synthetic():
    """Verify tuning and fitting functions execute cleanly on synthetic data."""
    rng = np.random.RandomState(42)
    X_tr = rng.randn(100, 4)
    y_tr = rng.randint(0, 2, 100)
    X_tu = rng.randn(40, 4)
    y_tu = rng.randint(0, 2, 40)

    # RF
    rf, rf_meta, rf_t_tune, rf_t_fit = tune_and_fit_rf(X_tr, y_tr, X_tu, y_tu, seed=42)
    assert hasattr(rf, "predict_proba")
    assert "best_params" in rf_meta
    assert rf_t_tune > 0.0

    # HGB
    hgb, hgb_meta, hgb_t_tune, hgb_t_fit = tune_and_fit_hgb(X_tr, y_tr, X_tu, y_tu, seed=42)
    assert hasattr(hgb, "predict_proba")
    assert "best_params" in hgb_meta
    assert hgb_t_tune > 0.0

    # SVM
    svm, scaler, svm_meta, s_prep, s_tune, s_fit = tune_and_fit_svm(X_tr, y_tr, X_tu, y_tu)
    assert hasattr(svm, "decision_function")
    assert hasattr(scaler, "mean_")
    assert "best_params" in svm_meta
    assert s_tune > 0.0
