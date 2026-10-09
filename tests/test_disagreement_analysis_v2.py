"""Unit tests for Stage 7b Step 8: Disagreement and Error Overlap Analysis."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.disagreement_analysis_v2 import (
    load_predictions_v2,
    compute_pairwise_agreements,
    compute_three_way_agreements,
    compute_conditional_correctness,
    compute_error_overlap_matrix,
    compute_majority_vote_ensemble,
    run_disagreement_analysis,
    save_disagreement_report,
    DISCLAIMER_TEXT,
    PREDICTIONS_CSV_PATH,
)


@pytest.fixture
def synthetic_predictions_df() -> pd.DataFrame:
    """Fixture with 10 synthetic flows for exact arithmetic verification."""
    # sample_id, y_true, qsvc_pred, svm_pred, rf_pred
    data = [
        # 1-3: Unanimous correct (3 flows)
        {"sample_id": 1, "y_true": 1, "qsvc_pred": 1, "svm_pred": 1, "rf_pred": 1},
        {"sample_id": 2, "y_true": 0, "qsvc_pred": 0, "svm_pred": 0, "rf_pred": 0},
        {"sample_id": 3, "y_true": 1, "qsvc_pred": 1, "svm_pred": 1, "rf_pred": 1},
        # 4: Unanimous incorrect (1 flow)
        {"sample_id": 4, "y_true": 1, "qsvc_pred": 0, "svm_pred": 0, "rf_pred": 0},
        # 5: QSVC only failed (SVM & RF correct) (1 flow)
        {"sample_id": 5, "y_true": 1, "qsvc_pred": 0, "svm_pred": 1, "rf_pred": 1},
        # 6: SVM only failed (QSVC & RF correct) (1 flow)
        {"sample_id": 6, "y_true": 1, "qsvc_pred": 1, "svm_pred": 0, "rf_pred": 1},
        # 7: RF only failed (QSVC & SVM correct) (1 flow)
        {"sample_id": 7, "y_true": 1, "qsvc_pred": 1, "svm_pred": 1, "rf_pred": 0},
        # 8: QSVC & SVM failed (RF alone correct) (1 flow)
        {"sample_id": 8, "y_true": 1, "qsvc_pred": 0, "svm_pred": 0, "rf_pred": 1},
        # 9: QSVC & RF failed (SVM alone correct) (1 flow)
        {"sample_id": 9, "y_true": 1, "qsvc_pred": 0, "svm_pred": 1, "rf_pred": 0},
        # 10: SVM & RF failed (QSVC alone correct) (1 flow)
        {"sample_id": 10, "y_true": 1, "qsvc_pred": 1, "svm_pred": 0, "rf_pred": 0},
    ]
    return pd.DataFrame(data)


def test_synthetic_pairwise_agreements(synthetic_predictions_df):
    results = compute_pairwise_agreements(synthetic_predictions_df)
    assert "SVM vs RF" in results
    assert "SVM vs QSVC" in results
    assert "RF vs QSVC" in results

    # Verify SVM vs RF:
    # flow 1: s=1, r=1 (agree)
    # flow 2: s=0, r=0 (agree)
    # flow 3: s=1, r=1 (agree)
    # flow 4: s=0, r=0 (agree)
    # flow 5: s=1, r=1 (agree)
    # flow 6: s=0, r=1 (disagree)
    # flow 7: s=1, r=0 (disagree)
    # flow 8: s=0, r=1 (disagree)
    # flow 9: s=1, r=0 (disagree)
    # flow 10: s=0, r=0 (agree)
    # Total agree = 6, disagree = 4
    svm_rf = results["SVM vs RF"]
    assert svm_rf["agreement_count"] == 6
    assert svm_rf["disagreement_count"] == 4
    assert svm_rf["agreement_rate"] == 0.6
    assert svm_rf["disagreement_rate"] == 0.4


def test_synthetic_three_way_agreements(synthetic_predictions_df):
    results = compute_three_way_agreements(synthetic_predictions_df)
    assert results["total_samples"] == 10
    # Unanimous flows: 1, 2, 3 (correct) + 4 (incorrect) = 4 flows
    unanimous = results["unanimous_agreement"]
    assert unanimous["count"] == 4
    assert unanimous["rate"] == 0.4
    assert unanimous["unanimous_correct_count"] == 3
    assert unanimous["unanimous_incorrect_count"] == 1

    dis = results["three_way_disagreement"]
    assert dis["count"] == 6
    assert dis["rate"] == 0.6


def test_synthetic_conditional_correctness(synthetic_predictions_df):
    cond = compute_conditional_correctness(synthetic_predictions_df)
    assert cond["disagreement_subset_size"] == 6
    perf = cond["model_performance_on_disagreements"]

    # In flows 5-10:
    # y=1 for all 5-10
    # QSVC preds: [0, 1, 1, 0, 0, 1] -> 3 correct out of 6
    # SVM preds:  [1, 0, 1, 0, 1, 0] -> 3 correct out of 6
    # RF preds:   [1, 1, 0, 1, 0, 0] -> 3 correct out of 6
    assert perf["qsvc"]["correct_count"] == 3
    assert perf["svm"]["correct_count"] == 3
    assert perf["rf"]["correct_count"] == 3
    assert perf["qsvc"]["accuracy_on_disagreement"] == 0.5


def test_synthetic_error_overlap_matrix_exactness(synthetic_predictions_df):
    matrix = compute_error_overlap_matrix(synthetic_predictions_df)
    assert matrix["partition_is_exact"] is True
    assert matrix["partition_verification_sum"] == 10

    # In our fixture: exactly 1 flow per error subset:
    # all_3_wrong: flow 4 (1)
    # qsvc_only: flow 5 (1)
    # svm_only: flow 6 (1)
    # rf_only: flow 7 (1)
    # qsvc_svm_wrong: flow 8 (1)
    # qsvc_rf_wrong: flow 9 (1)
    # svm_rf_wrong: flow 10 (1)
    # all_3_right: flows 1, 2, 3 (3)
    # Total = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 3 = 10
    assert matrix["overall_error_coverage"]["all_three_correct"] == 3
    assert matrix["overall_error_coverage"]["all_three_incorrect"] == 1
    disjoint = matrix["disjoint_error_breakdown"]
    assert disjoint["unique_errors_single_model"]["qsvc_only_failed"] == 1
    assert disjoint["unique_errors_single_model"]["svm_only_failed"] == 1
    assert disjoint["unique_errors_single_model"]["rf_only_failed"] == 1
    assert disjoint["shared_errors_two_models"]["qsvc_and_svm_failed_rf_alone_correct"] == 1
    assert disjoint["shared_errors_two_models"]["qsvc_and_rf_failed_svm_alone_correct"] == 1
    assert disjoint["shared_errors_two_models"]["svm_and_rf_failed_qsvc_alone_correct"] == 1


def test_majority_vote_synthetic(synthetic_predictions_df):
    maj = compute_majority_vote_ensemble(synthetic_predictions_df)
    assert maj["majority_vote_correct_count"] <= 10
    assert 0.0 <= maj["majority_vote_accuracy"] <= 1.0


def test_real_predictions_disagreement_analysis(tmp_path):
    if not PREDICTIONS_CSV_PATH.is_file():
        pytest.skip("predictions_v2.csv not found on disk")

    df = load_predictions_v2(PREDICTIONS_CSV_PATH)
    assert len(df) == 20000

    report = run_disagreement_analysis(PREDICTIONS_CSV_PATH)

    # Check top-level keys
    assert "disclaimer" in report
    assert "Do not claim models complement one another" in report["disclaimer"]
    assert "pairwise_agreements" in report
    assert "three_way_agreements" in report
    assert "conditional_correctness_on_disagreements" in report
    assert "error_overlap_matrix" in report
    assert "majority_vote_ensemble_evaluation" in report

    # Verify partition is exact on real dataset
    matrix = report["error_overlap_matrix"]
    assert matrix["partition_is_exact"] is True
    assert matrix["partition_verification_sum"] == 20000

    # Verify exact counts
    coverage = matrix["overall_error_coverage"]
    assert coverage["all_three_correct"] == 15327
    assert coverage["all_three_incorrect"] == 875

    disjoint = matrix["disjoint_error_breakdown"]
    assert disjoint["unique_errors_single_model"]["qsvc_only_failed"] == 948
    assert disjoint["unique_errors_single_model"]["svm_only_failed"] == 343
    assert disjoint["unique_errors_single_model"]["rf_only_failed"] == 736
    assert disjoint["shared_errors_two_models"]["qsvc_and_svm_failed_rf_alone_correct"] == 1186
    assert disjoint["shared_errors_two_models"]["qsvc_and_rf_failed_svm_alone_correct"] == 359
    assert disjoint["shared_errors_two_models"]["svm_and_rf_failed_qsvc_alone_correct"] == 226

    # Verify majority vote degradation
    maj = report["majority_vote_ensemble_evaluation"]
    assert maj["majority_vote_accuracy"] < maj["single_model_accuracies"]["rf"]

    # Verify JSON saving
    temp_json = tmp_path / "test_disagreement.json"
    save_disagreement_report(report, temp_json)
    assert temp_json.is_file()
    with open(temp_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["sample_size"] == 20000
