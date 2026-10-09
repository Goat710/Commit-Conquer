"""Unit tests for Stage 7b Step 6 paired statistical analysis."""

import numpy as np
import pytest

from src.statistical_analysis_v2 import (
    holm_bonferroni_correction,
    mcnemar_test_paired,
    paired_bootstrap_ci,
)


def test_mcnemar_test_paired():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    # Model A: correct on indices 0, 1, 2, 3, 4, 5 (wrong on 6, 7)
    # Model B: correct on indices 0, 1, 6, 7 (wrong on 2, 3, 4, 5)
    y_pred_A = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred_B = np.array([0, 0, 0, 0, 1, 0, 0, 1])

    res = mcnemar_test_paired(y_true, y_pred_A, y_pred_B)

    # b: A wrong, B correct -> indices 6, 7 -> b = 2
    # c: A correct, B wrong -> indices 2, 3, 4, 5 -> c = 4
    assert res["contingency_table"]["A_wrong_B_correct_b"] == 2
    assert res["contingency_table"]["A_correct_B_wrong_c"] == 4
    assert res["contingency_table"]["total_discordant"] == 6

    # chi2 with continuity correction: (|2 - 4| - 1)^2 / 6 = 1 / 6 = 0.1667
    expected_chi2 = 1.0 / 6.0
    assert abs(res["chi2_statistic"] - expected_chi2) < 1e-3
    assert 0.0 <= res["raw_p_value"] <= 1.0


def test_holm_bonferroni_correction():
    raw_p = [0.01, 0.04, 0.03]
    # Sorted order: 0.01 (m=3 -> 0.03), 0.03 (m=2 -> 0.06), 0.04 (m=1 -> max(0.06, 0.04)=0.06)
    adj = holm_bonferroni_correction(raw_p)

    assert adj[0] == 0.03  # 0.01 * 3
    assert adj[2] == 0.06  # 0.03 * 2
    assert adj[1] == 0.06  # max(0.06, 0.04 * 1)


def test_paired_bootstrap_ci_structure():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=100)
    p_A = y_true.copy()
    p_A[:10] = 1 - p_A[:10]  # 90% accuracy
    p_B = y_true.copy()
    p_B[:20] = 1 - p_B[:20]  # 80% accuracy

    s_A = p_A.astype(float)
    s_B = p_B.astype(float)

    boot = paired_bootstrap_ci(
        y_true, p_A, s_A, p_B, s_B,
        n_resamples=50,
        seed=42,
    )

    for k in ["accuracy_diff", "f1_diff", "roc_auc_diff", "fpr_diff"]:
        assert k in boot
        assert "ci_95_percent" in boot[k]
        assert boot[k]["ci_95_percent"][0] <= boot[k]["ci_95_percent"][1]
