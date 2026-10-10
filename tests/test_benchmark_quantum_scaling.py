"""Unit and integration tests for Stage 8F-B quantum scaling and Nyström approximation."""

import numpy as np
import pytest

from src.benchmark_quantum_scaling import (
    NYSTROEM_LANDMARK_BUDGETS,
    PEAK_MEMORY_BUDGET_BYTES,
    TARGET_QUBIT_COUNTS,
    estimate_statevector_memory_demand,
    get_deterministic_feature_subsets,
    run_experiment_a_scaling,
    run_experiment_b_nystroem,
)
from src.dataset_v2 import load_raw_datasets


def test_target_qubit_and_landmark_definitions():
    """Verify target qubit counts and landmark budgets."""
    assert TARGET_QUBIT_COUNTS == [4, 6, 8, 10, 12, 14, 16]
    assert NYSTROEM_LANDMARK_BUDGETS == [5, 10, 20, 50, 100]
    assert PEAK_MEMORY_BUDGET_BYTES == 500 * 1024 * 1024


def test_deterministic_feature_subsets():
    """Verify feature subsets match qubit counts and maintain baseline 4 features."""
    train_df, _ = load_raw_datasets()
    subsets = get_deterministic_feature_subsets(train_df, [4, 6, 8, 10, 12, 14, 16])

    assert len(subsets) == 7
    for q, feats in subsets.items():
        assert len(feats) == q
        assert feats[:4] == ["sbytes", "sload", "sttl", "smean"]
        assert len(set(feats)) == q  # All unique


def test_memory_demand_estimation():
    """Verify conservative memory estimation formula."""
    # N=100, q=4: 100 * 16 * 16 = 25.6 KB + overhead
    mem_q4 = estimate_statevector_memory_demand(100, 4)
    assert mem_q4 < 20 * 1024 * 1024  # < 20 MB

    # N=100, q=16: 100 * 65536 * 16 = 104.85 MB + overhead
    mem_q16 = estimate_statevector_memory_demand(100, 16)
    assert mem_q16 < PEAK_MEMORY_BUDGET_BYTES  # Within 500 MB budget

    # N=4000, q=20: should exceed budget
    mem_q20 = estimate_statevector_memory_demand(4000, 20)
    assert mem_q20 > PEAK_MEMORY_BUDGET_BYTES


def test_nystroem_mathematical_formula_synthetic():
    """Verify Nyström approximation on a synthetic low-rank Gram matrix."""
    rng = np.random.RandomState(42)
    # Rank 5 matrix of size 30x30
    X = rng.randn(30, 5)
    K = X @ X.T  # Exact rank 5

    # With m=5 landmarks, Nyström should reconstruct K with near-zero error
    landmarks = np.array([0, 1, 2, 3, 4])
    C = K[:, landmarks]
    W = K[np.ix_(landmarks, landmarks)]
    W_pinv = np.linalg.pinv(W, rcond=1e-15)
    K_hat = C @ W_pinv @ C.T

    frob_err = np.linalg.norm(K - K_hat, ord="fro") / np.linalg.norm(K, ord="fro")
    spec_err = np.linalg.norm(K - K_hat, ord=2) / np.linalg.norm(K, ord=2)

    assert frob_err < 1e-10
    assert spec_err < 1e-10


def test_mock_scaling_and_nystroem_integration():
    """Verify integration of scaling and Nyström functions on miniature subset."""
    train_df, _ = load_raw_datasets()
    train_pool_idx = np.arange(200)
    eval_idx = train_pool_idx[:15]  # 15 samples

    # Test scaling on q=[4, 6]
    res_a = run_experiment_a_scaling(
        train_df=train_df,
        train_pool_indices=train_pool_idx,
        eval_row_indices=eval_idx,
        qubit_counts=[4, 6],
    )
    assert "q_4" in res_a
    assert "q_6" in res_a
    assert res_a["q_4"]["status"] == "SUCCESS"
    assert res_a["q_6"]["status"] == "SUCCESS"
    assert res_a["q_4"]["numerical_validation"]["is_square"] is True

    # Test Nyström on m=[3, 5, 15]
    res_b = run_experiment_b_nystroem(
        train_df=train_df,
        train_pool_indices=train_pool_idx,
        eval_row_indices=eval_idx,
        landmark_budgets=[3, 5, 15],
    )
    assert "reference_kernel" in res_b
    assert "landmark_approximations" in res_b
    assert res_b["reference_kernel"]["numerical_properties"]["is_symmetric"] is True
    assert "m_3" in res_b["landmark_approximations"]
    assert "m_15" in res_b["landmark_approximations"]
    assert res_b["landmark_approximations"]["m_15"]["is_full_landmark_boundary"] is True
