"""Tests for chunked kernel computation and validated caching — Step 3."""

import numpy as np
import pytest

from src.kernel_cache_v2 import (
    KernelCacheManager,
    compute_kernel_matrix_chunked,
    compute_statevector_embeddings_chunked,
    hash_array,
)
from src.quantum_model_v2 import (
    compute_statevector_embeddings,
    compute_statevector_kernel,
)


def test_hash_array():
    arr1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    arr3 = np.array([[1.0, 2.0], [3.0, 4.0001]])

    assert hash_array(arr1) == hash_array(arr2)
    assert hash_array(arr1) != hash_array(arr3)


def test_cache_manager_put_and_get(tmp_path):
    mgr = KernelCacheManager(cache_dir=tmp_path)
    K = np.eye(5, dtype=np.float64)
    key = mgr.generate_cache_key(
        split_name="train",
        n_samples_A=5,
        n_samples_B=5,
        seed=42,
        features=["f1", "f2"],
        data_hash_A="abc123",
    )

    meta = {"features": ["f1", "f2"], "seed": 42, "data_hash_A": "abc123"}
    mgr.put(key, K, meta)

    # Valid get
    loaded = mgr.get(key, expected_shape=(5, 5), expected_data_hash="abc123")
    assert loaded is not None
    np.testing.assert_array_equal(loaded, K)

    # Shape mismatch invalidation
    bad_shape = mgr.get(key, expected_shape=(5, 6), expected_data_hash="abc123")
    assert bad_shape is None

    # Hash mismatch invalidation
    bad_hash = mgr.get(key, expected_shape=(5, 5), expected_data_hash="wrong_hash")
    assert bad_hash is None


def test_chunked_embeddings_and_kernel():
    np.random.seed(42)
    X_A = np.random.uniform(0, np.pi, size=(12, 4))
    X_B = np.random.uniform(0, np.pi, size=(6, 4))

    # Embeddings: chunked vs direct
    states_A_direct = compute_statevector_embeddings(X_A)
    states_A_chunked = compute_statevector_embeddings_chunked(X_A, chunk_size=5)
    np.testing.assert_allclose(states_A_chunked, states_A_direct, atol=1e-12)

    states_B_direct = compute_statevector_embeddings(X_B)

    # Kernel: chunked vs direct
    K_direct = compute_statevector_kernel(states_A_direct, states_B_direct)
    K_chunked = compute_kernel_matrix_chunked(states_A_chunked, states_B_direct, chunk_size=4)
    np.testing.assert_allclose(K_chunked, K_direct, atol=1e-12)
