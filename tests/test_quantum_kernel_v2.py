"""Tests for exact statevector quantum kernel — Stage 7b.

Tests cover:
  - Statevector embedding generation, shape, dtype, unit norm
  - Input validation (1D, empty, NaN, dimension mismatch)
  - Kernel invariants (square, diagonal ~ 1, symmetry, range [0, 1], PSD)
  - Rectangular kernel computation
  - Strict equivalence with Qiskit's FidelityQuantumKernel (< 1e-6)
  - Comparison with historical Stage 5/7 kernel (shot noise analysis)
"""

import numpy as np
import pytest
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.quantum_model import (
    build_feature_map,
    build_quantum_kernel,
    verify_kernel_matrix,
)
from src.quantum_model_v2 import (
    compute_exact_quantum_kernel,
    compute_statevector_embeddings,
    compute_statevector_kernel,
    verify_exact_kernel,
)


class TestStatevectorEmbeddings:
    """Tests for compute_statevector_embeddings."""

    def test_shape_and_dtype(self):
        np.random.seed(42)
        X = np.random.uniform(0, np.pi, size=(7, 4))
        states = compute_statevector_embeddings(X)

        assert states.shape == (7, 16)
        assert states.dtype == np.complex128
        assert np.all(np.isfinite(states))

    def test_unit_norm(self):
        np.random.seed(42)
        X = np.random.uniform(0, np.pi, size=(10, 4))
        states = compute_statevector_embeddings(X)

        norms = np.linalg.norm(states, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_invalid_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="2-D array"):
            compute_statevector_embeddings(X)

    def test_empty_input(self):
        X = np.empty((0, 4))
        with pytest.raises(ValueError, match="0 samples"):
            compute_statevector_embeddings(X)

    def test_nan_input(self):
        X = np.array([[1.0, np.nan, 2.0, 3.0]])
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_statevector_embeddings(X)

    def test_feature_dimension_mismatch(self):
        feature_map = build_feature_map(n_features=4)
        X_wrong = np.random.uniform(0, np.pi, size=(5, 3))
        with pytest.raises(ValueError, match="expects 4 features"):
            compute_statevector_embeddings(X_wrong, feature_map=feature_map)


class TestStatevectorKernelInvariants:
    """Tests for kernel computation and mathematical invariants."""

    def test_square_kernel_invariants(self):
        np.random.seed(100)
        X = np.random.uniform(0, np.pi, size=(8, 4))
        states = compute_statevector_embeddings(X)
        K = compute_statevector_kernel(states)

        assert K.shape == (8, 8)
        assert K.dtype == np.float64

        # Diagonal exactly 1.0
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

        # Symmetry
        np.testing.assert_allclose(K, K.T, atol=1e-12)

        # Range [0, 1]
        assert np.all(K >= 0.0)
        assert np.all(K <= 1.0)

        # Diagnostics check via verify_exact_kernel
        diag = verify_exact_kernel(
            K,
            tol_diag=1e-10,
            tol_sym=1e-10,
            tol_range=1e-10,
            tol_psd=-1e-10,
        )
        assert diag["all_passed"] is True
        assert diag["strictly_psd"] is True

    def test_rectangular_kernel(self):
        np.random.seed(2024)
        X_A = np.random.uniform(0, np.pi, size=(5, 4))
        X_B = np.random.uniform(0, np.pi, size=(3, 4))

        states_A = compute_statevector_embeddings(X_A)
        states_B = compute_statevector_embeddings(X_B)

        K_rect = compute_statevector_kernel(states_A, states_B)
        assert K_rect.shape == (5, 3)
        assert np.all(K_rect >= 0.0)
        assert np.all(K_rect <= 1.0)

    def test_dimension_mismatch(self):
        s_A = np.zeros((4, 16), dtype=complex)
        s_B = np.zeros((4, 8), dtype=complex)
        with pytest.raises(ValueError, match="Hilbert space dimension mismatch"):
            compute_statevector_kernel(s_A, s_B)


class TestFidelityQuantumKernelValidation:
    """Validation against Qiskit's FidelityQuantumKernel and Stage 7 comparisons."""

    def test_validation_against_qiskit_fidelity_quantum_kernel(self):
        """Require maximum absolute difference below 1e-6 against FidelityQuantumKernel."""
        np.random.seed(777)
        X = np.random.uniform(0, np.pi, size=(5, 4))
        feature_map = build_feature_map(n_features=4, reps=2)

        # Exact statevector kernel
        K_sv = compute_exact_quantum_kernel(X, feature_map=feature_map)

        # Qiskit FidelityQuantumKernel default
        qiskit_kernel = FidelityQuantumKernel(feature_map=feature_map)
        K_qiskit = qiskit_kernel.evaluate(X)

        max_abs_diff = float(np.max(np.abs(K_sv - K_qiskit)))
        assert max_abs_diff < 1e-6, f"Kernel difference {max_abs_diff} exceeds 1e-6"
        # In fact it matches to floating-point precision (< 1e-10)
        assert max_abs_diff < 1e-10

    def test_comparison_against_historical_stage7_kernel(self):
        """Document differences between exact statevector and historical shot-noise kernel."""
        np.random.seed(999)
        X = np.random.uniform(0, np.pi, size=(6, 4))
        feature_map = build_feature_map(n_features=4, reps=2)

        # Exact statevector kernel
        K_sv = compute_exact_quantum_kernel(X, feature_map=feature_map)

        # Historical Stage 5/7 kernel (StatevectorSampler with default 1024 shots)
        k_hist = build_quantum_kernel(feature_map, seed=42, enforce_psd=False)
        K_hist = k_hist.evaluate(X)

        max_abs_diff = float(np.max(np.abs(K_sv - K_hist)))

        # Historical kernel has shot noise ~ 1/sqrt(1024) ~ 0.03
        # Both share unit diagonals:
        np.testing.assert_allclose(np.diag(K_sv), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.diag(K_hist), 1.0, atol=1e-10)

        # Differences exist off-diagonal due to shot noise in StatevectorSampler
        assert max_abs_diff > 1e-4, "Expected shot noise difference between historical and exact"
        assert max_abs_diff < 0.15, "Shot noise difference should be bounded by ~3 sigma of 1/sqrt(shots)"
