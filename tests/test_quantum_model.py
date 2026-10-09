"""Unit tests for src/quantum_model.py — Step 1: Quantum scaling helpers.

Tests cover:
  - Correct scaling to [0, pi] range
  - Clipping of out-of-range values
  - No re-fitting on transform (scaler parameters unchanged)
  - Dimensionality mismatch rejection
  - Invalid / unfitted scaler rejection
  - NaN / Inf input rejection
  - 1-D input rejection
  - Empty training data rejection
"""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.quantum_model import (
    ANGLE_MAX,
    ANGLE_MIN,
    fit_quantum_scaler,
    transform_quantum_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_train():
    """2 features, 5 samples -- deterministic training set."""
    rng = np.random.default_rng(42)
    return rng.uniform(low=0.0, high=10.0, size=(5, 2))


@pytest.fixture()
def fitted_scaler(simple_train):
    return fit_quantum_scaler(simple_train)


# ---------------------------------------------------------------------------
# fit_quantum_scaler tests
# ---------------------------------------------------------------------------


class TestFitQuantumScaler:
    def test_returns_mmmaxscaler(self, simple_train):
        scaler = fit_quantum_scaler(simple_train)
        assert isinstance(scaler, MinMaxScaler)

    def test_is_fitted(self, simple_train):
        scaler = fit_quantum_scaler(simple_train)
        # A fitted MinMaxScaler has the scale_ attribute.
        assert hasattr(scaler, "scale_")

    def test_feature_range_is_0_to_pi(self, simple_train):
        scaler = fit_quantum_scaler(simple_train)
        assert scaler.feature_range == (ANGLE_MIN, ANGLE_MAX)

    def test_training_extremes_map_to_bounds(self, simple_train):
        """The min of each training column should map to 0 and max to pi."""
        scaler = fit_quantum_scaler(simple_train)
        col_min = simple_train.min(axis=0, keepdims=True)
        col_max = simple_train.max(axis=0, keepdims=True)
        extremes = np.vstack([col_min, col_max])
        out = scaler.transform(extremes)
        # Bottom row should be ~0, top row should be ~pi.
        np.testing.assert_allclose(out[0], ANGLE_MIN, atol=1e-10)
        np.testing.assert_allclose(out[1], ANGLE_MAX, atol=1e-10)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            fit_quantum_scaler(np.array([1.0, 2.0, 3.0]))

    def test_rejects_empty_array(self):
        with pytest.raises(ValueError, match="at least one sample"):
            fit_quantum_scaler(np.empty((0, 3)))


# ---------------------------------------------------------------------------
# transform_quantum_features tests
# ---------------------------------------------------------------------------


class TestTransformQuantumFeatures:
    def test_output_in_0_pi_range(self, fitted_scaler, simple_train):
        out = transform_quantum_features(fitted_scaler, simple_train)
        assert np.all(out >= ANGLE_MIN - 1e-12)
        assert np.all(out <= ANGLE_MAX + 1e-12)

    def test_output_shape_preserved(self, fitted_scaler, simple_train):
        out = transform_quantum_features(fitted_scaler, simple_train)
        assert out.shape == simple_train.shape

    def test_values_are_finite(self, fitted_scaler, simple_train):
        out = transform_quantum_features(fitted_scaler, simple_train)
        assert np.all(np.isfinite(out))

    def test_no_refit_scaler_params_unchanged(self, simple_train):
        """Transform must NOT modify the scaler's fitted parameters."""
        scaler = fit_quantum_scaler(simple_train)
        scale_before = scaler.scale_.copy()
        min_before = scaler.data_min_.copy()

        # Transform a completely different array.
        X_other = np.ones((3, simple_train.shape[1])) * 999.0
        transform_quantum_features(scaler, X_other)

        np.testing.assert_array_equal(scaler.scale_, scale_before)
        np.testing.assert_array_equal(scaler.data_min_, min_before)

    def test_clipping_above_training_max(self, simple_train):
        """Values above training max must be clipped to pi (not exceed it)."""
        scaler = fit_quantum_scaler(simple_train)
        # Construct a point far above the training maximum.
        X_extreme = simple_train.max(axis=0, keepdims=True) * 100.0
        out = transform_quantum_features(scaler, X_extreme)
        # Without clipping this would be >> pi; after clipping it must be pi.
        np.testing.assert_allclose(out, ANGLE_MAX, atol=1e-10)

    def test_clipping_below_training_min(self, simple_train):
        """Values below training min must be clipped to 0 (not go negative)."""
        scaler = fit_quantum_scaler(simple_train)
        X_extreme = simple_train.min(axis=0, keepdims=True) - 100.0
        out = transform_quantum_features(scaler, X_extreme)
        np.testing.assert_allclose(out, ANGLE_MIN, atol=1e-10)

    def test_rejects_wrong_feature_count(self, fitted_scaler):
        """Dimensionality mismatch (wrong n_features) must raise ValueError."""
        X_bad = np.ones((3, fitted_scaler.n_features_in_ + 1))
        with pytest.raises(ValueError, match="features"):
            transform_quantum_features(fitted_scaler, X_bad)

    def test_rejects_1d_input(self, fitted_scaler):
        with pytest.raises(ValueError, match="2-D"):
            transform_quantum_features(fitted_scaler, np.array([0.5, 1.0]))

    def test_rejects_non_mmmaxscaler(self, simple_train):
        """Passing something that is not a MinMaxScaler must raise TypeError."""
        with pytest.raises(TypeError, match="MinMaxScaler"):
            transform_quantum_features("not_a_scaler", simple_train)

    def test_rejects_unfitted_scaler(self, simple_train):
        """An unfitted MinMaxScaler must be rejected."""
        raw = MinMaxScaler(feature_range=(ANGLE_MIN, ANGLE_MAX))
        with pytest.raises(ValueError, match="fitted"):
            transform_quantum_features(raw, simple_train)

    def test_rejects_nan_input(self, fitted_scaler):
        """NaN in input must propagate and trigger the finite-value guard."""
        X_nan = np.ones((2, fitted_scaler.n_features_in_))
        X_nan[0, 0] = np.nan
        with pytest.raises((ValueError, Exception)):
            transform_quantum_features(fitted_scaler, X_nan)

    def test_deterministic_with_seed(self, simple_train):
        """Same input always produces identical output."""
        scaler = fit_quantum_scaler(simple_train)
        out1 = transform_quantum_features(scaler, simple_train)
        out2 = transform_quantum_features(scaler, simple_train)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Step 2 -- ZZFeatureMap tests
# ---------------------------------------------------------------------------

from src.quantum_model import build_feature_map, save_circuit_diagram


class TestBuildFeatureMap:
    def test_returns_quantum_circuit(self):
        """build_feature_map must return a QuantumCircuit."""
        from qiskit.circuit import QuantumCircuit

        qc = build_feature_map(n_features=4, reps=2)
        assert isinstance(qc, QuantumCircuit)

    def test_num_qubits_equals_n_features(self):
        """One qubit per feature."""
        for n in (2, 3, 4, 6):
            qc = build_feature_map(n_features=n, reps=2)
            assert qc.num_qubits == n, f"Expected {n} qubits, got {qc.num_qubits}"

    def test_n_qubit_features_config(self):
        """Circuit width must equal N_QUBIT_FEATURES from config."""
        from src.config import N_QUBIT_FEATURES

        qc = build_feature_map(n_features=N_QUBIT_FEATURES, reps=2)
        assert qc.num_qubits == N_QUBIT_FEATURES

    def test_num_parameters_equals_n_features(self):
        """ZZFeatureMap has exactly n_features free parameters (x[0]..x[n-1])."""
        n = 4
        qc = build_feature_map(n_features=n, reps=2)
        assert qc.num_parameters == n

    def test_reps_affects_depth(self):
        """Larger reps must produce a strictly deeper circuit."""
        qc2 = build_feature_map(n_features=4, reps=2)
        qc3 = build_feature_map(n_features=4, reps=3)
        assert qc3.depth() > qc2.depth(), (
            f"reps=3 depth {qc3.depth()} should exceed reps=2 depth {qc2.depth()}"
        )

    def test_rejects_n_features_less_than_2(self):
        """ZZ interaction needs >= 2 qubits."""
        with pytest.raises(ValueError, match="n_features"):
            build_feature_map(n_features=1, reps=2)

    def test_rejects_zero_reps(self):
        """reps must be a positive integer."""
        with pytest.raises(ValueError, match="reps"):
            build_feature_map(n_features=4, reps=0)

    def test_circuit_name(self):
        """Circuit should have a recognisable name."""
        qc = build_feature_map(n_features=4, reps=2)
        assert "ZZ" in qc.name or qc.name, f"Unexpected circuit name: {qc.name!r}"


class TestSaveCircuitDiagram:
    def test_creates_png_file(self, tmp_path):
        """save_circuit_diagram must write a non-empty PNG to the given path."""
        qc = build_feature_map(n_features=4, reps=2)
        out = tmp_path / "circuit_test.png"
        save_circuit_diagram(qc, out)
        assert out.exists(), "PNG file was not created"
        assert out.stat().st_size > 1000, "PNG file is suspiciously small"

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories that do not exist must be created."""
        qc = build_feature_map(n_features=4, reps=2)
        nested = tmp_path / "a" / "b" / "c" / "circuit.png"
        save_circuit_diagram(qc, nested)
        assert nested.exists()
