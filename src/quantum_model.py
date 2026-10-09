"""Quantum model utilities for Quantum CyberShield — Stage 5.

This module provides:
  - Step 1 : Quantum feature scaling helpers (MinMaxScaler -> [0, pi])
  - Step 2 : ZZFeatureMap builder  (added after approval)
  - Step 3 : FidelityQuantumKernel builder (added after approval)

Scaling design note
-------------------
The ZZFeatureMap / PauliFeatureMap encodes each feature value as a rotation
angle in a qubit.  Valid rotation angles lie in [0, pi].  We achieve this by
fitting a MinMaxScaler(feature_range=(0, pi)) on the TRAINING subsample only,
then applying it to all subsequent partitions (validation / test) WITHOUT
refitting.  Clipping the output to [0, pi] guards against tiny floating-point
overflows from unseen values outside the training range; it does NOT use test
label information and therefore introduces no leakage.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Rotation angle bounds for the feature map.
ANGLE_MIN: float = 0.0
ANGLE_MAX: float = float(np.pi)  # approx 3.14159 radians


# ---------------------------------------------------------------------------
# Step 1 -- Quantum scaling helpers
# ---------------------------------------------------------------------------


def fit_quantum_scaler(X_train: np.ndarray) -> MinMaxScaler:
    """Fit a MinMaxScaler that maps training features to the angle range [0, pi].

    Parameters
    ----------
    X_train:
        2-D float array of shape (n_samples, n_features).  Must be from the
        TRAINING partition only -- never pass validation or test data here.

    Returns
    -------
    MinMaxScaler
        A fitted scaler.  Call transform_quantum_features to apply it to
        any subsequent partition without refitting.

    Raises
    ------
    ValueError
        If X_train is empty or not 2-D.
    """
    X = _validate_2d_float(X_train, "X_train")

    if X.shape[0] == 0:
        raise ValueError(
            "fit_quantum_scaler: X_train must contain at least one sample; "
            "got 0 rows."
        )

    # MinMaxScaler(feature_range=(0, pi)) maps each feature independently so
    # that the training minimum -> 0 and training maximum -> pi.  This keeps
    # quantum rotation angles in the canonical range the ZZFeatureMap expects.
    scaler = MinMaxScaler(feature_range=(ANGLE_MIN, ANGLE_MAX))
    scaler.fit(X)
    return scaler


def transform_quantum_features(
    scaler: MinMaxScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Apply a pre-fitted quantum scaler to a feature matrix.

    This function ONLY transforms -- it never re-fits the scaler, preserving
    strict train-only scaling discipline.  After scaling, values are clipped
    to [0, pi] to handle rare numerical overflows caused by unseen test values
    that fall marginally outside the training range.  Clipping does not use
    label information and therefore introduces no data leakage.

    Parameters
    ----------
    scaler:
        A scaler already fitted by fit_quantum_scaler.
    X:
        2-D float array of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Transformed array with all values in [0, pi], shape identical to X.

    Raises
    ------
    ValueError
        If X is not 2-D, contains non-finite values after clipping, or has
        a different number of features than the scaler was fitted on.
    TypeError
        If scaler is not a fitted MinMaxScaler.
    """
    # --- type guard ---------------------------------------------------------
    if not isinstance(scaler, MinMaxScaler):
        raise TypeError(
            f"transform_quantum_features: expected a fitted MinMaxScaler, "
            f"got {type(scaler).__name__}."
        )
    if not hasattr(scaler, "scale_"):
        raise ValueError(
            "transform_quantum_features: scaler has not been fitted yet.  "
            "Call fit_quantum_scaler(X_train) first."
        )

    X_arr = _validate_2d_float(X, "X")

    # --- dimensionality guard -----------------------------------------------
    n_features_in = scaler.n_features_in_
    if X_arr.shape[1] != n_features_in:
        raise ValueError(
            f"transform_quantum_features: expected {n_features_in} features "
            f"(matching the scaler's training data), got {X_arr.shape[1]}."
        )

    # --- transform (no re-fit) ----------------------------------------------
    X_scaled = scaler.transform(X_arr)  # shape preserved, values in ~[0, pi]

    # --- clip to [0, pi] to absorb tiny floating-point overflows from unseen
    #     test values.  This is safe: it uses no label information.
    X_clipped = np.clip(X_scaled, ANGLE_MIN, ANGLE_MAX)

    # --- finite-value guard -------------------------------------------------
    if not np.all(np.isfinite(X_clipped)):
        bad_cols = np.where(~np.all(np.isfinite(X_clipped), axis=0))[0].tolist()
        raise ValueError(
            f"transform_quantum_features: non-finite values found after "
            f"scaling in column indices {bad_cols}.  Check for NaN/Inf in the "
            f"input data."
        )

    return X_clipped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_2d_float(arr: np.ndarray, name: str) -> np.ndarray:
    """Return a 2-D float64 copy of arr, or raise ValueError."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array (n_samples, n_features); "
            f"got ndim={arr.ndim}."
        )
    return arr


# ---------------------------------------------------------------------------
# Step 2 -- ZZFeatureMap builder
# ---------------------------------------------------------------------------

# Qiskit 2.1+ deprecates the ZZFeatureMap *class* in favour of the
# zz_feature_map *function* which returns a plain QuantumCircuit.
# We use the function to avoid DeprecationWarnings and to stay compatible
# with the Qiskit 3.0 roadmap.
from qiskit.circuit.library import zz_feature_map  # new function-based API
from qiskit.circuit import QuantumCircuit


def build_feature_map(n_features: int, reps: int = 2) -> QuantumCircuit:
    """Return a ZZFeatureMap circuit with one qubit per feature.

    Parameters
    ----------
    n_features:
        Number of features (= number of qubits).  Must be >= 2 because the
        ZZ interaction (two-qubit ZZ gates) requires at least two qubits.
    reps:
        Number of repetition layers of the feature map (default 2).
        Higher reps increase circuit depth and expressibility but also cost.

    Returns
    -------
    QuantumCircuit
        A parameterised ZZFeatureMap circuit.  The N_QUBIT_FEATURES parameters
        (named x[0] .. x[n-1]) are Qiskit ``Parameter`` objects; they are
        bound later during kernel computation.

    Notes
    -----
    Entanglement is set to 'linear' to keep two-qubit gate count proportional
    to n_features rather than O(n_features^2).  Linear entanglement creates
    ZZ interactions between adjacent qubit pairs: (0,1), (1,2), ...,
    giving a shallow, hardware-friendly circuit while retaining non-trivial
    feature correlations in the kernel.
    """
    if not isinstance(n_features, int) or n_features < 2:
        raise ValueError(
            f"build_feature_map: n_features must be an integer >= 2; "
            f"got {n_features!r}."
        )
    if not isinstance(reps, int) or reps < 1:
        raise ValueError(
            f"build_feature_map: reps must be a positive integer; "
            f"got {reps!r}."
        )

    # zz_feature_map(feature_dimension=n, reps=r, entanglement='linear'):
    #   - Creates n qubits (one per feature).
    #   - Applies Hadamard + single-qubit RZ rotations with angles x[i].
    #   - Applies CNOT + two-qubit ZZ-rotation gates on adjacent pairs.
    #   - Repeats the above block `reps` times.
    # The returned circuit has n parameters named x[0]..x[n-1].
    circuit = zz_feature_map(
        feature_dimension=n_features,  # 1 qubit per feature
        reps=reps,                     # repetition depth
        entanglement="linear",         # adjacent-pair ZZ interactions only
    )
    return circuit


def save_circuit_diagram(circuit: QuantumCircuit, save_path) -> None:
    """Render the circuit as a matplotlib figure and save to PNG.

    Parameters
    ----------
    circuit:
        The QuantumCircuit to draw (e.g. the output of build_feature_map).
    save_path:
        Path-like object or string pointing to the output PNG file.
        Parent directories are created automatically.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend -- safe for scripts
    import matplotlib.pyplot as plt
    from pathlib import Path

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # draw(output='mpl') requires pylatexenc for LaTeX-style gate labels.
    # fold=-1 disables line-wrapping so the full circuit fits in one row.
    fig = circuit.draw(output="mpl", fold=-1)

    fig.suptitle(
        f"ZZFeatureMap  |  {circuit.num_qubits} qubits, "
        f"reps=2, entanglement=linear",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# Step 3 -- FidelityQuantumKernel builder and kernel verification
# ---------------------------------------------------------------------------

# Qiskit ML 0.9.x: FidelityQuantumKernel accepts a feature_map (QuantumCircuit)
# and optionally a fidelity primitive.  The default fidelity is ComputeUncompute
# backed by Qiskit's reference StatevectorSampler.  StatevectorSampler gives
# exact (noiseless) statevector simulation, which is ideal for correctness
# verification and matches Aer's statevector backend without needing a separate
# Aer primitive.
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel


def build_quantum_kernel(
    feature_map: QuantumCircuit,
    seed: int = 42,
    enforce_psd: bool = True,
) -> FidelityQuantumKernel:
    """Construct a FidelityQuantumKernel backed by exact statevector simulation.

    Parameters
    ----------
    feature_map:
        A parameterised QuantumCircuit (e.g. from build_feature_map).
    seed:
        Random seed for the StatevectorSampler (controls shot noise; the
        statevector itself is deterministic but measurement sampling is not).
    enforce_psd:
        If True (default), project the kernel matrix onto the nearest
        positive-semidefinite matrix.  Set to False for raw kernel
        verification where you want to inspect the unmodified shot-noise
        matrix.

    Returns
    -------
    FidelityQuantumKernel
        Ready to call .evaluate(X_train) or .evaluate(X_train, X_test).
    """
    # StatevectorSampler: Qiskit's built-in exact simulator (noiseless
    # statevector, but measurement outcomes are sampled with shot noise).
    sampler = StatevectorSampler(seed=seed)

    # ComputeUncompute fidelity: computes |<psi|phi>|^2 by appending the
    # inverse of one circuit to the other, then measuring the all-zeros state
    # probability.  This is the standard technique for quantum kernel matrices.
    fidelity = ComputeUncompute(sampler=sampler)

    # FidelityQuantumKernel wraps fidelity computation and provides
    # .evaluate() for computing kernel matrices.
    # enforce_psd=True projects onto the nearest PSD matrix; False leaves
    # the raw (possibly slightly non-PSD due to shot noise) matrix intact.
    kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        enforce_psd=enforce_psd,
    )
    return kernel


def verify_kernel_matrix(
    K: np.ndarray,
    tol_diag: float = 1e-6,
    tol_sym: float = 1e-10,
    tol_range: float = 1e-6,
    tol_psd: float = -1e-8,
) -> dict:
    """Validate invariants of a quantum kernel matrix and return diagnostics.

    Checks performed:
      1. Shape is square (n x n).
      2. Diagonal entries are approximately 1.0 (within tol_diag).
      3. Matrix is symmetric (within tol_sym).
      4. All values lie in [0 - tol_range, 1 + tol_range].
      5. The symmetrized matrix (K + K^T)/2 is positive semidefinite:
         smallest eigenvalue >= tol_psd.

    Parameters
    ----------
    K : np.ndarray
        Square kernel matrix of shape (n, n).
    tol_diag : float
        Maximum allowed deviation of diagonal elements from 1.0.
    tol_sym : float
        Maximum allowed asymmetry |K - K^T|.
    tol_range : float
        Maximum allowed deviation outside [0, 1].
    tol_psd : float
        Minimum allowed eigenvalue of the symmetrized matrix.  Slightly
        negative to absorb floating-point rounding.

    Returns
    -------
    dict
        Diagnostics with keys: shape, diag_min, diag_max, diag_max_dev,
        is_symmetric, max_asymmetry, value_min, value_max, min_eigenvalue,
        is_psd, all_passed, tolerances, and any failure_reasons.

    Raises
    ------
    ValueError
        If any invariant fails.  The error message lists all failures.
    """
    # 1. Square check
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(
            f"Kernel matrix verification FAILED:\n  - Matrix is not square: shape={K.shape}"
        )

    failures = []
    n = K.shape[0]

    # 2. Diagonal ~ 1.0
    diag = np.diag(K)
    diag_min = float(np.min(diag))
    diag_max = float(np.max(diag))
    diag_max_dev = float(np.max(np.abs(diag - 1.0)))
    if diag_max_dev > tol_diag:
        failures.append(
            f"Diagonal deviation from 1.0: max_dev={diag_max_dev:.2e} "
            f"> tol_diag={tol_diag:.2e}"
        )

    # 3. Symmetry
    max_asym = float(np.max(np.abs(K - K.T)))
    is_symmetric = max_asym <= tol_sym
    if not is_symmetric:
        failures.append(
            f"Matrix is not symmetric: max|K - K^T|={max_asym:.2e} "
            f"> tol_sym={tol_sym:.2e}"
        )

    # 4. Values in [0, 1]
    v_min = float(np.min(K))
    v_max = float(np.max(K))
    if v_min < -tol_range or v_max > 1.0 + tol_range:
        failures.append(
            f"Values outside [0,1]: min={v_min:.6f}, max={v_max:.6f} "
            f"(tolerance={tol_range:.2e})"
        )

    # 5. Positive semidefiniteness via eigenvalues of symmetrized matrix
    K_sym = (K + K.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(K_sym)
    min_eig = float(np.min(eigenvalues))
    strictly_psd = bool(min_eig >= -1e-12)
    passes_psd_tolerance = bool(min_eig >= tol_psd)
    is_psd = passes_psd_tolerance

    if not passes_psd_tolerance:
        failures.append(
            f"Not positive semidefinite: min_eigenvalue={min_eig:.2e} "
            f"< tol_psd={tol_psd:.2e}"
        )

    all_passed = len(failures) == 0

    diagnostics = {
        "shape": list(K.shape),
        "n_samples": n,
        "diag_min": round(diag_min, 10),
        "diag_max": round(diag_max, 10),
        "diag_max_deviation": round(diag_max_dev, 10),
        "is_symmetric": is_symmetric,
        "max_asymmetry": round(max_asym, 12),
        "value_min": round(v_min, 10),
        "value_max": round(v_max, 10),
        "min_eigenvalue": round(min_eig, 10),
        "strictly_psd": strictly_psd,
        "passes_psd_tolerance": passes_psd_tolerance,
        "is_psd": is_psd,
        "all_passed": all_passed,
        "tolerances": {
            "tol_diag": tol_diag,
            "tol_sym": tol_sym,
            "tol_range": tol_range,
            "tol_psd": tol_psd,
        },
    }

    if failures:
        diagnostics["failure_reasons"] = failures
        raise ValueError(
            "Kernel matrix verification FAILED:\n  - "
            + "\n  - ".join(failures)
        )

    return diagnostics
