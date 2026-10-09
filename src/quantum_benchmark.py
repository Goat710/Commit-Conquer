"""Quantum kernel benchmarking module for Quantum CyberShield — Stage 5 Step 4.

Runs restricted sequential benchmarks across candidate training sample sizes
n in [50, 100, 200, 400] on the official UNSW-NB15 training dataset.

Enforces strict compute limits:
  - Training data only; test labels are never accessed or leaked.
  - Sequential execution (one size at a time).
  - Max training sample size <= 400.
  - Stopping condition: halt if completed run > 120s or projected next run > 180s.
  - Clear separation of empirical measurements from extrapolations.
  - Results written to results/quantum_benchmark.json.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn
import qiskit
import qiskit_machine_learning

from src.config import (
    N_QUBIT_FEATURES,
    RANDOM_SEED,
    RAW_TRAIN_CSV,
    RESULTS_DIR,
)
from src.quantum_model import (
    build_feature_map,
    build_quantum_kernel,
    fit_quantum_scaler,
    transform_quantum_features,
)

BENCHMARK_SIZES = [50, 100, 200, 400]
MAX_RUNTIME_PER_RUN_S = 120.0
MAX_PROJECTED_NEXT_RUN_S = 180.0
MAX_SAMPLES = 400


def get_software_versions() -> Dict[str, str]:
    """Collect installed library and environment versions."""
    try:
        import qiskit_aer
        aer_ver = getattr(qiskit_aer, "__version__", "unknown")
    except ImportError:
        aer_ver = "not_installed"

    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "qiskit": qiskit.__version__,
        "qiskit_machine_learning": qiskit_machine_learning.__version__,
        "qiskit_aer": aer_ver,
    }


def estimate_matrix_memory(n_rows: int, n_cols: int) -> Dict[str, Any]:
    """Calculate exact memory requirements for an n_rows x n_cols float64 matrix."""
    entries = int(n_rows * n_cols)
    bytes_count = entries * 8  # 8 bytes per float64
    kb = bytes_count / 1024.0
    mb = bytes_count / (1024.0 * 1024.0)
    gb = bytes_count / (1024.0 * 1024.0 * 1024.0)
    return {
        "entries": entries,
        "bytes": bytes_count,
        "kb": round(kb, 4),
        "mb": round(mb, 4),
        "gb": round(gb, 6),
    }


def project_next_runtime(
    prev_n: int,
    prev_time_s: float,
    next_n: int,
) -> float:
    """Project next runtime assuming O(n^2) scaling for kernel matrix evaluation.

    Parameters
    ----------
    prev_n : int
        Sample count of the completed benchmark.
    prev_time_s : float
        Elapsed wall-clock time of the completed benchmark in seconds.
    next_n : int
        Sample count of the proposed benchmark.

    Returns
    -------
    float
        Projected wall-clock time in seconds.
    """
    if prev_n <= 0 or prev_time_s < 0:
        raise ValueError(f"Invalid inputs: prev_n={prev_n}, prev_time_s={prev_time_s}")
    scaling_factor = (next_n / prev_n) ** 2
    return prev_time_s * scaling_factor


def should_stop_benchmark(
    prev_time_s: Optional[float],
    prev_n: Optional[int],
    next_n: int,
    max_run_time_s: float = MAX_RUNTIME_PER_RUN_S,
    max_projected_time_s: float = MAX_PROJECTED_NEXT_RUN_S,
) -> Tuple[bool, Optional[str], Optional[float]]:
    """Determine whether the next benchmark should be skipped based on safety thresholds.

    Returns
    -------
    (should_stop, reason, projected_time_s)
    """
    if prev_time_s is None or prev_n is None:
        return False, None, None

    if prev_time_s > max_run_time_s:
        reason = (
            f"Previous run (n={prev_n}) elapsed time {prev_time_s:.2f}s "
            f"exceeded threshold of {max_run_time_s:.1f}s."
        )
        return True, reason, None

    projected_time = project_next_runtime(prev_n, prev_time_s, next_n)
    if projected_time > max_projected_time_s:
        reason = (
            f"Projected runtime for n={next_n} is {projected_time:.2f}s, "
            f"which exceeds limit of {max_projected_time_s:.1f}s "
            f"(based on n={prev_n} taking {prev_time_s:.2f}s)."
        )
        return True, reason, projected_time

    return False, None, projected_time


def compute_extrapolations(
    completed_runs: List[Dict[str, Any]],
    n_full_train: int = 82332,
    n_full_test: int = 175341,
) -> Dict[str, Any]:
    """Compute empirical cost per entry and project full-scale costs.

    Explicitly labeled as estimates with stated assumptions.
    """
    if not completed_runs:
        return {"error": "No completed runs to extrapolate from"}

    # Use the largest completed run for empirical cost per entry
    largest_run = max(completed_runs, key=lambda r: r["n_samples"])
    n = largest_run["n_samples"]
    t = largest_run["elapsed_time_s"]
    entries = n * n
    cost_per_entry_s = t / entries

    # Full training kernel (82,332 x 82,332)
    full_train_mem = estimate_matrix_memory(n_full_train, n_full_train)
    full_train_proj_s = cost_per_entry_s * full_train_mem["entries"]

    # Full test evaluation kernel (82,332 x 175,341)
    full_eval_mem = estimate_matrix_memory(n_full_train, n_full_test)
    full_eval_proj_s = cost_per_entry_s * full_eval_mem["entries"]

    return {
        "basis_run": {
            "n_samples": n,
            "elapsed_time_s": t,
            "entries_computed": entries,
            "cost_per_entry_seconds": round(cost_per_entry_s, 8),
        },
        "assumptions": (
            "Extrapolations assume strictly quadratic O(n^2) scaling of kernel "
            "evaluations with constant overhead per entry on identical single-thread "
            "CPU hardware. In reality, memory pressure and cache misses would increase "
            "cost further for large matrices."
        ),
        "full_dataset_projections": {
            "full_training_kernel": {
                "shape": [n_full_train, n_full_train],
                "memory": full_train_mem,
                "projected_time_seconds": round(full_train_proj_s, 2),
                "projected_time_hours": round(full_train_proj_s / 3600.0, 2),
                "projected_time_days": round(full_train_proj_s / 86400.0, 2),
                "feasible": False,
                "reason": (
                    f"Requires {full_train_mem['gb']:.2f} GB RAM and ~"
                    f"{full_train_proj_s / 86400.0:.1f} days of compute on CPU."
                ),
            },
            "full_test_evaluation_kernel": {
                "shape": [n_full_train, n_full_test],
                "memory": full_eval_mem,
                "projected_time_seconds": round(full_eval_proj_s, 2),
                "projected_time_hours": round(full_eval_proj_s / 3600.0, 2),
                "projected_time_days": round(full_eval_proj_s / 86400.0, 2),
                "feasible": False,
                "reason": (
                    f"Requires {full_eval_mem['gb']:.2f} GB RAM and ~"
                    f"{full_eval_proj_s / 86400.0:.1f} days of compute on CPU."
                ),
            },
        },
        "subset_projections": {
            "candidate_n100": {
                "train_eval": {
                    "shape": [100, 100],
                    "memory_mb": estimate_matrix_memory(100, 100)["mb"],
                    "projected_time_s": round(cost_per_entry_s * 10000, 2),
                },
                "test_eval_100_samples": {
                    "shape": [100, 100],
                    "memory_mb": estimate_matrix_memory(100, 100)["mb"],
                    "projected_time_s": round(cost_per_entry_s * 10000, 2),
                },
            },
            "candidate_n200": {
                "train_eval": {
                    "shape": [200, 200],
                    "memory_mb": estimate_matrix_memory(200, 200)["mb"],
                    "projected_time_s": round(cost_per_entry_s * 40000, 2),
                },
                "test_eval_100_samples": {
                    "shape": [100, 200],
                    "memory_mb": estimate_matrix_memory(100, 200)["mb"],
                    "projected_time_s": round(cost_per_entry_s * 20000, 2),
                },
            },
        },
    }


def generate_recommendations(
    completed_runs: List[Dict[str, Any]],
    skipped_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Recommend N_train_q and N_test for Stage 5 Step 5 based on measured costs."""
    if not completed_runs:
        return {"error": "No completed runs to base recommendations on"}

    # Find the largest completed size
    largest_n = max(r["n_samples"] for r in completed_runs)
    largest_run = next(r for r in completed_runs if r["n_samples"] == largest_n)

    # If n=200 completed within reasonable time, recommend up to 200, else 100
    if largest_n >= 200 and largest_run["elapsed_time_s"] <= 120.0:
        recommended_n_train = 200
        recommended_n_test = 100
        recommended_n_val = 50
        rationale = (
            f"n=200 completed in {largest_run['elapsed_time_s']:.2f}s (< 120s limit). "
            f"Training kernel (200x200) takes ~{largest_run['elapsed_time_s']:.1f}s, and test "
            f"inference (100x200) takes approximately half that (~{largest_run['elapsed_time_s']/2:.1f}s). "
            f"This fits comfortably within a practical 3-minute turnaround for Stage 5 Step 5."
        )
    elif largest_n >= 100:
        recommended_n_train = 100
        recommended_n_test = 100
        recommended_n_val = 50
        rationale = (
            f"n={largest_n} completed successfully. For rapid parameter tuning across C in "
            f"[0.1, 1.0, 10.0] with precomputed kernel caching, n_train=100 provides high confidence "
            f"with low runtime risk."
        )
    else:
        recommended_n_train = 50
        recommended_n_test = 50
        recommended_n_val = 25
        rationale = "Only small sizes completed within time limits."

    return {
        "recommended_N_train_q": recommended_n_train,
        "recommended_N_test": recommended_n_test,
        "recommended_N_val": recommended_n_val,
        "practical_runtime_budget_minutes": 5,
        "rationale": rationale,
        "hard_ceiling_compliance": {
            "max_training_subsample_limit": 200,
            "max_validation_subsample_limit": 50,
            "max_test_subsample_limit": 100,
            "complies_with_limits": bool(
                recommended_n_train <= 200
                and recommended_n_val <= 50
                and recommended_n_test <= 100
            ),
        },
    }


def run_quantum_benchmark(
    sizes: Optional[List[int]] = None,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the restricted quantum benchmark sequentially.

    Parameters
    ----------
    sizes : list of int, optional
        Sample sizes to benchmark. Defaults to BENCHMARK_SIZES [50, 100, 200, 400].
    save_path : Path, optional
        Path where results JSON is saved. Defaults to results/quantum_benchmark.json.

    Returns
    -------
    dict
        Full benchmark results including timings, memory, versions, and recommendations.
    """
    if sizes is None:
        sizes = BENCHMARK_SIZES
    if save_path is None:
        save_path = RESULTS_DIR / "quantum_benchmark.json"

    # Enforce hard upper bound on samples
    if any(s > MAX_SAMPLES for s in sizes):
        raise ValueError(f"Requested size exceeds hard limit of {MAX_SAMPLES}")

    print("=" * 70)
    print("STAGE 5 STEP 4: RESTRICTED QUANTUM KERNEL BENCHMARK")
    print(f"Candidate sizes in sequence: {sizes}")
    print(f"Safety limits: max_run <= {MAX_RUNTIME_PER_RUN_S}s, max_projected <= {MAX_PROJECTED_NEXT_RUN_S}s")
    print("=" * 70)

    # 1. Load training data (TRAINING ONLY - NO TEST DATA)
    print("\n[1/3] Loading UNSW-NB15 training dataset...")
    train_df = pd.read_csv(RAW_TRAIN_CSV)
    fs_path = RESULTS_DIR / "feature_selection.json"
    with open(fs_path, "r", encoding="utf-8") as f:
        fs_data = json.load(f)
    selected_features = fs_data["runs"]["all_features"]["selected_features"]
    print(f"       Features ({len(selected_features)}): {selected_features}")

    X_train_full = train_df[selected_features].values.astype(np.float64)
    y_train_full = train_df["label"].values.astype(int)
    n_full_train = len(X_train_full)
    print(f"       Training set available: {n_full_train} samples")

    # 2. Build feature map & kernel
    feature_map = build_feature_map(n_features=N_QUBIT_FEATURES, reps=2)
    kernel = build_quantum_kernel(feature_map, seed=RANDOM_SEED, enforce_psd=True)

    completed_runs: List[Dict[str, Any]] = []
    skipped_runs: List[Dict[str, Any]] = []

    prev_time: Optional[float] = None
    prev_n: Optional[int] = None

    print("\n[2/3] Running sequential benchmarks...")
    for idx, n in enumerate(sizes, start=1):
        print(f"\n--- Benchmark {idx}/{len(sizes)}: n = {n} ---")

        # Check stopping criteria
        should_stop, stop_reason, projected_time = should_stop_benchmark(
            prev_time_s=prev_time,
            prev_n=prev_n,
            next_n=n,
        )

        mem_info = estimate_matrix_memory(n, n)

        if should_stop:
            print(f"  [SKIPPED] {stop_reason}")
            skipped_run = {
                "n_samples": n,
                "status": "skipped",
                "skip_reason": stop_reason,
                "projected_time_s": round(projected_time, 2) if projected_time else None,
                "matrix_shape": [n, n],
                "matrix_memory": mem_info,
            }
            skipped_runs.append(skipped_run)
            # Once a size is skipped, all subsequent larger sizes must also be skipped
            for remaining_n in sizes[idx:]:
                rem_mem = estimate_matrix_memory(remaining_n, remaining_n)
                rem_proj = project_next_runtime(prev_n, prev_time, remaining_n) if (prev_n and prev_time) else None
                skipped_runs.append({
                    "n_samples": remaining_n,
                    "status": "skipped",
                    "skip_reason": f"Skipped because predecessor n={n} was stopped.",
                    "projected_time_s": round(rem_proj, 2) if rem_proj else None,
                    "matrix_shape": [remaining_n, remaining_n],
                    "matrix_memory": rem_mem,
                })
            break

        # Select stratified subset strictly from training data
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=RANDOM_SEED)
        sample_idx, _ = next(splitter.split(X_train_full, y_train_full))
        X_sub = X_train_full[sample_idx]
        y_sub = y_train_full[sample_idx]

        # Scale strictly on this training subsample
        scaler = fit_quantum_scaler(X_sub)
        X_scaled = transform_quantum_features(scaler, X_sub)

        # Compute kernel matrix with precise timing
        print(f"  Evaluating {n}x{n} kernel matrix ({mem_info['entries']} entries, {mem_info['mb']:.4f} MB)...")
        t_start = time.perf_counter()
        K = kernel.evaluate(X_scaled)
        t_elapsed = time.perf_counter() - t_start

        print(f"  [COMPLETED] Wall-clock time: {t_elapsed:.2f}s (shape={K.shape})")

        completed_run = {
            "n_samples": n,
            "status": "completed",
            "elapsed_time_s": round(t_elapsed, 4),
            "matrix_shape": list(K.shape),
            "matrix_entries": mem_info["entries"],
            "matrix_memory": mem_info,
            "label_distribution": {
                "benign_0": int(np.sum(y_sub == 0)),
                "attack_1": int(np.sum(y_sub == 1)),
            },
        }
        completed_runs.append(completed_run)

        prev_time = t_elapsed
        prev_n = n

    # 3. Assemble report
    print("\n[3/3] Compiling report and extrapolations...")
    software_versions = get_software_versions()
    backend_info = {
        "primitive": "StatevectorSampler",
        "simulation": "exact_statevector",
        "shots": 1024,
        "seed": RANDOM_SEED,
        "enforce_psd": True,
        "feature_map": {
            "type": "ZZFeatureMap",
            "n_qubits": N_QUBIT_FEATURES,
            "reps": 2,
            "entanglement": "linear",
        },
    }

    extrapolations = compute_extrapolations(completed_runs, n_full_train=n_full_train)
    recommendations = generate_recommendations(completed_runs, skipped_runs)

    report = {
        "benchmark_title": "Stage 5 Step 4 - Restricted Quantum Kernel Benchmark",
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "safety_limits": {
            "candidate_sizes": sizes,
            "max_runtime_per_run_s": MAX_RUNTIME_PER_RUN_S,
            "max_projected_next_run_s": MAX_PROJECTED_NEXT_RUN_S,
            "max_samples_hard_limit": MAX_SAMPLES,
            "data_source": "UNSW-NB15 training-set only (no test data)",
        },
        "software_versions": software_versions,
        "backend": backend_info,
        "completed_runs": completed_runs,
        "skipped_runs": skipped_runs,
        "extrapolations": extrapolations,
        "recommendations": recommendations,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nBenchmark report successfully saved to: {save_path}")

    return report


if __name__ == "__main__":
    run_quantum_benchmark()
