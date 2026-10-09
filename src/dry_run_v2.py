"""Stage 7b Step 3 — Dry run and scaling benchmark script.

Executes a single end-to-end dry run:
  - Seed: 42
  - N_train: 500 samples (from disjoint training pool)
  - N_test: 20,000 samples (from fixed held-out test subset)
  - Features: ['sbytes', 'sload', 'sttl', 'smean']
  - Train-only scaling with MinMaxScaler(0, pi)
  - Chunked test kernel generation (chunk_size = 10,000)
  - Precomputed-kernel SVC(C=1.0) fitting and evaluation
  - Detailed resource profiling (time, memory, cache size)
  - Rigorous scaling projections for N_train in [100, 500, 1000, 2000, 4000]

Usage:
    python -m src.dry_run_v2
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC

from src.config import RESULTS_DIR
from src.dataset_v2 import (
    create_disjoint_partitions,
    load_raw_datasets,
)
from src.kernel_cache_v2 import (
    KernelCacheManager,
    compute_kernel_matrix_chunked,
    compute_statevector_embeddings_chunked,
    hash_array,
)
from src.quantum_model import (
    fit_quantum_scaler,
    transform_quantum_features,
)
from src.quantum_model_v2 import (
    compute_statevector_embeddings,
    compute_statevector_kernel,
)

logger = logging.getLogger(__name__)


def run_dry_run(
    n_train: int = 500,
    n_test: int = 20000,
    seed: int = 42,
    chunk_size: int = 10000,
) -> Dict[str, Any]:
    print(f"[1/6] Loading datasets and generating disjoint partitions (N_train={n_train}, N_test={n_test}, seed={seed})...")
    train_df, test_df = load_raw_datasets()
    features = ["sbytes", "sload", "sttl", "smean"]

    bundle = create_disjoint_partitions(
        train_df=train_df,
        test_df=test_df,
        features=features,
        n_train=n_train,
        seed=seed,
        n_tune=1000,
        n_cal=2000,
        n_test=n_test,
    )

    tracemalloc.start()
    t_start_all = time.perf_counter()

    # 1. Preprocessing (Train-only fit)
    print("[2/6] Fitting quantum scaler strictly on X_train...")
    t0_scale = time.perf_counter()
    scaler = fit_quantum_scaler(bundle.X_train)
    X_train_scaled = transform_quantum_features(scaler, bundle.X_train)
    X_test_scaled = transform_quantum_features(scaler, bundle.X_test)
    t_scale = time.perf_counter() - t0_scale

    # 2. Train Statevector & Kernel
    print(f"[3/6] Computing training statevectors and {n_train}x{n_train} Gram matrix...")
    t0_train_embed = time.perf_counter()
    states_train = compute_statevector_embeddings(X_train_scaled)
    t_train_embed = time.perf_counter() - t0_train_embed

    t0_train_kernel = time.perf_counter()
    K_train = compute_statevector_kernel(states_train)
    t_train_kernel = time.perf_counter() - t0_train_kernel

    # 3. Test Statevector & Kernel (Chunked)
    print(f"[4/6] Computing test statevectors and {n_test}x{n_train} test kernel in chunks of {chunk_size}...")
    t0_test_embed = time.perf_counter()
    states_test = compute_statevector_embeddings_chunked(
        X_test_scaled, chunk_size=chunk_size
    )
    t_test_embed = time.perf_counter() - t0_test_embed

    t0_test_kernel = time.perf_counter()
    K_test = compute_kernel_matrix_chunked(
        states_test, states_train, chunk_size=chunk_size
    )
    t_test_kernel = time.perf_counter() - t0_test_kernel

    # 4. Cache Kernels
    cache_mgr = KernelCacheManager()
    train_hash = hash_array(X_train_scaled)
    test_hash = hash_array(X_test_scaled)

    key_train = cache_mgr.generate_cache_key("train", n_train, n_train, seed, features, train_hash)
    key_test = cache_mgr.generate_cache_key("test", n_test, n_train, seed, features, test_hash, train_hash)

    cache_mgr.put(key_train, K_train, {"split": "train", "seed": seed, "features": features, "data_hash_A": train_hash})
    cache_mgr.put(key_test, K_test, {"split": "test", "seed": seed, "features": features, "data_hash_A": test_hash, "data_hash_B": train_hash})

    train_cache_size_kb = (cache_mgr.cache_dir / f"{key_train}.npy").stat().st_size / 1024.0
    test_cache_size_kb = (cache_mgr.cache_dir / f"{key_test}.npy").stat().st_size / 1024.0

    # 5. Model Fitting & Evaluation
    print(f"[5/6] Fitting precomputed SVC(C=1.0) and predicting on {n_test} test samples...")
    clf = SVC(kernel="precomputed", C=1.0)
    t0_fit = time.perf_counter()
    clf.fit(K_train, bundle.y_train)
    t_fit = time.perf_counter() - t0_fit

    t0_pred = time.perf_counter()
    y_pred = clf.predict(K_test)
    y_score = clf.decision_function(K_test)
    t_pred = time.perf_counter() - t0_pred

    # Metrics
    acc = float(accuracy_score(bundle.y_test, y_pred))
    prec = float(precision_score(bundle.y_test, y_pred, zero_division=0))
    rec = float(recall_score(bundle.y_test, y_pred, zero_division=0))
    f1 = float(f1_score(bundle.y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(bundle.y_test, y_score))
    tn, fp, fn, tp = confusion_matrix(bundle.y_test, y_pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Memory profiling
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024.0 * 1024.0)

    # 6. Scaling Projections
    print("[6/6] Computing scaling projections for full learning curve...")
    embedding_rate_per_sample = t_test_embed / n_test  # seconds per sample
    train_sizes_grid = [100, 500, 1000, 2000, 4000]
    scaling_projections = {}

    for n in train_sizes_grid:
        est_train_embed = n * embedding_rate_per_sample
        # matmul scales as O(N_test * N_train)
        est_test_matmul = t_test_kernel * (n / n_train)
        # Train kernel matmul
        est_train_matmul = t_train_kernel * ((n / n_train) ** 2)
        # SVC fit time typically O(N^2) to O(N^3) for precomputed kernel
        est_fit_time = t_fit * ((n / n_train) ** 2)
        total_time_per_seed = est_train_embed + est_train_matmul + est_test_matmul + est_fit_time + t_pred
        # Memory for test matrix: N_test * N_train * 8 bytes
        test_matrix_mb = (n_test * n * 8) / (1024.0 * 1024.0)

        scaling_projections[f"N_{n}"] = {
            "n_train": n,
            "n_test": n_test,
            "est_train_embed_s": round(est_train_embed, 3),
            "est_train_kernel_s": round(est_train_matmul, 4),
            "est_test_kernel_s": round(est_test_matmul, 3),
            "est_fit_time_s": round(est_fit_time, 3),
            "est_pred_time_s": round(t_pred, 3),
            "est_total_per_seed_s": round(total_time_per_seed, 2),
            "test_matrix_memory_mb": round(test_matrix_mb, 2),
        }

    total_time_5_seeds_all_sizes = sum(
        scaling_projections[f"N_{n}"]["est_total_per_seed_s"] for n in train_sizes_grid
    ) * 5.0

    report = {
        "step": "Stage 7b Step 3 — Dry Run and Scaling Benchmark",
        "dry_run_parameters": {
            "seed": seed,
            "n_train": n_train,
            "n_tune": 1000,
            "n_cal": 2000,
            "n_test": n_test,
            "chunk_size": chunk_size,
            "features": features,
            "classifier": "SVC(kernel='precomputed', C=1.0)",
        },
        "timings_s": {
            "preprocessing_fit_transform": round(t_scale, 4),
            "train_statevector_embedding": round(t_train_embed, 4),
            "train_kernel_generation": round(t_train_kernel, 4),
            "test_statevector_embedding": round(t_test_embed, 4),
            "test_kernel_generation": round(t_test_kernel, 4),
            "model_fit": round(t_fit, 4),
            "model_predict": round(t_pred, 4),
            "total_dry_run_time": round(time.perf_counter() - t_start_all, 4),
        },
        "resource_usage": {
            "peak_memory_mb": round(peak_mem_mb, 2),
            "train_cache_size_kb": round(train_cache_size_kb, 2),
            "test_cache_size_kb": round(test_cache_size_kb, 2),
            "test_kernel_memory_mb": round(K_test.nbytes / (1024.0 * 1024.0), 2),
        },
        "classification_performance": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "scaling_projections": scaling_projections,
        "full_experiment_estimate": {
            "total_estimated_compute_5_seeds_all_sizes_s": round(total_time_5_seeds_all_sizes, 1),
            "total_estimated_compute_minutes": round(total_time_5_seeds_all_sizes / 60.0, 2),
            "max_memory_per_run_mb": round(scaling_projections["N_4000"]["test_matrix_memory_mb"] + 50.0, 1),
            "feasibility_assessment": (
                "Highly feasible. Evaluating all 5 seeds across all 5 training sizes "
                f"([100, 500, 1000, 2000, 4000]) on 20,000 test samples will take approximately "
                f"{total_time_5_seeds_all_sizes / 60.0:.1f} minutes total, with peak RAM < 700 MB. "
                "The test embeddings (20,000 samples) need only be computed once and reused across "
                "different training sizes, further reducing actual compute time."
            ),
        },
    }

    out_path = RESULTS_DIR / "dry_run_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nDry run completed! Report saved to {out_path}")
    print(f"Total dry run time: {report['timings_s']['total_dry_run_time']}s")
    print(f"Test Accuracy: {report['classification_performance']['accuracy']:.4f}, F1: {report['classification_performance']['f1']:.4f}")
    print(f"Peak memory: {report['resource_usage']['peak_memory_mb']} MB")
    print(f"Projected full 5-seed grid runtime: {report['full_experiment_estimate']['total_estimated_compute_minutes']} min")

    return report


if __name__ == "__main__":
    run_dry_run()
