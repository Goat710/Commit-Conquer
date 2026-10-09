"""Stage 7b Step 5 — Learning Curves Sweep across training sizes and seeds.

Evaluates QSVC, Classical SVM (RBF), and Classical Random Forest across:
  - Training sizes: [100, 500, 1000, 2000, 4000]
  - Seeds: [42, 100, 2024, 777, 999]
  - Fixed stratified test set: N_test = 20,000 (RAW_TEST_CSV)
  - Identical test examples for all models and sizes
  - Strict train-only preprocessing discipline
  - Real measured execution timings (fit, infer, kernel) and peak memory
  - Predictions saved for paired statistical analysis (Step 6) and disagreement audit (Step 8)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import RESULTS_DIR
from src.dataset_v2 import (
    DEFAULT_SEEDS,
    FIXED_TEST_SIZE,
    PRIMARY_FEATURES,
    TRAIN_SIZES,
    create_disjoint_partitions,
    load_raw_datasets,
)
from src.kernel_cache_v2 import (
    KernelCacheManager,
    compute_kernel_matrix_chunked,
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

LEARNING_CURVES_JSON: Path = RESULTS_DIR / "learning_curves_v2.json"
PREDICTIONS_V2_CSV: Path = RESULTS_DIR / "predictions_v2.csv"

BEST_QSVC_C = 10.0
BEST_SVM_C = 100.0
BEST_SVM_GAMMA = "scale"
BEST_RF_TREES = 100
BEST_RF_DEPTH = 10


def evaluate_single_run(
    bundle,
    n_train: int,
    seed: int,
    cache_mgr: KernelCacheManager,
) -> Dict[str, Any]:
    """Train and evaluate QSVC, SVM, and RF on identical partitions."""
    X_train = bundle.X_train[:n_train]
    y_train = bundle.y_train[:n_train]
    X_test = bundle.X_test
    y_test = bundle.y_test

    run_results: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # 1. Classical Random Forest
    # -------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=BEST_RF_TREES,
        max_depth=BEST_RF_DEPTH,
        random_state=seed,
        n_jobs=-1,
    )
    t0_rf_fit = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_fit_time = time.perf_counter() - t0_rf_fit

    t0_rf_infer = time.perf_counter()
    rf_preds = rf.predict(X_test)
    rf_scores = rf.predict_proba(X_test)[:, 1]
    rf_infer_time = time.perf_counter() - t0_rf_infer

    tn, fp, fn, tp = confusion_matrix(y_test, rf_preds).ravel()
    run_results["rf"] = {
        "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
        "precision": round(float(precision_score(y_test, rf_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, rf_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, rf_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, rf_scores)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0,
        "fit_time_s": round(rf_fit_time, 4),
        "infer_time_s": round(rf_infer_time, 4),
        "y_pred": rf_preds,
        "y_score": rf_scores,
    }

    # -------------------------------------------------------------
    # 2. Classical SVM (StandardScaler on X_train only)
    # -------------------------------------------------------------
    scaler_svm = StandardScaler()
    scaler_svm.fit(X_train)
    X_tr_svm = scaler_svm.transform(X_train)
    X_te_svm = scaler_svm.transform(X_test)

    svm = SVC(kernel="rbf", C=BEST_SVM_C, gamma=BEST_SVM_GAMMA)
    t0_svm_fit = time.perf_counter()
    svm.fit(X_tr_svm, y_train)
    svm_fit_time = time.perf_counter() - t0_svm_fit

    t0_svm_infer = time.perf_counter()
    svm_preds = svm.predict(X_te_svm)
    svm_scores = svm.decision_function(X_te_svm)
    svm_infer_time = time.perf_counter() - t0_svm_infer

    tn, fp, fn, tp = confusion_matrix(y_test, svm_preds).ravel()
    run_results["svm"] = {
        "accuracy": round(float(accuracy_score(y_test, svm_preds)), 4),
        "precision": round(float(precision_score(y_test, svm_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, svm_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, svm_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, svm_scores)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0,
        "fit_time_s": round(svm_fit_time, 4),
        "infer_time_s": round(svm_infer_time, 4),
        "y_pred": svm_preds,
        "y_score": svm_scores,
    }

    # -------------------------------------------------------------
    # 3. Quantum Support Vector Classifier (Exact Statevector Kernel)
    # -------------------------------------------------------------
    scaler_q = fit_quantum_scaler(X_train)
    X_tr_q = transform_quantum_features(scaler_q, X_train)
    X_te_q = transform_quantum_features(scaler_q, X_test)

    # Embedding & Kernel
    t0_k_tr = time.perf_counter()
    states_train = compute_statevector_embeddings(X_tr_q)
    K_train = compute_statevector_kernel(states_train)
    k_tr_time = time.perf_counter() - t0_k_tr

    t0_k_te = time.perf_counter()
    states_test = compute_statevector_embeddings(X_te_q)
    K_test = compute_statevector_kernel(states_test, states_train)
    k_te_time = time.perf_counter() - t0_k_te

    qsvc = SVC(kernel="precomputed", C=BEST_QSVC_C)
    t0_q_fit = time.perf_counter()
    qsvc.fit(K_train, y_train)
    qsvc_fit_time = time.perf_counter() - t0_q_fit

    t0_q_infer = time.perf_counter()
    qsvc_preds = qsvc.predict(K_test)
    qsvc_scores = qsvc.decision_function(K_test)
    qsvc_infer_time = time.perf_counter() - t0_q_infer

    tn, fp, fn, tp = confusion_matrix(y_test, qsvc_preds).ravel()
    run_results["qsvc"] = {
        "accuracy": round(float(accuracy_score(y_test, qsvc_preds)), 4),
        "precision": round(float(precision_score(y_test, qsvc_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, qsvc_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, qsvc_preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, qsvc_scores)), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0,
        "fit_time_s": round(qsvc_fit_time, 4),
        "infer_time_s": round(qsvc_infer_time, 4),
        "kernel_train_time_s": round(k_tr_time, 4),
        "kernel_test_time_s": round(k_te_time, 4),
        "total_train_time_s": round(k_tr_time + qsvc_fit_time, 4),
        "total_infer_time_s": round(k_te_time + qsvc_infer_time, 4),
        "y_pred": qsvc_preds,
        "y_score": qsvc_scores,
    }

    return run_results


def run_learning_curves_sweep(
    train_sizes: List[int] = TRAIN_SIZES,
    seeds: List[int] = DEFAULT_SEEDS,
    save_predictions_size: int = 4000,
) -> Dict[str, Any]:
    """Execute complete learning curves sweep across all seeds and sizes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_raw_datasets()
    cache_mgr = KernelCacheManager()

    tracemalloc.start()
    t_start_sweep = time.perf_counter()

    attempted_runs = []
    runs_by_size: Dict[int, Dict[str, List[Dict[str, Any]]]] = {
        s: {"qsvc": [], "svm": [], "rf": []} for s in train_sizes
    }

    # Store predictions for paired analysis (from seed 42 at save_predictions_size)
    saved_predictions_data = None

    for seed in seeds:
        print(f"\n[Processing seed {seed}] Initializing maximum training pool (N=4000)...")
        # Generate base partition bundle with max training size (4000)
        bundle_max = create_disjoint_partitions(
            train_df=train_df,
            test_df=test_df,
            features=PRIMARY_FEATURES,
            n_train=max(train_sizes),
            seed=seed,
            n_tune=1000,
            n_cal=2000,
            n_test=FIXED_TEST_SIZE,
        )

        for n_train in train_sizes:
            print(f"  -> Evaluating N_train={n_train:4d} (seed={seed})...", end="", flush=True)
            t0_run = time.perf_counter()
            try:
                res = evaluate_single_run(
                    bundle=bundle_max,
                    n_train=n_train,
                    seed=seed,
                    cache_mgr=cache_mgr,
                )
                t_run = time.perf_counter() - t0_run

                # Record attempt
                attempted_runs.append({
                    "n_train": n_train,
                    "seed": seed,
                    "status": "success",
                    "elapsed_s": round(t_run, 4),
                })

                # Store clean metrics without arrays
                for model_key in ["qsvc", "svm", "rf"]:
                    m_data = dict(res[model_key])
                    y_pred = m_data.pop("y_pred")
                    y_score = m_data.pop("y_score")
                    runs_by_size[n_train][model_key].append(m_data)

                    # Save predictions for paired analysis at save_predictions_size, seed 42
                    if seed == 42 and n_train == save_predictions_size:
                        if saved_predictions_data is None:
                            saved_predictions_data = {
                                "sample_id": bundle_max.test_indices,
                                "y_true": bundle_max.y_test,
                            }
                        saved_predictions_data[f"{model_key}_pred"] = y_pred
                        saved_predictions_data[f"{model_key}_score"] = y_score

                q_f1 = res["qsvc"]["f1"]
                s_f1 = res["svm"]["f1"]
                r_f1 = res["rf"]["f1"]
                print(f" Done ({t_run:.2f}s) | F1 -> QSVC: {q_f1:.4f}, SVM: {s_f1:.4f}, RF: {r_f1:.4f}")

            except Exception as e:
                print(f" FAILED ({e})")
                attempted_runs.append({
                    "n_train": n_train,
                    "seed": seed,
                    "status": "failed",
                    "error": str(e),
                })

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem / (1024.0 * 1024.0)
    total_sweep_time = time.perf_counter() - t_start_sweep

    # Save predictions DataFrame
    if saved_predictions_data is not None:
        pred_df = pd.DataFrame(saved_predictions_data)
        pred_df.to_csv(PREDICTIONS_V2_CSV, index=False)
        print(f"\nSaved paired predictions to {PREDICTIONS_V2_CSV} (shape: {pred_df.shape})")

    # Aggregate summaries across seeds
    summary_by_size = {}
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "fit_time_s", "infer_time_s"]
    for n in train_sizes:
        summary_by_size[f"N_{n}"] = {}
        for m in ["qsvc", "svm", "rf"]:
            m_runs = runs_by_size[n][m]
            summary_by_size[f"N_{n}"][m] = {}
            for k in metric_keys:
                vals = [r[k] for r in m_runs if k in r]
                summary_by_size[f"N_{n}"][m][f"{k}_mean"] = round(float(np.mean(vals)), 4)
                summary_by_size[f"N_{n}"][m][f"{k}_std"] = round(float(np.std(vals)), 4)

    report = {
        "step": "Stage 7b Step 5 — Learning Curves Evaluation",
        "configuration": {
            "train_sizes": train_sizes,
            "seeds": seeds,
            "n_test_subset": FIXED_TEST_SIZE,
            "features": PRIMARY_FEATURES,
            "models": {
                "qsvc": {"kernel": "exact_statevector", "C": BEST_QSVC_C},
                "svm": {"kernel": "rbf", "C": BEST_SVM_C, "gamma": BEST_SVM_GAMMA},
                "rf": {"n_estimators": BEST_RF_TREES, "max_depth": BEST_RF_DEPTH},
            },
        },
        "resources": {
            "total_sweep_time_s": round(total_sweep_time, 2),
            "total_sweep_minutes": round(total_sweep_time / 60.0, 2),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "total_runs_attempted": len(attempted_runs),
            "successful_runs": sum(1 for r in attempted_runs if r["status"] == "success"),
            "failed_runs": sum(1 for r in attempted_runs if r["status"] == "failed"),
            "skipped_runs": 0,
        },
        "summary_by_training_size": summary_by_size,
        "per_seed_runs": runs_by_size,
        "attempted_runs_log": attempted_runs,
    }

    with open(LEARNING_CURVES_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nLearning curves report saved to {LEARNING_CURVES_JSON}")
    return report


if __name__ == "__main__":
    run_learning_curves_sweep()
