"""Stage 5 Step 5 — Restricted QSVC Experiment.

Executes a small, reproducible Quantum Support Vector Classifier (QSVC)
experiment on a restricted subset of UNSW-NB15 with strict safety and anti-leakage controls:
  - Maximum 100 training samples, 50 validation samples, 100 test samples.
  - Training and validation drawn strictly from the official training partition.
  - Official test partition untouched until model choices are frozen.
  - Quantum feature scaler fitted exclusively on the 100 training samples.
  - ZZFeatureMap (4 qubits, reps=2, linear entanglement).
  - FidelityQuantumKernel with StatevectorSampler (1024 shots, seed 42, enforce_psd=True).
  - Precomputed Gram matrix caching to evaluate C in [0.1, 1.0, 10.0] efficiently.
  - Selection metric: validation F1-score for ATTACK class (tie-break on smaller C).
  - Final frozen model evaluated once on the 100 test samples.
  - Results saved to results/quantum_experiment.json and associated plots.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from qiskit_machine_learning.algorithms import QSVC

from src.config import (
    N_QUBIT_FEATURES,
    RANDOM_SEED,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
    RESULTS_DIR,
)
from src.evaluation import (
    evaluate,
    plot_confusion_matrix,
    plot_roc_curve,
    save_evaluation_results,
)
from src.quantum_benchmark import get_software_versions
from src.quantum_model import (
    build_feature_map,
    build_quantum_kernel,
    fit_quantum_scaler,
    transform_quantum_features,
)

# Hard Experiment Ceilings
MAX_TRAIN_SAMPLES = 100
MAX_VAL_SAMPLES = 50
MAX_TEST_SAMPLES = 100
CANDIDATE_C_VALUES = [0.1, 1.0, 10.0]
MAX_PROJECTED_KERNEL_TIME_S = 180.0

QSVC_RESULTS_JSON: Path = RESULTS_DIR / "quantum_experiment.json"
QSVC_CM_PLOT: Path = RESULTS_DIR / "quantum_qsvc_confusion.png"
QSVC_ROC_PLOT: Path = RESULTS_DIR / "quantum_qsvc_roc.png"


def split_quantum_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: List[str],
    n_train: int = MAX_TRAIN_SAMPLES,
    n_val: int = MAX_VAL_SAMPLES,
    n_test: int = MAX_TEST_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Split train/val strictly from train_df and test from test_df using stratified sampling.

    Guarantees:
      - Validation subset comes from the official training partition.
      - Training and validation subsets are strictly disjoint.
      - Test partition is sampled separately from test_df.
      - Never uses test labels or test data during train/val selection.
    """
    if n_train > MAX_TRAIN_SAMPLES:
        raise ValueError(f"n_train ({n_train}) exceeds hard ceiling of {MAX_TRAIN_SAMPLES}")
    if n_val > MAX_VAL_SAMPLES:
        raise ValueError(f"n_val ({n_val}) exceeds hard ceiling of {MAX_VAL_SAMPLES}")
    if n_test > MAX_TEST_SAMPLES:
        raise ValueError(f"n_test ({n_test}) exceeds hard ceiling of {MAX_TEST_SAMPLES}")

    X_train_full = train_df[selected_features].values.astype(np.float64)
    y_train_full = train_df["label"].values.astype(int)

    # 1. Stratified draw of (n_train + n_val) from official train partition
    pool_size = n_train + n_val
    split_pool = StratifiedShuffleSplit(n_splits=1, train_size=pool_size, random_state=random_seed)
    pool_idx, _ = next(split_pool.split(X_train_full, y_train_full))

    X_pool = X_train_full[pool_idx]
    y_pool = y_train_full[pool_idx]

    # 2. Split pool into disjoint train and validation subsets
    split_train_val = StratifiedShuffleSplit(
        n_splits=1,
        train_size=n_train,
        test_size=n_val,
        random_state=random_seed,
    )
    tr_rel_idx, val_rel_idx = next(split_train_val.split(X_pool, y_pool))

    X_train_sub = X_pool[tr_rel_idx]
    y_train_sub = y_pool[tr_rel_idx]
    X_val_sub = X_pool[val_rel_idx]
    y_val_sub = y_pool[val_rel_idx]

    # 3. Stratified draw of n_test from official test partition
    X_test_full = test_df[selected_features].values.astype(np.float64)
    y_test_full = test_df["label"].values.astype(int)

    split_test = StratifiedShuffleSplit(n_splits=1, train_size=n_test, random_state=random_seed)
    test_idx, _ = next(split_test.split(X_test_full, y_test_full))

    X_test_sub = X_test_full[test_idx]
    y_test_sub = y_test_full[test_idx]

    return {
        "X_train": X_train_sub,
        "y_train": y_train_sub,
        "X_val": X_val_sub,
        "y_val": y_val_sub,
        "X_test": X_test_sub,
        "y_test": y_test_sub,
        "meta": {
            "n_train": len(y_train_sub),
            "train_benign_0": int(np.sum(y_train_sub == 0)),
            "train_attack_1": int(np.sum(y_train_sub == 1)),
            "n_val": len(y_val_sub),
            "val_benign_0": int(np.sum(y_val_sub == 0)),
            "val_attack_1": int(np.sum(y_val_sub == 1)),
            "n_test": len(y_test_sub),
            "test_benign_0": int(np.sum(y_test_sub == 0)),
            "test_attack_1": int(np.sum(y_test_sub == 1)),
            "random_seed": random_seed,
            "selected_features": selected_features,
        },
    }


def select_best_c_candidate(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Select the best candidate C using validation F1 score with deterministic tie-breaking.

    Tie-break rule: If validation F1 scores are identical, select the smaller C
    (preferring simpler, higher-regularization models).
    """
    if not candidates:
        raise ValueError("Cannot select from empty candidate list.")

    # Sort primarily by validation F1 descending, secondarily by C ascending
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (-c["val_f1"], c["C"]),
    )
    return sorted_candidates[0]


def tune_qsvc_hyperparameters(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_val: np.ndarray,
    y_val: np.ndarray,
    candidate_c_values: Optional[List[float]] = None,
    random_seed: int = RANDOM_SEED,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluate candidate C values on precomputed kernel matrices.

    Parameters
    ----------
    K_train : np.ndarray
        Precomputed training Gram matrix of shape (n_train, n_train).
    y_train : np.ndarray
        Training labels (0=BENIGN, 1=ATTACK).
    K_val : np.ndarray
        Precomputed validation kernel matrix of shape (n_val, n_train).
    y_val : np.ndarray
        Validation labels (0=BENIGN, 1=ATTACK).
    candidate_c_values : list of float, optional
        C values to evaluate. Defaults to [0.1, 1.0, 10.0].
    random_seed : int
        Deterministic seed.

    Returns
    -------
    (best_C, candidate_results)
    """
    if candidate_c_values is None:
        candidate_c_values = CANDIDATE_C_VALUES

    candidate_results: List[Dict[str, Any]] = []

    for C_val in candidate_c_values:
        # Fit QSVC with precomputed kernel
        clf = QSVC(quantum_kernel="precomputed", C=C_val, random_state=random_seed)

        t_fit_start = time.perf_counter()
        clf.fit(K_train, y_train)
        fit_time_s = time.perf_counter() - t_fit_start

        # Predict on validation kernel
        t_eval_start = time.perf_counter()
        y_val_pred = clf.predict(K_val)
        y_val_score = clf.decision_function(K_val)
        eval_time_s = time.perf_counter() - t_eval_start

        # Compute metrics (ATTACK = class 1)
        acc = float(accuracy_score(y_val, y_val_pred))
        prec = float(precision_score(y_val, y_val_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_val, y_val_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_val, y_val_pred, pos_label=1, zero_division=0))

        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        try:
            auc = float(roc_auc_score(y_val, y_val_score))
        except ValueError:
            auc = 0.5

        candidate_results.append({
            "C": C_val,
            "val_accuracy": round(acc, 6),
            "val_precision": round(prec, 6),
            "val_recall": round(rec, 6),
            "val_f1": round(f1, 6),
            "val_roc_auc": round(auc, 6),
            "val_confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "fit_time_s": round(fit_time_s, 6),
            "eval_time_s": round(eval_time_s, 6),
        })

    best_cand = select_best_c_candidate(candidate_results)
    return best_cand["C"], candidate_results


def run_qsvc_experiment(
    n_train: int = MAX_TRAIN_SAMPLES,
    n_val: int = MAX_VAL_SAMPLES,
    n_test: int = MAX_TEST_SAMPLES,
    candidate_c_values: Optional[List[float]] = None,
    random_seed: int = RANDOM_SEED,
    results_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the complete Stage 5 Step 5 QSVC experiment.

    Enforces all safety, anti-leakage, and reporting constraints.
    """
    if candidate_c_values is None:
        candidate_c_values = CANDIDATE_C_VALUES
    if results_path is None:
        results_path = QSVC_RESULTS_JSON

    t_total_start = time.perf_counter()

    print("=" * 70)
    print("STAGE 5 STEP 5: RESTRICTED QSVC EXPERIMENT")
    print(f"Sample limits: n_train={n_train}, n_val={n_val}, n_test={n_test}")
    print(f"Candidate C values: {candidate_c_values}")
    print("=" * 70)

    # 1. Load data and features
    print("\n[1/6] Loading data and feature selection metadata...")
    train_df = pd.read_csv(RAW_TRAIN_CSV)
    test_df = pd.read_csv(RAW_TEST_CSV)

    fs_path = RESULTS_DIR / "feature_selection.json"
    with open(fs_path, "r", encoding="utf-8") as f:
        fs_data = json.load(f)
    selected_features = fs_data["runs"]["all_features"]["selected_features"]
    print(f"       Features ({len(selected_features)}): {selected_features}")

    # 2. Anti-leakage data splitting
    print("\n[2/6] Splitting subsets (training partition for train/val; test for test)...")
    splits = split_quantum_data(
        train_df=train_df,
        test_df=test_df,
        selected_features=selected_features,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        random_seed=random_seed,
    )
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    split_meta = splits["meta"]

    print(f"       Train: {split_meta['n_train']} samples (0={split_meta['train_benign_0']}, 1={split_meta['train_attack_1']})")
    print(f"       Val:   {split_meta['n_val']} samples (0={split_meta['val_benign_0']}, 1={split_meta['val_attack_1']})")
    print(f"       Test:  {split_meta['n_test']} samples (0={split_meta['test_benign_0']}, 1={split_meta['test_attack_1']})")

    # 3. Fit scaler ONLY on training subset
    print("\n[3/6] Fitting quantum scaler strictly on training subset [0, pi]...")
    scaler = fit_quantum_scaler(X_train)
    X_train_scaled = transform_quantum_features(scaler, X_train)
    X_val_scaled = transform_quantum_features(scaler, X_val)
    X_test_scaled = transform_quantum_features(scaler, X_test)

    # 4. Build feature map and compute training and validation kernels
    print("\n[4/6] Building quantum kernel and evaluating train & val Gram matrices...")
    feature_map = build_feature_map(n_features=N_QUBIT_FEATURES, reps=2)
    kernel = build_quantum_kernel(feature_map, seed=random_seed, enforce_psd=True)

    # Train kernel (n_train x n_train)
    print(f"       Evaluating training kernel ({n_train}x{n_train})...")
    t0 = time.perf_counter()
    K_train = kernel.evaluate(X_train_scaled)
    t_train_kernel = time.perf_counter() - t0
    print(f"       Training kernel computed in {t_train_kernel:.2f}s (shape={K_train.shape})")

    # Validation kernel (n_val x n_train)
    print(f"       Evaluating validation kernel ({n_val}x{n_train})...")
    t0 = time.perf_counter()
    K_val = kernel.evaluate(X_val_scaled, X_train_scaled)
    t_val_kernel = time.perf_counter() - t0
    print(f"       Validation kernel computed in {t_val_kernel:.2f}s (shape={K_val.shape})")

    # 5. Hyperparameter selection across candidate C values
    print("\n[5/6] Tuning C on validation data with precomputed kernel caching...")
    best_C, candidate_results = tune_qsvc_hyperparameters(
        K_train=K_train,
        y_train=y_train,
        K_val=K_val,
        y_val=y_val,
        candidate_c_values=candidate_c_values,
        random_seed=random_seed,
    )

    print("       Validation results:")
    for cand in candidate_results:
        print(f"         C = {cand['C']:<5}: F1 = {cand['val_f1']:.4f}, "
              f"Acc = {cand['val_accuracy']:.4f}, Prec = {cand['val_precision']:.4f}, "
              f"Rec = {cand['val_recall']:.4f}")
    print(f"       => Selected best C: {best_C} (Selection criterion: Validation F1 for ATTACK class)")

    # 6. Freeze model and evaluate test subset ONCE
    print("\n[6/6] Freezing model choices and evaluating test subset ONCE...")
    print(f"       Evaluating test kernel ({n_test}x{n_train})...")
    t0 = time.perf_counter()
    K_test = kernel.evaluate(X_test_scaled, X_train_scaled)
    t_test_kernel = time.perf_counter() - t0
    print(f"       Test kernel computed in {t_test_kernel:.2f}s (shape={K_test.shape})")

    # Fit final model with best_C
    final_clf = QSVC(quantum_kernel="precomputed", C=best_C, random_state=random_seed)
    t_fit_start = time.perf_counter()
    final_clf.fit(K_train, y_train)
    t_fit_final = time.perf_counter() - t_fit_start

    # Predict once on test subset
    t_infer_start = time.perf_counter()
    y_test_pred = final_clf.predict(K_test)
    y_test_score = final_clf.decision_function(K_test)
    t_infer_final = time.perf_counter() - t_infer_start

    # Compute comprehensive evaluation metrics using src.evaluation
    eval_metrics = evaluate(
        y_true=y_test,
        y_pred=y_test_pred,
        y_score=y_test_score,
        train_time=t_train_kernel + t_fit_final,
        infer_time=t_test_kernel + t_infer_final,
    )

    t_total = time.perf_counter() - t_total_start

    # Save diagnostic plots
    plot_confusion_matrix(
        cm=eval_metrics["confusion_matrix"],
        model_name=f"QSVC (C={best_C}) — Restricted Subset",
        save_path=QSVC_CM_PLOT,
        labels=["Benign (0)", "Attack (1)"],
    )

    plot_roc_curve(
        y_true=y_test,
        y_score=y_test_score,
        model_name=f"QSVC (C={best_C})",
        save_path=QSVC_ROC_PLOT,
        auc_val=eval_metrics["roc_auc"],
    )

    # Assemble report
    report = {
        "schema_version": "1.0.0",
        "experiment_title": "Stage 5 Step 5 - Restricted QSVC Experiment",
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "model_name": f"Quantum Support Vector Classifier (QSVC, C={best_C})",
        "is_restricted_subset_experiment": True,
        "disclaimer": (
            "These metrics reflect a restricted subset experiment (N_train=100, N_test=100) "
            "and must NOT be directly compared as equivalent to full-dataset classical models "
            "evaluated on the complete 175,341 official test samples."
        ),
        "quantum_advantage_claimed": False,
        "quantum_advantage_statement": (
            "No quantum advantage is claimed. Classical baselines on full datasets achieved "
            "ROC-AUC > 0.98. The QSVC experiment demonstrates functional quantum kernel "
            "classification feasibility on NISQ-era simulation constraints."
        ),
        "configuration": {
            "random_seed": random_seed,
            "selected_features": selected_features,
            "n_qubits": N_QUBIT_FEATURES,
            "feature_map": {
                "type": "ZZFeatureMap",
                "n_qubits": N_QUBIT_FEATURES,
                "reps": 2,
                "entanglement": "linear",
            },
            "backend": {
                "primitive": "StatevectorSampler",
                "simulation": "exact_statevector",
                "shots": 1024,
                "seed": random_seed,
                "enforce_psd": True,
            },
            "software_versions": get_software_versions(),
        },
        "data_subsets": {
            "train": {
                "n_samples": split_meta["n_train"],
                "benign_0": split_meta["train_benign_0"],
                "attack_1": split_meta["train_attack_1"],
                "source": "official_UNSW_NB15_training-set.csv",
            },
            "val": {
                "n_samples": split_meta["n_val"],
                "benign_0": split_meta["val_benign_0"],
                "attack_1": split_meta["val_attack_1"],
                "source": "official_UNSW_NB15_training-set.csv",
            },
            "test": {
                "n_samples": split_meta["n_test"],
                "benign_0": split_meta["test_benign_0"],
                "attack_1": split_meta["test_attack_1"],
                "source": "official_UNSW_NB15_testing-set.csv",
            },
        },
        "kernel_evaluations": {
            "training_kernel": {
                "shape": list(K_train.shape),
                "entries": int(K_train.size),
                "memory_kb": round(K_train.nbytes / 1024.0, 4),
                "wall_clock_time_s": round(t_train_kernel, 4),
            },
            "validation_kernel": {
                "shape": list(K_val.shape),
                "entries": int(K_val.size),
                "memory_kb": round(K_val.nbytes / 1024.0, 4),
                "wall_clock_time_s": round(t_val_kernel, 4),
            },
            "test_kernel": {
                "shape": list(K_test.shape),
                "entries": int(K_test.size),
                "memory_kb": round(K_test.nbytes / 1024.0, 4),
                "wall_clock_time_s": round(t_test_kernel, 4),
            },
        },
        "hyperparameter_selection": {
            "candidates_evaluated": candidate_results,
            "selection_metric": "validation_f1_attack_class",
            "tie_break_rule": "smaller_C",
            "selected_C": best_C,
        },
        "test_evaluation": {
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "roc_auc": eval_metrics["roc_auc"],
            "false_positive_rate": eval_metrics["false_positive_rate"],
            "confusion_matrix": eval_metrics["confusion_matrix"],
            "classification_report": eval_metrics["classification_report"],
        },
        "timings": {
            "train_kernel_computation_s": round(t_train_kernel, 4),
            "val_kernel_computation_s": round(t_val_kernel, 4),
            "test_kernel_computation_s": round(t_test_kernel, 4),
            "final_model_fit_s": round(t_fit_final, 6),
            "final_test_infer_s": round(t_infer_final, 6),
            "total_experiment_wall_clock_s": round(t_total, 4),
        },
        "artifacts": {
            "results_json": str(results_path),
            "confusion_matrix_plot": str(QSVC_CM_PLOT),
            "roc_curve_plot": str(QSVC_ROC_PLOT),
        },
    }

    save_evaluation_results(report, results_path)
    print(f"\nReport successfully saved to: {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESTRICTED QSVC EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Selected C         : {best_C}")
    print(f"  Test Accuracy      : {eval_metrics['accuracy']:.4f}")
    print(f"  Test Precision     : {eval_metrics['precision']:.4f}")
    print(f"  Test Recall        : {eval_metrics['recall']:.4f}")
    print(f"  Test F1 Score      : {eval_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC       : {eval_metrics['roc_auc']:.4f}")
    print(f"  Test FPR           : {eval_metrics['false_positive_rate']:.4f}")
    print(f"  Confusion Matrix   : {eval_metrics['confusion_matrix']}")
    print(f"  Total Wall Clock   : {t_total:.2f}s")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run_qsvc_experiment()
