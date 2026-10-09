"""Stage 6 — Unified Experiments & Stability Analysis.

Orchestrates unified benchmarking and comparisons across classical models (SVM, RF)
and Quantum Support Vector Classifier (QSVC) on UNSW-NB15:
  - Experiment A: Primary comparison on identical selected 4 features, identical 100-sample
    training subset, and identical 100-sample test subset (seed 42).
  - Experiment C: Stability analysis repeated across 5 configured seeds with mean ± std.
  - Experiment D: Feature ablation using Stage 3 TTL-excluded feature set across seeds.
  - Stage 4 References: Loaded from existing Experiment B results (full-test reference).
  - Predictions: results/predictions_A.csv with shared sample IDs and continuous scores.
  - Artifacts: results/stage6_unified_experiments.json and formatted comparison table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure immediate unbuffered console logging for task monitoring
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC

from src.config import (
    RANDOM_SEED,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
    RESULTS_DIR,
)
from src.evaluation import evaluate
from src.quantum_benchmark import get_software_versions
from src.quantum_model import (
    build_feature_map,
    build_quantum_kernel,
    fit_quantum_scaler,
    transform_quantum_features,
)

DEFAULT_SEEDS = [42, 100, 2024, 777, 999]
DEFAULT_N_TRAIN = 100
DEFAULT_N_TEST = 100

PREDICTIONS_A_CSV: Path = RESULTS_DIR / "predictions_A.csv"
UNIFIED_RESULTS_JSON: Path = RESULTS_DIR / "stage6_unified_experiments.json"
EXP_B_SVM_PATH: Path = RESULTS_DIR / "experiment_B_svm_selected4.json"
EXP_B_RF_PATH: Path = RESULTS_DIR / "experiment_B_rf_selected4.json"


def load_feature_sets() -> Tuple[List[str], List[str]]:
    """Load primary 4 features and Stage 3 TTL-excluded features from results/feature_selection.json."""
    fs_file = RESULTS_DIR / "feature_selection.json"
    if not fs_file.exists():
        raise FileNotFoundError(f"Missing required feature selection artifact: {fs_file}")

    with open(fs_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    primary_features = data["runs"]["all_features"]["selected_features"]
    ablation_features = data["runs"]["TTL_excluded"]["selected_features"]
    return primary_features, ablation_features


def get_stratified_subsets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    n_train: int = DEFAULT_N_TRAIN,
    n_test: int = DEFAULT_N_TEST,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Draw reproducible stratified training and test subsets strictly from respective partitions.

    Training data is drawn exclusively from train_df.
    Test data is drawn exclusively from test_df.
    Test labels are never inspected for selection or preprocessing.
    """
    X_train_full = train_df[features].values.astype(np.float64)
    y_train_full = train_df["label"].values.astype(int)

    split_tr = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
    train_idx, _ = next(split_tr.split(X_train_full, y_train_full))

    X_test_full = test_df[features].values.astype(np.float64)
    y_test_full = test_df["label"].values.astype(int)

    split_te = StratifiedShuffleSplit(n_splits=1, train_size=n_test, random_state=seed)
    test_idx, _ = next(split_te.split(X_test_full, y_test_full))

    return {
        "X_train": X_train_full[train_idx],
        "y_train": y_train_full[train_idx],
        "train_indices": train_idx,
        "X_test": X_test_full[test_idx],
        "y_test": y_test_full[test_idx],
        "test_indices": test_idx,
        "seed": seed,
        "features": features,
    }


def evaluate_classical_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = RANDOM_SEED,
    C: float = 5.0,
    gamma: str = "auto",
) -> Dict[str, Any]:
    """Train and evaluate classical SVM (RBF kernel) on the subset with train-only scaling."""
    # Preprocessing: StandardScaler fitted ONLY on training subset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = SVC(
        C=C,
        kernel="rbf",
        gamma=gamma,
        probability=True,
        random_state=seed,
    )

    t0_train = time.perf_counter()
    clf.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0_train

    t0_infer = time.perf_counter()
    y_pred = clf.predict(X_test_scaled)
    # Decision scores: continuous signed distance to separating hyperplane
    y_score = clf.decision_function(X_test_scaled)
    infer_time = time.perf_counter() - t0_infer

    metrics = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        train_time=train_time,
        infer_time=infer_time,
    )

    metrics["y_pred_arr"] = y_pred.tolist()
    metrics["y_score_arr"] = y_score.tolist()
    metrics["model_name"] = "SVM (RBF)"
    return metrics


def evaluate_classical_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = RANDOM_SEED,
    n_estimators: int = 100,
    max_depth: int = 20,
    min_samples_leaf: int = 2,
) -> Dict[str, Any]:
    """Train and evaluate classical Random Forest on the subset."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )

    t0_train = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0_train

    t0_infer = time.perf_counter()
    y_pred = clf.predict(X_test)
    # Decision score: continuous probability of positive class (ATTACK)
    y_score = clf.predict_proba(X_test)[:, 1]
    infer_time = time.perf_counter() - t0_infer

    metrics = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        train_time=train_time,
        infer_time=infer_time,
    )

    metrics["y_pred_arr"] = y_pred.tolist()
    metrics["y_score_arr"] = y_score.tolist()
    metrics["model_name"] = "Random Forest"
    return metrics


def evaluate_qsvc_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = RANDOM_SEED,
    C: float = 10.0,
) -> Dict[str, Any]:
    """Train and evaluate QSVC on the subset using train-only quantum scaling [0, pi] and statevector simulation."""
    n_features = X_train.shape[1]

    # Preprocessing: MinMaxScaler fitted ONLY on training subset, clipped to [0, pi]
    scaler = fit_quantum_scaler(X_train)
    X_train_scaled = transform_quantum_features(scaler, X_train)
    X_test_scaled = transform_quantum_features(scaler, X_test)

    feature_map = build_feature_map(n_features=n_features, reps=2)
    kernel = build_quantum_kernel(feature_map, seed=seed, enforce_psd=True)

    # Compute training kernel Gram matrix
    t0_k_train = time.perf_counter()
    K_train = kernel.evaluate(X_train_scaled)
    t_k_train = time.perf_counter() - t0_k_train

    # Fit QSVC on precomputed training kernel
    clf = QSVC(quantum_kernel="precomputed", C=C, random_state=seed)
    t0_fit = time.perf_counter()
    clf.fit(K_train, y_train)
    t_fit = time.perf_counter() - t0_fit
    train_time = t_k_train + t_fit

    # Compute test kernel matrix
    t0_k_test = time.perf_counter()
    K_test = kernel.evaluate(X_test_scaled, X_train_scaled)
    t_k_test = time.perf_counter() - t0_k_test

    # Inference on test kernel
    t0_infer = time.perf_counter()
    y_pred = clf.predict(K_test)
    # Decision scores: continuous signed distance to hyperplane in quantum Hilbert space
    y_score = clf.decision_function(K_test)
    t_infer = time.perf_counter() - t0_infer
    infer_time = t_k_test + t_infer

    metrics = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        train_time=train_time,
        infer_time=infer_time,
    )

    metrics["y_pred_arr"] = y_pred.tolist()
    metrics["y_score_arr"] = y_score.tolist()
    metrics["model_name"] = "QSVC"
    metrics["kernel_train_time_s"] = round(t_k_train, 4)
    metrics["kernel_test_time_s"] = round(t_k_test, 4)
    return metrics


def compute_metrics_aggregation(
    runs: List[Dict[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute mean and sample standard deviation across multiple seeds for each metric."""
    if metric_keys is None:
        metric_keys = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "false_positive_rate",
            "train_time",
            "infer_time",
        ]

    agg: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        values = [r[key] for r in runs if key in r and r[key] is not None]
        if values:
            val_arr = np.array(values, dtype=float)
            agg[key] = {
                "mean": round(float(np.mean(val_arr)), 6),
                "std": round(float(np.std(val_arr, ddof=1)) if len(val_arr) > 1 else 0.0, 6),
            }
        else:
            agg[key] = {"mean": 0.0, "std": 0.0}

    return agg


def load_stage4_baselines() -> Dict[str, Any]:
    """Load existing Stage 4 Experiment B results as full-test reference baselines."""
    baselines = {}
    if EXP_B_SVM_PATH.exists():
        with open(EXP_B_SVM_PATH, "r", encoding="utf-8") as f:
            baselines["svm_selected4_full_test"] = json.load(f)
    if EXP_B_RF_PATH.exists():
        with open(EXP_B_RF_PATH, "r", encoding="utf-8") as f:
            baselines["rf_selected4_full_test"] = json.load(f)
    return baselines


def run_stage6_experiments(
    seeds: Optional[List[int]] = None,
    output_json_path: Path = UNIFIED_RESULTS_JSON,
    predictions_csv_path: Path = PREDICTIONS_A_CSV,
) -> Dict[str, Any]:
    """Execute the full Stage 6 workflow: Experiment A, C (Stability), D (Ablation), and reporting."""
    if seeds is None:
        seeds = DEFAULT_SEEDS

    print("=" * 75)
    print("STAGE 6: UNIFIED EXPERIMENTS & STABILITY ANALYSIS")
    print(f"Seeds: {seeds}")
    print(f"Sample limits: n_train={DEFAULT_N_TRAIN}, n_test={DEFAULT_N_TEST}")
    print("=" * 75)

    primary_features, ablation_features = load_feature_sets()
    print(f"Primary features ({len(primary_features)}): {primary_features}")
    print(f"Ablation features ({len(ablation_features)}): {ablation_features}")

    train_df = pd.read_csv(RAW_TRAIN_CSV)
    test_df = pd.read_csv(RAW_TEST_CSV)

    stage4_refs = load_stage4_baselines()

    experiment_A_results: Dict[str, Any] = {}
    experiment_C_runs: List[Dict[str, Any]] = []
    experiment_D_runs: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Primary Comparison (A) & Stability Analysis (C) across seeds
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT A & C (PRIMARY 4 FEATURES ACROSS SEEDS)")
    print("=" * 60)

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n--- Seed {idx}/{len(seeds)} (seed={seed}) ---")
        subsets = get_stratified_subsets(
            train_df=train_df,
            test_df=test_df,
            features=primary_features,
            n_train=DEFAULT_N_TRAIN,
            n_test=DEFAULT_N_TEST,
            seed=seed,
        )

        X_tr, y_tr = subsets["X_train"], subsets["y_train"]
        X_te, y_te = subsets["X_test"], subsets["y_test"]

        # 1. Classical SVM
        print("  Evaluating Classical SVM (RBF)...")
        svm_res = evaluate_classical_svm(X_tr, y_tr, X_te, y_te, seed=seed)
        print(f"    SVM Accuracy: {svm_res['accuracy']:.4f}, F1: {svm_res['f1']:.4f}")

        # 2. Classical Random Forest
        print("  Evaluating Classical Random Forest...")
        rf_res = evaluate_classical_rf(X_tr, y_tr, X_te, y_te, seed=seed)
        print(f"    RF  Accuracy: {rf_res['accuracy']:.4f}, F1: {rf_res['f1']:.4f}")

        # 3. QSVC
        print("  Evaluating QSVC (Quantum Kernel)...")
        qsvc_res = evaluate_qsvc_model(X_tr, y_tr, X_te, y_te, seed=seed)
        print(f"    QSVC Accuracy: {qsvc_res['accuracy']:.4f}, F1: {qsvc_res['f1']:.4f}")

        seed_entry = {
            "seed": seed,
            "SVM": svm_res,
            "Random_Forest": rf_res,
            "QSVC": qsvc_res,
        }
        experiment_C_runs.append(seed_entry)

        # Record Experiment A from primary seed (seeds[0] == 42)
        if seed == seeds[0]:
            experiment_A_results = {
                "seed": seed,
                "features": primary_features,
                "SVM": svm_res,
                "Random_Forest": rf_res,
                "QSVC": qsvc_res,
                "subsets_meta": {
                    "train_benign_0": int(np.sum(y_tr == 0)),
                    "train_attack_1": int(np.sum(y_tr == 1)),
                    "test_benign_0": int(np.sum(y_te == 0)),
                    "test_attack_1": int(np.sum(y_te == 1)),
                },
            }

            # Generate predictions_A.csv
            print(f"\nWriting shared predictions to {predictions_csv_path}...")
            test_indices = subsets["test_indices"]
            pred_df = pd.DataFrame({
                "sample_id": test_df.loc[test_indices, "id"].values if "id" in test_df.columns else test_indices,
                "original_test_index": test_indices,
                "y_true": y_te,
                "attack_cat": test_df.loc[test_indices, "attack_cat"].values if "attack_cat" in test_df.columns else "N/A",
                "svm_pred": svm_res["y_pred_arr"],
                "svm_score": svm_res["y_score_arr"],
                "rf_pred": rf_res["y_pred_arr"],
                "rf_score": rf_res["y_score_arr"],
                "qsvc_pred": qsvc_res["y_pred_arr"],
                "qsvc_score": qsvc_res["y_score_arr"],
            })
            predictions_csv_path.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(predictions_csv_path, index=False)
            print(f"[OK] Saved predictions_A.csv with {len(pred_df)} aligned rows.")

    # -------------------------------------------------------------------------
    # 2. Ablation Experiment (D) across seeds (TTL-excluded features)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENT D (FEATURE ABLATION: TTL-EXCLUDED)")
    print("=" * 60)

    for idx, seed in enumerate(seeds, start=1):
        print(f"\n--- Ablation Seed {idx}/{len(seeds)} (seed={seed}) ---")
        subsets_abl = get_stratified_subsets(
            train_df=train_df,
            test_df=test_df,
            features=ablation_features,
            n_train=DEFAULT_N_TRAIN,
            n_test=DEFAULT_N_TEST,
            seed=seed,
        )

        X_tr, y_tr = subsets_abl["X_train"], subsets_abl["y_train"]
        X_te, y_te = subsets_abl["X_test"], subsets_abl["y_test"]

        # Classical SVM
        svm_abl = evaluate_classical_svm(X_tr, y_tr, X_te, y_te, seed=seed)
        # Classical Random Forest
        rf_abl = evaluate_classical_rf(X_tr, y_tr, X_te, y_te, seed=seed)
        # QSVC
        qsvc_abl = evaluate_qsvc_model(X_tr, y_tr, X_te, y_te, seed=seed)

        experiment_D_runs.append({
            "seed": seed,
            "SVM": svm_abl,
            "Random_Forest": rf_abl,
            "QSVC": qsvc_abl,
        })
        print(f"    SVM  F1: {svm_abl['f1']:.4f} | RF F1: {rf_abl['f1']:.4f} | QSVC F1: {qsvc_abl['f1']:.4f}")

    # -------------------------------------------------------------------------
    # 3. Aggregate Summaries (Mean ± Std)
    # -------------------------------------------------------------------------
    stability_summary = {
        "SVM": compute_metrics_aggregation([r["SVM"] for r in experiment_C_runs]),
        "Random_Forest": compute_metrics_aggregation([r["Random_Forest"] for r in experiment_C_runs]),
        "QSVC": compute_metrics_aggregation([r["QSVC"] for r in experiment_C_runs]),
    }

    ablation_summary = {
        "SVM": compute_metrics_aggregation([r["SVM"] for r in experiment_D_runs]),
        "Random_Forest": compute_metrics_aggregation([r["Random_Forest"] for r in experiment_D_runs]),
        "QSVC": compute_metrics_aggregation([r["QSVC"] for r in experiment_D_runs]),
    }

    # Clean raw array pointers from JSON serialization
    for r in experiment_C_runs:
        for m in ["SVM", "Random_Forest", "QSVC"]:
            r[m].pop("y_pred_arr", None)
            r[m].pop("y_score_arr", None)

    for r in experiment_D_runs:
        for m in ["SVM", "Random_Forest", "QSVC"]:
            r[m].pop("y_pred_arr", None)
            r[m].pop("y_score_arr", None)

    experiment_A_clean = {
        "seed": experiment_A_results["seed"],
        "features": experiment_A_results["features"],
        "SVM": {k: v for k, v in experiment_A_results["SVM"].items() if not k.endswith("_arr")},
        "Random_Forest": {k: v for k, v in experiment_A_results["Random_Forest"].items() if not k.endswith("_arr")},
        "QSVC": {k: v for k, v in experiment_A_results["QSVC"].items() if not k.endswith("_arr")},
        "subsets_meta": experiment_A_results["subsets_meta"],
    }

    # -------------------------------------------------------------------------
    # 4. Assemble and Save Master Results JSON
    # -------------------------------------------------------------------------
    master_report = {
        "schema_version": "1.0.0",
        "experiment_title": "Stage 6 — Unified Experiments & Stability Analysis",
        "created_at_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "disclaimer": (
            "Experiments A, C, and D are evaluated on restricted subsets (N_train=100, N_test=100) "
            "and must NOT be directly compared as equivalent to full-dataset classical models "
            "evaluated on the complete 175,341 official test samples."
        ),
        "quantum_advantage_claimed": False,
        "configuration": {
            "n_train_samples": DEFAULT_N_TRAIN,
            "n_test_samples": DEFAULT_N_TEST,
            "primary_seed": seeds[0],
            "stability_seeds": seeds,
            "primary_features": primary_features,
            "ablation_features": ablation_features,
            "feature_map": {
                "type": "ZZFeatureMap",
                "n_qubits": 4,
                "reps": 2,
                "entanglement": "linear",
            },
            "backend": {
                "primitive": "StatevectorSampler",
                "shots": 1024,
                "enforce_psd": True,
            },
            "software_versions": get_software_versions(),
            "model_hyperparameters": {
                "SVM": {"C": 5.0, "gamma": "auto", "kernel": "rbf"},
                "Random_Forest": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2},
                "QSVC": {"C": 10.0, "quantum_kernel": "precomputed"},
            },
            "decision_score_conventions": {
                "SVM": "signed distance from decision boundary (decision_function)",
                "Random_Forest": "probability of ATTACK class (predict_proba[:, 1])",
                "QSVC": "signed distance from hyperplane in quantum Hilbert space (decision_function)",
            },
        },
        "experiment_A_primary_comparison": experiment_A_clean,
        "experiment_C_stability_analysis": {
            "seeds": seeds,
            "features": primary_features,
            "per_seed_runs": experiment_C_runs,
            "summary_mean_std": stability_summary,
        },
        "experiment_D_feature_ablation": {
            "seeds": seeds,
            "features": ablation_features,
            "per_seed_runs": experiment_D_runs,
            "summary_mean_std": ablation_summary,
        },
        "stage4_ceiling_references": stage4_refs,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)
    print(f"\n[OK] Master report saved to {output_json_path}")

    # -------------------------------------------------------------------------
    # 5. Print Comparison Table Computed Directly from Saved JSON
    # -------------------------------------------------------------------------
    print_unified_comparison_table(output_json_path)

    return master_report


def print_unified_comparison_table(json_path: Path) -> str:
    """Read saved JSON and print formatted comparison table with mean ± std."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exp_a = data["experiment_A_primary_comparison"]
    exp_c_sum = data["experiment_C_stability_analysis"]["summary_mean_std"]
    exp_d_sum = data["experiment_D_feature_ablation"]["summary_mean_std"]
    st4_refs = data.get("stage4_ceiling_references", {})

    header = (
        f"{'Experiment / Model':<34} "
        f"{'Scope':<12} "
        f"{'Accuracy':<17} "
        f"{'F1 Score':<17} "
        f"{'ROC-AUC':<17} "
        f"{'FPR':<17}"
    )
    sep = "=" * len(header)
    subsep = "-" * len(header)

    lines = [
        "\n" + sep,
        "STAGE 6 — UNIFIED EXPERIMENTS & STABILITY COMPARISON TABLE",
        "(Computed directly from saved JSON artifact)",
        sep,
        header,
        sep,
        "--- [Experiment A: Primary Comparison (Seed 42, 4 Features)] ---",
    ]

    for m_key, m_label in [("SVM", "SVM (RBF)"), ("Random_Forest", "Random Forest"), ("QSVC", "QSVC (C=10.0)")]:
        m = exp_a[m_key]
        lines.append(
            f"{'Exp A: ' + m_label:<34} "
            f"{'N=100 subset':<12} "
            f"{m['accuracy']:<17.4f} "
            f"{m['f1']:<17.4f} "
            f"{m['roc_auc']:<17.4f} "
            f"{m['false_positive_rate']:<17.4f}"
        )

    lines.extend([
        subsep,
        "--- [Experiment C: Stability Analysis (Mean ± Std over 5 Seeds, 4 Features)] ---",
    ])

    for m_key, m_label in [("SVM", "SVM (RBF)"), ("Random_Forest", "Random Forest"), ("QSVC", "QSVC (C=10.0)")]:
        s = exp_c_sum[m_key]
        acc_str = f"{s['accuracy']['mean']:.4f} ± {s['accuracy']['std']:.4f}"
        f1_str = f"{s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}"
        auc_str = f"{s['roc_auc']['mean']:.4f} ± {s['roc_auc']['std']:.4f}"
        fpr_str = f"{s['false_positive_rate']['mean']:.4f} ± {s['false_positive_rate']['std']:.4f}"
        lines.append(
            f"{'Exp C: ' + m_label:<34} "
            f"{'5-seed avg':<12} "
            f"{acc_str:<17} "
            f"{f1_str:<17} "
            f"{auc_str:<17} "
            f"{fpr_str:<17}"
        )

    lines.extend([
        subsep,
        "--- [Experiment D: Feature Ablation — TTL-Excluded (Mean ± Std over 5 Seeds)] ---",
    ])

    for m_key, m_label in [("SVM", "SVM (RBF)"), ("Random_Forest", "Random Forest"), ("QSVC", "QSVC (C=10.0)")]:
        s = exp_d_sum[m_key]
        acc_str = f"{s['accuracy']['mean']:.4f} ± {s['accuracy']['std']:.4f}"
        f1_str = f"{s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}"
        auc_str = f"{s['roc_auc']['mean']:.4f} ± {s['roc_auc']['std']:.4f}"
        fpr_str = f"{s['false_positive_rate']['mean']:.4f} ± {s['false_positive_rate']['std']:.4f}"
        lines.append(
            f"{'Exp D: ' + m_label:<34} "
            f"{'5-seed avg':<12} "
            f"{acc_str:<17} "
            f"{f1_str:<17} "
            f"{auc_str:<17} "
            f"{fpr_str:<17}"
        )

    if st4_refs:
        lines.extend([
            subsep,
            "--- [Stage 4 Ceiling References (Full Official Test Set: N_train=55,945, N_test=175,341)] ---",
        ])
        if "svm_selected4_full_test" in st4_refs:
            ref = st4_refs["svm_selected4_full_test"]
            lines.append(
                f"{'Ref: SVM (RBF) — Full Test':<34} "
                f"{'Full Test':<12} "
                f"{ref['accuracy']:<17.4f} "
                f"{ref['f1']:<17.4f} "
                f"{ref['roc_auc']:<17.4f} "
                f"{ref['false_positive_rate']:<17.4f}"
            )
        if "rf_selected4_full_test" in st4_refs:
            ref = st4_refs["rf_selected4_full_test"]
            lines.append(
                f"{'Ref: Random Forest — Full Test':<34} "
                f"{'Full Test':<12} "
                f"{ref['accuracy']:<17.4f} "
                f"{ref['f1']:<17.4f} "
                f"{ref['roc_auc']:<17.4f} "
                f"{ref['false_positive_rate']:<17.4f}"
            )

    lines.append(sep + "\n")
    table_str = "\n".join(lines)
    print(table_str)
    return table_str


if __name__ == "__main__":
    run_stage6_experiments()
