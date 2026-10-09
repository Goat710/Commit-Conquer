"""Stage 4 — Classical Baselines & Experiment B (Classical Ceiling).

Implements:
1. Support Vector Classifier (RBF kernel, probability=True)
2. Random Forest Classifier
Hyperparameter tuning strictly on TRAIN using stratified cross-validation.
Final models fitted on the full deduplicated training set, evaluated exactly once on the official test set.

Experiment B configurations:
A. SVM — Stage 3 selected 4 features
B. Random Forest — Stage 3 selected 4 features
C. SVM — all eligible features (Stage 2 preprocessed)
D. Random Forest — all eligible features (Stage 2 preprocessed)
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Suppress sklearn 1.9+ future warning for SVC(probability=True) to keep terminal logs clean
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm._base")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.evaluation import (
    evaluate,
    plot_combined_roc_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    print_comparison_table,
    save_evaluation_results,
)
from src.feature_selection import FEATURE_SELECTION_JSON
from src.preprocessing import (
    build_classical_preprocessor,
    deduplicate_train,
    load_data,
    separate_features_target_and_metadata,
)

# Predefined output paths for Experiment B
EXP_B_SVM_SEL4_JSON: Path = RESULTS_DIR / "experiment_B_svm_selected4.json"
EXP_B_RF_SEL4_JSON: Path = RESULTS_DIR / "experiment_B_rf_selected4.json"
EXP_B_SVM_ALL_JSON: Path = RESULTS_DIR / "experiment_B_svm_all.json"
EXP_B_RF_ALL_JSON: Path = RESULTS_DIR / "experiment_B_rf_all.json"

EXP_B_ROC_PLOT: Path = RESULTS_DIR / "experiment_B_roc_curves.png"
EXP_B_CM_PLOT: Path = RESULTS_DIR / "experiment_B_confusion_matrices.png"


def load_stage3_selected_features(
    fs_path: Path = FEATURE_SELECTION_JSON,
) -> List[str]:
    """Reads the exact Stage 3 selected 4-feature list from results/feature_selection.json.
    
    Guarantees no re-selection or re-computation.
    """
    path_obj = Path(fs_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Stage 3 feature selection JSON not found at: {path_obj}")

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected_features = data["runs"]["all_features"]["selected_features"]
    if len(selected_features) != 4:
        raise ValueError(
            f"Expected exactly 4 selected features, found {len(selected_features)}: {selected_features}"
        )
    return list(selected_features)


def tune_svm_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_SEED,
    cv_subsample_size: int = 5000,
    n_splits: int = 3,
) -> Dict[str, Any]:
    """Tunes SVM (RBF) hyperparameters strictly on TRAIN using stratified cross-validation.
    
    Uses decision_function scoring during CV for rapid hackathon iteration,
    then returns the optimal hyperparameter dictionary.
    """
    param_grid = {
        "C": [0.5, 1.0, 5.0],
        "gamma": ["scale", "auto"],
    }

    # Stratified subsampling for fast CV grid search on hackathon runtime
    n_train = len(y_train)
    if n_train > cv_subsample_size:
        skf_sub = StratifiedKFold(n_splits=int(n_train // cv_subsample_size), shuffle=True, random_state=random_state)
        sub_idx, _ = next(skf_sub.split(X_train, y_train))
        X_cv = X_train[sub_idx]
        y_cv = y_train[sub_idx]
    else:
        X_cv = X_train
        y_cv = y_train

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_svc = SVC(kernel="rbf", random_state=random_state, cache_size=1000)
    grid = GridSearchCV(base_svc, param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
    grid.fit(X_cv, y_cv)

    return grid.best_params_


def tune_rf_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_SEED,
    cv_subsample_size: int = 5000,
    n_splits: int = 3,
) -> Dict[str, Any]:
    """Tunes Random Forest hyperparameters strictly on TRAIN using stratified cross-validation."""
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20],
        "min_samples_leaf": [1, 2],
    }

    n_train = len(y_train)
    if n_train > cv_subsample_size:
        skf_sub = StratifiedKFold(n_splits=int(n_train // cv_subsample_size), shuffle=True, random_state=random_state)
        sub_idx, _ = next(skf_sub.split(X_train, y_train))
        X_cv = X_train[sub_idx]
        y_cv = y_train[sub_idx]
    else:
        X_cv = X_train
        y_cv = y_train

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=1)
    grid = GridSearchCV(base_rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
    grid.fit(X_cv, y_cv)

    return grid.best_params_


def train_and_evaluate_model(
    model_instance: Any,
    hyperparameters: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_name: str,
    model_name: str,
    feature_set: str,
    features_list: List[str],
    save_json_path: Path,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Fits the final model on full train data, evaluates once on official test data,

    and saves full metrics and plots.
    """
    print(f"\n--- Fitting Final {model_name} on Full Train Set ({len(y_train):,} samples) ---")
    print(f"Hyperparameters: {hyperparameters}")

    # 1. Final model training timing
    t0_train = time.perf_counter()
    model_instance.fit(X_train, y_train)
    train_time = time.perf_counter() - t0_train
    print(f"Final training completed in {train_time:.2f} seconds.")

    # 2. Final test set inference timing (evaluated exactly once)
    print(f"Evaluating inference on untouched official test set ({len(y_test):,} samples)...")
    t0_infer = time.perf_counter()
    y_pred = model_instance.predict(X_test)
    y_score = model_instance.predict_proba(X_test)[:, 1]
    infer_time = time.perf_counter() - t0_infer
    print(f"Inference completed in {infer_time:.2f} seconds.")

    # 3. Compute metrics
    eval_metrics = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        train_time=train_time,
        infer_time=infer_time,
    )

    # 4. Construct complete result record
    full_result: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "feature_set": feature_set,
        "features": features_list,
        "selected_hyperparameters": hyperparameters,
        "accuracy": eval_metrics["accuracy"],
        "precision": eval_metrics["precision"],
        "recall": eval_metrics["recall"],
        "f1": eval_metrics["f1"],
        "roc_auc": eval_metrics["roc_auc"],
        "false_positive_rate": eval_metrics["false_positive_rate"],
        "confusion_matrix": eval_metrics["confusion_matrix"],
        "classification_report": eval_metrics["classification_report"],
        "train_time": eval_metrics["train_time"],
        "total_inference_time": eval_metrics["infer_time"],
        "inference_time_per_1000_samples": eval_metrics["inference_time_per_1000_samples"],
        "random_seed": random_state,
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
    }

    # Save JSON artifact
    save_evaluation_results(full_result, save_json_path)
    print(f"[OK] Saved results to: {save_json_path}")

    # Save individual confusion matrix and ROC curve
    cm_path = save_json_path.parent / f"{save_json_path.stem}_cm.png"
    roc_path = save_json_path.parent / f"{save_json_path.stem}_roc.png"
    plot_confusion_matrix(eval_metrics["confusion_matrix"], f"{model_name} ({feature_set})", cm_path)
    plot_roc_curve(y_test, y_score, f"{model_name} ({feature_set})", roc_path, auc_val=eval_metrics["roc_auc"])

    # Attach raw predictions and probabilities for consolidated plotting
    full_result["_y_score"] = y_score
    full_result["_y_pred"] = y_pred

    return full_result


def plot_consolidated_confusion_matrices(
    results_list: List[Dict[str, Any]],
    save_path: Path = EXP_B_CM_PLOT,
) -> None:
    """Plots a 2x2 grid of confusion matrices for all four Experiment B configurations."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    axes = axes.flatten()

    for idx, res in enumerate(results_list):
        ax = axes[idx]
        ax.set_facecolor("#FFFFFF")
        cm_arr = np.asarray(res["confusion_matrix"])
        cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)

        total = np.sum(cm_arr)
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                val = cm_arr[i, j]
                pct = (val / total) * 100 if total > 0 else 0
                color = "white" if val > (cm_arr.max() / 2) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:,}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10.5,
                    fontweight="bold",
                )

        labels = ["Benign (0)", "Attack (1)"]
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10.5, fontweight="bold", labelpad=6)
        ax.set_ylabel("True Label", fontsize=10.5, fontweight="bold", labelpad=6)
        ax.set_title(
            f"{res['model_name']} — {res['feature_set']}\n(Acc: {res['accuracy']:.4f}, F1: {res['f1']:.4f})",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )

    plt.suptitle("Experiment B — Classical Baselines Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def run_experiment_b(
    fs_json_path: Path = FEATURE_SELECTION_JSON,
    results_dir: Path = RESULTS_DIR,
    random_state: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Executes the complete Experiment B (Classical Ceiling) suite:

    A. SVM — Stage 3 selected 4 features
    B. Random Forest — Stage 3 selected 4 features
    C. SVM — all eligible features
    D. Random Forest — all eligible features
    """
    print("=" * 70)
    print("STAGE 4: EXPERIMENT B — CLASSICAL CEILING EVALUATION")
    print("=" * 70)

    # 1. Load data
    train_raw, test_raw = load_data()
    train_dedup = deduplicate_train(train_raw, test_df=test_raw)
    X_train_full, y_train, _ = separate_features_target_and_metadata(train_dedup)
    X_test_full, y_test, _ = separate_features_target_and_metadata(test_raw)

    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    # 2. Load Stage 3 selected 4 features (without recomputation)
    selected_4_features = load_stage3_selected_features(fs_json_path)
    print(f"\nLoaded Stage 3 Selected 4 Features: {selected_4_features}")

    # Prepare 4-feature matrices with train-only StandardScaler
    scaler_4 = StandardScaler()
    X_train_4_scaled = scaler_4.fit_transform(X_train_full[selected_4_features])
    X_test_4_scaled = scaler_4.transform(X_test_full[selected_4_features])

    # 3. Prepare All-Feature matrices using Stage 2 ColumnTransformer
    print("\nPreparing All-Feature space using Stage 2 ColumnTransformer...")
    preprocessor_all, prep_meta = build_classical_preprocessor(X_train_full)
    X_train_all_transformed = preprocessor_all.transform(X_train_full)
    X_test_all_transformed = preprocessor_all.transform(X_test_full)
    all_feature_names = prep_meta["fitted_feature_names_out"]
    print(f"Total Transformed Features: {X_train_all_transformed.shape[1]}")

    results_all: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Configuration A: SVM — Stage 3 Selected 4 Features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONFIG A: SVM — Stage 3 Selected 4 Features")
    print("=" * 60)
    print("Tuning SVM on TRAIN ONLY...")
    svm_4_params = tune_svm_hyperparameters(X_train_4_scaled, y_train_arr, random_state=random_state)
    svm_4_model = SVC(
        kernel="rbf",
        probability=True,
        cache_size=2000,
        random_state=random_state,
        **svm_4_params,
    )
    res_a = train_and_evaluate_model(
        model_instance=svm_4_model,
        hyperparameters=svm_4_params,
        X_train=X_train_4_scaled,
        y_train=y_train_arr,
        X_test=X_test_4_scaled,
        y_test=y_test_arr,
        experiment_name="Experiment B - Classical Ceiling",
        model_name="SVM (RBF)",
        feature_set="selected_4",
        features_list=selected_4_features,
        save_json_path=EXP_B_SVM_SEL4_JSON,
        random_state=random_state,
    )
    results_all.append(res_a)

    # ------------------------------------------------------------------
    # Configuration B: Random Forest — Stage 3 Selected 4 Features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONFIG B: Random Forest — Stage 3 Selected 4 Features")
    print("=" * 60)
    print("Tuning Random Forest on TRAIN ONLY...")
    rf_4_params = tune_rf_hyperparameters(X_train_4_scaled, y_train_arr, random_state=random_state)
    rf_4_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1,
        **rf_4_params,
    )
    res_b = train_and_evaluate_model(
        model_instance=rf_4_model,
        hyperparameters=rf_4_params,
        X_train=X_train_4_scaled,
        y_train=y_train_arr,
        X_test=X_test_4_scaled,
        y_test=y_test_arr,
        experiment_name="Experiment B - Classical Ceiling",
        model_name="Random Forest",
        feature_set="selected_4",
        features_list=selected_4_features,
        save_json_path=EXP_B_RF_SEL4_JSON,
        random_state=random_state,
    )
    results_all.append(res_b)

    # ------------------------------------------------------------------
    # Configuration C: SVM — All Eligible Features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONFIG C: SVM — All Eligible Features")
    print("=" * 60)
    print("Tuning SVM on TRAIN ONLY...")
    svm_all_params = tune_svm_hyperparameters(X_train_all_transformed, y_train_arr, random_state=random_state)
    svm_all_model = SVC(
        kernel="rbf",
        probability=True,
        cache_size=1000,
        random_state=random_state,
        **svm_all_params,
    )
    res_c = train_and_evaluate_model(
        model_instance=svm_all_model,
        hyperparameters=svm_all_params,
        X_train=X_train_all_transformed,
        y_train=y_train_arr,
        X_test=X_test_all_transformed,
        y_test=y_test_arr,
        experiment_name="Experiment B - Classical Ceiling",
        model_name="SVM (RBF)",
        feature_set="all_features",
        features_list=all_feature_names,
        save_json_path=EXP_B_SVM_ALL_JSON,
        random_state=random_state,
    )
    results_all.append(res_c)

    # ------------------------------------------------------------------
    # Configuration D: Random Forest — All Eligible Features
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONFIG D: Random Forest — All Eligible Features")
    print("=" * 60)
    print("Tuning Random Forest on TRAIN ONLY...")
    rf_all_params = tune_rf_hyperparameters(X_train_all_transformed, y_train_arr, random_state=random_state)
    rf_all_model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1,
        **rf_all_params,
    )
    res_d = train_and_evaluate_model(
        model_instance=rf_all_model,
        hyperparameters=rf_all_params,
        X_train=X_train_all_transformed,
        y_train=y_train_arr,
        X_test=X_test_all_transformed,
        y_test=y_test_arr,
        experiment_name="Experiment B - Classical Ceiling",
        model_name="Random Forest",
        feature_set="all_features",
        features_list=all_feature_names,
        save_json_path=EXP_B_RF_ALL_JSON,
        random_state=random_state,
    )
    results_all.append(res_d)

    # ------------------------------------------------------------------
    # Consolidated Plots & Comparison Table
    # ------------------------------------------------------------------
    print("\nGenerating consolidated ROC curves and Confusion Matrices...")
    roc_dict = {
        f"{r['model_name']} ({r['feature_set']})": (y_test_arr, r["_y_score"], r["roc_auc"])
        for r in results_all
    }
    plot_combined_roc_curves(roc_dict, save_path=EXP_B_ROC_PLOT)
    plot_consolidated_confusion_matrices(results_all, save_path=EXP_B_CM_PLOT)
    print(f"[OK] Saved consolidated ROC plot to: {EXP_B_ROC_PLOT}")
    print(f"[OK] Saved consolidated Confusion Matrix plot to: {EXP_B_CM_PLOT}")

    # Remove temporary internal keys before returning
    for r in results_all:
        r.pop("_y_score", None)
        r.pop("_y_pred", None)

    print("\n" + "=" * 70)
    print("EXPERIMENT B — FINAL PERFORMANCE COMPARISON TABLE")
    print("=" * 70)
    print_comparison_table(results_all)

    return results_all


if __name__ == "__main__":
    run_experiment_b()
