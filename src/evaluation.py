"""Evaluation and metrics module for Quantum CyberShield.

Provides standard evaluation functions, timing helpers, ROC and confusion matrix
plotting utilities, JSON persistence, and comparison table printing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_score: Union[np.ndarray, List[float]],
    train_time: float,
    infer_time: float,
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics and inference timing.
    
    ROC-AUC uses continuous y_score (positive class probability/score), not hard predictions.
    False positive rate is explicitly calculated as FP / (FP + TN).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)

    n_samples = len(y_true_arr)
    if n_samples == 0:
        raise ValueError("Cannot evaluate on empty test set.")

    # Core metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        roc_auc = 0.5

    # Confusion matrix and False Positive Rate (explicitly with labels=[0, 1] to guarantee 2x2 shape)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Timing metrics
    infer_per_1000 = (float(infer_time) / n_samples) * 1000.0

    # Classification report as dict
    report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)

    return {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(roc_auc, 6),
        "false_positive_rate": round(fpr, 6),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": report,
        "train_time": round(float(train_time), 4),
        "infer_time": round(float(infer_time), 4),
        "inference_time_per_1000_samples": round(infer_per_1000, 6),
    }


def plot_confusion_matrix(
    cm: Union[List[List[int]], np.ndarray],
    model_name: str,
    save_path: Path,
    labels: Optional[List[str]] = None,
) -> None:
    """Plots and saves a high-resolution, styled confusion matrix."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm_arr = np.asarray(cm)
    display_labels = labels or ["Benign (0)", "Attack (1)"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    # Heatmap
    cax = ax.matshow(cm_arr, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Annotate counts and percentages
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
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(display_labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_roc_curve(
    y_true: Union[np.ndarray, List[int]],
    y_score: Union[np.ndarray, List[float]],
    model_name: str,
    save_path: Path,
    auc_val: Optional[float] = None,
) -> None:
    """Plots and saves a single-model ROC curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    calculated_auc = auc_val if auc_val is not None else roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    ax.plot(
        fpr,
        tpr,
        color="#00ADB5",
        lw=2.5,
        label=f"{model_name} (AUC = {calculated_auc:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_combined_roc_curves(
    models_roc_data: Dict[str, Tuple[Union[np.ndarray, List[int]], Union[np.ndarray, List[float]], float]],
    save_path: Path,
    title: str = "Experiment B — Classical Baselines ROC Curves",
) -> None:
    """Plots all model ROC curves on a single consolidated high-DPI figure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    palette = ["#00ADB5", "#3B82F6", "#10B981", "#8B5CF6"]

    for idx, (m_name, (y_true, y_score, auc_val)) in enumerate(models_roc_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        color = palette[idx % len(palette)]
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.2,
            label=f"{m_name} (AUC = {auc_val:.4f})",
        )

    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Chance (AUC = 0.5000)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_evaluation_results(
    results_dict: Dict[str, Any],
    file_path: Path,
) -> None:
    """Persists evaluation results cleanly to JSON."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)


def print_comparison_table(results_list: List[Dict[str, Any]]) -> str:
    """Prints and returns a formatted ASCII comparison table for evaluated models."""
    header = (
        f"{'Model / Experiment':<32} "
        f"{'Features':<14} "
        f"{'Accuracy':<10} "
        f"{'F1 Score':<10} "
        f"{'ROC-AUC':<10} "
        f"{'FPR':<10} "
        f"{'Train (s)':<11} "
        f"{'Infer/1k (s)':<12}"
    )
    separator = "-" * len(header)
    rows = [separator, header, separator]

    for res in results_list:
        model_exp = f"{res.get('model_name', 'Model')} ({res.get('experiment_name', '')})"[:32]
        feat_name = str(res.get('feature_set', 'features'))[:14]
        acc_str = f"{res.get('accuracy', 0.0):.4f}"
        f1_str = f"{res.get('f1', 0.0):.4f}"
        auc_str = f"{res.get('roc_auc', 0.0):.4f}"
        fpr_str = f"{res.get('false_positive_rate', 0.0):.4f}"
        t_train_str = f"{res.get('train_time', 0.0):.2f}"
        t_inf_1k_str = f"{res.get('inference_time_per_1000_samples', 0.0):.4f}"

        row_line = (
            f"{model_exp:<32} "
            f"{feat_name:<14} "
            f"{acc_str:<10} "
            f"{f1_str:<10} "
            f"{auc_str:<10} "
            f"{fpr_str:<10} "
            f"{t_train_str:<11} "
            f"{t_inf_1k_str:<12}"
        )
        rows.append(row_line)

    rows.append(separator)
    table_str = "\n".join(rows)
    print("\n" + table_str + "\n")
    return table_str
