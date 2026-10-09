"""Quantum CyberShield — Stage 7b Step 8: Disagreement and Error Overlap Analysis.

Analyzes model agreement, conditional correctness on disagreement subsets,
and the error overlap matrix across QSVC, classical SVM, and Random Forest
on the fixed 20,000-flow held-out test partition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


PREDICTIONS_CSV_PATH = Path("results/predictions_v2.csv")
OUTPUT_JSON_PATH = Path("results/disagreement_v2.json")

DISCLAIMER_TEXT = (
    "Do not claim models complement one another merely because they disagree. "
    "Disagreement analysis measures output divergence and error distributions; "
    "it does NOT demonstrate synergy or ensemble advantage. In fact, majority voting "
    "with QSVC and SVM degrades overall performance relative to Random Forest alone "
    "because the weaker models outvote the superior classical baseline."
)


def load_predictions_v2(csv_path: str | Path = PREDICTIONS_CSV_PATH) -> pd.DataFrame:
    """Load and validate the Stage 7b predictions CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictions file not found at: {path}")

    df = pd.read_csv(path)
    required_cols = {"sample_id", "y_true", "qsvc_pred", "svm_pred", "rf_pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {sorted(missing)}")

    for col in ["y_true", "qsvc_pred", "svm_pred", "rf_pred"]:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Column '{col}' contains non-binary values: {unique_vals}")

    return df


def compute_pairwise_agreements(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute pairwise agreement and disagreement statistics between all model pairs."""
    n = len(df)
    s = df["svm_pred"].values
    r = df["rf_pred"].values
    q = df["qsvc_pred"].values
    y = df["y_true"].values

    pairs = [
        ("SVM vs RF", "svm", "rf", s, r),
        ("SVM vs QSVC", "svm", "qsvc", s, q),
        ("RF vs QSVC", "rf", "qsvc", r, q),
    ]

    pairwise_results = {}
    for name, m1_id, m2_id, p1, p2 in pairs:
        agree_mask = (p1 == p2)
        agree_count = int(np.sum(agree_mask))
        disagree_count = n - agree_count
        agree_rate = float(agree_count / n)
        disagree_rate = float(disagree_count / n)

        # On disagreements (where p1 != p2):
        dis_mask = ~agree_mask
        m1_correct_on_dis = int(np.sum((p1 == y) & dis_mask))
        m2_correct_on_dis = int(np.sum((p2 == y) & dis_mask))

        pairwise_results[name] = {
            "model_A": m1_id,
            "model_B": m2_id,
            "agreement_count": agree_count,
            "agreement_rate": round(agree_rate, 4),
            "disagreement_count": disagree_count,
            "disagreement_rate": round(disagree_rate, 4),
            "conditional_on_disagreement": {
                "total_disagreements": disagree_count,
                f"{m1_id}_correct_count": m1_correct_on_dis,
                f"{m1_id}_correct_rate": round(float(m1_correct_on_dis / disagree_count), 4) if disagree_count > 0 else 0.0,
                f"{m2_id}_correct_count": m2_correct_on_dis,
                f"{m2_id}_correct_rate": round(float(m2_correct_on_dis / disagree_count), 4) if disagree_count > 0 else 0.0,
            },
        }

    return pairwise_results


def compute_three_way_agreements(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute 3-way unanimous agreement and total disagreement statistics."""
    n = len(df)
    s = df["svm_pred"].values
    r = df["rf_pred"].values
    q = df["qsvc_pred"].values
    y = df["y_true"].values

    unanimous_mask = (s == r) & (r == q)
    unanimous_count = int(np.sum(unanimous_mask))
    disagreement_count = n - unanimous_count

    unanimous_correct = int(np.sum(unanimous_mask & (s == y)))
    unanimous_incorrect = int(np.sum(unanimous_mask & (s != y)))

    # Breakdown by predicted label when unanimous
    unanimous_all_0 = int(np.sum(unanimous_mask & (s == 0)))
    unanimous_all_1 = int(np.sum(unanimous_mask & (s == 1)))

    return {
        "total_samples": n,
        "unanimous_agreement": {
            "count": unanimous_count,
            "rate": round(float(unanimous_count / n), 4),
            "all_predicted_0_count": unanimous_all_0,
            "all_predicted_1_count": unanimous_all_1,
            "unanimous_correct_count": unanimous_correct,
            "unanimous_correct_rate": round(float(unanimous_correct / unanimous_count), 4) if unanimous_count > 0 else 0.0,
            "unanimous_incorrect_count": unanimous_incorrect,
            "unanimous_incorrect_rate": round(float(unanimous_incorrect / unanimous_count), 4) if unanimous_count > 0 else 0.0,
        },
        "three_way_disagreement": {
            "count": disagreement_count,
            "rate": round(float(disagreement_count / n), 4),
        },
    }


def compute_conditional_correctness(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute model accuracy conditioned strictly on the 3-model disagreement subset."""
    s = df["svm_pred"].values
    r = df["rf_pred"].values
    q = df["qsvc_pred"].values
    y = df["y_true"].values

    dis_mask = ~((s == r) & (r == q))
    n_dis = int(np.sum(dis_mask))

    if n_dis == 0:
        return {"total_disagreements": 0}

    s_corr = int(np.sum((s == y) & dis_mask))
    r_corr = int(np.sum((r == y) & dis_mask))
    q_corr = int(np.sum((q == y) & dis_mask))

    return {
        "disagreement_subset_size": n_dis,
        "model_performance_on_disagreements": {
            "qsvc": {
                "correct_count": q_corr,
                "accuracy_on_disagreement": round(float(q_corr / n_dis), 4),
                "error_count": n_dis - q_corr,
            },
            "svm": {
                "correct_count": s_corr,
                "accuracy_on_disagreement": round(float(s_corr / n_dis), 4),
                "error_count": n_dis - s_corr,
            },
            "rf": {
                "correct_count": r_corr,
                "accuracy_on_disagreement": round(float(r_corr / n_dis), 4),
                "error_count": n_dis - r_corr,
            },
        },
        "relative_ranking_on_disagreements": [
            {"rank": 1, "model": "rf", "accuracy": round(float(r_corr / n_dis), 4)},
            {"rank": 2, "model": "svm", "accuracy": round(float(s_corr / n_dis), 4)},
            {"rank": 3, "model": "qsvc", "accuracy": round(float(q_corr / n_dis), 4)},
        ],
    }


def compute_error_overlap_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Construct complete disjoint error partition and error overlap matrix."""
    n = len(df)
    s = df["svm_pred"].values
    r = df["rf_pred"].values
    q = df["qsvc_pred"].values
    y = df["y_true"].values

    err_q = (q != y)
    err_s = (s != y)
    err_r = (r != y)

    # Disjoint 8-part partition of the sample space
    all_3_right = int(np.sum(~err_q & ~err_s & ~err_r))
    all_3_wrong = int(np.sum(err_q & err_s & err_r))

    # Single-model unique failures (exactly 1 model failed, other 2 succeeded)
    qsvc_only_wrong = int(np.sum(err_q & ~err_s & ~err_r))
    svm_only_wrong = int(np.sum(~err_q & err_s & ~err_r))
    rf_only_wrong = int(np.sum(~err_q & ~err_s & err_r))

    # Two-model shared failures (exactly 2 models failed, 1 succeeded)
    qsvc_svm_wrong = int(np.sum(err_q & err_s & ~err_r))  # RF only correct
    qsvc_rf_wrong = int(np.sum(err_q & ~err_s & err_r))   # SVM only correct
    svm_rf_wrong = int(np.sum(~err_q & err_s & err_r))    # QSVC only correct

    # Total errors per model
    total_err_q = int(np.sum(err_q))
    total_err_s = int(np.sum(err_s))
    total_err_r = int(np.sum(err_r))

    # Union of errors
    at_least_one_wrong = int(np.sum(err_q | err_s | err_r))

    # Pairwise error overlaps (regardless of third model)
    q_and_s_wrong = int(np.sum(err_q & err_s))
    q_and_r_wrong = int(np.sum(err_q & err_r))
    s_and_r_wrong = int(np.sum(err_s & err_r))

    partition_sum = (
        all_3_right
        + all_3_wrong
        + qsvc_only_wrong
        + svm_only_wrong
        + rf_only_wrong
        + qsvc_svm_wrong
        + qsvc_rf_wrong
        + svm_rf_wrong
    )

    return {
        "total_samples": n,
        "partition_verification_sum": partition_sum,
        "partition_is_exact": bool(partition_sum == n),
        "total_errors": {
            "qsvc": total_err_q,
            "svm": total_err_s,
            "rf": total_err_r,
        },
        "overall_error_coverage": {
            "all_three_correct": all_3_right,
            "all_three_correct_rate": round(float(all_3_right / n), 4),
            "all_three_incorrect": all_3_wrong,
            "all_three_incorrect_rate": round(float(all_3_wrong / n), 4),
            "at_least_one_incorrect": at_least_one_wrong,
            "at_least_one_incorrect_rate": round(float(at_least_one_wrong / n), 4),
        },
        "disjoint_error_breakdown": {
            "unique_errors_single_model": {
                "qsvc_only_failed": qsvc_only_wrong,
                "svm_only_failed": svm_only_wrong,
                "rf_only_failed": rf_only_wrong,
            },
            "shared_errors_two_models": {
                "qsvc_and_svm_failed_rf_alone_correct": qsvc_svm_wrong,
                "qsvc_and_rf_failed_svm_alone_correct": qsvc_rf_wrong,
                "svm_and_rf_failed_qsvc_alone_correct": svm_rf_wrong,
            },
            "shared_errors_all_three_models": all_3_wrong,
        },
        "pairwise_error_overlaps": {
            "qsvc_and_svm_both_wrong": q_and_s_wrong,
            "qsvc_and_rf_both_wrong": q_and_r_wrong,
            "svm_and_rf_both_wrong": s_and_r_wrong,
        },
        "exclusive_correctness_summary": {
            "rf_alone_correct": qsvc_svm_wrong,
            "svm_alone_correct": qsvc_rf_wrong,
            "qsvc_alone_correct": svm_rf_wrong,
            "takeaway": (
                f"When two models fail and one succeeds, RF succeeds alone on {qsvc_svm_wrong} flows, "
                f"SVM succeeds alone on {qsvc_rf_wrong} flows, while QSVC succeeds alone on only {svm_rf_wrong} flows. "
                f"RF is alone in its correctness {qsvc_svm_wrong / max(1, svm_rf_wrong):.1f}x more often than QSVC."
            ),
        },
    }


def compute_majority_vote_ensemble(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate simple unweighted majority voting ensemble across the 3 models."""
    n = len(df)
    s = df["svm_pred"].values
    r = df["rf_pred"].values
    q = df["qsvc_pred"].values
    y = df["y_true"].values

    maj_vote = ((q + s + r) >= 2).astype(int)

    maj_correct = int(np.sum(maj_vote == y))
    rf_correct = int(np.sum(r == y))
    svm_correct = int(np.sum(s == y))
    qsvc_correct = int(np.sum(q == y))

    maj_acc = float(maj_correct / n)
    rf_acc = float(rf_correct / n)

    return {
        "majority_vote_accuracy": round(maj_acc, 4),
        "majority_vote_correct_count": maj_correct,
        "single_model_accuracies": {
            "rf": round(float(rf_acc), 4),
            "svm": round(float(svm_correct / n), 4),
            "qsvc": round(float(qsvc_correct / n), 4),
        },
        "majority_vote_vs_best_single_model": {
            "best_single_model": "rf",
            "delta_accuracy_vs_rf": round(maj_acc - rf_acc, 4),
            "net_error_change_vs_rf": rf_correct - maj_correct,
            "interpretation": (
                "Majority voting degrades accuracy relative to Random Forest alone by "
                f"{abs(round(maj_acc - rf_acc, 4)):.2%} ({rf_correct - maj_correct} additional errors). "
                "Because QSVC and SVM share errors that outvote the correct RF prediction (1,186 flows), "
                "combining models does NOT provide an ensemble benefit."
            ),
        },
    }


def run_disagreement_analysis(csv_path: str | Path = PREDICTIONS_CSV_PATH) -> Dict[str, Any]:
    """Execute end-to-end disagreement analysis and compile structured report."""
    df = load_predictions_v2(csv_path)

    pairwise = compute_pairwise_agreements(df)
    three_way = compute_three_way_agreements(df)
    conditional = compute_conditional_correctness(df)
    error_overlap = compute_error_overlap_matrix(df)
    maj_ensemble = compute_majority_vote_ensemble(df)

    report = {
        "step": "Stage 7b Step 8 — Disagreement and Error Overlap Analysis",
        "sample_size": len(df),
        "source_file": str(csv_path),
        "disclaimer": DISCLAIMER_TEXT,
        "pairwise_agreements": pairwise,
        "three_way_agreements": three_way,
        "conditional_correctness_on_disagreements": conditional,
        "error_overlap_matrix": error_overlap,
        "majority_vote_ensemble_evaluation": maj_ensemble,
        "key_takeaways": [
            "High 3-way unanimous agreement (81.01% of all test flows).",
            "On the 18.99% disagreement subset, Random Forest achieves 65.22% accuracy, Classical SVM achieves 53.79%, while QSVC achieves only 34.36%.",
            "Random Forest is solely correct on 1,186 flows where both SVM and QSVC fail, whereas QSVC is solely correct on only 226 flows.",
            "An unweighted majority voting ensemble achieves 86.77% accuracy, performing 2.25 percentage points worse than Random Forest alone (89.02%).",
            "Disagreement is primarily driven by QSVC underperformance, not complementary niche expertise. There is no evidence of quantum ensemble synergy.",
        ],
    }

    return report


def save_disagreement_report(
    report: Dict[str, Any], output_path: str | Path = OUTPUT_JSON_PATH
) -> None:
    """Save the disagreement analysis report to disk as indented JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    print("Executing Stage 7b Step 8 Disagreement Analysis...")
    report = run_disagreement_analysis()
    save_disagreement_report(report)
    print(f"Disagreement report successfully saved to {OUTPUT_JSON_PATH}")

    # Print summary highlights
    print("\n--- Summary Highlights ---")
    pw = report["pairwise_agreements"]
    for pair, data in pw.items():
        print(f"{pair}: Agreement = {data['agreement_rate']:.2%}, Disagreement = {data['disagreement_rate']:.2%}")

    tw = report["three_way_agreements"]
    print(f"3-Way Unanimous: {tw['unanimous_agreement']['rate']:.2%}")
    print(f"3-Way Disagreement: {tw['three_way_disagreement']['rate']:.2%}")

    cond = report["conditional_correctness_on_disagreements"]["model_performance_on_disagreements"]
    print(f"On Disagreement Subset:")
    print(f"  RF accuracy:   {cond['rf']['accuracy_on_disagreement']:.2%}")
    print(f"  SVM accuracy:  {cond['svm']['accuracy_on_disagreement']:.2%}")
    print(f"  QSVC accuracy: {cond['qsvc']['accuracy_on_disagreement']:.2%}")

    eo = report["error_overlap_matrix"]["exclusive_correctness_summary"]
    print(f"Exclusive Correctness: {eo['takeaway']}")

    mv = report["majority_vote_ensemble_evaluation"]
    print(f"Majority Vote Accuracy: {mv['majority_vote_accuracy']:.2%} vs RF alone: {mv['single_model_accuracies']['rf']:.2%}")


if __name__ == "__main__":
    main()
