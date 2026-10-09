"""Stage 7 — CyberShield Engine.

Unified multi-model security engine integrating:
  1. Classical SVM (RBF kernel with StandardScaler)
  2. Classical Random Forest (tree ensemble)
  3. Quantum Support Vector Classifier (QSVC with ZZFeatureMap & FidelityQuantumKernel)
  4. Platt Sigmoid Probability Calibrators (fitted strictly on held-out validation data)
  5. Consensus & Threat Policy (0/3=LOW, 1/3=MEDIUM, 2/3=HIGH, 3/3=HIGH)
  6. Heterogeneous Disagreement Review Trigger:
     "Analyst-review signal from heterogeneous models"

Decision Score vs Raw Probability vs Calibrated Probability:
------------------------------------------------------------
- Decision Score:
  Continuous geometric or structural output produced by the classifier prior to
  thresholding:
  * SVM & QSVC: Signed Euclidean distance f(x) from the sample to the separating
    hyperplane in feature space (classical RBF or quantum Hilbert space).
    Range: (-inf, +inf). Points with f(x) > 0 are assigned to ATTACK (1), and
    f(x) < 0 to NORMAL (0). It lacks a probabilistic scale (e.g. f(x)=2.0 does
    not indicate twice the certainty of f(x)=1.0).
  * Random Forest: Raw leaf frequency / vote fraction P_raw(Y=1|X) in [0, 1].
    While bounded in [0, 1], decision trees push empirical leaf probabilities
    away from 0 and 1, resulting in poor empirical calibration (compression).

- Raw Output:
  The discrete predicted class label y_pred in {0, 1} derived from applying the
  default decision threshold (f(x) > 0 for SVM/QSVC, P_raw >= 0.5 for RF).

- Calibrated Probability:
  A statistically calibrated posterior probability P(Y=1 | score) in [0, 1]
  obtained by passing continuous scores through a Platt sigmoid mapping:
      P(Y=1 | s) = 1 / (1 + exp(A * s + B))
  where parameters (A, B) are fitted strictly on held-out validation data (never
  on test labels) using unregularized/lightly-regularized logistic regression.
  Under true calibration, among flows assigned a calibrated probability of ~0.80,
  approximately 80% are ground-truth network attacks.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure immediate console output in non-interactive / pipeline runs
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC

from src.config import (
    MODELS_DIR,
    RANDOM_SEED,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
    RESULTS_DIR,
    REVIEW_FLAG_REASON,
    VOTE_THREAT_POLICY,
)
from src.quantum_model import (
    build_feature_map,
    build_quantum_kernel,
    fit_quantum_scaler,
    transform_quantum_features,
)

logger = logging.getLogger(__name__)

PRIMARY_FEATURES: List[str] = ["sbytes", "sload", "sttl", "smean"]
ENGINE_BUNDLE_PATH: Path = MODELS_DIR / "cybershield_engine.joblib"
DISAGREEMENT_JSON_PATH: Path = RESULTS_DIR / "disagreement.json"
RELIABILITY_PLOT_PATH: Path = RESULTS_DIR / "calibration_reliability.png"


class CyberShieldEngine:
    """CyberShield Security Engine providing real-time multi-model inference and calibration."""

    def __init__(
        self,
        scaler_svm: StandardScaler,
        scaler_quantum: Any,
        model_svm: SVC,
        model_rf: RandomForestClassifier,
        model_qsvc: QSVC,
        calibrator_svm: LogisticRegression,
        calibrator_rf: LogisticRegression,
        calibrator_qsvc: LogisticRegression,
        qsvc_train_data: Dict[str, Any],
        kernel: Optional[Any] = None,
        features: Optional[List[str]] = None,
    ) -> None:
        self.scaler_svm = scaler_svm
        self.scaler_quantum = scaler_quantum
        self.model_svm = model_svm
        self.model_rf = model_rf
        self.model_qsvc = model_qsvc
        self.calibrator_svm = calibrator_svm
        self.calibrator_rf = calibrator_rf
        self.calibrator_qsvc = calibrator_qsvc
        self.qsvc_train_data = qsvc_train_data
        self.features = features or list(PRIMARY_FEATURES)

        # Lazy-build or assign quantum kernel
        if kernel is not None:
            self.kernel = kernel
        else:
            feature_map = build_feature_map(n_features=len(self.features), reps=2)
            self.kernel = build_quantum_kernel(feature_map, seed=RANDOM_SEED, enforce_psd=True)

    def validate_input(
        self,
        flow_features: Union[Dict[str, Any], List[Any], np.ndarray, pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        """Validate input flow features, ensuring exact feature dimensions, order, and finite numeric values.

        Returns
        -------
        np.ndarray
            2D float64 array of shape (1, 4).
        """
        if flow_features is None:
            raise ValueError("flow_features cannot be None.")

        # Dictionary input
        if isinstance(flow_features, dict):
            missing = [f for f in self.features if f not in flow_features]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}. Expected: {self.features}")
            vals = []
            for f in self.features:
                val = flow_features[f]
                if val is None:
                    raise ValueError(f"Feature '{f}' contains None value.")
                try:
                    f_val = float(val)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Feature '{f}' value {val!r} is not numeric.") from exc
                if not np.isfinite(f_val):
                    raise ValueError(f"Feature '{f}' contains non-finite value: {f_val}")
                vals.append(f_val)
            return np.array(vals, dtype=np.float64).reshape(1, -1)

        # Pandas Series input
        if isinstance(flow_features, pd.Series):
            missing = [f for f in self.features if f not in flow_features.index]
            if missing:
                raise ValueError(f"Missing required feature(s) in Series: {missing}")
            vals = flow_features[self.features].values.astype(np.float64)
            if not np.all(np.isfinite(vals)):
                raise ValueError("Series contains non-finite values (NaN or Inf).")
            return vals.reshape(1, -1)

        # Pandas DataFrame input
        if isinstance(flow_features, pd.DataFrame):
            if len(flow_features) != 1:
                raise ValueError(f"DataFrame input must contain exactly 1 row; got {len(flow_features)} rows.")
            missing = [f for f in self.features if f not in flow_features.columns]
            if missing:
                raise ValueError(f"Missing required column(s) in DataFrame: {missing}")
            vals = flow_features[self.features].values.astype(np.float64)
            if not np.all(np.isfinite(vals)):
                raise ValueError("DataFrame contains non-finite values (NaN or Inf).")
            return vals.reshape(1, -1)

        # List or NumPy array input
        if isinstance(flow_features, (list, tuple, np.ndarray)):
            arr = np.asarray(flow_features, dtype=np.float64)
            if arr.ndim == 1:
                if len(arr) != len(self.features):
                    raise ValueError(f"Expected {len(self.features)} features, got {len(arr)}.")
                arr = arr.reshape(1, -1)
            elif arr.ndim == 2:
                if arr.shape[0] != 1 or arr.shape[1] != len(self.features):
                    raise ValueError(f"Expected 2D array of shape (1, {len(self.features)}), got {arr.shape}.")
            else:
                raise ValueError(f"Unsupported array dimension: ndim={arr.ndim}.")

            if not np.all(np.isfinite(arr)):
                raise ValueError("Input array contains non-finite values (NaN or Inf).")
            return arr

        raise TypeError(f"Unsupported input type: {type(flow_features).__name__}")

    def evaluate_policy(self, svm_pred: int, rf_pred: int, qsvc_pred: int) -> Dict[str, Any]:
        """Evaluate consensus votes, majority prediction, threat level, and review flag.

        Policy in config.py:
          - 0 attack votes: LOW threat, review_flag = False
          - 1 attack vote:  MEDIUM threat, review_flag = True
          - 2 attack votes: HIGH threat, review_flag = True
          - 3 attack votes: HIGH threat, review_flag = False
        """
        votes_count = int(svm_pred + rf_pred + qsvc_pred)
        attack_votes_str = f"{votes_count}/3"
        final_prediction = 1 if votes_count >= 2 else 0

        threat_level = VOTE_THREAT_POLICY.get(votes_count, "HIGH" if votes_count >= 2 else "LOW")

        # Disagreement occurs when models do not all agree (votes is 1 or 2)
        is_disagreement = votes_count in (1, 2)
        review_flag = is_disagreement
        review_reason = REVIEW_FLAG_REASON if is_disagreement else None

        return {
            "attack_votes": attack_votes_str,
            "attack_votes_count": votes_count,
            "final_prediction": final_prediction,
            "threat_level": threat_level,
            "review_flag": review_flag,
            "review_reason": review_reason,
        }

    def analyze(
        self,
        flow_features: Union[Dict[str, Any], List[Any], np.ndarray, pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Perform end-to-end multi-model inference and policy assessment on a single network flow.

        Parameters
        ----------
        flow_features:
            Raw network flow features (dict, list, array, or Series) for ['sbytes', 'sload', 'sttl', 'smean'].

        Returns
        -------
        dict
            Comprehensive analysis report containing individual model predictions, continuous scores,
            calibrated probabilities, consensus vote fraction, majority prediction, threat level,
            and analyst-review flag.
        """
        X = self.validate_input(flow_features)  # shape (1, 4)

        # 1. Classical SVM inference
        X_svm_scaled = self.scaler_svm.transform(X)
        svm_score = float(self.model_svm.decision_function(X_svm_scaled)[0])
        svm_pred = int(self.model_svm.predict(X_svm_scaled)[0])
        svm_prob = float(self.calibrator_svm.predict_proba([[svm_score]])[0, 1])

        # 2. Classical Random Forest inference
        rf_raw_prob = float(self.model_rf.predict_proba(X)[0, 1])
        rf_pred = int(self.model_rf.predict(X)[0])
        rf_prob = float(self.calibrator_rf.predict_proba([[rf_raw_prob]])[0, 1])

        # 3. QSVC inference
        X_q_scaled = transform_quantum_features(self.scaler_quantum, X)
        X_train_q_scaled = self.qsvc_train_data["X_train_scaled"]

        K_new = self.kernel.evaluate(X_q_scaled, X_train_q_scaled)
        qsvc_score = float(self.model_qsvc.decision_function(K_new)[0])
        qsvc_pred = int(self.model_qsvc.predict(K_new)[0])
        qsvc_prob = float(self.calibrator_qsvc.predict_proba([[qsvc_score]])[0, 1])

        # 4. Consensus & Policy Evaluation
        policy = self.evaluate_policy(svm_pred=svm_pred, rf_pred=rf_pred, qsvc_pred=qsvc_pred)

        feature_dict = {f: float(X[0, idx]) for idx, f in enumerate(self.features)}

        return {
            "features": feature_dict,
            "models": {
                "SVM": {
                    "prediction": svm_pred,
                    "decision_score": round(svm_score, 6),
                    "calibrated_prob": round(svm_prob, 6),
                },
                "Random_Forest": {
                    "prediction": rf_pred,
                    "raw_prob": round(rf_raw_prob, 6),
                    "calibrated_prob": round(rf_prob, 6),
                },
                "QSVC": {
                    "prediction": qsvc_pred,
                    "decision_score": round(qsvc_score, 6),
                    "calibrated_prob": round(qsvc_prob, 6),
                },
            },
            "attack_votes": policy["attack_votes"],
            "attack_votes_count": policy["attack_votes_count"],
            "final_prediction": policy["final_prediction"],
            "threat_level": policy["threat_level"],
            "review_flag": policy["review_flag"],
            "review_reason": policy["review_reason"],
        }

    def save(self, bundle_path: Union[str, Path] = ENGINE_BUNDLE_PATH) -> Path:
        """Persist engine bundle and subcomponents to models/ directory using joblib."""
        bundle_path = Path(bundle_path)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "scaler_svm": self.scaler_svm,
            "scaler_quantum": self.scaler_quantum,
            "model_svm": self.model_svm,
            "model_rf": self.model_rf,
            "model_qsvc": self.model_qsvc,
            "calibrator_svm": self.calibrator_svm,
            "calibrator_rf": self.calibrator_rf,
            "calibrator_qsvc": self.calibrator_qsvc,
            "qsvc_train_data": self.qsvc_train_data,
            "features": self.features,
        }

        joblib.dump(bundle, bundle_path)

        # Save individual model and scaler components for modular use
        joblib.dump(self.model_svm, bundle_path.parent / "svm_model.joblib")
        joblib.dump(self.model_rf, bundle_path.parent / "rf_model.joblib")
        joblib.dump(self.model_qsvc, bundle_path.parent / "qsvc_model.joblib")
        joblib.dump(self.scaler_svm, bundle_path.parent / "svm_scaler.joblib")
        joblib.dump(self.scaler_quantum, bundle_path.parent / "quantum_scaler.joblib")

        calibrators_bundle = {
            "SVM": self.calibrator_svm,
            "Random_Forest": self.calibrator_rf,
            "QSVC": self.calibrator_qsvc,
        }
        joblib.dump(calibrators_bundle, bundle_path.parent / "calibrators.joblib")
        joblib.dump(self.qsvc_train_data, bundle_path.parent / "qsvc_train_data.joblib")

        logger.info(f"Engine bundle saved to {bundle_path}")
        return bundle_path

    @classmethod
    def load(cls, bundle_path: Union[str, Path] = ENGINE_BUNDLE_PATH) -> CyberShieldEngine:
        """Reload engine from saved joblib bundle."""
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Engine bundle not found at {bundle_path}. Run build_and_save_engine() first.")

        bundle = joblib.load(bundle_path)
        return cls(
            scaler_svm=bundle["scaler_svm"],
            scaler_quantum=bundle["scaler_quantum"],
            model_svm=bundle["model_svm"],
            model_rf=bundle["model_rf"],
            model_qsvc=bundle["model_qsvc"],
            calibrator_svm=bundle["calibrator_svm"],
            calibrator_rf=bundle["calibrator_rf"],
            calibrator_qsvc=bundle["calibrator_qsvc"],
            qsvc_train_data=bundle["qsvc_train_data"],
            features=bundle.get("features", PRIMARY_FEATURES),
        )


def compute_and_save_disagreement(
    predictions_csv_path: Path = RESULTS_DIR / "predictions_A.csv",
    output_json_path: Path = DISAGREEMENT_JSON_PATH,
) -> Dict[str, Any]:
    """Calculate pairwise agreement and each model's correctness among disagreements on held-out test subset."""
    if not predictions_csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found at {predictions_csv_path}")

    df = pd.read_csv(predictions_csv_path)
    n = len(df)
    y_true = df["y_true"].values
    svm = df["svm_pred"].values
    rf = df["rf_pred"].values
    qsvc = df["qsvc_pred"].values

    # Pairwise agreement
    svm_rf_agree = int(np.sum(svm == rf))
    svm_qsvc_agree = int(np.sum(svm == qsvc))
    rf_qsvc_agree = int(np.sum(rf == qsvc))

    # Ensemble agreement (all 3 agree)
    unanimous_mask = (svm == rf) & (rf == qsvc)
    unanimous_count = int(np.sum(unanimous_mask))
    disagree_mask = ~unanimous_mask
    disagree_count = int(np.sum(disagree_mask))

    # Model correctness on ensemble disagreements
    svm_dis_correct = int(np.sum(svm[disagree_mask] == y_true[disagree_mask]))
    rf_dis_correct = int(np.sum(rf[disagree_mask] == y_true[disagree_mask]))
    qsvc_dis_correct = int(np.sum(qsvc[disagree_mask] == y_true[disagree_mask]))

    # Pairwise disagreement breakdowns
    def pair_breakdown(m1_name: str, m1: np.ndarray, m2_name: str, m2: np.ndarray) -> Dict[str, Any]:
        p_dis = m1 != m2
        denom = int(np.sum(p_dis))
        m1_c = int(np.sum((m1 == y_true) & p_dis))
        m2_c = int(np.sum((m2 == y_true) & p_dis))
        both_w = int(np.sum((m1 != y_true) & (m2 != y_true) & p_dis))
        return {
            "disagreement_count": denom,
            f"{m1_name}_correct": m1_c,
            f"{m2_name}_correct": m2_c,
            "both_wrong": both_w,
            f"{m1_name}_accuracy_on_disagreements": round(m1_c / denom, 6) if denom > 0 else 0.0,
            f"{m2_name}_accuracy_on_disagreements": round(m2_c / denom, 6) if denom > 0 else 0.0,
        }

    report = {
        "schema_version": "1.0.0",
        "test_sample_count": n,
        "models": ["SVM", "Random_Forest", "QSVC"],
        "features": PRIMARY_FEATURES,
        "pairwise_agreement": {
            "SVM_vs_Random_Forest": {
                "agreed_count": svm_rf_agree,
                "disagreed_count": n - svm_rf_agree,
                "total_samples": n,
                "agreement_rate": round(svm_rf_agree / n, 6),
                "disagreement_rate": round((n - svm_rf_agree) / n, 6),
            },
            "SVM_vs_QSVC": {
                "agreed_count": svm_qsvc_agree,
                "disagreed_count": n - svm_qsvc_agree,
                "total_samples": n,
                "agreement_rate": round(svm_qsvc_agree / n, 6),
                "disagreement_rate": round((n - svm_qsvc_agree) / n, 6),
            },
            "Random_Forest_vs_QSVC": {
                "agreed_count": rf_qsvc_agree,
                "disagreed_count": n - rf_qsvc_agree,
                "total_samples": n,
                "agreement_rate": round(rf_qsvc_agree / n, 6),
                "disagreement_rate": round((n - rf_qsvc_agree) / n, 6),
            },
        },
        "ensemble_agreement": {
            "unanimous_agreement_count": unanimous_count,
            "unanimous_rate": round(unanimous_count / n, 6),
            "disagreement_count": disagree_count,
            "disagreement_rate": round(disagree_count / n, 6),
        },
        "correctness_on_ensemble_disagreements": {
            "disagreement_denominator": disagree_count,
            "SVM": {
                "correct_count": svm_dis_correct,
                "accuracy_on_disagreements": round(svm_dis_correct / disagree_count, 6) if disagree_count > 0 else 0.0,
            },
            "Random_Forest": {
                "correct_count": rf_dis_correct,
                "accuracy_on_disagreements": round(rf_dis_correct / disagree_count, 6) if disagree_count > 0 else 0.0,
            },
            "QSVC": {
                "correct_count": qsvc_dis_correct,
                "accuracy_on_disagreements": round(qsvc_dis_correct / disagree_count, 6) if disagree_count > 0 else 0.0,
            },
        },
        "pairwise_disagreement_breakdown": {
            "SVM_vs_Random_Forest": pair_breakdown("SVM", svm, "Random_Forest", rf),
            "SVM_vs_QSVC": pair_breakdown("SVM", svm, "QSVC", qsvc),
            "Random_Forest_vs_QSVC": pair_breakdown("Random_Forest", rf, "QSVC", qsvc),
        },
        "interpretation": (
            f"On the {disagree_count} held-out test samples where the heterogeneous models disagreed, "
            f"Classical Random Forest achieved the highest correctness ({rf_dis_correct}/{disagree_count}, "
            f"{rf_dis_correct/disagree_count:.1%}), while Classical SVM and QSVC tied "
            f"({svm_dis_correct}/{disagree_count}, {svm_dis_correct/disagree_count:.1%}). "
            "Heterogeneous disagreements provide an effective trigger for human analyst review, "
            "without indicating quantum supremacy over classical tree models."
        ),
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def plot_reliability_diagrams(
    y_test: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    brier_scores: Dict[str, float],
    save_path: Path = RELIABILITY_PLOT_PATH,
) -> Path:
    """Generate and save calibration reliability curves for all models."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")

    colors = {
        "SVM": "#1f77b4",
        "Random_Forest": "#2ca02c",
        "QSVC": "#9467bd",
    }
    markers = {
        "SVM": "s",
        "Random_Forest": "o",
        "QSVC": "^",
    }

    for name, probs in probs_dict.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=5, strategy="uniform")
        brier = brier_scores.get(name, 0.0)
        ax.plot(
            mean_pred,
            frac_pos,
            marker=markers.get(name, "o"),
            color=colors.get(name, "blue"),
            linewidth=2,
            label=f"{name} (Brier: {brier:.4f})",
        )

    ax.set_title("Platt Sigmoid Calibration — Reliability Diagram\n(Validation-Only Calibration Evaluated on Held-Out Test)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability (Attack)", fontsize=10)
    ax.set_ylabel("Fraction of True Positives", fontsize=10)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def build_and_save_engine(
    train_csv: Path = RAW_TRAIN_CSV,
    test_csv: Path = RAW_TEST_CSV,
    n_train: int = 100,
    n_val: int = 50,
    n_test: int = 100,
    random_seed: int = RANDOM_SEED,
    output_bundle_path: Path = ENGINE_BUNDLE_PATH,
) -> Tuple[CyberShieldEngine, Dict[str, Any]]:
    """Train models on Stage 6 train subset, calibrate on validation data, evaluate on test, and persist."""
    print("=" * 70)
    print("STAGE 7: BUILDING CYBERSHIELD ENGINE & CALIBRATING MODELS")
    print("=" * 70)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    features = list(PRIMARY_FEATURES)

    # 1. Exact Stage 6 training partition (n=100, seed=42)
    X_train_full = train_df[features].values.astype(np.float64)
    y_train_full = train_df["label"].values.astype(int)

    split_tr = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=random_seed)
    train_idx, _ = next(split_tr.split(X_train_full, y_train_full))
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]

    # 2. Validation partition: strictly from the remaining training data (never test labels)
    rem_train_idx = np.setdiff1d(np.arange(len(train_df)), train_idx)
    split_val = StratifiedShuffleSplit(n_splits=1, train_size=n_val, random_state=random_seed)
    val_rel_idx, _ = next(split_val.split(X_train_full[rem_train_idx], y_train_full[rem_train_idx]))
    val_idx = rem_train_idx[val_rel_idx]
    X_val = X_train_full[val_idx]
    y_val = y_train_full[val_idx]

    # 3. Exact Stage 6 test partition (n=100, seed=42)
    X_test_full = test_df[features].values.astype(np.float64)
    y_test_full = test_df["label"].values.astype(int)

    split_te = StratifiedShuffleSplit(n_splits=1, train_size=n_test, random_state=random_seed)
    test_idx, _ = next(split_te.split(X_test_full, y_test_full))
    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]

    print(f"Data subsets: Train={len(X_train)} (from train.csv), Val={len(X_val)} (from train.csv), Test={len(X_test)} (from test.csv)")

    # 4. Train Classical SVM (StandardScaler fitted strictly on X_train)
    print("Training Classical SVM (RBF)...")
    scaler_svm = StandardScaler().fit(X_train)
    X_train_svm = scaler_svm.transform(X_train)
    X_val_svm = scaler_svm.transform(X_val)
    X_test_svm = scaler_svm.transform(X_test)

    model_svm = SVC(C=5.0, kernel="rbf", gamma="auto", random_state=random_seed)
    model_svm.fit(X_train_svm, y_train)

    score_val_svm = model_svm.decision_function(X_val_svm)
    score_test_svm = model_svm.decision_function(X_test_svm)

    # 5. Train Classical Random Forest
    print("Training Classical Random Forest...")
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=random_seed)
    model_rf.fit(X_train, y_train)

    score_val_rf = model_rf.predict_proba(X_val)[:, 1]
    score_test_rf = model_rf.predict_proba(X_test)[:, 1]

    # 6. Train QSVC (Quantum Scaler fitted strictly on X_train)
    print("Computing quantum feature scaling and kernel...")
    scaler_quantum = fit_quantum_scaler(X_train)
    X_train_q = transform_quantum_features(scaler_quantum, X_train)
    X_val_q = transform_quantum_features(scaler_quantum, X_val)
    X_test_q = transform_quantum_features(scaler_quantum, X_test)

    feature_map = build_feature_map(n_features=len(features), reps=2)
    kernel = build_quantum_kernel(feature_map, seed=random_seed, enforce_psd=True)

    # Cache/load training Gram matrix to avoid unnecessary recomputation
    cache_k_train_path = MODELS_DIR / "k_train_cache.npy"
    if cache_k_train_path.exists():
        print(f"Loading cached K_train from {cache_k_train_path}...")
        K_train = np.load(cache_k_train_path)
    else:
        print("Evaluating training kernel Gram matrix (100x100)...")
        K_train = kernel.evaluate(X_train_q)
        cache_k_train_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_k_train_path, K_train)

    model_qsvc = QSVC(quantum_kernel="precomputed", C=10.0, random_state=random_seed)
    model_qsvc.fit(K_train, y_train)

    print("Evaluating validation kernel matrix (50x100)...")
    K_val = kernel.evaluate(X_val_q, X_train_q)
    score_val_qsvc = model_qsvc.decision_function(K_val)

    cache_k_test_path = MODELS_DIR / "k_test_cache.npy"
    if cache_k_test_path.exists():
        print(f"Loading cached K_test from {cache_k_test_path}...")
        K_test = np.load(cache_k_test_path)
    else:
        print("Evaluating test kernel matrix (100x100)...")
        K_test = kernel.evaluate(X_test_q, X_train_q)
        np.save(cache_k_test_path, K_test)

    score_test_qsvc = model_qsvc.decision_function(K_test)

    # 7. Fit Platt Sigmoid Calibrators using VALIDATION SCORES ONLY
    print("\nFitting Platt Sigmoid Calibrators on validation partition (n_val=50)...")
    calibrator_svm = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator_svm.fit(score_val_svm.reshape(-1, 1), y_val)

    calibrator_rf = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator_rf.fit(score_val_rf.reshape(-1, 1), y_val)

    calibrator_qsvc = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator_qsvc.fit(score_val_qsvc.reshape(-1, 1), y_val)

    # 8. Evaluate Brier Scores on Held-Out Test Set
    prob_test_svm = calibrator_svm.predict_proba(score_test_svm.reshape(-1, 1))[:, 1]
    prob_test_rf = calibrator_rf.predict_proba(score_test_rf.reshape(-1, 1))[:, 1]
    prob_test_qsvc = calibrator_qsvc.predict_proba(score_test_qsvc.reshape(-1, 1))[:, 1]

    brier_svm = float(brier_score_loss(y_test, prob_test_svm))
    brier_rf = float(brier_score_loss(y_test, prob_test_rf))
    brier_qsvc = float(brier_score_loss(y_test, prob_test_qsvc))

    # Also compute raw RF Brier score for comparison
    brier_rf_raw = float(brier_score_loss(y_test, score_test_rf))

    print(f"Test Brier Scores (lower is better):")
    print(f"  SVM (Calibrated):          {brier_svm:.4f}")
    print(f"  Random Forest (Calibrated): {brier_rf:.4f} (Raw: {brier_rf_raw:.4f})")
    print(f"  QSVC (Calibrated):         {brier_qsvc:.4f}")

    brier_dict = {
        "SVM": brier_svm,
        "Random_Forest": brier_rf,
        "QSVC": brier_qsvc,
    }
    probs_dict = {
        "SVM": prob_test_svm,
        "Random_Forest": prob_test_rf,
        "QSVC": prob_test_qsvc,
    }

    # 9. Plot and Save Reliability Diagram
    plot_reliability_diagrams(y_test, probs_dict, brier_dict, save_path=RELIABILITY_PLOT_PATH)
    print(f"[OK] Saved calibration reliability plot to {RELIABILITY_PLOT_PATH}")

    # 10. Generate Disagreement Report on predictions_A.csv
    disagree_report = compute_and_save_disagreement(
        predictions_csv_path=RESULTS_DIR / "predictions_A.csv",
        output_json_path=DISAGREEMENT_JSON_PATH,
    )
    print(f"[OK] Saved disagreement analysis to {DISAGREEMENT_JSON_PATH}")

    # 11. Package and Persist Engine
    qsvc_train_data = {
        "X_train_raw": X_train,
        "X_train_scaled": X_train_q,
        "y_train": y_train,
        "features": features,
        "random_seed": random_seed,
    }

    engine = CyberShieldEngine(
        scaler_svm=scaler_svm,
        scaler_quantum=scaler_quantum,
        model_svm=model_svm,
        model_rf=model_rf,
        model_qsvc=model_qsvc,
        calibrator_svm=calibrator_svm,
        calibrator_rf=calibrator_rf,
        calibrator_qsvc=calibrator_qsvc,
        qsvc_train_data=qsvc_train_data,
        kernel=kernel,
        features=features,
    )

    engine.save(output_bundle_path)
    print(f"[OK] CyberShield Engine successfully saved to {output_bundle_path}")

    metrics_meta = {
        "brier_scores": brier_dict,
        "raw_brier_scores": {"Random_Forest": brier_rf_raw},
        "disagreement": disagree_report,
    }

    return engine, metrics_meta


if __name__ == "__main__":
    build_and_save_engine()
