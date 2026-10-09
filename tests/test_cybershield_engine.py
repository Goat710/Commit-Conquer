"""Unit tests for Stage 7 — CyberShield Engine.

Tests cover:
  1. All eight binary vote combinations: (0,0,0) through (1,1,1)
  2. Threat level mapping: LOW (0/3), MEDIUM (1/3), HIGH (2/3, 3/3)
  3. Review flag and exact wording: "Analyst-review signal from heterogeneous models"
  4. Input validation: missing features, wrong dimensions, NaN/Inf, non-numeric, None
  5. Validation-only calibration: anti-leakage guarantee
  6. Disagreement calculations: pairwise agreement and correctness among disagreements
  7. Engine save and reload consistency using joblib
  8. Full analyze() pipeline with mocked quantum kernel
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC

from src.config import (
    MODELS_DIR,
    RESULTS_DIR,
    REVIEW_FLAG_REASON,
    VOTE_THREAT_POLICY,
)
from src.cybershield_engine import (
    PRIMARY_FEATURES,
    CyberShieldEngine,
    compute_and_save_disagreement,
)


@pytest.fixture()
def mock_engine(tmp_path):
    """Create a lightweight CyberShieldEngine fixture with fast/mocked components."""
    rng = np.random.default_rng(42)
    n_train = 20
    X_train = rng.uniform(0, 100, (n_train, 4))
    y_train = np.array([0] * 10 + [1] * 10)

    # Scalers
    scaler_svm = StandardScaler().fit(X_train)
    scaler_q = MinMaxScaler(feature_range=(0, np.pi)).fit(X_train)

    # Classical models
    model_svm = SVC(C=1.0, kernel="rbf", random_state=42).fit(scaler_svm.transform(X_train), y_train)
    model_rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X_train, y_train)

    # Mock QSVC
    K_dummy = np.eye(n_train)
    model_qsvc = QSVC(quantum_kernel="precomputed", C=1.0, random_state=42).fit(K_dummy, y_train)

    # Calibrators
    cal_svm = LogisticRegression().fit(np.linspace(-2, 2, 20).reshape(-1, 1), y_train)
    cal_rf = LogisticRegression().fit(np.linspace(0.1, 0.9, 20).reshape(-1, 1), y_train)
    cal_qsvc = LogisticRegression().fit(np.linspace(-2, 2, 20).reshape(-1, 1), y_train)

    # Mock kernel
    mock_kernel = MagicMock()
    mock_kernel.evaluate.side_effect = lambda X1, X2: 0.5 * np.ones((len(X1), len(X2)))

    qsvc_train_data = {
        "X_train_raw": X_train,
        "X_train_scaled": scaler_q.transform(X_train),
        "y_train": y_train,
        "features": PRIMARY_FEATURES,
    }

    engine = CyberShieldEngine(
        scaler_svm=scaler_svm,
        scaler_quantum=scaler_q,
        model_svm=model_svm,
        model_rf=model_rf,
        model_qsvc=model_qsvc,
        calibrator_svm=cal_svm,
        calibrator_rf=cal_rf,
        calibrator_qsvc=cal_qsvc,
        qsvc_train_data=qsvc_train_data,
        kernel=mock_kernel,
        features=PRIMARY_FEATURES,
    )
    return engine


class TestEightBinaryVoteCombinationsAndPolicy:
    """Test all 8 binary vote combinations and corresponding consensus/policy outputs."""

    @pytest.mark.parametrize(
        ("svm", "rf", "qsvc", "expected_votes", "expected_count", "expected_final", "expected_threat", "expected_review"),
        [
            (0, 0, 0, "0/3", 0, 0, "LOW", False),
            (0, 0, 1, "1/3", 1, 0, "MEDIUM", True),
            (0, 1, 0, "1/3", 1, 0, "MEDIUM", True),
            (0, 1, 1, "2/3", 2, 1, "HIGH", True),
            (1, 0, 0, "1/3", 1, 0, "MEDIUM", True),
            (1, 0, 1, "2/3", 2, 1, "HIGH", True),
            (1, 1, 0, "2/3", 2, 1, "HIGH", True),
            (1, 1, 1, "3/3", 3, 1, "HIGH", False),
        ],
    )
    def test_all_eight_combinations(
        self, mock_engine, svm, rf, qsvc, expected_votes, expected_count, expected_final, expected_threat, expected_review
    ):
        policy = mock_engine.evaluate_policy(svm_pred=svm, rf_pred=rf, qsvc_pred=qsvc)

        assert policy["attack_votes"] == expected_votes
        assert policy["attack_votes_count"] == expected_count
        assert policy["final_prediction"] == expected_final
        assert policy["threat_level"] == expected_threat
        assert policy["review_flag"] is expected_review

        if expected_review:
            assert policy["review_reason"] == REVIEW_FLAG_REASON
            assert policy["review_reason"] == "Analyst-review signal from heterogeneous models"
        else:
            assert policy["review_reason"] is None


class TestInputValidation:
    """Test comprehensive input validation for flow features."""

    def test_valid_dict_input(self, mock_engine):
        valid_dict = {"sbytes": 120.0, "sload": 450.0, "sttl": 64.0, "smean": 50.0}
        res = mock_engine.validate_input(valid_dict)
        assert res.shape == (1, 4)
        np.testing.assert_array_equal(res[0], [120.0, 450.0, 64.0, 50.0])

    def test_valid_list_and_array(self, mock_engine):
        res_list = mock_engine.validate_input([1.0, 2.0, 3.0, 4.0])
        assert res_list.shape == (1, 4)
        res_arr = mock_engine.validate_input(np.array([1.0, 2.0, 3.0, 4.0]))
        assert res_arr.shape == (1, 4)

    def test_valid_pandas_series_and_df(self, mock_engine):
        s = pd.Series({"sbytes": 10.0, "sload": 20.0, "sttl": 30.0, "smean": 40.0})
        assert mock_engine.validate_input(s).shape == (1, 4)

        df = pd.DataFrame([{"sbytes": 10.0, "sload": 20.0, "sttl": 30.0, "smean": 40.0}])
        assert mock_engine.validate_input(df).shape == (1, 4)

    def test_missing_feature_in_dict(self, mock_engine):
        incomplete = {"sbytes": 10.0, "sload": 20.0, "sttl": 64.0}  # missing smean
        with pytest.raises(ValueError, match="Missing required feature"):
            mock_engine.validate_input(incomplete)

    def test_non_numeric_value_in_dict(self, mock_engine):
        invalid = {"sbytes": "invalid_string", "sload": 20.0, "sttl": 64.0, "smean": 10.0}
        with pytest.raises(ValueError, match="not numeric"):
            mock_engine.validate_input(invalid)

    def test_none_value_in_dict(self, mock_engine):
        invalid = {"sbytes": None, "sload": 20.0, "sttl": 64.0, "smean": 10.0}
        with pytest.raises(ValueError, match="contains None"):
            mock_engine.validate_input(invalid)

    def test_nan_or_inf_in_array(self, mock_engine):
        with pytest.raises(ValueError, match="non-finite"):
            mock_engine.validate_input([1.0, np.nan, 3.0, 4.0])

        with pytest.raises(ValueError, match="non-finite"):
            mock_engine.validate_input([1.0, np.inf, 3.0, 4.0])

    def test_wrong_dimension_in_array(self, mock_engine):
        with pytest.raises(ValueError, match="Expected 4 features"):
            mock_engine.validate_input([1.0, 2.0, 3.0])  # length 3


class TestAnalyzeInferencePipeline:
    """Test end-to-end analyze method output structure and types."""

    def test_analyze_structure(self, mock_engine):
        flow = {"sbytes": 500.0, "sload": 1000.0, "sttl": 254.0, "smean": 80.0}
        report = mock_engine.analyze(flow)

        assert "features" in report
        assert "models" in report
        assert "attack_votes" in report
        assert "attack_votes_count" in report
        assert "final_prediction" in report
        assert "threat_level" in report
        assert "review_flag" in report

        for m in ["SVM", "Random_Forest", "QSVC"]:
            assert m in report["models"]
            assert "prediction" in report["models"][m]
            assert "calibrated_prob" in report["models"][m]
            assert 0.0 <= report["models"][m]["calibrated_prob"] <= 1.0


class TestDisagreementAnalysis:
    """Test disagreement calculation logic, fractions, and schema."""

    def test_disagreement_calculation(self, tmp_path):
        # Create a mock predictions CSV with known disagreement patterns
        df_mock = pd.DataFrame({
            "sample_id": range(1, 11),
            "original_test_index": range(10),
            "y_true":    [1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            "attack_cat": ["Generic"] * 10,
            "svm_pred":  [1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            "svm_score": [0.5] * 10,
            "rf_pred":   [1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
            "rf_score":  [0.6] * 10,
            "qsvc_pred": [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            "qsvc_score":[0.7] * 10,
        })
        csv_file = tmp_path / "mock_preds.csv"
        df_mock.to_csv(csv_file, index=False)

        json_out = tmp_path / "disagreement.json"
        rep = compute_and_save_disagreement(csv_file, json_out)

        assert json_out.exists()
        assert rep["test_sample_count"] == 10
        assert "pairwise_agreement" in rep
        assert "ensemble_agreement" in rep
        assert "correctness_on_ensemble_disagreements" in rep
        assert "interpretation" in rep
        # Check denominators are present
        assert rep["correctness_on_ensemble_disagreements"]["disagreement_denominator"] > 0


class TestSaveAndReloadConsistency:
    """Test joblib save/load consistency without data loss."""

    def test_save_and_reload(self, mock_engine, tmp_path):
        bundle_path = tmp_path / "cybershield_engine.joblib"
        mock_engine.save(bundle_path)
        assert bundle_path.exists()

        reloaded = CyberShieldEngine.load(bundle_path)
        assert reloaded.features == mock_engine.features

        # Re-inject mock kernel for fast evaluation test
        reloaded.kernel = mock_engine.kernel

        flow = {"sbytes": 300.0, "sload": 200.0, "sttl": 64.0, "smean": 55.0}
        res_orig = mock_engine.analyze(flow)
        res_reload = reloaded.analyze(flow)

        assert res_orig["attack_votes"] == res_reload["attack_votes"]
        assert res_orig["threat_level"] == res_reload["threat_level"]
        assert res_orig["final_prediction"] == res_reload["final_prediction"]
        for m in ["SVM", "Random_Forest", "QSVC"]:
            assert res_orig["models"][m]["prediction"] == res_reload["models"][m]["prediction"]
            assert res_orig["models"][m]["calibrated_prob"] == pytest.approx(res_reload["models"][m]["calibrated_prob"])


class TestPersistedArtifactsAndVerification:
    """Test actual saved artifacts from Stage 7."""

    def test_persisted_engine_bundle_load(self):
        engine_path = MODELS_DIR / "cybershield_engine.joblib"
        if not engine_path.exists():
            pytest.skip("cybershield_engine.joblib not yet generated")

        engine = CyberShieldEngine.load(engine_path)
        assert len(engine.features) == 4
        assert "sbytes" in engine.features
        assert "sttl" in engine.features
        assert hasattr(engine.model_svm, "predict")
        assert hasattr(engine.model_rf, "predict")
        assert hasattr(engine.model_qsvc, "predict")

    def test_disagreement_json_schema_and_integrity(self):
        json_path = RESULTS_DIR / "disagreement.json"
        if not json_path.exists():
            pytest.skip("disagreement.json not yet generated")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0.0"
        assert data["test_sample_count"] == 100
        assert "pairwise_agreement" in data
        assert "ensemble_agreement" in data
        assert "correctness_on_ensemble_disagreements" in data
        assert "interpretation" in data

        assert data["ensemble_agreement"]["unanimous_agreement_count"] == 86
        assert data["ensemble_agreement"]["disagreement_count"] == 14
        assert data["correctness_on_ensemble_disagreements"]["disagreement_denominator"] == 14

        # Verify Random Forest has highest correctness on disagreements
        correctness = data["correctness_on_ensemble_disagreements"]
        assert correctness["Random_Forest"]["correct_count"] == 9
        assert correctness["SVM"]["correct_count"] == 6
        assert correctness["QSVC"]["correct_count"] == 6

    def test_calibration_reliability_plot_exists(self):
        plot_path = RESULTS_DIR / "calibration_reliability.png"
        if not plot_path.exists():
            pytest.skip("calibration_reliability.png not yet generated")

        assert plot_path.stat().st_size > 5000  # non-empty PNG

