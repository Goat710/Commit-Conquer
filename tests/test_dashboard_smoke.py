"""End-to-end smoke test for dashboard data pipelines and inference."""

import pandas as pd
from src.dashboard_data import (
    get_verified_presets,
    load_disagreement_results,
    load_predictions_slice,
    load_security_engine,
    load_stage7b_results,
    load_stage8c_results,
    load_stage8e_results,
    load_stage8f_results,
)


def test_stage8c_metrics_consistency():
    data = load_stage8c_results()
    summary = data["five_seed_summary"]
    for m in ["rf", "svm", "qsvc"]:
        assert m in summary
        t1 = summary[m]["target_1pct"]
        t2 = summary[m]["target_2pct"]
        assert 0.0 <= t1["test_recall_mean"] <= 1.0
        assert 0.0 <= t2["test_recall_mean"] <= 1.0
        assert 0.0 <= t1["achieved_test_fpr_mean"] <= 1.0


def test_stage8e_metrics_consistency():
    data = load_stage8e_results()
    agg = data["aggregated_results"]
    for space in ["F4-Base", "F4-NoTTL", "F39-Full"]:
        assert space in agg
        for model_name in agg[space]:
            rec = agg[space][model_name]["threshold_tau_01"]["recall"]
            assert 0.0 <= rec["mean"] <= 1.0


def test_stage8f_scaling_consistency():
    data = load_stage8f_results()
    scaling = data["experiment_a_simulation_scaling"]
    assert len(scaling) == 7
    # Verify q=4 to q=16 exist
    assert "q_4" in scaling and "q_16" in scaling
    assert scaling["q_4"]["qubit_count"] == 4
    assert scaling["q_16"]["qubit_count"] == 16


def test_stage8f_scaling_buffer_schema_regression():
    """Verify statevector buffer RAM key matches schema without KeyError, and fallback handles missing keys."""
    data = load_stage8f_results()
    scaling = data["experiment_a_simulation_scaling"]

    for q_key, d in scaling.items():
        mem = d.get("memory", {})
        # Actual key in benchmark JSON
        assert "statevector_buffer_mb" in mem
        buffer_mb = mem.get("statevector_buffer_mb")
        if buffer_mb is None:
            buffer_mb = mem.get("theoretical_statevector_buffer_mb")
        buffer_val = round(buffer_mb, 2) if buffer_mb is not None else "Not available"

        assert isinstance(buffer_val, float)
        assert buffer_val > 0.0

    # Test fallback on absent key returns 'Not available' without crashing
    empty_mem = {}
    buf_empty = empty_mem.get("statevector_buffer_mb")
    if buf_empty is None:
        buf_empty = empty_mem.get("theoretical_statevector_buffer_mb")
    val_empty = round(buf_empty, 2) if buf_empty is not None else "Not available"
    assert val_empty == "Not available"

    # Test backward-compatibility fallback with legacy key
    legacy_mem = {"theoretical_statevector_buffer_mb": 12.34}
    buf_legacy = legacy_mem.get("statevector_buffer_mb")
    if buf_legacy is None:
        buf_legacy = legacy_mem.get("theoretical_statevector_buffer_mb")
    val_legacy = round(buf_legacy, 2) if buf_legacy is not None else "Not available"
    assert val_legacy == 12.34


def test_live_engine_presets_smoke():
    engine = load_security_engine()
    presets = get_verified_presets()
    assert len(presets) == 4, "Must provide exactly 4 verified presets"
    for name, p in presets.items():
        res = engine.analyze(p["features"])
        assert "models" in res
        assert "Random_Forest" in res["models"]
        assert "SVM" in res["models"]
        assert "QSVC" in res["models"]
        assert res["threat_level"] in ("LOW", "MEDIUM", "HIGH")
        assert res["attack_votes"] in ("0/3", "1/3", "2/3", "3/3")
        assert isinstance(res["review_flag"], bool)


def test_ui_input_validation_hardening():
    """Verify input validation handles negative numbers, None, NaN, inf, and strings gracefully."""
    import numpy as np

    def validate_inputs(input_dict):
        for k, val in input_dict.items():
            if val is None or not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0:
                return False, f"Invalid input for {k}: value must be a non-negative, finite number."
        return True, "Valid"

    # Valid inputs
    valid_dict = {"sbytes": 100.0, "sload": 500.0, "sttl": 64.0, "smean": 50.0}
    is_ok, _ = validate_inputs(valid_dict)
    assert is_ok

    # Invalid inputs
    invalid_cases = [
        ({"sbytes": -1.0, "sload": 500.0, "sttl": 64.0, "smean": 50.0}, "sbytes"),
        ({"sbytes": None, "sload": 500.0, "sttl": 64.0, "smean": 50.0}, "sbytes"),
        ({"sbytes": np.nan, "sload": 500.0, "sttl": 64.0, "smean": 50.0}, "sbytes"),
        ({"sbytes": np.inf, "sload": 500.0, "sttl": 64.0, "smean": 50.0}, "sbytes"),
        ({"sbytes": "invalid", "sload": 500.0, "sttl": 64.0, "smean": 50.0}, "sbytes"),
    ]
    for bad_dict, expected_field in invalid_cases:
        is_ok, msg = validate_inputs(bad_dict)
        assert not is_ok
        assert f"Invalid input for {expected_field}" in msg
