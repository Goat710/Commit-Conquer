"""Tests for dashboard data loaders and verified presets."""

from src.dashboard_data import (
    get_verified_presets,
    load_disagreement_results,
    load_predictions_slice,
    load_stage7b_results,
    load_stage8c_results,
    load_stage8e_results,
    load_stage8f_results,
)


def test_load_stage8c_results():
    data = load_stage8c_results()
    assert "five_seed_summary" in data
    assert "rf" in data["five_seed_summary"]
    assert "svm" in data["five_seed_summary"]
    assert "qsvc" in data["five_seed_summary"]
    assert "target_1pct" in data["five_seed_summary"]["rf"]
    assert "target_2pct" in data["five_seed_summary"]["rf"]


def test_load_stage8e_results():
    data = load_stage8e_results()
    assert "aggregated_results" in data
    assert "F4-Base" in data["aggregated_results"]
    assert "F4-NoTTL" in data["aggregated_results"]
    assert "F39-Full" in data["aggregated_results"]


def test_load_stage8f_results():
    data = load_stage8f_results()
    assert "experiment_a_simulation_scaling" in data
    assert "experiment_b_nystroem_approximation" in data


def test_load_stage7b_results():
    data = load_stage7b_results()
    assert "verified_benchmarks" in data
    assert "disagreement_and_ensemble_summary" in data


def test_load_disagreement_results():
    data = load_disagreement_results()
    assert "pairwise_agreements" in data
    assert "three_way_agreements" in data


def test_load_predictions_slice():
    df = load_predictions_slice(n_samples=50)
    assert len(df) == 50
    assert "y_true" in df.columns
    assert "qsvc_pred" in df.columns
    assert "rf_pred" in df.columns
    assert "svm_pred" in df.columns


def test_get_verified_presets():
    presets = get_verified_presets()
    assert len(presets) >= 3
    for name, p in presets.items():
        assert "features" in p
        assert set(p["features"].keys()) == {"sbytes", "sload", "sttl", "smean"}
        assert p["ground_truth"] in (0, 1)
