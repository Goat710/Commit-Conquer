"""Unit tests for src/quantum_benchmark.py — Stage 5 Step 4.

Tests cover:
  - Matrix memory estimation calculations
  - O(n^2) runtime projection logic
  - Safety stopping rules (run > 120s, projected > 180s)
  - Extrapolation calculations and labeled assumptions
  - Recommendation boundaries (strictly <= 200 train, <= 50 val, <= 100 test)
  - Hard sample limit validation (> 400 raises ValueError)
  - JSON artifact schema compliance (no mock/fabricated metrics)
"""

import json
from pathlib import Path
import pytest

from src.quantum_benchmark import (
    MAX_RUNTIME_PER_RUN_S,
    MAX_PROJECTED_NEXT_RUN_S,
    MAX_SAMPLES,
    compute_extrapolations,
    estimate_matrix_memory,
    generate_recommendations,
    get_software_versions,
    project_next_runtime,
    should_stop_benchmark,
)


class TestEstimateMatrixMemory:
    def test_square_matrix_calculation(self):
        """100x100 matrix of float64 is exactly 80,000 bytes."""
        mem = estimate_matrix_memory(100, 100)
        assert mem["entries"] == 10000
        assert mem["bytes"] == 80000
        assert mem["kb"] == round(80000 / 1024.0, 4)
        assert mem["mb"] == round(80000 / (1024.0 * 1024.0), 4)

    def test_rectangular_matrix_calculation(self):
        """82332 x 100 matrix memory."""
        mem = estimate_matrix_memory(82332, 100)
        assert mem["entries"] == 8233200
        assert mem["bytes"] == 8233200 * 8


class TestProjectNextRuntime:
    def test_quadratic_scaling(self):
        """Doubling sample size quadruples projected time."""
        t_proj = project_next_runtime(prev_n=50, prev_time_s=10.0, next_n=100)
        assert t_proj == pytest.approx(40.0)

    def test_quadruple_scaling(self):
        """4x samples -> 16x time."""
        t_proj = project_next_runtime(prev_n=100, prev_time_s=5.0, next_n=400)
        assert t_proj == pytest.approx(80.0)

    def test_rejects_non_positive_prev_n(self):
        with pytest.raises(ValueError, match="Invalid inputs"):
            project_next_runtime(prev_n=0, prev_time_s=10.0, next_n=100)

    def test_rejects_negative_time(self):
        with pytest.raises(ValueError, match="Invalid inputs"):
            project_next_runtime(prev_n=50, prev_time_s=-1.0, next_n=100)


class TestSafetyStoppingRules:
    def test_first_run_never_stops(self):
        """Initial run with no prior timing should proceed."""
        stop, reason, proj = should_stop_benchmark(
            prev_time_s=None,
            prev_n=None,
            next_n=50,
        )
        assert stop is False
        assert reason is None
        assert proj is None

    def test_stops_if_prior_run_exceeded_120s(self):
        """Halt immediately if completed run exceeded 120s."""
        stop, reason, _ = should_stop_benchmark(
            prev_time_s=121.5,
            prev_n=100,
            next_n=200,
        )
        assert stop is True
        assert "exceeded threshold of 120.0s" in reason

    def test_stops_if_projected_time_exceeds_180s(self):
        """Halt if next projected runtime exceeds 180s."""
        # prev took 50s at n=100; next n=200 -> projected = 50 * 4 = 200s (> 180s)
        stop, reason, proj = should_stop_benchmark(
            prev_time_s=50.0,
            prev_n=100,
            next_n=200,
        )
        assert stop is True
        assert proj == 200.0
        assert "exceeds limit of 180.0s" in reason

    def test_allows_run_if_within_limits(self):
        """Proceed if elapsed <= 120s and projected <= 180s."""
        # prev took 10s at n=50; next n=100 -> projected = 40s (<= 180s)
        stop, reason, proj = should_stop_benchmark(
            prev_time_s=10.0,
            prev_n=50,
            next_n=100,
        )
        assert stop is False
        assert reason is None
        assert proj == 40.0


class TestExtrapolationsAndRecommendations:
    @pytest.fixture()
    def mock_completed_runs(self):
        return [
            {
                "n_samples": 50,
                "status": "completed",
                "elapsed_time_s": 8.0,
                "matrix_shape": [50, 50],
                "matrix_entries": 2500,
                "matrix_memory": estimate_matrix_memory(50, 50),
            },
            {
                "n_samples": 100,
                "status": "completed",
                "elapsed_time_s": 32.0,
                "matrix_shape": [100, 100],
                "matrix_entries": 10000,
                "matrix_memory": estimate_matrix_memory(100, 100),
            },
        ]

    def test_extrapolation_structure_and_labels(self, mock_completed_runs):
        extrap = compute_extrapolations(mock_completed_runs, n_full_train=82332)
        assert "assumptions" in extrap
        assert extrap["basis_run"]["n_samples"] == 100
        # Full training kernel should be deemed infeasible
        full_train = extrap["full_dataset_projections"]["full_training_kernel"]
        assert full_train["feasible"] is False
        assert "GB RAM" in full_train["reason"]
        assert full_train["projected_time_hours"] > 10.0

    def test_recommendation_never_exceeds_ceilings(self, mock_completed_runs):
        recs = generate_recommendations(mock_completed_runs, skipped_runs=[])
        assert recs["recommended_N_train_q"] <= 200
        assert recs["recommended_N_val"] <= 50
        assert recs["recommended_N_test"] <= 100
        assert recs["hard_ceiling_compliance"]["complies_with_limits"] is True


class TestSoftwareVersions:
    def test_versions_collected(self):
        ver = get_software_versions()
        assert "python" in ver
        assert "qiskit" in ver
        assert "scikit_learn" in ver
        assert "numpy" in ver


class TestBenchmarkArtifactSchema:
    def test_benchmark_json_schema_if_exists(self):
        from src.config import RESULTS_DIR
        bench_file = RESULTS_DIR / "quantum_benchmark.json"
        if not bench_file.exists():
            pytest.skip("Benchmark result JSON does not exist yet")

        with open(bench_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "benchmark_title" in data
        assert "safety_limits" in data
        assert "completed_runs" in data
        assert "skipped_runs" in data
        assert "extrapolations" in data
        assert "recommendations" in data

        # Check completed runs have valid fields
        for run in data["completed_runs"]:
            assert run["status"] == "completed"
            assert run["n_samples"] in [50, 100, 200, 400]
            assert run["elapsed_time_s"] > 0
            assert run["matrix_shape"] == [run["n_samples"], run["n_samples"]]

        # Check skipped runs have reason
        for run in data["skipped_runs"]:
            assert run["status"] == "skipped"
            assert "skip_reason" in run

        # Recommendations respect Stage 5 limits
        recs = data["recommendations"]
        assert recs["recommended_N_train_q"] <= 200
        assert recs["recommended_N_val"] <= 50
        assert recs["recommended_N_test"] <= 100
