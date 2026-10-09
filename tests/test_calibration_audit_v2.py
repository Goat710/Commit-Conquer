"""Unit tests for Stage 7b Step 7 calibration audit."""

import numpy as np
import pytest

from src.calibration_audit_v2 import compute_ece


def test_compute_ece_perfect_calibration():
    # Construct synthetic data where in bin [0, 0.5], acc=conf=0.25; in [0.5, 1], acc=conf=0.75
    y_prob = np.array([0.25] * 100 + [0.75] * 100)
    y_true = np.array([0] * 75 + [1] * 25 + [0] * 25 + [1] * 75)

    ece, bin_details = compute_ece(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-3)
    assert len(bin_details) == 10


def test_compute_ece_poor_calibration():
    # All predictions 0.9, but true labels are all 0
    y_prob = np.array([0.9] * 100)
    y_true = np.array([0] * 100)

    ece, bin_details = compute_ece(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.9, abs=1e-3)
