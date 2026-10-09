"""Tests for Stage 7b dataset partitioning and discipline — Step 2."""

import numpy as np
import pytest

from src.dataset_v2 import (
    CALIBRATION_SIZE,
    DEFAULT_SEEDS,
    FIXED_TEST_SIZE,
    TUNE_SIZE,
    create_disjoint_partitions,
    load_raw_datasets,
)

FEATURES = ["sbytes", "sload", "sttl", "smean"]


@pytest.fixture(scope="module")
def raw_data():
    return load_raw_datasets()


def test_disjoint_partitions_indices(raw_data):
    train_df, test_df = raw_data
    bundle = create_disjoint_partitions(
        train_df=train_df,
        test_df=test_df,
        features=FEATURES,
        n_train=500,
        seed=42,
        n_tune=TUNE_SIZE,
        n_cal=CALIBRATION_SIZE,
        n_test=FIXED_TEST_SIZE,
    )

    # Check shapes
    assert bundle.X_train.shape == (500, 4)
    assert bundle.y_train.shape == (500,)
    assert bundle.X_tune.shape == (TUNE_SIZE, 4)
    assert bundle.y_tune.shape == (TUNE_SIZE,)
    assert bundle.X_cal.shape == (CALIBRATION_SIZE, 4)
    assert bundle.y_cal.shape == (CALIBRATION_SIZE,)
    assert bundle.X_test.shape == (FIXED_TEST_SIZE, 4)
    assert bundle.y_test.shape == (FIXED_TEST_SIZE,)

    # Check disjointness within training dataframe
    s_train = set(bundle.train_indices)
    s_tune = set(bundle.tune_indices)
    s_cal = set(bundle.cal_indices)

    assert len(s_train.intersection(s_tune)) == 0, "Train and Tune overlap!"
    assert len(s_train.intersection(s_cal)) == 0, "Train and Cal overlap!"
    assert len(s_tune.intersection(s_cal)) == 0, "Tune and Cal overlap!"


def test_stratification_preservation(raw_data):
    train_df, test_df = raw_data
    bundle = create_disjoint_partitions(
        train_df=train_df,
        test_df=test_df,
        features=FEATURES,
        n_train=1000,
        seed=100,
    )

    # Training set prevalence is ~0.5506
    train_prev = np.mean(train_df["label"].values)
    assert abs(np.mean(bundle.y_train) - train_prev) < 0.02
    assert abs(np.mean(bundle.y_tune) - train_prev) < 0.02
    assert abs(np.mean(bundle.y_cal) - train_prev) < 0.02

    # Test set prevalence is ~0.6806
    test_prev = np.mean(test_df["label"].values)
    assert abs(np.mean(bundle.y_test) - test_prev) < 0.01


def test_seed_reproducibility(raw_data):
    train_df, test_df = raw_data
    bundle_a = create_disjoint_partitions(train_df, test_df, FEATURES, n_train=100, seed=42)
    bundle_b = create_disjoint_partitions(train_df, test_df, FEATURES, n_train=100, seed=42)
    bundle_c = create_disjoint_partitions(train_df, test_df, FEATURES, n_train=100, seed=100)

    np.testing.assert_array_equal(bundle_a.train_indices, bundle_b.train_indices)
    np.testing.assert_array_equal(bundle_a.test_indices, bundle_b.test_indices)
    assert not np.array_equal(bundle_a.train_indices, bundle_c.train_indices)
