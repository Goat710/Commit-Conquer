"""Stage 7b dataset partitioning and discipline utilities.

Ensures rigorous experimental separation across partitions:
  - Training Partition (N_train in [100, 500, 1000, 2000, 4000]): Model fitting
  - Tuning Partition (N_tune = 1000): Hyperparameter validation (strictly train-derived)
  - Calibration Partition (N_cal = 2000): Platt sigmoid calibration (strictly train-derived)
  - Evaluation Partition (N_test = 20000): Fixed stratified held-out test subset (from official test CSV)
  - Disjointness invariant: Train, Tune, Cal partitions have empty set intersections
  - Preprocessing discipline: Scalers are fitted ONLY on the model training partition
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_SEED,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
)
from src.quantum_model import (
    fit_quantum_scaler,
    transform_quantum_features,
)

logger = logging.getLogger(__name__)

PRIMARY_FEATURES = ["sbytes", "sload", "sttl", "smean"]
DEFAULT_SEEDS = [42, 100, 2024, 777, 999]
TRAIN_SIZES = [100, 500, 1000, 2000, 4000]
FIXED_TEST_SIZE = 20000
TUNE_SIZE = 1000
CALIBRATION_SIZE = 2000


@dataclass
class PartitionBundle:
    """Container holding strictly disjoint feature arrays and labels."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_tune: np.ndarray
    y_tune: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    features: List[str]
    train_indices: np.ndarray
    tune_indices: np.ndarray
    cal_indices: np.ndarray
    test_indices: np.ndarray
    seed: int


def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load official training and test datasets."""
    train_df = pd.read_csv(RAW_TRAIN_CSV)
    test_df = pd.read_csv(RAW_TEST_CSV)
    return train_df, test_df


def get_fixed_test_subset(
    test_df: pd.DataFrame,
    features: List[str],
    n_test: int = FIXED_TEST_SIZE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a deterministic, stratified held-out test subset from RAW_TEST_CSV.

    Fixed across all model comparisons and training sizes.
    """
    y_full = test_df["label"].values.astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_test, random_state=seed)
    test_idx, _ = next(sss.split(test_df, y_full))
    X_test = test_df[features].iloc[test_idx].values.astype(np.float64)
    y_test = y_full[test_idx]
    return X_test, y_test, test_idx


def create_disjoint_partitions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    n_train: int,
    seed: int,
    n_tune: int = TUNE_SIZE,
    n_cal: int = CALIBRATION_SIZE,
    n_test: int = FIXED_TEST_SIZE,
    test_seed: int = 42,
) -> PartitionBundle:
    """Generate strictly disjoint stratified partitions for Stage 7b experiments.

    Parameters
    ----------
    train_df:
        Official UNSW-NB15 training dataframe (82,332 rows).
    test_df:
        Official UNSW-NB15 test dataframe (175,341 rows).
    features:
        List of feature column names (e.g. ['sbytes', 'sload', 'sttl', 'smean']).
    n_train:
        Number of training samples for the model (e.g. 100, 500, 1000, 2000, 4000).
    seed:
        Random seed for sampling training, tuning, and calibration partitions.
    n_tune:
        Number of samples reserved for hyperparameter tuning (default 1000).
    n_cal:
        Number of samples reserved for probability calibration (default 2000).
    n_test:
        Number of test samples (default 20,000 from RAW_TEST_CSV).
    test_seed:
        Fixed seed for held-out test selection (default 42).

    Returns
    -------
    PartitionBundle:
        Structured object containing all partitions and verified disjoint index sets.
    """
    y_train_full = train_df["label"].values.astype(int)

    # 1. Total pool needed from training dataframe
    total_train_needed = n_train + n_tune + n_cal
    if total_train_needed > len(train_df):
        raise ValueError(
            f"Requested {total_train_needed} samples from training set of size {len(train_df)}."
        )

    # Initial stratified split from full training set
    sss_pool = StratifiedShuffleSplit(n_splits=1, train_size=total_train_needed, random_state=seed)
    pool_idx, _ = next(sss_pool.split(train_df, y_train_full))
    pool_y = y_train_full[pool_idx]

    # Split calibration partition out of the pool
    sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=n_cal, random_state=seed)
    rem_idx_rel, cal_idx_rel = next(sss_cal.split(pool_idx, pool_y))
    cal_idx = pool_idx[cal_idx_rel]
    rem_idx = pool_idx[rem_idx_rel]
    rem_y = y_train_full[rem_idx]

    # Split tuning partition out of remainder
    sss_tune = StratifiedShuffleSplit(n_splits=1, test_size=n_tune, random_state=seed)
    train_idx_rel, tune_idx_rel = next(sss_tune.split(rem_idx, rem_y))
    tune_idx = rem_idx[tune_idx_rel]
    train_idx = rem_idx[train_idx_rel]

    # Verify disjointness
    set_train = set(train_idx)
    set_tune = set(tune_idx)
    set_cal = set(cal_idx)
    if set_train.intersection(set_tune) or set_train.intersection(set_cal) or set_tune.intersection(set_cal):
        raise RuntimeError("Partitioning failure: Training partitions overlap!")

    # Extract arrays
    X_train = train_df[features].iloc[train_idx].values.astype(np.float64)
    y_train = y_train_full[train_idx]

    X_tune = train_df[features].iloc[tune_idx].values.astype(np.float64)
    y_tune = y_train_full[tune_idx]

    X_cal = train_df[features].iloc[cal_idx].values.astype(np.float64)
    y_cal = y_train_full[cal_idx]

    # Extract fixed test subset
    X_test, y_test, test_idx = get_fixed_test_subset(
        test_df=test_df,
        features=features,
        n_test=n_test,
        seed=test_seed,
    )

    return PartitionBundle(
        X_train=X_train,
        y_train=y_train,
        X_tune=X_tune,
        y_tune=y_tune,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        features=features,
        train_indices=train_idx,
        tune_indices=tune_idx,
        cal_indices=cal_idx,
        test_indices=test_idx,
        seed=seed,
    )
