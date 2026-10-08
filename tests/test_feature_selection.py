"""Unit tests for Stage 3 - Train-Only Feature Selection.

Verifies:
1. Feature exclusion happens strictly before selection and excluded features are never selected.
2. Selection pipeline is 100% deterministic with seed 42.
3. Train-only behavior: test data is never loaded, inspected, or used.
4. Correctness of k-feature selection (k=2, k=4, etc.).
5. Near-constant filtering and correlation pruning (|r| > 0.9 retains higher MI).
6. Output JSON schema and metadata compliance.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    N_QUBIT_FEATURES,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.feature_selection import (
    FEATURE_SELECTION_JSON,
    MI_SCORES_PLOT_PATH,
    STABILITY_PLOT_PATH,
    TTL_FEATURES_TO_EXCLUDE,
    compute_mutual_information,
    load_train_only_pool,
    remove_highly_correlated_features,
    remove_near_constant_features,
    run_feature_selection_pipeline,
    run_single_selection,
    run_stability_selection,
    select_mrmr_features,
)
from src.preprocessing import load_data, separate_features_target_and_metadata, deduplicate_train, get_quantum_feature_pool


@pytest.fixture(scope="module")
def train_sample_data():
    """Provides a deterministic sample of train data for fast, reliable unit testing."""
    train_raw, _ = load_data()
    train_dedup = deduplicate_train(train_raw)
    X_train, y_train, _ = separate_features_target_and_metadata(train_dedup)
    pool = get_quantum_feature_pool(X_train)
    # 500 rows is fast for MI and rich enough for correlation testing
    X_sample = X_train[pool].iloc[:500].copy()
    y_sample = y_train.iloc[:500].copy()
    return X_sample, y_sample, pool


def test_feature_exclusion_happens_before_selection(train_sample_data):
    """Proves: exclude_features drops specified columns before any scoring,

    and excluded features NEVER appear in MI rankings or selected subsets.
    """
    X_sample, y_sample, pool = train_sample_data
    exclude_list = ["sttl", "dttl", "ct_state_ttl"]

    res = run_single_selection(
        X=X_sample,
        y=y_sample,
        candidate_pool=pool,
        exclude_features=exclude_list,
        k=N_QUBIT_FEATURES,
        random_state=RANDOM_SEED,
    )

    # 1. None of the excluded features appear in selected_features
    for excl in exclude_list:
        assert excl not in res["selected_features"], f"Excluded feature {excl} was selected!"
        assert excl not in res["mi_scores"], f"Excluded feature {excl} was scored in MI!"
        assert excl not in res["surviving_features"], f"Excluded feature {excl} survived selection!"

    # 2. Active pool size must be exactly original pool minus exclusions present
    assert res["active_pool_size"] == len(pool) - len(exclude_list)


def test_selection_determinism(train_sample_data):
    """Proves: Running feature selection with fixed RANDOM_SEED=42 is 100% deterministic."""
    X_sample, y_sample, pool = train_sample_data

    res_run1 = run_single_selection(
        X=X_sample,
        y=y_sample,
        candidate_pool=pool,
        exclude_features=None,
        k=4,
        random_state=RANDOM_SEED,
    )
    res_run2 = run_single_selection(
        X=X_sample,
        y=y_sample,
        candidate_pool=pool,
        exclude_features=None,
        k=4,
        random_state=RANDOM_SEED,
    )

    assert res_run1["selected_features"] == res_run2["selected_features"]
    assert res_run1["surviving_features"] == res_run2["surviving_features"]
    for col, score1 in res_run1["mi_scores"].items():
        assert col in res_run2["mi_scores"]
        assert score1 == pytest.approx(res_run2["mi_scores"][col], abs=1e-6)


def test_train_only_behavior_never_accesses_test(tmp_path):
    """Proves: load_train_only_pool strictly loads TRAIN data, without touching test files."""
    # Create a dummy train file
    df_train = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "label": [0, 1, 0, 1],
        "attack_cat": ["Normal", "Generic", "Normal", "Exploits"],
        "num1": [10.0, 20.0, 30.0, 40.0],
        "num2": [1.0, 2.0, 3.0, 4.0],
    })
    train_file = tmp_path / "dummy_train.csv"
    df_train.to_csv(train_file, index=False)

    # Notice: No test file exists in tmp_path at all
    X_pool, y_train, pool = load_train_only_pool(train_file)
    assert len(X_pool) == 4
    assert len(y_train) == 4
    assert set(pool) == {"num1", "num2"}


def test_k_feature_selection_cardinality(train_sample_data):
    """Proves: mRMR selects exactly k features for different values of k."""
    X_sample, y_sample, pool = train_sample_data

    for test_k in [2, 3, 4, 6]:
        res = run_single_selection(
            X=X_sample,
            y=y_sample,
            candidate_pool=pool,
            exclude_features=None,
            k=test_k,
            random_state=RANDOM_SEED,
        )
        assert len(res["selected_features"]) == test_k
        assert len(set(res["selected_features"])) == test_k  # all unique


def test_near_constant_and_correlation_pruning():
    """Proves:

    1. Features with zero/near-zero variance are removed.
    2. For pairs with |Spearman r| > 0.9, the one with lower MI is dropped.
    """
    np.random.seed(42)
    n = 200
    y = pd.Series(np.random.binomial(1, 0.5, size=n))

    # Feature 1: correlated with target (higher MI)
    f_high_mi = y * 2.0 + np.random.normal(0, 0.2, size=n)
    # Feature 2: collinear with Feature 1 (|r| ~ 1.0) but with more noise relative to target
    f_collinear = f_high_mi + np.random.normal(0, 0.01, size=n)
    # Feature 3: zero variance
    f_constant = np.zeros(n)
    # Feature 4: independent informative feature
    f_indep = y * 1.5 + np.random.normal(0, 0.5, size=n)

    df = pd.DataFrame({
        "f_high_mi": f_high_mi,
        "f_collinear": f_collinear,
        "f_constant": f_constant,
        "f_indep": f_indep,
    })

    # 1. Near constant removal
    retained, dropped = remove_near_constant_features(df, threshold=1e-5)
    assert "f_constant" in dropped
    assert "f_constant" not in retained

    # 2. Compute MI
    mi = compute_mutual_information(df, y, features=retained, random_state=42)
    assert mi["f_high_mi"] > 0
    assert mi["f_indep"] > 0

    # 3. Correlation pruning
    surviving, dropped_info = remove_highly_correlated_features(df, mi, correlation_threshold=0.9)
    # One of f_high_mi or f_collinear must be dropped, and the survivor must have higher or equal MI
    assert ("f_high_mi" in surviving) ^ ("f_collinear" in surviving)
    dropped_feat = "f_collinear" if "f_high_mi" in surviving else "f_high_mi"
    kept_feat = "f_high_mi" if "f_high_mi" in surviving else "f_collinear"
    assert mi[kept_feat] >= mi[dropped_feat]
    assert dropped_feat in dropped_info


def test_output_json_and_plots_generation(tmp_path):
    """Proves: feature selection pipeline produces valid JSON and PNG outputs matching schema."""
    np.random.seed(42)
    n = 100
    y = pd.Series(np.random.binomial(1, 0.5, size=n), name="label")
    df = pd.DataFrame({
        "id": range(n),
        "label": y,
        "attack_cat": ["Normal"] * n,
        "feat_a": np.random.normal(0, 1, n),
        "feat_b": np.random.normal(0, 1, n),
        "feat_c": np.random.normal(0, 1, n),
        "feat_d": np.random.normal(0, 1, n),
        "sttl": np.random.normal(0, 1, n),
        "dttl": np.random.normal(0, 1, n),
        "ct_state_ttl": np.random.normal(0, 1, n),
    })

    train_path = tmp_path / "dummy_train.csv"
    json_path = tmp_path / "feature_selection.json"
    mi_plot_path = tmp_path / "mi_plot.png"
    stab_plot_path = tmp_path / "stab_plot.png"
    df.to_csv(train_path, index=False)

    res = run_feature_selection_pipeline(
        train_path=train_path,
        k=2,
        random_state=42,
        output_json_path=json_path,
        output_mi_plot_path=mi_plot_path,
        output_stability_plot_path=stab_plot_path,
    )

    # 1. Files must exist
    assert json_path.exists()
    assert mi_plot_path.exists()
    assert stab_plot_path.exists()

    # 2. JSON must contain both runs and metadata
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "metadata" in loaded
    assert loaded["metadata"]["random_seed"] == 42
    assert loaded["metadata"]["k_qubit_features"] == 2
    assert "runs" in loaded
    assert "all_features" in loaded["runs"]
    assert "TTL_excluded" in loaded["runs"]

    all_run = loaded["runs"]["all_features"]
    ttl_run = loaded["runs"]["TTL_excluded"]

    assert len(all_run["selected_features"]) == 2
    assert len(ttl_run["selected_features"]) == 2
    for excl in TTL_FEATURES_TO_EXCLUDE:
        assert excl not in ttl_run["selected_features"]
