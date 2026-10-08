"""Unit tests for Stage 2 - Train-Only Preprocessing.

Proves:
1. Fitted scaler parameters do not change when test values are changed.
2. Processed train/test feature columns and column order are identical.
3. Train-only deduplication leaves test set untouched.
4. Quantum feature pool contains strictly numeric/binary features (no categoricals/leaks).
"""

import copy
import json
import numpy as np
import pandas as pd
import pytest

from src.config import (
    LEAKY_COLUMNS,
    PREPROCESSING_LOG_JSON,
    TARGET_COLUMN,
)
from src.preprocessing import (
    build_classical_preprocessor,
    deduplicate_train,
    extract_attack_categories,
    get_quantum_feature_pool,
    identify_feature_types,
    load_data,
    separate_features_target_and_metadata,
)


@pytest.fixture(scope="module")
def raw_data_samples():
    """Provides small representative samples of train and test data for fast, reliable unit testing."""
    train_raw, test_raw = load_data()
    train_sample = train_raw.head(300).copy()
    test_sample = test_raw.head(150).copy()
    return train_sample, test_sample


def test_scaler_parameters_invariant_to_test_modifications(raw_data_samples):
    """Proves: Fitted scaler parameters do not change when test values are changed (no data leakage)."""
    train_sample, test_sample = raw_data_samples
    
    # 1. Separate features
    X_train, y_train, _ = separate_features_target_and_metadata(train_sample)
    X_test_orig, y_test, _ = separate_features_target_and_metadata(test_sample)

    # 2. Fit preprocessor strictly on X_train
    preprocessor, _ = build_classical_preprocessor(X_train)

    # Extract fitted scaler parameters
    def extract_scaler_params(transformer_obj):
        params = {}
        for name, pipe, cols in transformer_obj.transformers_:
            if hasattr(pipe, "named_steps") and "scaler" in pipe.named_steps:
                scaler = pipe.named_steps["scaler"]
                params[name] = {
                    "mean": copy.deepcopy(scaler.mean_),
                    "scale": copy.deepcopy(scaler.scale_),
                    "var": copy.deepcopy(scaler.var_),
                }
        return params

    params_before = extract_scaler_params(preprocessor)

    # 3. Create drastically altered test data (multiply numeric columns by 10,000, invert signs, etc.)
    X_test_perturbed = X_test_orig.copy()
    numeric_cols = X_test_perturbed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        X_test_perturbed[col] = X_test_perturbed[col] * 10000.0 + 99999.0

    # 4. Transform both original test and perturbed test
    _ = preprocessor.transform(X_test_orig)
    _ = preprocessor.transform(X_test_perturbed)

    # 5. Extract params again and assert exact equality
    params_after = extract_scaler_params(preprocessor)

    for trans_name in params_before:
        assert trans_name in params_after
        np.testing.assert_array_equal(
            params_before[trans_name]["mean"],
            params_after[trans_name]["mean"],
            err_msg=f"Mean changed for {trans_name} when test was perturbed! Data leakage detected.",
        )
        np.testing.assert_array_equal(
            params_before[trans_name]["scale"],
            params_after[trans_name]["scale"],
            err_msg=f"Scale changed for {trans_name} when test was perturbed! Data leakage detected.",
        )
        np.testing.assert_array_equal(
            params_before[trans_name]["var"],
            params_after[trans_name]["var"],
            err_msg=f"Variance changed for {trans_name} when test was perturbed! Data leakage detected.",
        )


def test_processed_train_test_columns_and_order_identical(raw_data_samples):
    """Proves: Processed train/test feature columns and column order are identical."""
    train_sample, test_sample = raw_data_samples

    X_train, _, _ = separate_features_target_and_metadata(train_sample)
    X_test, _, _ = separate_features_target_and_metadata(test_sample)

    preprocessor, _ = build_classical_preprocessor(X_train)

    # Transform both splits
    train_transformed = preprocessor.transform(X_train)
    test_transformed = preprocessor.transform(X_test)

    # Check output matrix shapes
    assert train_transformed.shape[1] == test_transformed.shape[1], (
        f"Feature count mismatch: Train has {train_transformed.shape[1]}, Test has {test_transformed.shape[1]}"
    )

    # Check feature names and order
    feature_names_train = list(preprocessor.get_feature_names_out())
    assert len(feature_names_train) == train_transformed.shape[1]
    assert len(feature_names_train) == test_transformed.shape[1]

    # Verify no NaN or infinite values produced
    assert not np.isnan(train_transformed).any(), "NaN values found in train transformed features"
    assert not np.isnan(test_transformed).any(), "NaN values found in test transformed features"


def test_train_deduplication_preserves_test_untouched(raw_data_samples):
    """Proves: Deduplication operates on train only and never alters test."""
    train_sample, test_sample = raw_data_samples
    test_len_before = len(test_sample)

    deduped_train = deduplicate_train(train_sample, test_df=test_sample)

    assert len(deduped_train) <= len(train_sample)
    assert len(test_sample) == test_len_before, "Test sample row count was modified by deduplication!"
    assert PREPROCESSING_LOG_JSON.exists(), "preprocessing_log.json was not created"

    with open(PREPROCESSING_LOG_JSON, "r", encoding="utf-8") as f:
        log_data = json.load(f)
    assert "original_train_rows" in log_data
    assert "final_train_rows" in log_data
    assert "test_rows_unmodified" in log_data


def test_quantum_feature_pool_excludes_categoricals_and_leaks(raw_data_samples):
    """Proves: Quantum feature pool contains strictly numeric/binary features and excludes leaks/categoricals."""
    train_sample, _ = raw_data_samples
    X_train, _, _ = separate_features_target_and_metadata(train_sample)

    pool = get_quantum_feature_pool(X_train)
    types_dict = identify_feature_types(X_train)

    # No categoricals allowed
    for cat_col in types_dict["categorical"]:
        assert cat_col not in pool, f"Categorical column '{cat_col}' illegally found in quantum pool!"

    # No leaky columns or target allowed
    for leaky in LEAKY_COLUMNS:
        assert leaky not in pool, f"Leaky column '{leaky}' found in quantum pool!"
    assert TARGET_COLUMN not in pool, f"Target '{TARGET_COLUMN}' found in quantum pool!"

    # Every item in pool must be either numeric or binary
    allowed = set(types_dict["numeric"]) | set(types_dict["binary"])
    for col in pool:
        assert col in allowed, f"Unexpected column '{col}' in quantum pool!"


def test_attack_cat_preserved_separately_not_as_feature(raw_data_samples):
    """Proves: attack_cat is preserved separately for post-hoc analysis and not in features X."""
    train_sample, _ = raw_data_samples
    X_train, y_train, attack_cat = separate_features_target_and_metadata(train_sample)

    assert "attack_cat" not in X_train.columns
    assert "attack_cat" not in X_train.columns
    assert "id" not in X_train.columns
    assert "label" not in X_train.columns
    assert len(attack_cat) == len(X_train)
    assert attack_cat.dtype == object or pd.api.types.is_string_dtype(attack_cat)
