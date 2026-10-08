"""Quantum CyberShield - Preprocessing Module.

Strict Train-Only Preprocessing Pipeline:
- Data loading with target extraction and post-hoc attack_cat preservation
- Train-only deduplication (test set remains strictly untouched)
- Programmatic feature typing (numeric, binary, categorical)
- Classical ColumnTransformer reference pipeline (train skewness log1p + StandardScaler + OneHotEncoder)
- Quantum feature pool extractor (strictly continuous numeric + binary)
- Stratified validation splitting from train only
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    DROP_COLUMNS,
    LEAKY_COLUMNS,
    PREPROCESSING_LOG_JSON,
    RANDOM_SEED,
    RAW_TEST_CSV,
    RAW_TRAIN_CSV,
    RESULTS_DIR,
    SKEW_THRESHOLD,
    TARGET_COLUMN,
)


def load_data(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train and test DataFrames from raw storage.
    
    Treats raw files as immutable.
    """
    train_file = train_path or RAW_TRAIN_CSV
    test_file = test_path or RAW_TEST_CSV

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Raw train file not found at: {train_file}")
    if not Path(test_file).exists():
        raise FileNotFoundError(f"Raw test file not found at: {test_file}")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df


def extract_attack_categories(df: pd.DataFrame) -> pd.Series:
    """Extracts attack_cat series for post-hoc analysis.
    
    This metadata must NEVER be used as a feature in model training.
    """
    if "attack_cat" in df.columns:
        return df["attack_cat"].fillna("Normal").astype(str).str.strip().copy()
    return pd.Series(["Unknown"] * len(df), index=df.index, name="attack_cat")


def separate_features_target_and_metadata(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separates feature matrix X, binary target y, and metadata attack_cat.
    
    Excludes all configured LEAKY_COLUMNS ('id', 'attack_cat') and TARGET_COLUMN ('label').
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    # Target: 0 = benign, 1 = attack
    y = df[TARGET_COLUMN].astype(int).copy()

    # Post-hoc metadata
    attack_cat = extract_attack_categories(df)

    # Exclude configured leaky columns and target
    cols_to_exclude = set(LEAKY_COLUMNS) | {TARGET_COLUMN}
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    X = df[feature_cols].copy()

    return X, y, attack_cat


def deduplicate_train(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    log_path: Optional[Path] = PREPROCESSING_LOG_JSON,
) -> pd.DataFrame:
    """Deduplicates TRAIN ONLY using all feature columns except identifier 'id'.
    
    Never deduplicates or modifies the test set.
    Logs original train rows, removed rows, and final train rows to JSON.
    """
    original_train_rows = len(train_df)
    
    # Feature columns for duplicate check (exclude 'id' identifier)
    subset_cols = [c for c in train_df.columns if c != "id"]
    deduped_train_df = train_df.drop_duplicates(subset=subset_cols).copy()
    final_train_rows = len(deduped_train_df)
    removed_rows = original_train_rows - final_train_rows

    test_rows_count = len(test_df) if test_df is not None else None

    log_data: Dict[str, Any] = {
        "original_train_rows": original_train_rows,
        "removed_duplicate_rows": removed_rows,
        "final_train_rows": final_train_rows,
        "duplicate_removal_percentage": round(removed_rows / original_train_rows * 100, 2),
        "deduplication_subset_columns_count": len(subset_cols),
        "test_rows_unmodified": test_rows_count,
        "note": "Deduplication applied strictly to TRAIN partition. Test set left untouched.",
    }

    if log_path:
        log_path_obj = Path(log_path)
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path_obj, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

    return deduped_train_df


def identify_feature_types(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Programmatically identifies numeric, binary, and categorical columns.
    
    Does not hard-code feature names except explicitly configured exclusions.
    """
    excluded = set(LEAKY_COLUMNS) | {TARGET_COLUMN}
    valid_cols = [c for c in X.columns if c not in excluded]

    categorical_cols: List[str] = []
    binary_cols: List[str] = []
    numeric_cols: List[str] = []

    for col in valid_cols:
        series = X[col]
        # Check string / object / categorical types
        if (
            pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        ):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            # Programmatically check if values are strictly binary subset {0, 1}
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({0, 1}):
                binary_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            # Fallback for unhandled types
            categorical_cols.append(col)

    return {
        "categorical": sorted(categorical_cols),
        "binary": sorted(binary_cols),
        "numeric": sorted(numeric_cols),
    }


def get_quantum_feature_pool(X: pd.DataFrame) -> List[str]:
    """Returns the quantum feature candidate pool: strictly numeric + binary features.
    
    Excludes:
    - categorical columns
    - 'id'
    - 'attack_cat'
    - target 'label'
    - all configured leaky columns
    Does not one-hot encode categoricals for the quantum pool.
    """
    feature_types = identify_feature_types(X)
    quantum_pool = sorted(feature_types["numeric"] + feature_types["binary"])
    return quantum_pool


def build_classical_preprocessor(
    X_train: pd.DataFrame,
    skew_threshold: float = SKEW_THRESHOLD,
) -> Tuple[ColumnTransformer, Dict[str, Any]]:
    """Builds and fits an sklearn ColumnTransformer on TRAIN ONLY.
    
    Numeric features:
    - Identifies heavy-tailed, non-negative numeric columns using TRAIN skewness (skew > skew_threshold, min >= 0).
    - Applies log1p on those columns, followed by StandardScaler.
    - Standardizes remaining numeric and binary columns with StandardScaler.
    
    Categorical features:
    - Applies OneHotEncoder(handle_unknown='ignore', sparse_output=False).
    
    Returns the fitted ColumnTransformer and a metadata dictionary detailing the transformations.
    """
    feature_types = identify_feature_types(X_train)
    numeric_candidates = feature_types["numeric"]
    binary_cols = feature_types["binary"]
    categorical_cols = feature_types["categorical"]

    # Calculate skewness strictly on train partition
    train_numeric_data = X_train[numeric_candidates]
    skewness_series = train_numeric_data.skew()
    min_series = train_numeric_data.min()

    heavy_tailed_cols: List[str] = []
    standard_numeric_cols: List[str] = []

    for col in numeric_candidates:
        skew_val = float(skewness_series[col])
        min_val = float(min_series[col])
        # Heavy-tailed criteria: skewness > threshold AND non-negative values (safe for log1p)
        if skew_val > skew_threshold and min_val >= 0:
            heavy_tailed_cols.append(col)
        else:
            standard_numeric_cols.append(col)

    heavy_tailed_cols.sort()
    standard_numeric_cols.sort()
    standard_and_binary = sorted(standard_numeric_cols + binary_cols)

    # 1. Pipeline for heavy-tailed numeric: log1p -> StandardScaler
    heavy_tail_pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    # 2. Pipeline for standard numeric & binary: StandardScaler
    standard_num_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    # 3. Pipeline for categorical: OneHotEncoder
    categorical_pipeline = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []
    if heavy_tailed_cols:
        transformers.append(("heavy_tail_num", heavy_tail_pipeline, heavy_tailed_cols))
    if standard_and_binary:
        transformers.append(("standard_num", standard_num_pipeline, standard_and_binary))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
    )

    # Fit strictly on train
    preprocessor.fit(X_train)

    metadata: Dict[str, Any] = {
        "skew_threshold_used": skew_threshold,
        "heavy_tailed_numeric_features": heavy_tailed_cols,
        "standard_numeric_features": standard_numeric_cols,
        "binary_features": binary_cols,
        "categorical_features": categorical_cols,
        "fitted_feature_names_out": list(preprocessor.get_feature_names_out()),
        "total_transformed_features": len(preprocessor.get_feature_names_out()),
    }

    return preprocessor, metadata


def create_stratified_validation_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Creates a stratified validation split from TRAIN only.
    
    Test data remains completely untouched.
    """
    return train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )


def run_preprocessing_pipeline() -> Dict[str, Any]:
    """Executes the train-only preprocessing pipeline and logs summary."""
    print("=" * 60)
    print("STAGE 2: EXECUTING TRAIN-ONLY PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load data
    train_raw, test_raw = load_data()
    print(f"Loaded Raw Train: {train_raw.shape}, Raw Test: {test_raw.shape}")

    # 2. Train-only deduplication
    train_deduped = deduplicate_train(train_raw, test_df=test_raw)
    print(f"Train Deduplicated: {len(train_deduped):,} rows remaining")

    # 3. Separate features, target, and metadata
    X_train, y_train, attack_cat_train = separate_features_target_and_metadata(train_deduped)
    X_test, y_test, attack_cat_test = separate_features_target_and_metadata(test_raw)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # 4. Feature typing
    types_dict = identify_feature_types(X_train)
    print(f"Identified Features: {len(types_dict['numeric'])} numeric, "
          f"{len(types_dict['binary'])} binary, {len(types_dict['categorical'])} categorical")

    # 5. Quantum feature pool
    q_pool = get_quantum_feature_pool(X_train)
    print(f"Quantum Candidate Feature Pool: {len(q_pool)} features (strictly continuous + binary)")

    # 6. Fit classical ColumnTransformer on TRAIN ONLY
    preprocessor, meta = build_classical_preprocessor(X_train)
    print(f"Classical ColumnTransformer fitted on Train: "
          f"{len(meta['heavy_tailed_numeric_features'])} heavy-tail (log1p+scaler), "
          f"{len(meta['standard_numeric_features']) + len(meta['binary_features'])} standard (scaler), "
          f"{len(meta['categorical_features'])} categorical (OHE)")
    print(f"Total Transformed Features: {meta['total_transformed_features']}")

    # 7. Stratified validation split from TRAIN only
    X_tr_sub, X_val, y_tr_sub, y_val = create_stratified_validation_split(X_train, y_train)
    print(f"Train Subsplit: {X_tr_sub.shape}, Validation Split: {X_val.shape}")

    print("=" * 60)
    return {
        "metadata": meta,
        "feature_types": types_dict,
        "quantum_pool_size": len(q_pool),
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
