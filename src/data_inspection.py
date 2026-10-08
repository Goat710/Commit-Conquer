"""Dataset Inspection Script for UNSW-NB15.
Analyzes raw train and test splits, verifies schema, missing values,
class distributions, categorical features, and leakage columns.
Saves detailed inspection artifact to results/data_report.json.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np

from src.config import (
    RAW_TRAIN_CSV,
    RAW_TEST_CSV,
    DATA_REPORT_JSON,
    RESULTS_DIR,
    TARGET_COLUMN,
    DROP_COLUMNS,
)


def inspect_split(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Inspects a single split of the dataset."""
    n_rows, n_cols = df.shape
    
    # Missing values
    missing_counts = df.isnull().sum()
    cols_with_missing = {
        col: int(count) for col, count in missing_counts.items() if count > 0
    }
    
    # Duplicates (excluding 'id' if present)
    df_no_id = df.drop(columns=["id"], errors="ignore")
    n_duplicates = int(df_no_id.duplicated().sum())

    # Data types and categorization
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Class distribution (target)
    target_dist: Dict[str, Any] = {}
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts().to_dict()
        total = len(df)
        target_dist = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): round(float(v) / total * 100, 2) for k, v in counts.items()},
            "benign_count": int(counts.get(0, 0)),
            "attack_count": int(counts.get(1, 0)),
            "attack_ratio": round(float(counts.get(1, 0)) / total, 4),
        }

    # Attack category distribution (if present)
    attack_cat_dist: Dict[str, int] = {}
    if "attack_cat" in df.columns:
        attack_cat_dist = {
            str(k).strip(): int(v)
            for k, v in df["attack_cat"].fillna("Normal").value_counts().to_dict().items()
        }

    # Unique values for categorical features
    cat_cardinality = {col: int(df[col].nunique()) for col in cat_cols}

    return {
        "split_name": split_name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "categorical_columns": cat_cols,
        "numerical_columns": num_cols,
        "categorical_cardinality": cat_cardinality,
        "missing_columns": cols_with_missing,
        "total_missing_values": int(missing_counts.sum()),
        "duplicate_rows_excluding_id": n_duplicates,
        "target_distribution": target_dist,
        "attack_category_distribution": attack_cat_dist,
    }


def run_inspection() -> Dict[str, Any]:
    print("=" * 60)
    print("UNSW-NB15 RAW DATASET DETAILED INSPECTION")
    print("=" * 60)

    if not RAW_TRAIN_CSV.exists() or not RAW_TEST_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV files missing: {RAW_TRAIN_CSV} or {RAW_TEST_CSV}"
        )

    print(f"Loading {RAW_TRAIN_CSV.name}...")
    train_df = pd.read_csv(RAW_TRAIN_CSV)
    print(f"Loading {RAW_TEST_CSV.name}...")
    test_df = pd.read_csv(RAW_TEST_CSV)

    train_report = inspect_split(train_df, "official_train")
    test_report = inspect_split(test_df, "official_test")

    # Column consistency check
    cols_match = train_df.columns.tolist() == test_df.columns.tolist()
    
    # Shared / Leakage columns
    potential_leakage = [col for col in DROP_COLUMNS if col in train_df.columns]

    summary = {
        "train_file": str(RAW_TRAIN_CSV.name),
        "test_file": str(RAW_TEST_CSV.name),
        "train_shape": [train_report["n_rows"], train_report["n_cols"]],
        "test_shape": [test_report["n_rows"], test_report["n_cols"]],
        "columns_identical_across_splits": cols_match,
        "total_columns": len(train_df.columns),
        "target_column": TARGET_COLUMN,
        "leakage_identifier_columns_to_drop": potential_leakage,
        "categorical_features": train_report["categorical_columns"],
        "train": train_report,
        "test": test_report,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = DATA_REPORT_JSON
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Inspection report saved to: {report_file}")
    print("\n--- Summary ---")
    print(f"Train Shape: {train_report['n_rows']} rows x {train_report['n_cols']} columns")
    print(f"Test Shape:  {test_report['n_rows']} rows x {test_report['n_cols']} columns")
    print(f"Target Column: '{TARGET_COLUMN}'")
    print(f"Train Class Distribution: {train_report['target_distribution']['percentages']}")
    print(f"Test Class Distribution:  {test_report['target_distribution']['percentages']}")
    print(f"Categorical Features: {train_report['categorical_columns']}")
    print(f"Excluded Columns: {potential_leakage}")
    print(f"Missing Values: Train={train_report['total_missing_values']}, Test={test_report['total_missing_values']}")
    print(f"Duplicate Rows (excl id): Train={train_report['duplicate_rows_excluding_id']}, Test={test_report['duplicate_rows_excluding_id']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_inspection()
