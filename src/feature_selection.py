"""Stage 3 - Train-Only Feature Selection for Quantum Feature Map.

Selects the quantum feature subset from the quantum-eligible numeric/binary pool
using TRAIN data strictly. Model-agnostic: no classifiers (SVM, RF, QSVC) are used.

Pipeline:
1. Remove near-constant features (variance <= threshold or nunique <= 1).
2. Compute |Spearman| correlations. For pairs with |r| > 0.9, keep the feature
   with higher TRAIN-only mutual information (MI) with label.
3. Rank remaining features by TRAIN-only MI with label.
4. Greedily select k=N_QUBIT_FEATURES using an mRMR criterion (high MI relevance +
   low redundancy with already-selected features).
5. Stability selection: repeat steps 1-4 across 5 stratified TRAIN-only folds and
   record feature selection frequencies.

Runs:
- "all_features": Full quantum-eligible numeric/binary candidate pool.
- "TTL_excluded": Excludes 'sttl', 'dttl', 'ct_state_ttl' before selection.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold

from src.config import (
    N_QUBIT_FEATURES,
    RANDOM_SEED,
    RAW_TRAIN_CSV,
    RESULTS_DIR,
    TARGET_COLUMN,
)
from src.preprocessing import (
    deduplicate_train,
    get_quantum_feature_pool,
    separate_features_target_and_metadata,
)

# Output artifact paths
FEATURE_SELECTION_JSON: Path = RESULTS_DIR / "feature_selection.json"
MI_SCORES_PLOT_PATH: Path = RESULTS_DIR / "feature_selection_mi_scores.png"
STABILITY_PLOT_PATH: Path = RESULTS_DIR / "feature_selection_stability.png"

# Standard TTL domain features identified for exclusion run
TTL_FEATURES_TO_EXCLUDE: List[str] = ["sttl", "dttl", "ct_state_ttl"]


def load_train_only_pool(
    train_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Loads raw train split only, applies train-only deduplication, and extracts pool.
    
    Guarantees strict train-only discipline: TEST data is never loaded or accessed.
    """
    path = train_path or RAW_TRAIN_CSV
    if not Path(path).exists():
        raise FileNotFoundError(f"Raw train file not found at: {path}")

    raw_train = pd.read_csv(path)
    dedup_train = deduplicate_train(raw_train, test_df=None, log_path=None)
    X_train, y_train, _ = separate_features_target_and_metadata(dedup_train)
    quantum_pool = get_quantum_feature_pool(X_train)
    return X_train[quantum_pool], y_train, quantum_pool


def remove_near_constant_features(
    X: pd.DataFrame,
    threshold: float = 1e-5,
) -> Tuple[List[str], List[str]]:
    """Identifies and removes features with variance <= threshold or nunique <= 1.
    
    Returns (retained_features, dropped_features).
    """
    retained: List[str] = []
    dropped: List[str] = []

    for col in X.columns:
        series = X[col].dropna()
        if series.nunique() <= 1 or series.var() <= threshold:
            dropped.append(col)
        else:
            retained.append(col)

    return retained, dropped


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    random_state: int = RANDOM_SEED,
) -> pd.Series:
    """Computes mutual information between candidate features and binary target on TRAIN only.
    
    Deterministic via random_state. Returns pd.Series sorted descending by MI score.
    """
    if not features:
        return pd.Series(dtype=float)

    X_subset = X[features].copy()
    mi_values = mutual_info_classif(
        X_subset,
        y,
        random_state=random_state,
        n_neighbors=3,
    )
    mi_series = pd.Series(mi_values, index=features, name="mutual_info")
    return mi_series.sort_values(ascending=False)


def remove_highly_correlated_features(
    X: pd.DataFrame,
    mi_scores: pd.Series,
    correlation_threshold: float = 0.9,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Computes |Spearman| correlations on TRAIN data only.
    
    For pairs with |r| > correlation_threshold, keeps the feature with higher
    TRAIN-only mutual information (MI) with label.
    
    Returns (surviving_features, dropped_info_dict).
    """
    candidates = list(mi_scores.index)
    if len(candidates) <= 1:
        return candidates, {}

    corr_matrix = X[candidates].corr(method="spearman").abs()

    # Sort candidates strictly by MI descending (with feature name secondary for determinism)
    sorted_candidates = sorted(
        candidates,
        key=lambda col: (mi_scores[col], col),
        reverse=True,
    )

    surviving: List[str] = []
    dropped_info: Dict[str, Dict[str, Any]] = {}

    for cand in sorted_candidates:
        # Check if cand is highly correlated with any already-retained feature
        redundant_with = None
        for retained in surviving:
            r_val = corr_matrix.loc[cand, retained]
            if r_val > correlation_threshold:
                redundant_with = retained
                dropped_info[cand] = {
                    "dropped_in_favor_of": retained,
                    "spearman_correlation": round(float(r_val), 5),
                    "mi_candidate": round(float(mi_scores[cand]), 6),
                    "mi_retained": round(float(mi_scores[retained]), 6),
                }
                break

        if redundant_with is None:
            surviving.append(cand)

    return surviving, dropped_info


def select_mrmr_features(
    X: pd.DataFrame,
    candidate_features: List[str],
    mi_scores: pd.Series,
    k: int = N_QUBIT_FEATURES,
    criterion: str = "MID",
) -> List[str]:
    """Greedily selects k features using mRMR (high MI relevance + low Spearman redundancy).
    
    1st feature: highest MI relevance with target.
    Subsequent features: argmax [ Relevance(c) - Redundancy(c, S) ]
    where Relevance(c) = MI(c, y),
          Redundancy(c, S) = mean_{s in S} |r_spearman(c, s)|.
    """
    if not candidate_features:
        return []

    target_k = min(k, len(candidate_features))
    corr_matrix = X[candidate_features].corr(method="spearman").abs()

    # Step 1: Feature with highest MI
    # Sort deterministically
    ranked_initial = sorted(
        candidate_features,
        key=lambda c: (mi_scores[c], c),
        reverse=True,
    )
    selected: List[str] = [ranked_initial[0]]

    # Step 2: Iteratively select remaining k-1 features
    for _ in range(1, target_k):
        unselected = [c for c in candidate_features if c not in selected]
        best_cand: Optional[str] = None
        best_score = -float("inf")

        # Sort unselected by MI descending for deterministic iteration
        unselected_sorted = sorted(
            unselected,
            key=lambda c: (mi_scores[c], c),
            reverse=True,
        )

        for c in unselected_sorted:
            relevance = float(mi_scores[c])
            redundancy = float(np.mean([corr_matrix.loc[c, s] for s in selected]))

            if criterion.upper() == "MIQ":
                score = relevance / (redundancy + 1e-5)
            else:  # "MID" (Mutual Information Difference) default
                score = relevance - redundancy

            if score > best_score:
                best_score = score
                best_cand = c

        if best_cand is not None:
            selected.append(best_cand)

    return selected


def run_single_selection(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_pool: List[str],
    exclude_features: Optional[List[str]] = None,
    k: int = N_QUBIT_FEATURES,
    variance_threshold: float = 1e-5,
    correlation_threshold: float = 0.9,
    random_state: int = RANDOM_SEED,
    criterion: str = "MID",
) -> Dict[str, Any]:
    """Executes steps 1 to 4 of the feature selection pipeline on given (X, y).
    
    Exclusions happen BEFORE any selection steps.
    """
    exclusions = set(exclude_features or [])
    active_pool = [c for c in candidate_pool if c not in exclusions and c in X.columns]

    # Step 1: Remove near-constant features
    retained_after_var, dropped_near_constant = remove_near_constant_features(
        X[active_pool],
        threshold=variance_threshold,
    )

    # Step 2: Compute TRAIN-only MI with label for all variance-surviving candidates
    mi_scores = compute_mutual_information(
        X,
        y,
        features=retained_after_var,
        random_state=random_state,
    )

    # Step 3: Compute Spearman correlations and remove pairs with |r| > correlation_threshold
    surviving_features, dropped_correlated = remove_highly_correlated_features(
        X,
        mi_scores=mi_scores,
        correlation_threshold=correlation_threshold,
    )

    # Step 4: Rank remaining by MI and greedily select k using mRMR
    final_selected = select_mrmr_features(
        X,
        candidate_features=surviving_features,
        mi_scores=mi_scores,
        k=k,
        criterion=criterion,
    )

    return {
        "selected_features": final_selected,
        "mi_scores": {col: round(float(val), 6) for col, val in mi_scores.items()},
        "dropped_near_constant": dropped_near_constant,
        "dropped_correlated": dropped_correlated,
        "surviving_features": surviving_features,
        "surviving_count": len(surviving_features),
        "active_pool_size": len(active_pool),
    }


def run_stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    candidate_pool: List[str],
    exclude_features: Optional[List[str]] = None,
    k: int = N_QUBIT_FEATURES,
    n_splits: int = 5,
    variance_threshold: float = 1e-5,
    correlation_threshold: float = 0.9,
    random_state: int = RANDOM_SEED,
    criterion: str = "MID",
) -> Dict[str, Any]:
    """Runs 5-fold Stratified TRAIN-only stability selection.
    
    Evaluates steps 1-4 on each stratified fold and computes feature selection frequencies.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_selections: List[List[str]] = []
    feature_counts: Dict[str, int] = {}

    for fold_idx, (train_indices, _) in enumerate(skf.split(X, y)):
        X_fold = X.iloc[train_indices]
        y_fold = y.iloc[train_indices]

        fold_res = run_single_selection(
            X_fold,
            y_fold,
            candidate_pool=candidate_pool,
            exclude_features=exclude_features,
            k=k,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            random_state=random_state + fold_idx,
            criterion=criterion,
        )
        selected = fold_res["selected_features"]
        fold_selections.append(selected)

        for feat in selected:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    # Sort feature counts descending
    sorted_counts = dict(
        sorted(feature_counts.items(), key=lambda item: (-item[1], item[0]))
    )
    selection_frequencies = {
        feat: round(count / n_splits, 4) for feat, count in sorted_counts.items()
    }

    return {
        "fold_selections": fold_selections,
        "stability_selection_counts": sorted_counts,
        "stability_selection_frequencies": selection_frequencies,
        "n_folds": n_splits,
    }


def plot_mi_scores(
    all_runs_data: Dict[str, Any],
    output_path: Path = MI_SCORES_PLOT_PATH,
    top_n: int = 15,
) -> None:
    """Generates high-resolution visualization for Mutual Information rankings.
    
    Compares 'all_features' and 'TTL_excluded' side-by-side, highlighting selected k features.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    runs = [
        ("all_features", "Run 1: All Candidate Features (39 features)", axes[0]),
        ("TTL_excluded", "Run 2: TTL Excluded (sttl, dttl, ct_state_ttl dropped)", axes[1]),
    ]

    for run_key, title, ax in runs:
        run_data = all_runs_data[run_key]
        mi_scores = pd.Series(run_data["mutual_information_scores"])
        selected_feats = set(run_data["selected_features"])

        top_mi = mi_scores.head(top_n).sort_values(ascending=True)
        colors = ["#00ADB5" if f in selected_feats else "#94A3B8" for f in top_mi.index]

        y_pos = np.arange(len(top_mi))
        bars = ax.barh(y_pos, top_mi.values, color=colors, edgecolor="#1E293B", linewidth=0.8, height=0.65)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_mi.index, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mutual Information with Target (label)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Label values
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
            )

        ax.set_xlim(0, max(top_mi.values) * 1.18)

    # Custom legend
    custom_lines = [
        plt.Rectangle((0, 0), 1, 1, fc="#00ADB5", ec="#1E293B"),
        plt.Rectangle((0, 0), 1, 1, fc="#94A3B8", ec="#1E293B"),
    ]
    fig.legend(
        custom_lines,
        ["Selected Qubit Feature (mRMR k=4)", "Candidate Feature (Unselected)"],
        loc="upper center",
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=11,
        bbox_to_anchor=(0.5, 0.99),
    )

    plt.suptitle(
        "Quantum CyberShield — TRAIN-Only Mutual Information & Feature Selection",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_stability_counts(
    all_runs_data: Dict[str, Any],
    output_path: Path = STABILITY_PLOT_PATH,
) -> None:
    """Generates high-resolution visualization for 5-Fold Stability Selection counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")

    runs = [
        ("all_features", "Stability: All Features (5 Folds)", axes[0]),
        ("TTL_excluded", "Stability: TTL Excluded (5 Folds)", axes[1]),
    ]

    for run_key, title, ax in runs:
        run_data = all_runs_data[run_key]
        stability_counts = run_data["stability_selection_counts"]

        if not stability_counts:
            ax.text(0.5, 0.5, "No features selected", ha="center", va="center")
            continue

        counts_series = pd.Series(stability_counts).sort_values(ascending=True)
        colors = ["#10B981" if c == 5 else "#F59E0B" for c in counts_series.values]

        y_pos = np.arange(len(counts_series))
        bars = ax.barh(
            y_pos,
            counts_series.values,
            color=colors,
            edgecolor="#1E293B",
            linewidth=0.8,
            height=0.55,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(counts_series.index, fontsize=11, fontweight="bold")
        ax.set_xlabel("Selection Count (out of 5 Stratified Train Folds)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(0, 5.8)
        ax.set_xticks(range(6))
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            w = bar.get_width()
            pct = int((w / 5.0) * 100)
            ax.text(
                w + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{int(w)}/5 ({pct}%)",
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#0F172A",
            )

    custom_lines = [
        plt.Rectangle((0, 0), 1, 1, fc="#10B981", ec="#1E293B"),
        plt.Rectangle((0, 0), 1, 1, fc="#F59E0B", ec="#1E293B"),
    ]
    fig.legend(
        custom_lines,
        ["100% Stable (5/5 folds)", "Partially Selected (<5 folds)"],
        loc="upper center",
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=11,
        bbox_to_anchor=(0.5, 0.99),
    )

    plt.suptitle(
        "Quantum CyberShield — 5-Fold Stratified Stability Selection",
        fontsize=14,
        fontweight="bold",
        y=1.04,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


def run_feature_selection_pipeline(
    train_path: Optional[Path] = None,
    k: int = N_QUBIT_FEATURES,
    random_state: int = RANDOM_SEED,
    output_json_path: Path = FEATURE_SELECTION_JSON,
    output_mi_plot_path: Path = MI_SCORES_PLOT_PATH,
    output_stability_plot_path: Path = STABILITY_PLOT_PATH,
    variance_threshold: float = 1e-5,
    correlation_threshold: float = 0.9,
    criterion: str = "MID",
) -> Dict[str, Any]:
    """Orchestrates Stage 3 train-only feature selection across both benchmark runs.
    
    Run 1: 'all_features'
    Run 2: 'TTL_excluded' (sttl, dttl, ct_state_ttl excluded)
    
    Saves results to results/feature_selection.json and generates visualization PNGs.
    """
    print("=" * 60)
    print("STAGE 3: TRAIN-ONLY FEATURE SELECTION PIPELINE")
    print("=" * 60)

    # 1. Load Train data only (never touches test set)
    X_train_pool, y_train, full_pool = load_train_only_pool(train_path)
    print(f"Loaded Train Pool: {X_train_pool.shape[0]} rows x {len(full_pool)} quantum features")
    print(f"Target distribution: {dict(y_train.value_counts(normalize=True).round(4))}")

    run_specs = {
        "all_features": [],
        "TTL_excluded": TTL_FEATURES_TO_EXCLUDE,
    }

    results_data: Dict[str, Any] = {
        "metadata": {
            "random_seed": random_state,
            "k_qubit_features": k,
            "n_stability_folds": 5,
            "variance_threshold": variance_threshold,
            "correlation_threshold": correlation_threshold,
            "mrmr_criterion": criterion,
            "train_rows_evaluated": int(len(X_train_pool)),
            "quantum_candidate_pool_size": len(full_pool),
            "quantum_candidate_pool": full_pool,
        },
        "runs": {},
    }

    for run_name, exclude_list in run_specs.items():
        print(f"\n--- Running: {run_name} (Excluded: {exclude_list}) ---")

        # Full TRAIN selection
        single_sel = run_single_selection(
            X=X_train_pool,
            y=y_train,
            candidate_pool=full_pool,
            exclude_features=exclude_list,
            k=k,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            random_state=random_state,
            criterion=criterion,
        )

        print(f"Surviving features after correlation pruning: {single_sel['surviving_count']}")
        print(f"Selected k={k} features: {single_sel['selected_features']}")

        # Stability selection across 5 stratified folds
        print("Running 5-fold stratified stability selection...")
        stab_sel = run_stability_selection(
            X=X_train_pool,
            y=y_train,
            candidate_pool=full_pool,
            exclude_features=exclude_list,
            k=k,
            n_splits=5,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            random_state=random_state,
            criterion=criterion,
        )
        print(f"Stability selection counts: {stab_sel['stability_selection_counts']}")

        results_data["runs"][run_name] = {
            "excluded_features": exclude_list,
            "selected_features": single_sel["selected_features"],
            "mutual_information_scores": single_sel["mi_scores"],
            "stability_selection_counts": stab_sel["stability_selection_counts"],
            "stability_selection_frequencies": stab_sel["stability_selection_frequencies"],
            "fold_selections": stab_sel["fold_selections"],
            "dropped_near_constant": single_sel["dropped_near_constant"],
            "dropped_correlated": single_sel["dropped_correlated"],
            "surviving_features_count": single_sel["surviving_count"],
        }

    # Save JSON report
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[OK] Feature selection results saved to: {output_json_path}")

    # Generate plots
    plot_mi_scores(results_data["runs"], output_path=output_mi_plot_path)
    print(f"[OK] Mutual information plot saved to: {output_mi_plot_path}")

    plot_stability_counts(results_data["runs"], output_path=output_stability_plot_path)
    print(f"[OK] Stability selection plot saved to: {output_stability_plot_path}")

    print("=" * 60)
    return results_data


if __name__ == "__main__":
    run_feature_selection_pipeline()
