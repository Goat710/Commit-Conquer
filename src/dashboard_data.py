"""Dashboard Data Access and Caching Helpers for Quantum CyberShield.

This module provides verified data loading, caching, and preprocessing
routines for the Streamlit dashboard. All metrics are loaded directly from
existing, pre-computed benchmark artifacts; no experiments are run dynamically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Base project directory relative to this file
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODELS_DIR: Path = PROJECT_ROOT / "models"


@st.cache_data(show_spinner=False)
def load_stage8c_results() -> Dict[str, Any]:
    """Load Stage 8C controlled operational low-FPR benchmark results."""
    path = RESULTS_DIR / "stage8c_operational_benchmark.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 8C results file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_stage8e_results() -> Dict[str, Any]:
    """Load Stage 8E feature-space comparison benchmark results."""
    path = RESULTS_DIR / "stage8e_benchmark.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 8E results file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_stage8f_results() -> Dict[str, Any]:
    """Load Stage 8F quantum simulation scaling and Nystrom approximation results."""
    path = RESULTS_DIR / "stage8f_scaling_benchmark.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 8F results file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_stage7b_results() -> Dict[str, Any]:
    """Load Stage 7b research summary report."""
    path = RESULTS_DIR / "stage7b_final_report.json"
    if not path.exists():
        raise FileNotFoundError(f"Stage 7b results file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_disagreement_results() -> Dict[str, Any]:
    """Load Stage 7b / Stage 8 disagreement analysis data."""
    path = RESULTS_DIR / "disagreement_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"Disagreement results file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_predictions_slice(n_samples: int = 500) -> pd.DataFrame:
    """Load a small slice of predictions_v2.csv for interactive exploration."""
    path = RESULTS_DIR / "predictions_v2.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found at {path}")
    # Load first n_samples to maintain instant UI responsiveness
    return pd.read_csv(path, nrows=n_samples)


@st.cache_resource(show_spinner=False)
def load_security_engine() -> Any:
    """Load CyberShield multi-model engine bundle using joblib."""
    from src.cybershield_engine import CyberShieldEngine
    bundle_path = MODELS_DIR / "cybershield_engine.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Engine bundle not found at {bundle_path}")
    return CyberShieldEngine.load(bundle_path)


def get_verified_presets() -> Dict[str, Dict[str, Any]]:
    """Return verified network flow samples extracted from the official UNSW-NB15 test set."""
    return {
        "Verified Normal Flow (Consensus Low Threat)": {
            "source_idx": 39147,
            "ground_truth": 0,
            "category": "Normal",
            "features": {
                "sbytes": 17642.0,
                "sload": 59421.89,
                "sttl": 31.0,
                "smean": 53.0,
            },
            "description": "Legitimate enterprise flow. Low packet size, small TTL, balanced transfer.",
            "expected_votes": "0/3",
            "expected_threat": "LOW",
        },
        "Verified Attack Flow (Reconnaissance - Consensus High Threat)": {
            "source_idx": 69617,
            "ground_truth": 1,
            "category": "Reconnaissance",
            "features": {
                "sbytes": 168.0,
                "sload": 672000000.0,
                "sttl": 254.0,
                "smean": 84.0,
            },
            "description": "Port scanning and host probe. High transmission load, elevated TTL (254).",
            "expected_votes": "3/3",
            "expected_threat": "HIGH",
        },
        "Verified Attack Flow (Exploits - Model Disagreement Trigger)": {
            "source_idx": 107333,
            "ground_truth": 1,
            "category": "Exploits",
            "features": {
                "sbytes": 200.0,
                "sload": 400000000.0,
                "sttl": 254.0,
                "smean": 100.0,
            },
            "description": "Network exploit flow. Classical models detect attack while QSVC misclassifies, triggering analyst review.",
            "expected_votes": "2/3",
            "expected_threat": "HIGH",
        },
        "Verified Exploit Flow (Buffer Overflow Pattern - Disagreement)": {
            "source_idx": 119845,
            "ground_truth": 1,
            "category": "Exploits",
            "features": {
                "sbytes": 9354.0,
                "sload": 151190.03,
                "sttl": 254.0,
                "smean": 668.0,
            },
            "description": "High mean packet size (668 bytes) exploit triggering heterogeneous divergence.",
            "expected_votes": "2/3",
            "expected_threat": "HIGH",
        },
    }
