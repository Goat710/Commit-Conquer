"""Central configuration module for Quantum CyberShield.

Defines reproducible seeds, feature parameters, column exclusions,
experiment metadata, and filesystem paths.
"""

from pathlib import Path
from typing import List

# ==========================================
# Reproducibility & Experiment Meta
# ==========================================
RANDOM_SEED: int = 42
EXPERIMENT_NAME: str = "unsw_nb15_binary_v1"

# ==========================================
# Target & Feature Constraints
# ==========================================
TARGET_COLUMN: str = "label"
LEAKY_COLUMNS: List[str] = ["id", "attack_cat"]
DROP_COLUMNS: List[str] = LEAKY_COLUMNS
SKEW_THRESHOLD: float = 1.5

# Initial MVP feature dimension for quantum feature map encoding (4 qubits)
# Note: N_QUBIT_FEATURES=4 is only the initial MVP configuration; not assumed optimal.
N_QUBIT_FEATURES: int = 4

# Representative sample size for quantum kernel simulation (O(N^2) statevector scaling)
QUANTUM_TRAIN_SAMPLE_SIZE: int = 200
QUANTUM_TEST_SAMPLE_SIZE: int = 100

# ==========================================
# File System Paths
# ==========================================
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

RESULTS_DIR: Path = BASE_DIR / "results"
MODELS_DIR: Path = BASE_DIR / "models"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
DASHBOARD_DIR: Path = BASE_DIR / "dashboard"

RAW_TRAIN_CSV: Path = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
RAW_TEST_CSV: Path = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"
DATA_REPORT_JSON: Path = RESULTS_DIR / "data_report.json"
PREPROCESSING_LOG_JSON: Path = RESULTS_DIR / "preprocessing_log.json"

# Remote source URLs for UNSW-NB15 official benchmark splits
DATASET_SOURCE_URL_TRAIN: str = (
    "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/train.csv"
)
DATASET_SOURCE_URL_TEST: str = (
    "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main/test.csv"
)

# ==========================================
# CyberShield Policy Configuration (Stage 7)
# ==========================================
# Threat level mapped directly from model consensus votes (0 to 3)
VOTE_THREAT_POLICY: dict = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "HIGH",
}

# Review flag triggered whenever models disagree (votes 1 or 2)
REVIEW_FLAG_REASON: str = "Analyst-review signal from heterogeneous models"

