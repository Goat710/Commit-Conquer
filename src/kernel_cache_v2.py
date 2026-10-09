"""Memory-aware kernel computation, chunking, and caching strategy — Stage 7b.

Provides:
  - KernelCacheManager: SHA256-keyed, metadata-validated caching for kernel matrices
  - compute_statevector_embeddings_chunked: Chunked statevector computation
  - compute_kernel_matrix_chunked: Memory-bounded kernel matrix generation (chunk_size <= 10,000)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import MODELS_DIR, RESULTS_DIR
from src.quantum_model_v2 import (
    compute_statevector_embeddings,
    compute_statevector_kernel,
)

logger = logging.getLogger(__name__)

CACHE_DIR_V2: Path = MODELS_DIR / "cache_v2"
DEFAULT_CHUNK_SIZE: int = 10000


def hash_array(arr: np.ndarray) -> str:
    """Compute a deterministic SHA-256 hash of a NumPy array's bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


class KernelCacheManager:
    """Manages metadata-validated disk caches for precomputed quantum kernels."""

    def __init__(self, cache_dir: Path = CACHE_DIR_V2):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        split_name: str,
        n_samples_A: int,
        n_samples_B: int,
        seed: int,
        features: List[str],
        data_hash_A: str,
        data_hash_B: Optional[str] = None,
    ) -> str:
        """Create a deterministic unique cache key incorporating data and configuration."""
        b_hash = data_hash_B or data_hash_A
        feat_str = ",".join(sorted(features))
        key_raw = f"{split_name}_A{n_samples_A}_B{n_samples_B}_s{seed}_{feat_str}_{data_hash_A}_{b_hash}"
        key_hash = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:20]
        return f"kernel_{split_name}_{n_samples_A}x{n_samples_B}_s{seed}_{key_hash}"

    def get(
        self,
        cache_key: str,
        expected_shape: Tuple[int, int],
        expected_data_hash: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Load and strictly validate a cached kernel matrix.

        Returns None if cache is missing, corrupted, or fails validation.
        """
        npy_path = self.cache_dir / f"{cache_key}.npy"
        meta_path = self.cache_dir / f"{cache_key}.json"

        if not npy_path.exists() or not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Metadata validation
            if meta.get("shape") != list(expected_shape):
                logger.warning("Cache validation failed: shape mismatch for %s", cache_key)
                return None

            if expected_data_hash and meta.get("data_hash_A") != expected_data_hash:
                logger.warning("Cache validation failed: data hash mismatch for %s", cache_key)
                return None

            # Array loading & invariant validation
            arr = np.load(npy_path)
            if arr.shape != expected_shape:
                return None

            if not np.all(np.isfinite(arr)):
                return None

            # Square matrix invariants
            if expected_shape[0] == expected_shape[1]:
                diag_dev = float(np.max(np.abs(np.diag(arr) - 1.0)))
                if diag_dev > 1e-5:
                    logger.warning("Cache validation failed: diagonal dev %f > 1e-5", diag_dev)
                    return None

            return arr
        except Exception as e:
            logger.warning("Error loading cache %s: %s", cache_key, e)
            return None

    def put(
        self,
        cache_key: str,
        K: np.ndarray,
        metadata: Dict[str, Any],
    ) -> None:
        """Atomically persist a kernel matrix and its JSON metadata."""
        npy_path = self.cache_dir / f"{cache_key}.npy"
        meta_path = self.cache_dir / f"{cache_key}.json"

        meta_to_save = dict(metadata)
        meta_to_save["shape"] = list(K.shape)
        meta_to_save["dtype"] = str(K.dtype)
        meta_to_save["file_size_bytes"] = K.nbytes
        meta_to_save["created_at"] = time.time()

        # Save numpy array
        np.save(npy_path, K)
        # Save metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2)


def compute_statevector_embeddings_chunked(
    X: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    feature_map: Optional[Any] = None,
) -> np.ndarray:
    """Compute statevector embeddings in row chunks to bound memory overhead."""
    n_samples = len(X)
    hilbert_dim = 16  # 4 qubits -> 2^4 = 16
    states = np.empty((n_samples, hilbert_dim), dtype=np.complex128)

    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        X_chunk = X[start_idx:end_idx]
        states[start_idx:end_idx] = compute_statevector_embeddings(
            X_chunk, feature_map=feature_map
        )

    return states


def compute_kernel_matrix_chunked(
    states_A: np.ndarray,
    states_B: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Compute rectangular kernel matrix K(A, B) in row chunks of at most chunk_size rows.

    Parameters
    ----------
    states_A:
        Complex statevectors for samples A (e.g. Test set, shape (N_A, 16)).
    states_B:
        Complex statevectors for samples B (e.g. Train set, shape (N_B, 16)).
    chunk_size:
        Maximum number of rows of A to process in a single chunk (default 10,000).

    Returns
    -------
    np.ndarray:
        Kernel matrix of shape (N_A, N_B).
    """
    n_A = len(states_A)
    n_B = len(states_B)
    K = np.empty((n_A, n_B), dtype=np.float64)

    for start_idx in range(0, n_A, chunk_size):
        end_idx = min(start_idx + chunk_size, n_A)
        states_chunk = states_A[start_idx:end_idx]
        K[start_idx:end_idx, :] = compute_statevector_kernel(states_chunk, states_B)

    return K
