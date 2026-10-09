"""Unit tests for Stage 7b Step 5 learning curves sweep module."""

from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from src.kernel_cache_v2 import KernelCacheManager
from src.learning_curves_v2 import evaluate_single_run


@dataclass
class MockBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@pytest.fixture
def dummy_bundle():
    np.random.seed(42)
    X_train = np.random.uniform(0, np.pi, size=(30, 4))
    y_train = np.array([0] * 15 + [1] * 15)
    X_test = np.random.uniform(0, np.pi, size=(20, 4))
    y_test = np.array([0] * 10 + [1] * 10)
    return MockBundle(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def test_evaluate_single_run_structure(dummy_bundle, tmp_path):
    cache_mgr = KernelCacheManager(cache_dir=tmp_path)
    res = evaluate_single_run(dummy_bundle, n_train=20, seed=42, cache_mgr=cache_mgr)

    for m in ["qsvc", "svm", "rf"]:
        assert m in res
        m_res = res[m]
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate", "fit_time_s", "infer_time_s"]:
            assert metric in m_res
            assert isinstance(m_res[metric], float)

        assert len(m_res["y_pred"]) == 20
        assert len(m_res["y_score"]) == 20
