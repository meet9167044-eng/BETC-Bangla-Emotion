"""
tests/test_phase8_class_imbalance.py
=====================================
Regression tests for Phase 8 Class Imbalance utilities.
"""

import numpy as np
import scipy.sparse as sp
import pytest
import sys
import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from scripts.experiments.phase8_class_imbalance import (
    oversample_labels, build_custom_weights, NumpyEncoder
)
from src.models.betc import TARGET_COLS
import json


# ── oversample_labels ─────────────────────────────────────────────────────────

def test_oversample_labels_increases_rows():
    """Oversampling should increase total row count."""
    n = 100
    Y = np.zeros((n, 6), dtype=int)
    Y[:10, 2] = 1   # fear: 10 positives
    X = sp.eye(n, format="csr")
    X_aug, Y_aug = oversample_labels(X, Y, ["fear"], [2], multiplier=2)
    assert X_aug.shape[0] > n


def test_oversample_labels_exact_count():
    """2x multiplier should double the positives (adds 1x existing)."""
    n = 100
    Y = np.zeros((n, 6), dtype=int)
    Y[:10, 2] = 1   # fear: 10 positives
    X = sp.eye(n, format="csr")
    X_aug, Y_aug = oversample_labels(X, Y, ["fear"], [2], multiplier=2)
    # Should add 10 rows → total = 110
    assert X_aug.shape[0] == 110


def test_oversample_labels_preserves_negatives():
    """Oversampling should not change the original training rows (just add)."""
    n = 50
    Y = np.zeros((n, 6), dtype=int)
    Y[:5, 5] = 1   # surprise: 5 positives
    X = sp.eye(n, format="csr")
    X_aug, Y_aug = oversample_labels(X, Y, ["surprise"], [5], multiplier=2)
    # Total = 55 rows; positives in surprise column >= 10
    assert Y_aug[:, 5].sum() >= 10


def test_oversample_labels_no_leakage_structure():
    """Y_aug should only contain integer 0/1 values."""
    n = 80
    Y = np.zeros((n, 6), dtype=int)
    Y[:15, 1] = 1
    X = sp.random(n, 200, density=0.05, format="csr")
    X_aug, Y_aug = oversample_labels(X, Y, ["disgust"], [1], multiplier=2)
    assert set(np.unique(Y_aug)).issubset({0, 1})


def test_oversample_labels_multiple_targets():
    """Multiple label oversampling should augment all specified labels."""
    n = 100
    Y = np.zeros((n, 6), dtype=int)
    Y[:8, 2] = 1    # fear
    Y[:12, 5] = 1   # surprise
    X = sp.eye(n, format="csr")
    X_aug, Y_aug = oversample_labels(X, Y, ["fear", "surprise"], [2, 5], multiplier=2)
    # Should have added 8 + 12 = 20 new rows
    assert X_aug.shape[0] == 120


# ── build_custom_weights ──────────────────────────────────────────────────────

def test_build_custom_weights_default_one():
    """Samples not in any target label should have weight 1.0."""
    n = 20
    Y = np.zeros((n, 6), dtype=int)
    weights = build_custom_weights(Y, {"fear": 3.0})
    assert np.all(weights == 1.0)


def test_build_custom_weights_multiplier():
    """Samples in fear should get weight 3.0."""
    n = 20
    Y = np.zeros((n, 6), dtype=int)
    fear_idx = TARGET_COLS.index("fear")
    Y[:5, fear_idx] = 1
    weights = build_custom_weights(Y, {"fear": 3.0})
    assert np.all(weights[:5] == 3.0)
    assert np.all(weights[5:] == 1.0)


def test_build_custom_weights_max_rule():
    """Samples in multiple target labels should get the max weight."""
    n = 10
    Y = np.zeros((n, 6), dtype=int)
    fear_idx    = TARGET_COLS.index("fear")
    disgust_idx = TARGET_COLS.index("disgust")
    Y[0, fear_idx]    = 1
    Y[0, disgust_idx] = 1
    weights = build_custom_weights(Y, {"fear": 3.0, "disgust": 2.0})
    assert weights[0] == pytest.approx(3.0)


# ── NumpyEncoder ──────────────────────────────────────────────────────────────

def test_numpy_encoder_int64():
    data = {"val": np.int64(42)}
    result = json.loads(json.dumps(data, cls=NumpyEncoder))
    assert result["val"] == 42
    assert isinstance(result["val"], int)


def test_numpy_encoder_float32():
    data = {"val": np.float32(3.14)}
    result = json.loads(json.dumps(data, cls=NumpyEncoder))
    assert abs(result["val"] - 3.14) < 0.01
