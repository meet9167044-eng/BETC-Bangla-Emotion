"""
tests/test_phase12_final_eval.py
================================
Unit tests for Phase 12 final test evaluation utilities.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import TARGET_COLS
from src.models.evaluate import compute_metrics, apply_thresholds


class TestPhase12Evaluation:
    def test_target_cols_present_in_test_data(self):
        test_path = os.path.join(WORKSPACE, "Data", "processed", "test", "test_preprocessed.csv")
        assert os.path.exists(test_path), f"Test set missing at {test_path}"
        df = pd.read_csv(test_path, nrows=10)
        for col in TARGET_COLS:
            assert col in df.columns, f"Target col {col} missing from test set"

    def test_compute_metrics_returns_full_suite(self):
        rng = np.random.default_rng(42)
        Y_true = rng.integers(0, 2, size=(100, 6))
        Y_pred = rng.integers(0, 2, size=(100, 6))

        m = compute_metrics(Y_true, Y_pred, TARGET_COLS)
        expected_keys = {
            "macro_f1", "micro_f1", "macro_precision", "macro_recall",
            "hamming_loss", "jaccard", "subset_accuracy", "per_class"
        }
        assert expected_keys.issubset(set(m.keys()))
        assert set(m["per_class"].keys()) == set(TARGET_COLS)
        for col in TARGET_COLS:
            assert "f1" in m["per_class"][col]
            assert "precision" in m["per_class"][col]
            assert "recall" in m["per_class"][col]

    def test_apply_thresholds_exactness(self):
        P = np.array([
            [0.2, 0.4, 0.6, 0.8, 0.1, 0.5],
            [0.3, 0.2, 0.7, 0.4, 0.9, 0.1],
        ])
        thresholds = {
            "anger": 0.25,
            "disgust": 0.35,
            "fear": 0.55,
            "joy": 0.50,
            "sadness": 0.50,
            "surprise": 0.40,
        }
        pred = apply_thresholds(P, thresholds, TARGET_COLS)
        expected = np.array([
            [0, 1, 1, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ])
        np.testing.assert_array_equal(pred, expected)

    def test_soft_vote_weights(self):
        P_lr = np.array([[0.8, 0.2], [0.4, 0.6]])
        P_cnb = np.array([[0.2, 0.6], [0.8, 0.4]])
        P_blend = 0.70 * P_lr + 0.30 * P_cnb
        expected = np.array([[0.62, 0.32], [0.52, 0.54]])
        np.testing.assert_allclose(P_blend, expected, atol=1e-5)

