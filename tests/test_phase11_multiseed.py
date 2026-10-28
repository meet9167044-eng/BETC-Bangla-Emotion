"""
tests/test_phase11_multiseed.py
===============================
Unit tests for Phase 11 multi-seed robustness helpers.
Fast, mock data only, no heavy fitting.
"""
import sys
import os
import numpy as np
import pytest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from scripts.experiments.phase11_multiseed_robustness import (
    compute_summary_stats,
    sweep_thresholds,
    SEEDS,
    TARGET_COLS,
)


class TestMultiSeedSummaryStats:
    def test_summary_stats_calculation(self):
        runs = [
            {
                "seed": 42,
                "macro_f1": 0.5100,
                "micro_f1": 0.5700,
                "per_class": {col: 0.5000 for col in TARGET_COLS},
            },
            {
                "seed": 123,
                "macro_f1": 0.5200,
                "micro_f1": 0.5800,
                "per_class": {col: 0.5200 for col in TARGET_COLS},
            },
            {
                "seed": 999,
                "macro_f1": 0.5300,
                "micro_f1": 0.5900,
                "per_class": {col: 0.5400 for col in TARGET_COLS},
            },
        ]
        stats = compute_summary_stats(runs)

        assert stats["macro_mean"] == 0.5200
        assert stats["micro_mean"] == 0.5800
        assert np.isclose(stats["macro_std"], 0.0100)
        assert np.isclose(stats["micro_std"], 0.0100)
        assert set(stats["per_class"].keys()) == set(TARGET_COLS)
        assert stats["per_class"]["fear"]["mean"] == 0.5200
        assert np.isclose(stats["per_class"]["fear"]["std"], 0.0200)

    def test_single_seed_handling(self):
        runs = [
            {
                "seed": 42,
                "macro_f1": 0.5170,
                "micro_f1": 0.5746,
                "per_class": {col: 0.5000 for col in TARGET_COLS},
            }
        ]
        stats = compute_summary_stats(runs)
        assert stats["macro_mean"] == 0.5170
        assert stats["macro_std"] == 0.0
        assert stats["micro_std"] == 0.0


class TestMultiSeedConstants:
    def test_seeds_list(self):
        assert len(SEEDS) >= 3
        assert 42 in SEEDS
        assert 123 in SEEDS
        assert 999 in SEEDS

    def test_target_cols(self):
        assert len(TARGET_COLS) == 6
        assert set(TARGET_COLS) == {"anger", "disgust", "fear", "joy", "sadness", "surprise"}


class TestSweepThresholds:
    def test_sweep_returns_expected_structure(self):
        rng = np.random.default_rng(42)
        P = rng.uniform(0.0, 1.0, size=(50, 6))
        Y = (P > 0.5).astype(int)
        taus = sweep_thresholds(P, Y, step=0.05, label_names=TARGET_COLS)
        assert isinstance(taus, dict)
        assert set(taus.keys()) == set(TARGET_COLS)
        for val in taus.values():
            assert 0.05 <= val <= 0.95

