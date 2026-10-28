"""
tests/test_phase9_classifiers.py
===================================
Unit tests for Phase 9 classifier comparison helpers.
Fast, no model fitting on real data.
"""
import sys
import os
import numpy as np
import pytest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from scripts.experiments.phase9_classifier_comparison import (
    sweep_thresholds,
    CHAMPION_MACRO,
    CHAMPION_MICRO,
    CHAMPION_PER_CLASS,
    TARGET_COLS,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def tiny_proba_and_labels():
    """6-label prob matrix and binary ground truth (100 samples)."""
    rng = np.random.default_rng(42)
    P = rng.uniform(0.0, 1.0, size=(100, 6)).astype(np.float32)
    Y = (P > 0.5).astype(int)
    return P, Y


# ─── sweep_thresholds ────────────────────────────────────────────────────────
class TestSweepThresholds:
    def test_returns_dict(self, tiny_proba_and_labels):
        P, Y = tiny_proba_and_labels
        taus = sweep_thresholds(P, Y, step=0.05, label_names=TARGET_COLS)
        assert isinstance(taus, dict)

    def test_correct_keys(self, tiny_proba_and_labels):
        P, Y = tiny_proba_and_labels
        taus = sweep_thresholds(P, Y, step=0.05, label_names=TARGET_COLS)
        assert set(taus.keys()) == set(TARGET_COLS)

    def test_values_in_range(self, tiny_proba_and_labels):
        P, Y = tiny_proba_and_labels
        taus = sweep_thresholds(P, Y, step=0.05, label_names=TARGET_COLS)
        for k, v in taus.items():
            assert 0.05 <= v <= 0.95, f"Threshold {v} for {k} out of [0.05, 0.95]"

    def test_perfect_probabilities(self):
        """If P == Y exactly (0 or 1), optimal threshold should be in (0, 1)."""
        Y = np.array([[1, 0, 1, 0, 1, 0]] * 50, dtype=int)
        P = Y.astype(float)
        taus = sweep_thresholds(P, Y, step=0.05, label_names=TARGET_COLS)
        # With perfect P, any threshold in (0, 1) gives perfect F1
        for v in taus.values():
            assert 0.05 <= v <= 0.95

    def test_default_label_names(self, tiny_proba_and_labels):
        """Without label_names, should use label_0 .. label_5."""
        P, Y = tiny_proba_and_labels
        taus = sweep_thresholds(P, Y, step=0.1)
        for i in range(6):
            assert f"label_{i}" in taus

    def test_single_label(self):
        rng = np.random.default_rng(7)
        P = rng.uniform(0, 1, (50, 1))
        Y = (P > 0.5).astype(int)
        taus = sweep_thresholds(P, Y, step=0.05, label_names=["anger"])
        assert "anger" in taus
        assert 0.05 <= taus["anger"] <= 0.95


# ─── Champion constants ───────────────────────────────────────────────────────
class TestChampionConstants:
    def test_champion_macro_is_float(self):
        assert isinstance(CHAMPION_MACRO, float)
        assert 0.0 < CHAMPION_MACRO < 1.0

    def test_champion_micro_is_float(self):
        assert isinstance(CHAMPION_MICRO, float)
        assert 0.0 < CHAMPION_MICRO < 1.0

    def test_champion_per_class_has_all_labels(self):
        assert set(CHAMPION_PER_CLASS.keys()) == set(TARGET_COLS)

    def test_champion_per_class_values_in_range(self):
        for k, v in CHAMPION_PER_CLASS.items():
            assert 0.0 <= v <= 1.0, f"Per-class F1 {v} for {k} out of range"

    def test_target_cols_length(self):
        assert len(TARGET_COLS) == 6

    def test_target_cols_correct_names(self):
        expected = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
        assert set(TARGET_COLS) == expected


# ─── Regression-check logic ──────────────────────────────────────────────────
class TestRegressionCheck:
    """Test the regression detection logic (inline, mirrors phase9 script)."""
    REGRESSION_TOL = 0.010

    def _check_regressions(self, per_cls):
        return {
            k: round(per_cls[k] - CHAMPION_PER_CLASS[k], 4)
            for k in TARGET_COLS
            if round(per_cls[k] - CHAMPION_PER_CLASS[k], 4) < -self.REGRESSION_TOL
        }

    def test_no_regression_if_same_scores(self):
        per_cls = {k: round(v, 4) for k, v in CHAMPION_PER_CLASS.items()}
        assert self._check_regressions(per_cls) == {}

    def test_regression_detected_when_fear_drops(self):
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["fear"] = round(CHAMPION_PER_CLASS["fear"] - 0.02, 4)
        reg = self._check_regressions(per_cls)
        assert "fear" in reg

    def test_no_regression_for_small_drop(self):
        """Drop of exactly 0.005 should NOT trigger regression (tolerance=0.010)."""
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["disgust"] = round(CHAMPION_PER_CLASS["disgust"] - 0.005, 4)
        reg = self._check_regressions(per_cls)
        assert "disgust" not in reg

    def test_regression_at_boundary(self):
        """Drop of exactly 0.010 is at boundary — not strictly < -0.010, so no reg."""
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["anger"] = round(CHAMPION_PER_CLASS["anger"] - 0.010, 4)
        reg = self._check_regressions(per_cls)
        assert "anger" not in reg

    def test_regression_just_over_boundary(self):
        """Drop of 0.011 should trigger regression."""
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["anger"] = round(CHAMPION_PER_CLASS["anger"] - 0.011, 4)
        reg = self._check_regressions(per_cls)
        assert "anger" in reg

