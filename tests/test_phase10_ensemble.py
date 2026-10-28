"""
tests/test_phase10_ensemble.py
================================
Unit tests for Phase 10 ensemble comparison helpers.
Fast, no model fitting on real data.
"""
import sys
import os
import numpy as np
import pytest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from scripts.experiments.phase10_ensemble_comparison import (
    sweep_thresholds,
    evaluate_probs,
    CHAMPION_MACRO,
    CHAMPION_MICRO,
    CHAMPION_PER_CLASS,
    REGRESSION_TOLERANCE,
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


# ─── evaluate_probs ──────────────────────────────────────────────────────────
class TestEvaluateProbs:
    def test_evaluate_probs_returns_expected_keys(self, tiny_proba_and_labels):
        P, Y = tiny_proba_and_labels
        res = evaluate_probs("Test_Model", P, Y, extra_dims=100)
        expected_keys = {
            "name", "dims", "macro_f1", "micro_f1", "per_class",
            "thresholds", "delta_macro", "delta_micro", "regressions",
            "beats_s0", "sweep_time_s",
        }
        assert set(res.keys()) == expected_keys
        assert res["name"] == "Test_Model"
        assert res["dims"] == 100
        assert isinstance(res["macro_f1"], float)
        assert isinstance(res["micro_f1"], float)
        assert set(res["per_class"].keys()) == set(TARGET_COLS)
        assert set(res["thresholds"].keys()) == set(TARGET_COLS)


# ─── Blending & Ensembling ───────────────────────────────────────────────────
class TestBlendingLogic:
    def test_soft_vote_convex_combination(self, tiny_proba_and_labels):
        P_lr, _ = tiny_proba_and_labels
        rng = np.random.default_rng(123)
        P_cnb = rng.uniform(0.0, 1.0, size=P_lr.shape).astype(np.float32)

        # 50/50 blend
        P_blend_50 = 0.5 * P_lr + 0.5 * P_cnb
        assert P_blend_50.shape == P_lr.shape
        assert np.all((P_blend_50 >= 0.0) & (P_blend_50 <= 1.0))

        # 70/30 blend
        P_blend_70 = 0.7 * P_lr + 0.3 * P_cnb
        assert P_blend_70.shape == P_lr.shape
        assert np.all((P_blend_70 >= 0.0) & (P_blend_70 <= 1.0))

        # 30/70 blend
        P_blend_30 = 0.3 * P_lr + 0.7 * P_cnb
        assert P_blend_30.shape == P_lr.shape
        assert np.all((P_blend_30 >= 0.0) & (P_blend_30 <= 1.0))

    def test_oracle_per_label_composition(self, tiny_proba_and_labels):
        P_lr, _ = tiny_proba_and_labels
        P_cnb = np.ones_like(P_lr) * 0.9

        lr_better = ["anger", "joy", "sadness"]
        cnb_better = ["disgust", "fear", "surprise"]

        P_oracle = np.zeros_like(P_lr)
        for i, col in enumerate(TARGET_COLS):
            if col in lr_better:
                P_oracle[:, i] = P_lr[:, i]
            else:
                P_oracle[:, i] = P_cnb[:, i]

        for i, col in enumerate(TARGET_COLS):
            if col in lr_better:
                np.testing.assert_array_equal(P_oracle[:, i], P_lr[:, i])
            else:
                np.testing.assert_array_equal(P_oracle[:, i], P_cnb[:, i])


# ─── Regression Checks ───────────────────────────────────────────────────────
class TestRegressionCheck:
    def _check_regressions(self, per_cls):
        return {
            k: round(per_cls[k] - CHAMPION_PER_CLASS[k], 4)
            for k in TARGET_COLS
            if round(per_cls[k] - CHAMPION_PER_CLASS[k], 4) < -REGRESSION_TOLERANCE
        }

    def test_no_regression_if_same_scores(self):
        per_cls = {k: round(v, 4) for k, v in CHAMPION_PER_CLASS.items()}
        assert self._check_regressions(per_cls) == {}

    def test_regression_detected_when_joy_drops(self):
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["joy"] = round(CHAMPION_PER_CLASS["joy"] - 0.02, 4)
        reg = self._check_regressions(per_cls)
        assert "joy" in reg

    def test_no_regression_for_small_drop(self):
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["fear"] = round(CHAMPION_PER_CLASS["fear"] - 0.005, 4)
        reg = self._check_regressions(per_cls)
        assert "fear" not in reg

    def test_regression_at_boundary(self):
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["anger"] = round(CHAMPION_PER_CLASS["anger"] - 0.010, 4)
        reg = self._check_regressions(per_cls)
        assert "anger" not in reg

    def test_regression_just_over_boundary(self):
        per_cls = dict(CHAMPION_PER_CLASS)
        per_cls["anger"] = round(CHAMPION_PER_CLASS["anger"] - 0.011, 4)
        reg = self._check_regressions(per_cls)
        assert "anger" in reg

