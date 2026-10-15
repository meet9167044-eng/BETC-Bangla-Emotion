"""
tests/test_phase7_threshold_optimization.py
=============================================
Regression tests for Phase 7 Threshold Optimization utilities.
"""

import numpy as np
import pytest
import sys
import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from scripts.experiments.phase7_threshold_optimization import (
    sweep_thresholds,
    eval_with_thresholds,
    threshold_f1_curve,
)
from src.models.betc import TARGET_COLS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_probs_and_labels(n=200, n_labels=6, seed=42):
    rng = np.random.default_rng(seed)
    Y_true = (rng.random((n, n_labels)) > 0.7).astype(int)
    # Make P well-calibrated: higher prob where true=1
    P = np.clip(Y_true * 0.5 + rng.random((n, n_labels)) * 0.5, 0, 1).astype(np.float64)
    return P, Y_true


# ── sweep_thresholds ──────────────────────────────────────────────────────────

def test_sweep_thresholds_returns_all_labels():
    P, Y = make_probs_and_labels()
    thresholds, best_f1s = sweep_thresholds(P, Y, label_names=TARGET_COLS, tau_step=0.05)
    assert set(thresholds.keys()) == set(TARGET_COLS)
    assert set(best_f1s.keys()) == set(TARGET_COLS)


def test_sweep_thresholds_in_range():
    P, Y = make_probs_and_labels()
    thresholds, _ = sweep_thresholds(P, Y, label_names=TARGET_COLS, tau_min=0.1, tau_max=0.9, tau_step=0.05)
    for lbl, tau in thresholds.items():
        assert 0.09 <= tau <= 0.91, f"{lbl} threshold {tau} out of [0.1, 0.9]"


def test_sweep_thresholds_improves_over_fixed_05():
    """Optimized thresholds should achieve >= F1 of fixed 0.5 on same probabilities."""
    P, Y = make_probs_and_labels(n=500)
    thresholds_opt, _ = sweep_thresholds(P, Y, label_names=TARGET_COLS, tau_step=0.01)
    thresholds_fixed  = {c: 0.5 for c in TARGET_COLS}
    m_opt   = eval_with_thresholds(P, Y, thresholds_opt)
    m_fixed = eval_with_thresholds(P, Y, thresholds_fixed)
    assert m_opt["macro_f1"] >= m_fixed["macro_f1"] - 1e-6, (
        f"Optimized Macro-F1 {m_opt['macro_f1']} < fixed Macro-F1 {m_fixed['macro_f1']}"
    )


def test_sweep_thresholds_finer_step_better_or_equal():
    """Finer step should yield >= Macro-F1 compared to coarser step."""
    P, Y = make_probs_and_labels(n=300)
    thresholds_coarse, _ = sweep_thresholds(P, Y, label_names=TARGET_COLS, tau_step=0.05)
    thresholds_fine,   _ = sweep_thresholds(P, Y, label_names=TARGET_COLS, tau_step=0.005)
    m_coarse = eval_with_thresholds(P, Y, thresholds_coarse)
    m_fine   = eval_with_thresholds(P, Y, thresholds_fine)
    assert m_fine["macro_f1"] >= m_coarse["macro_f1"] - 1e-6


# ── eval_with_thresholds ──────────────────────────────────────────────────────

def test_eval_with_thresholds_keys():
    P, Y = make_probs_and_labels()
    thresholds = {c: 0.5 for c in TARGET_COLS}
    metrics = eval_with_thresholds(P, Y, thresholds)
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "per_class" in metrics
    assert set(metrics["per_class"].keys()) == set(TARGET_COLS)


def test_eval_with_thresholds_all_positive():
    """If threshold is 0 for all, everything is predicted positive."""
    n = 100
    P = np.ones((n, 6)) * 0.5
    Y = np.ones((n, 6), dtype=int)
    thresholds = {c: 0.0 for c in TARGET_COLS}
    metrics = eval_with_thresholds(P, Y, thresholds)
    assert metrics["macro_f1"] == pytest.approx(1.0, abs=1e-4)


def test_eval_with_thresholds_all_negative():
    """If threshold is 1.0 for all, everything is predicted negative → F1=0."""
    n = 100
    P = np.ones((n, 6)) * 0.9
    Y = np.ones((n, 6), dtype=int)
    thresholds = {c: 1.0 for c in TARGET_COLS}
    metrics = eval_with_thresholds(P, Y, thresholds)
    assert metrics["macro_f1"] == pytest.approx(0.0, abs=1e-4)


# ── threshold_f1_curve ────────────────────────────────────────────────────────

def test_threshold_f1_curve_output():
    P, Y = make_probs_and_labels(n=200)
    tau_range = np.round(np.arange(0.1, 0.9, 0.1), 3)
    curve = threshold_f1_curve(P, Y, "anger", tau_range)
    assert len(curve) == len(tau_range)
    for tau, f1 in curve:
        assert 0.0 <= tau <= 1.0
        assert 0.0 <= f1 <= 1.0


def test_threshold_f1_curve_peak_is_maximum():
    P, Y = make_probs_and_labels(n=300)
    tau_range = np.round(np.arange(0.05, 0.95, 0.01), 5)
    curve = threshold_f1_curve(P, Y, "joy", tau_range)
    peak  = max(curve, key=lambda x: x[1])
    all_f1s = [x[1] for x in curve]
    assert peak[1] == max(all_f1s)
