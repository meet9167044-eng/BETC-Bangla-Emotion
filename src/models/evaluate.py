"""
src/models/evaluate.py
=======================
BETC — Shared evaluation utilities for Phase 8 (Baselines) and beyond.

Computes all metrics defined in EVALUATION_PROTOCOL.md Section 4:
  - Macro-F1 (primary)
  - Micro-F1
  - Per-class F1, Precision, Recall
  - Hamming Loss
  - Jaccard similarity (averaged)
  - Subset Accuracy (exact match ratio)

Also implements per-class threshold sweep on validation-set probabilities
(PIPELINE_SPEC.md Section 3.7):
  - Sweep tau from 0.05 to 0.95 in steps of 0.01
  - Select threshold that maximizes per-class F1 on validation set

EVALUATION_PROTOCOL.md Leakage Rule:
  - Threshold selection uses ONLY validation-set probabilities.
  - Test-set probabilities must NEVER be used when choosing thresholds.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    jaccard_score,
    accuracy_score,
)

TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, label_names=None) -> dict:
    """
    Compute the full set of multi-label metrics defined in
    EVALUATION_PROTOCOL.md Section 4.

    Parameters
    ----------
    Y_true : np.ndarray, shape (n_samples, n_labels)
    Y_pred : np.ndarray, shape (n_samples, n_labels) — binary 0/1
    label_names : list[str], optional

    Returns
    -------
    dict with keys: macro_f1, micro_f1, hamming_loss, jaccard, subset_accuracy,
                    per_class (dict of {label: {f1, precision, recall}})
    """
    if label_names is None:
        label_names = [f"label_{i}" for i in range(Y_true.shape[1])]

    macro_f1    = float(f1_score(Y_true, Y_pred, average="macro",  zero_division=0))
    micro_f1    = float(f1_score(Y_true, Y_pred, average="micro",  zero_division=0))
    macro_prec  = float(precision_score(Y_true, Y_pred, average="macro", zero_division=0))
    macro_rec   = float(recall_score(Y_true, Y_pred, average="macro",    zero_division=0))
    h_loss      = float(hamming_loss(Y_true, Y_pred))
    jaccard     = float(jaccard_score(Y_true, Y_pred, average="samples", zero_division=0))
    subset_acc  = float(accuracy_score(Y_true, Y_pred))

    per_class_f1   = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(Y_true, Y_pred, average=None, zero_division=0)
    per_class_rec  = recall_score(Y_true, Y_pred, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "f1":        round(float(per_class_f1[i]),   4),
            "precision": round(float(per_class_prec[i]), 4),
            "recall":    round(float(per_class_rec[i]),  4),
        }

    return {
        "macro_f1":      round(macro_f1,   4),
        "micro_f1":      round(micro_f1,   4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall":  round(macro_rec,  4),
        "hamming_loss":  round(h_loss,     6),
        "jaccard":       round(jaccard,    4),
        "subset_accuracy": round(subset_acc, 4),
        "per_class":     per_class,
    }


def optimize_thresholds(
    P_val: np.ndarray,
    Y_val: np.ndarray,
    tau_min: float = 0.05,
    tau_max: float = 0.95,
    tau_step: float = 0.01,
    label_names=None,
) -> dict:
    """
    Per-class threshold optimization on the VALIDATION set only.

    For each label k, sweeps tau from tau_min to tau_max in tau_step increments,
    selects the tau that maximises F1 on Y_val[:,k].

    EVALUATION_PROTOCOL.md Section 3, Rule 2:
    "Per-class thresholds are selected using validation-set predicted
     probabilities ONLY. Test-set probabilities must NEVER be used."

    Parameters
    ----------
    P_val : np.ndarray, shape (n_val, n_labels)  — predicted probabilities
    Y_val : np.ndarray, shape (n_val, n_labels)  — true binary labels
    tau_min, tau_max, tau_step : float            — sweep range

    Returns
    -------
    dict  {label_name: float}  — one threshold per label
    """
    if label_names is None:
        label_names = [f"label_{i}" for i in range(Y_val.shape[1])]

    taus = np.round(np.arange(tau_min, tau_max + tau_step / 2, tau_step), 4)
    thresholds = {}

    for k, name in enumerate(label_names):
        best_tau = 0.5
        best_f1  = -1.0
        for tau in taus:
            y_pred_k = (P_val[:, k] >= tau).astype(int)
            f1 = f1_score(Y_val[:, k], y_pred_k, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_tau = float(tau)
        thresholds[name] = best_tau

    return thresholds


def apply_thresholds(P: np.ndarray, thresholds: dict, label_names=None) -> np.ndarray:
    """
    Convert probability matrix P to binary predictions using per-class thresholds.

    Parameters
    ----------
    P           : np.ndarray, shape (n_samples, n_labels)
    thresholds  : dict {label_name: float}
    label_names : list[str]

    Returns
    -------
    np.ndarray, shape (n_samples, n_labels), dtype int
    """
    if label_names is None:
        label_names = [f"label_{i}" for i in range(P.shape[1])]
    Y_pred = np.zeros_like(P, dtype=int)
    for k, name in enumerate(label_names):
        tau = thresholds.get(name, 0.5)
        Y_pred[:, k] = (P[:, k] >= tau).astype(int)
    return Y_pred
